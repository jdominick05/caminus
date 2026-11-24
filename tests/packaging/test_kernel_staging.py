"""CPU packaging flow with synthetic, deliberately nonexecutable kernel bytes.

python -I tests/packaging/test_kernel_staging.py --scratch DIR

Exercise the real stager, provenance verifier and setup.py's check_staged guard
without compiling, importing torch, initializing HIP, building or installing a
wheel. These fixtures establish packaging behavior, never GPU executability.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.device import kernels as K

spec = importlib.util.spec_from_file_location('stage_kernels_fixture', ROOT / 'tools/release/stage_kernels.py')
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)
SCRATCH = None


def setup_guard(staged):
    # Compile the real guard function without running setuptools' CLI frontend.
    # The guard itself imports this checkout's registry and inspects our fixture.
    tree = ast.parse((ROOT / 'setup.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_staged')
    scope = dict(os=os, sys=sys, hashlib=hashlib, json=json, HERE=str(ROOT), STAGED=str(staged))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / 'setup.py'), 'exec'), scope)
    scope['check_staged']()


class Staging(unittest.TestCase):
    def setUp(self):
        self.home = SCRATCH / self._testMethodName
        self.build, self.out = self.home / 'build', self.home / 'staged'
        self.build.mkdir(parents=True)
        self.info = dict(built='2026-11-13T00:00:00Z', arch=K.ARCH, toolchain='synthetic CPU fixture',
                         toolchain_version='fixture', compiler='not a compiler', source_commit='0' * 40,
                         source_dirty='0', kernel_tree_sha256=stager.kernel_tree_sha256(str(ROOT)),
                         ggml_commit='0' * 40, ggml_dirty='0')
        self.commands = {}
        for name in K.RUNTIME_SET + ('dec_q3m_trace', 'dec_q35_swift_trace'):
            file = name + '.hsaco'
            (self.build / file).write_bytes(b'NOT EXECUTABLE: synthetic CPU packaging fixture ' + name.encode())
            self.commands[file] = f'<ROCM_PATH>/bin/amdclang++ SYNTHETIC {name}'
        self.write_metadata()

    def write_metadata(self):
        (self.build / 'BUILD-INFO').write_text(''.join(f'{k}\t{v}\n' for k, v in self.info.items()))
        (self.build / 'BUILD-CMDS.tsv').write_text(''.join(f'{k}\t{v}\n' for k, v in self.commands.items()))

    def stage(self, *args):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'tools/release/stage_kernels.py'),
                               '--build', str(self.build), '--out', str(self.out), *args],
                              text=True, capture_output=True, timeout=30,
                              env=dict(os.environ, HIP_VISIBLE_DEVICES='', SOURCE_DATE_EPOCH='1794528000'))

    def staged(self):
        result = self.stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads((self.out / K.PROVENANCE).read_text())

    def refused_before_output(self, message):
        result = self.stage()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.out.exists(), 'a refused input must not publish a partial runtime set')

    def test_complete_set_copies_bytes_commands_and_verifies_without_experimental_objects(self):
        prov = self.staged()
        expected = {name + '.hsaco' for name in K.RUNTIME_SET}
        self.assertEqual(set(prov['objects']), expected)
        self.assertEqual({p.name for p in self.out.glob('*.hsaco')}, expected)
        for file, record in prov['objects'].items():
            self.assertEqual((self.out / file).read_bytes(), (self.build / file).read_bytes())
            self.assertEqual(record['command'], self.commands[file])
            self.assertEqual(record['bytes'], (self.out / file).stat().st_size)
        self.assertEqual(set(prov['not_shipped']), {'dec_q3m_trace.hsaco', 'dec_q35_swift_trace.hsaco'})
        _, checks = K.verify(str(self.out))
        self.assertTrue(checks and all(ok for _, ok, _ in checks))
        setup_guard(self.out)

    def test_old_runtime_only_build_is_refused_before_staging(self):
        legacy = set(K.K3 + K.PREFILL + ('samp_topk', 'smoke'))
        for name in set(K.RUNTIME_SET) - legacy:
            (self.build / (name + '.hsaco')).unlink()
        self.refused_before_output('missing objects or commands')

    def test_missing_new_decode_object_is_refused_before_staging(self):
        (self.build / 'dec_q3m.hsaco').unlink()
        self.refused_before_output('dec_q3m.hsaco')

    def test_missing_drafter_compile_command_is_refused_before_staging(self):
        del self.commands['dec_q35_tcap_mtp.hsaco']
        self.write_metadata()
        self.refused_before_output('dec_q35_tcap_mtp.hsaco')

    def test_each_missing_gsq_draft_object_is_refused_before_staging(self):
        for name in ('dec_q35_gsq_ver', 'dec_q35_gsq_ver_m', 'dec_q35_gsq_mtp', 'dec_q35_gsq_dflash'):
            with self.subTest(module=name):
                path = self.build / (name + '.hsaco')
                saved = path.read_bytes()
                path.unlink()
                try:
                    self.refused_before_output(name + '.hsaco')
                finally:
                    path.write_bytes(saved)

    def test_otherwise_complete_pre_gsq_manifest_is_refused(self):
        prov = self.staged()
        missing = {'dec_q35_gsq_ver.hsaco', 'dec_q35_gsq_ver_m.hsaco',
                   'dec_q35_gsq_mtp.hsaco', 'dec_q35_gsq_dflash.hsaco'}
        for file in missing:
            (self.out / file).unlink()
            del prov['objects'][file]
        (self.out / K.PROVENANCE).write_text(json.dumps(prov))
        _, checks = K.verify(str(self.out))
        self.assertEqual({file for file, ok, _ in checks if not ok}, missing)
        with self.assertRaisesRegex(SystemExit, 'staged code object .* missing'):
            setup_guard(self.out)

    def test_wrong_kernel_source_hash_is_refused_before_staging(self):
        self.info['kernel_tree_sha256'] = 'f' * 64
        self.write_metadata()
        self.refused_before_output('not from these sources')

    def test_dirty_build_refused_and_explicit_allow_dirty_recorded(self):
        self.info['source_dirty'] = '1'
        self.write_metadata()
        self.refused_before_output('dirty')
        result = self.stage('--allow-dirty')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIs(json.loads((self.out / K.PROVENANCE).read_text())['source']['dirty'], True)

    def test_provenance_and_setup_find_corrupt_or_missing_own_objects(self):
        self.staged()
        (self.out / 'dec_q35_swift.hsaco').write_bytes(b'corrupt')
        (self.out / 'pf_q3m.hsaco').unlink()
        _, checks = K.verify(str(self.out))
        self.assertEqual({file for file, ok, _ in checks if not ok}, {'dec_q35_swift.hsaco', 'pf_q3m.hsaco'})
        with self.assertRaisesRegex(SystemExit, 'does not match|missing'):
            setup_guard(self.out)

    def test_setup_refuses_old_runtime_manifest_with_otherwise_matching_files(self):
        prov = self.staged()
        for name in set(K.RUNTIME_SET) - set(K.K3 + K.PREFILL + ('samp_topk', 'smoke')):
            file = name + '.hsaco'
            (self.out / file).unlink()
            del prov['objects'][file]
        (self.out / K.PROVENANCE).write_text(json.dumps(prov))
        with self.assertRaisesRegex(SystemExit, 'staged code object .* missing'):
            setup_guard(self.out)

    def test_extra_staged_object_refused_and_existing_provenance_not_replaced(self):
        self.staged()
        old = (self.out / K.PROVENANCE).read_bytes()
        (self.out / 'dec_q35_swift_trace.hsaco').write_bytes(b'extra')
        result = self.stage()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn('outside the runtime set', result.stderr)
        self.assertEqual((self.out / K.PROVENANCE).read_bytes(), old)
        with self.assertRaisesRegex(SystemExit, 'outside the runtime set'):
            setup_guard(self.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scratch', required=True)
    args, rest = ap.parse_known_args()
    SCRATCH = Path(args.scratch).resolve()
    if SCRATCH.exists() and any(SCRATCH.iterdir()):
        ap.error('--scratch must be fresh/empty')
    result = unittest.main(argv=[sys.argv[0]] + rest, exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(not result.wasSuccessful())
