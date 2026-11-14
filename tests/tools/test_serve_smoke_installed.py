"""Installed smoke admission/commands on synthetic package files; stdlib, no server or GPU.

python -I tests/tools/test_serve_smoke_installed.py --scratch DIR

The package fixture loads the real config/kernel helpers from scratch copies.
Its code-object bytes are deliberately nonexecutable. Distribution discovery,
CLI dispatch, Popen and reader threads are mocked; no sockets or installs.
"""
import argparse
import builtins
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import serve_smoke1 as smoke
import serve_smoke_installed as installed
from release import stage_kernels

SCRATCH = None
TREE = 'a' * 64


class Commands(unittest.TestCase):
    def args(self, **over):
        return NS(**dict(dict(model='synthetic model.gguf', max_ctx=8192, scratch=str(SCRATCH / 'server arguments'),
                              alias='synthetic-alias', cards=2, draft='mtp', card='1,0', block_imports=''), **over))

    def serve_args(self, a):
        args = ['--port', '0', '--model', a.model, '--max-ctx', str(a.max_ctx),
                '--cache-dir', str(Path(a.scratch) / 'kv'), '--flight-dir', str(Path(a.scratch) / 'flight')]
        if a.alias:
            args += ['--model-alias', a.alias]
        args += ['--cards', str(a.cards)]
        if a.draft:
            args += ['--draft', a.draft]
        return args

    def test_checkout_command_and_blocked_wrapper_are_unchanged(self):
        for block in ('', 'gguf,tokenizers'):
            with self.subTest(block=block), patch.object(smoke.S, 'INSTALLED', False):
                a = self.args(block_imports=block)
                expected = [sys.executable, '-I', str(ROOT / 'tools/caminus_serve.py'), *self.serve_args(a)]
                if block:
                    expected = expected[:2] + ['-c', smoke.BLOCKER.format(block=('gguf', 'tokenizers')), expected[2]] + expected[3:]
                self.assertEqual(smoke.server_command(a), expected)

    def test_installed_command_is_isolated_and_preserves_all_serve_arguments(self):
        for options in ({}, dict(alias=None, draft=None, cards=1, max_ctx='auto'), dict(draft='off'), dict(draft='dflash')):
            with self.subTest(options=options), patch.object(smoke.S, 'INSTALLED', True):
                a = self.args(**options)
                cmd = smoke.server_command(a)
                self.assertEqual(cmd, [sys.executable, '-I', str(ROOT / 'tools/serve_smoke_installed.py'), '--', *self.serve_args(a)])
                self.assertNotIn('-c', cmd)
                self.assertFalse(any('sys.path' in x for x in cmd))

    def test_installed_own_gate_passes_only_the_audit_digest(self):
        digest = stage_kernels.kernel_tree_sha256(smoke.ROOT)
        with patch.object(smoke.S, 'INSTALLED', True):
            a = self.args(block_imports='gguf,tokenizers')
            cmd = smoke.server_command(a)
        self.assertEqual(cmd, [sys.executable, '-I', str(ROOT / 'tools/serve_smoke_installed.py'),
                               '--block-imports', 'gguf,tokenizers', '--kernel-tree-sha256', digest, '--', *self.serve_args(a)])
        self.assertFalse(any('sys.path' in x for x in cmd))

    def test_both_server_and_restart_popen_use_the_helper_and_visibility(self):
        a = self.args(block_imports='gguf,tokenizers')
        expected = [sys.executable, '-I', str(ROOT / 'tools/serve_smoke_installed.py'), '--block-imports', a.block_imports,
                    '--kernel-tree-sha256', stage_kernels.kernel_tree_sha256(smoke.ROOT), '--', *self.serve_args(a)]
        processes = [NS(stdout=[]), NS(stdout=[])]
        with patch.object(smoke.S, 'INSTALLED', True), patch.object(smoke.S, 'LIVE', []), \
                patch.object(smoke.subprocess, 'Popen', side_effect=processes) as popen, \
                patch.object(smoke.threading, 'Thread') as thread, \
                patch.object(smoke, 'server_command', wraps=smoke.server_command) as command, \
                contextlib.redirect_stdout(io.StringIO()):
            first, restart = smoke.Server1('server', a), smoke.Server1('restart', a)
            self.assertEqual(smoke.S.LIVE, [first, restart])
        self.assertEqual(command.call_count, 2)
        self.assertEqual(popen.call_count, 2)
        self.assertIs(first.p, processes[0])
        self.assertIs(restart.p, processes[1])
        for call in popen.call_args_list:
            cmd, = call.args
            self.assertEqual(cmd, expected)
            self.assertEqual(call.kwargs['env']['HIP_VISIBLE_DEVICES'], '1,0')
            self.assertEqual(call.kwargs['env']['PYTHONUNBUFFERED'], '1')
            self.assertEqual(call.kwargs['stdout'], smoke.subprocess.PIPE)
            self.assertEqual(call.kwargs['stderr'], smoke.subprocess.STDOUT)
        self.assertEqual(thread.call_count, 2)
        self.assertEqual(thread.return_value.start.call_count, 2)   # Mock.start never creates a thread.


class Preflight(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.home = SCRATCH / self._testMethodName
        self.pkg = self.home / 'site' / 'caminus'
        (self.pkg / 'device').mkdir(parents=True)
        for rel in ('__init__.py', 'device/__init__.py', '__main__.py'):
            (self.pkg / rel).write_text('# synthetic package fixture; never a server\n')
        self.stack.enter_context(patch.dict(os.environ, dict(HOME=str(self.home), CAMINUS_GUARD='synthetic-outer-guard',
                                                            CAMINUS_CONFIG=str(self.home / 'config.toml')), clear=True))
        # Restore every prior module and finder, including any checkout modules
        # imported by the command-building tests, without touching real settings.
        self.stack.enter_context(patch.dict(sys.modules))
        for name in list(sys.modules):
            if name.split('.')[0] in ('caminus', 'gguf', 'tokenizers'):
                del sys.modules[name]
        old_meta = list(sys.meta_path)
        self.stack.callback(lambda: sys.meta_path.__setitem__(slice(None), old_meta))
        self.cam = self.package('caminus', self.pkg)
        self.device = self.package('caminus.device', self.pkg / 'device')
        for name, rel in (('caminus.config', 'config.py'), ('caminus.device.kernels', 'device/kernels.py')):
            path = self.pkg / rel
            path.write_text((ROOT / 'python/caminus' / rel).read_text())
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if name.endswith('.config'):
                self.config = module
                self.cam.config = module
            else:
                self.kernels = module
                self.device.kernels = module
        self.kdir = Path(self.kernels.PACKAGE_DIR)
        self.kdir.mkdir(parents=True)
        objects = {}
        for name in self.kernels.RUNTIME_SET:
            file = name + '.hsaco'
            data = b'NONEXECUTABLE installed-package CPU fixture: ' + name.encode()
            (self.kdir / file).write_bytes(data)
            objects[file] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        self.prov = dict(objects=objects, source=dict(kernel_tree_sha256=TREE))
        self.write_provenance()
        self.direct = None
        self.recorded = self.pkg / '__init__.py'
        self.dist = NS(version='0.0.synthetic', read_text=lambda name: self.direct,
                       locate_file=lambda name: self.recorded)
        self.distribution = self.stack.enter_context(patch.object(installed.importlib.metadata, 'distribution', return_value=self.dist))
        self.entry = ModuleType('caminus.__main__')
        self.entry.__file__ = str(self.pkg / '__main__.py')
        self.dispatch = self.entry.main = Mock(return_value=17)
        sys.modules['caminus.__main__'] = self.entry
        self.out, self.err = io.StringIO(), io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.out))
        self.stack.enter_context(contextlib.redirect_stderr(self.err))

    def package(self, name, path):
        module = ModuleType(name)
        module.__file__ = str(path / '__init__.py')
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return module

    def write_provenance(self, value=None):
        (self.kdir / self.kernels.PROVENANCE).write_text(json.dumps(self.prov if value is None else value))

    def refused(self, message, expected=TREE):
        with self.assertRaisesRegex(installed.PreflightError, message):
            installed.preflight(expected)
        self.dispatch.assert_not_called()

    def test_packaged_files_are_verified_by_real_helpers(self):
        self.assertIsNone(self.kernels.checkout_build())
        self.assertEqual(self.kernels.kernel_dir(), (str(self.kdir), 'package'))
        prov, checks = self.kernels.verify()
        self.assertEqual(prov, self.prov)
        self.assertEqual(len(checks), len(self.kernels.RUNTIME_SET))
        self.assertTrue(all(ok for _, ok, _ in checks))
        installed.preflight(TREE)
        self.distribution.assert_called_once_with('caminus')
        self.assertIn('packaged kernel objects verified', self.out.getvalue())

    def test_outer_guard_is_required_before_import_or_dispatch(self):
        os.environ.pop('CAMINUS_GUARD')
        self.refused('outer guard is required')
        self.distribution.assert_not_called()

    def test_editable_and_invalid_direct_url_metadata_are_refused(self):
        for value in ('{"dir_info":{"editable":true}}', '{broken', '[]', 'null', '{"dir_info":null}'):
            with self.subTest(value=value):
                self.direct = value
                self.refused('editable|invalid installed direct_url')

    def test_noneditable_direct_url_is_accepted(self):
        self.direct = '{"dir_info":{"editable":false}}'
        installed.preflight(TREE)

    def test_distribution_and_import_origin_must_match(self):
        self.recorded = self.home / 'different-install' / 'caminus' / '__init__.py'
        self.refused('does not match its installed distribution')

    def test_source_checkout_is_refused_even_with_matching_distribution(self):
        (self.home / 'build.sh').write_text('# synthetic checkout marker\n')
        (self.home / 'kernels').mkdir()
        self.refused('source checkout')

    def test_environment_and_config_kernel_overrides_are_refused(self):
        # Even an override naming the same physical package directory is refused.
        with patch.dict(os.environ, CAMINUS_KERNEL_DIR=str(self.kdir)):
            self.refused('remove kernel_dir overrides')
        Path(os.environ['CAMINUS_CONFIG']).write_text('kernel_dir = ' + json.dumps(str(self.kdir)) + '\n')
        self.config._FILE.clear()
        self.refused('remove kernel_dir overrides')

    def test_missing_and_empty_provenance_are_refused(self):
        (self.kdir / self.kernels.PROVENANCE).unlink()
        self.refused('provenance is missing or fails')
        self.write_provenance({})
        self.refused('provenance is missing or fails')
        with patch.object(self.kernels, 'verify', return_value=(self.prov, [])):
            self.refused('provenance is missing or fails')

    def test_missing_and_corrupt_object_bytes_fail_verification(self):
        path = self.kdir / (self.kernels.RUNTIME_SET[0] + '.hsaco')
        original = path.read_bytes()
        path.unlink()
        self.refused('provenance is missing or fails')
        path.write_bytes(original + b'corrupt')
        self.refused('provenance is missing or fails')

    def test_unlisted_physical_object_cannot_pass_provenance_verification(self):
        (self.kdir / 'unlisted.hsaco').write_bytes(b'NONEXECUTABLE extra CPU fixture')
        self.assertTrue(all(ok for _, ok, _ in self.kernels.verify()[1]))
        self.refused('must match exactly')

    def test_extra_manifest_object_cannot_expand_the_runtime_set(self):
        data = b'NONEXECUTABLE extra CPU fixture'
        (self.kdir / 'extra.hsaco').write_bytes(data)
        self.prov['objects']['extra.hsaco'] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        self.write_provenance()
        self.assertTrue(all(ok for _, ok, _ in self.kernels.verify()[1]))
        self.refused('must match exactly')

    def test_new_runtime_requirement_absent_from_manifest_is_refused(self):
        self.kernels.RUNTIME_SET += ('new-runtime-object',)
        self.refused('provenance is missing or fails')

    def test_expected_audit_digest_must_be_well_formed_and_match(self):
        for digest in ('A' * 64, 'a' * 63, 'not-a-digest', ' ' + TREE):
            with self.subTest(digest=digest):
                self.refused('lowercase SHA256', expected=digest)
        self.refused('do not match the checkout sources', expected='b' * 64)

    def test_own_gate_refuses_missing_or_malformed_manifest_source_digest(self):
        for digest in (None, 'not-a-digest', 'B' * 64, 'b' * 64):
            with self.subTest(digest=digest):
                self.prov['source']['kernel_tree_sha256'] = digest
                self.write_provenance()
                self.assertEqual(installed.main(['--block-imports', 'gguf', '--kernel-tree-sha256', TREE, '--', '--help']), 2)
        self.dispatch.assert_not_called()

    def test_main_dispatches_installed_cli_with_import_hook_and_exact_args(self):
        args = ['--model', 'synthetic model.gguf', '--cards', '2', '--draft', 'off', '--model-alias', 'synthetic-alias',
                '--cache-dir', 'scratch-kv', '--flight-dir', 'scratch-flight']

        def dispatch(argv):
            self.assertTrue(any(isinstance(f, installed.ImportBlocker) and f.names == {'gguf', 'tokenizers'} for f in sys.meta_path))
            self.assertEqual(argv, ['serve', *args])
            return 17

        self.dispatch.side_effect = dispatch
        before = list(sys.path)
        rc = installed.main(['--block-imports', 'gguf,tokenizers', '--kernel-tree-sha256', TREE, '--', *args])
        self.assertEqual(rc, 17)
        self.assertEqual(sys.path, before)
        self.dispatch.assert_called_once_with(['serve', *args])

    def test_forced_submodule_import_is_denied_during_installed_dispatch(self):
        with self.assertRaisesRegex(ImportError, 'blocked by serve_smoke1: gguf.sub'):
            installed.ImportBlocker(('gguf',)).find_spec('gguf.sub')
        # The parent is first imported during dispatch, after the hook exists.
        # A preloaded root is a separate refusal, since it bypasses find_spec.
        gguf_path = self.home / 'synthetic-imports' / 'gguf'
        gguf_path.mkdir(parents=True)
        (gguf_path / '__init__.py').write_text('# synthetic forbidden package\n')
        self.stack.enter_context(patch.object(sys, 'path', [str(gguf_path.parent), *sys.path]))
        self.dispatch.side_effect = lambda argv: importlib.import_module('gguf.sub')
        with self.assertRaisesRegex(ImportError, 'blocked by serve_smoke1: gguf'):
            installed.main(['--block-imports', 'gguf', '--kernel-tree-sha256', TREE, '--', '--help'])
        self.assertIn('import refused: gguf', self.out.getvalue())
        self.dispatch.assert_called_once_with(['serve', '--help'])

    def test_preloaded_forbidden_package_is_refused_before_cli_dispatch(self):
        # Import finders cannot block modules already cached by site startup.
        self.package('gguf', self.home / 'synthetic-preloaded-gguf')
        self.assertEqual(installed.main(['--block-imports', 'gguf', '--kernel-tree-sha256', TREE, '--', '--help']), 2)
        self.assertIn('already imported before the smoke hook', self.err.getvalue())
        self.dispatch.assert_not_called()

    def test_own_gate_without_digest_or_invalid_block_names_never_dispatches(self):
        for value in ('gguf', ',', ' ', 'gguf.sub', 'bad-name'):
            with self.subTest(value=value):
                self.assertEqual(installed.main(['--block-imports', value, '--', '--help']), 2)
        self.dispatch.assert_not_called()
        self.distribution.assert_not_called()

    def test_bad_preflight_has_no_cli_import_or_checkout_fallback(self):
        (self.kdir / self.kernels.PROVENANCE).write_text('{malformed')
        imports = []
        original = builtins.__import__

        def track(name, *args, **kwargs):
            imports.append(name)
            if name == 'caminus.__main__' or name.startswith(('caminus.cmd.', 'caminus.serve.')):
                raise AssertionError('CLI import attempted after failed preflight')
            return original(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=track):
            self.assertEqual(installed.main(['--', '--help']), 2)
        self.assertNotIn('caminus.__main__', imports)
        self.assertIn('installed preflight refused', self.err.getvalue())
        self.dispatch.assert_not_called()

    def test_missing_distribution_refuses_without_cli_dispatch(self):
        self.distribution.side_effect = installed.importlib.metadata.PackageNotFoundError('synthetic missing distribution')
        self.assertEqual(installed.main(['--', '--help']), 2)
        self.assertIn('must be installed', self.err.getvalue())
        self.dispatch.assert_not_called()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scratch', required=True)
    args, rest = ap.parse_known_args()
    SCRATCH = Path(args.scratch).resolve()
    if SCRATCH.exists() and (not SCRATCH.is_dir() or any(SCRATCH.iterdir())):
        ap.error('--scratch must be fresh/empty')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    result = unittest.main(argv=[sys.argv[0]] + rest, exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(not result.wasSuccessful())
