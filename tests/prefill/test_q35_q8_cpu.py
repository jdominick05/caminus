"""Dense Q8 prefill construction, operation ordering and failure drains (CPU only).

Synthetic shape/format descriptors exercise the real constructor and chunk
driver. Tensor storage stays on CPU; launchers are recorded, never executed.
These checks establish no device arithmetic, model quality or speed result.
"""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.device import hip
from caminus.errors import InvalidKVEncoding, NonFiniteLogits
from caminus.engine import k3
from caminus.prefill import attn, gdn, glue, q35_kvq8, q35own


class Module:
    loaded = []

    def __init__(self, name):
        self.name = name
        self.loaded.append(name)

    def __getattr__(self, name):
        raise AssertionError('device function requested in CPU integration test: ' + name)


def target(line='swift', q8=True):
    m = NS(D=5120, ffn=512, nh=6, nkv=1, hd=256, S=128, NK=1, NV=2, nl=2, nl_build=2,
           dev=0, max_ctx=64, kv_rows=67, q8=q8, tcap=line != 'swift', vocab=64, nrot=64,
           theta_scale=0.9, eps=1e-6, broken=None, poisoned=None, fresh=False)
    m.module = 'k3_q35_' + line + ('_kvq8' if q8 else '')
    m.emb = NS(type='Q4_K' if line == 'swift' else 'IQ2_S')
    m.emb_dev = torch.zeros(1, dtype=torch.uint8)
    m.out_norm = torch.ones(m.D)
    fmt = {'swift': 'Q4_K', 'tcap': 'IQ3_S', 'gsq': 'Q2_K'}[line]
    m.L = []
    for _ in range(m.nl):
        row = dict(kind='attn', attn_norm=torch.ones(m.D), post_norm=torch.ones(m.D),
                   kn=torch.ones(m.hd), qn=torch.ones(m.hd))
        row.update({name: dict(fmt=fmt) for name in ('q', 'k', 'v', 'o', 'fg', 'fu', 'fd')})
        size = m.kv_rows * k3.kv_pos_bytes(m.nkv * m.hd, q8)
        row['cache'] = torch.zeros(size if q8 else size // 2, dtype=torch.uint8 if q8 else torch.float16)
        m.L.append(row)

    def broken(why, error_cls=NonFiniteLogits):
        m.broken, m._broken_error = why, error_cls
        raise error_cls(why)

    m.mark_broken = broken
    return m


class Record:
    def __init__(self, name, calls, real=None):
        self.name, self.calls, self.real = name, calls, real

    def __getattr__(self, name):
        if name == 'attn_blocks':
            return self.real.attn_blocks
        def call(*args, **kwargs):
            self.calls.append((self.name + '.' + name, args, kwargs))
        return call


class DenseQ8Prefill(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        zeros = torch.zeros
        self.stack.enter_context(patch('torch.zeros', side_effect=lambda *a, **k:
            zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})))
        self.stack.enter_context(patch('torch.cuda.device', side_effect=lambda *a: contextlib.nullcontext()))
        self.stack.enter_context(patch('torch.cuda.get_device_properties', return_value=NS(multi_processor_count=42)))
        self.stack.enter_context(patch('torch.cuda.Event', side_effect=lambda: NS(record=lambda: None, synchronize=lambda: None)))
        self.sync = self.stack.enter_context(patch('torch.cuda.synchronize'))
        self.stack.enter_context(patch('torch.cuda.is_available', return_value=False))
        for name in ('init', '_lazy_init'):
            self.stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError('GPU initialization forbidden')))
        for owner in (hip, attn, gdn, glue, q35_kvq8):
            self.stack.enter_context(patch.object(owner, 'Module', Module))
        self.cap = self.stack.enter_context(patch.object(q35own.capacity, 'check'))
        Module.loaded = []

    def prefill(self, line='swift', q8=True, export_h=True):
        return q35own.Q35OwnPrefill(target(line, q8), chunk=16, export_h=export_h, feature_layers=(1,))

    def record(self, pf):
        calls = []
        for name in ('O', 'A', 'Gl', 'Gd', 'KV'):
            if getattr(pf, name, None) is not None:
                setattr(pf, name, Record(name, calls, getattr(pf, name)))
        return calls

    def test_constructor_keeps_weight_modules_and_projects_existing_buffers_for_all_lines(self):
        for line in ('swift', 'tcap', 'gsq'):
            with self.subTest(line=line):
                Module.loaded = []
                pf = self.prefill(line)
                expected = {'swift': 'q35_prefill', 'tcap': 'q35_prefill_tcap', 'gsq': 'q35_prefill_gsq'}[line]
                self.assertEqual(pf.O.module, expected)
                self.assertEqual(pf.KV.mod.name, 'q35_prefill_kvq8')
                self.assertEqual(Module.loaded.count('q35_prefill_kvq8'), 1)
                self.assertEqual(pf.need_mib, q35own.buffer_need_mib(pf.m, 16, export_h=True, nfeat=1))
                self.assertEqual(self.cap.call_args.args[1], [pf.need_mib])
                self.assertEqual(pf.features.shape, (1, 16, 5120))
                self.assertIn('row-fp32', pf.numerics()['dense_kv'])
        old = self.prefill(q8=False)
        self.assertIsNone(old.KV)
        self.assertNotIn('dense_kv', old.numerics())

    def test_capacity_refusal_precedes_module_load_or_prefill_buffers(self):
        self.cap.side_effect = ValueError('synthetic no headroom')
        with self.assertRaisesRegex(ValueError, 'no headroom'):
            self.prefill()
        self.assertEqual(Module.loaded, [])

    def test_real_chunk_rotates_before_append_and_restores_output_before_channel_gate(self):
        pf = self.prefill()
        calls = self.record(pf)
        pf._chunk([1, 2, 3], 0, 3)
        names = [name for name, _, _ in calls]
        rotations = [(i, args[1:]) for i, (name, args, _) in enumerate(calls) if name == 'A.hadamard']
        self.assertEqual([args for _, args in rotations], [(256, 3), (64, 12), (256, 18), (64, 72)] * 2)
        appends = [i for i, name in enumerate(names) if name == 'KV.append']
        gates = [i for i, name in enumerate(names) if name == 'O.gate_mul16']
        for layer in range(2):
            r = [i for i, _ in rotations[layer * 4:layer * 4 + 4]]
            self.assertLess(r[0], r[1]); self.assertLess(r[1], appends[layer])
            self.assertLess(appends[layer], r[2]); self.assertLess(r[2], r[3]); self.assertLess(r[3], gates[layer])
        self.assertNotIn('O.kv_append16', names)
        self.assertNotIn('O.attn16', names)
        self.assertNotIn('O.attn_wm', names)

    def test_last_cache_only_layer_keeps_kv_rotations_without_query_or_output(self):
        pf = self.prefill(export_h=False)
        calls = self.record(pf)
        pf._chunk([1, 2], 0, 2)
        self.assertEqual(sum(name == 'A.hadamard' for name, _, _ in calls), 6)
        self.assertEqual(sum(name == 'KV.append' for name, _, _ in calls), 2)
        self.assertEqual(sum(name == 'KV.attn_wm' for name, _, _ in calls), 1)

    def test_numerical_flags_win_over_bounds_and_block_handoff_checkpoint_and_draft_exports(self):
        for kind in ('mtp', 'dflash35'):
            with self.subTest(kind=kind):
                pf = self.prefill()
                pf._chunk, pf._handoff = Mock(), Mock()
                dr = NS(kind=kind, fill_chunk=Mock(), inject_chunk=Mock(), d=NS(layers_tgt=[1]))
                at_stop = Mock()
                self.sync.side_effect = lambda *a: pf.err.fill_(3)
                with self.assertRaises(InvalidKVEncoding):
                    pf.run([1] * 8, 0, 0, 8, draft=dr, stops=[4], at_stop=at_stop)
                self.assertEqual(int(pf.err.item()), 3)
                self.assertTrue(pf.m.broken)
                self.assertIsNone(pf.m.poisoned)
                dr.fill_chunk.assert_not_called(); dr.inject_chunk.assert_not_called()
                pf._handoff.assert_not_called(); at_stop.assert_not_called()
                with self.assertRaises(InvalidKVEncoding):
                    pf.run([1] * 8, 0, 0, 8, validated=True)

    def test_error_discovered_during_host_exception_drain_supersedes_recovery(self):
        pf = self.prefill()
        pf._handoff = Mock()
        def host_error(*args):
            pf.err.fill_(2)
            raise OSError('synthetic host failure after launch')
        pf._chunk = host_error
        with self.assertRaises(InvalidKVEncoding):
            pf.run([1] * 4, 0, 0, 4)
        self.assertIsNone(pf.m.poisoned)
        pf._handoff.assert_not_called()
        self.assertEqual(int(pf.err.item()), 2)

    def test_successful_chunk_checks_before_shifted_mtp_or_dflash_exports(self):
        for kind in ('mtp', 'dflash35'):
            pf = self.prefill()
            order = []
            pf._chunk = lambda *args: order.append('chunk')
            self.sync.side_effect = lambda *args: order.append('wait')
            dr = NS(kind=kind, d=NS(layers_tgt=[1]),
                    fill_chunk=Mock(side_effect=lambda *a: order.append('mtp')),
                    inject_chunk=Mock(side_effect=lambda *a: order.append('dflash')))
            pf.check_drafter(dr)
            pf._run_chunk([10, 11, 12, 13], 0, 0, 0, 4, [], [])
            self.assertEqual(order, ['chunk', 'wait', 'mtp' if kind == 'mtp' else 'dflash'])
            if kind == 'mtp':
                self.assertEqual(dr.fill_chunk.call_args.args[1:], ([11, 12, 13, 13], 1))

    def test_terminal_engine_rejected_before_constructor_allocation_and_validated_run(self):
        m = target()
        m.broken, m._broken_error = 'synthetic terminal state', InvalidKVEncoding
        with self.assertRaises(InvalidKVEncoding):
            q35own.Q35OwnPrefill(m, chunk=16)
        self.assertEqual(Module.loaded, [])
        self.cap.assert_not_called()
        pf = self.prefill()
        pf.m.broken, pf.m._broken_error = m.broken, InvalidKVEncoding
        pf._run_start = Mock()
        with self.assertRaises(InvalidKVEncoding):
            pf.run([1] * 4, 0, 0, 4, validated=True)
        pf._run_start.assert_not_called()

    def test_original_numerical_failure_keeps_type_and_bypasses_ordinary_poisoning(self):
        pf = self.prefill(q8=False)
        error = NonFiniteLogits('synthetic nonfinite state')
        pf._chunk = Mock(side_effect=error)
        pf._run_failure_check = Mock()
        with self.assertRaises(NonFiniteLogits) as caught:
            pf.run([1] * 4, 0, 0, 4)
        self.assertIs(caught.exception, error)
        self.assertIsNone(pf.m.poisoned)
        pf._run_failure_check.assert_not_called()

    def test_ordinary_host_failure_still_poisoned_and_bounds_flag_still_cleared(self):
        for q8 in (False, True):
            pf = self.prefill(q8=q8)
            pf._chunk = Mock(side_effect=OSError('synthetic ordinary host failure'))
            with self.assertRaises(OSError):
                pf.run([1] * 4, 0, 0, 4)
            self.assertIn('OSError', pf.m.poisoned)
            pf.err.fill_(1)
            with self.assertRaises(k3.ContextBoundsError):
                pf.check_flags()
            self.assertEqual(int(pf.err.item()), 0)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(0 if result.wasSuccessful() else 1)
