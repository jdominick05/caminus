"""Two-card Q8 target/verify allocation, tapes and terminal failures, CPU only.

Real constructors read synthetic GGUFs and build their tapes. GPU initialization
and HIP loading are forbidden; no device arithmetic or model quality is proved.
"""
import argparse
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_q35
from caminus.cache.prompt import Layout, PromptCache
from caminus.device import hip
from caminus.engine import k3, q35_dense_tp
from caminus.engine.q35_dense_tp import Dense35TP
from caminus.engine.q35_dense_verify_tp import Dense35VerifyTP


class Slots:
    made = 0

    def __init__(self, dev, n, N):
        Slots.made += 1
        self.base, self.n, self.stride = Slots.made << 40, n, (16 + N) * 4

    def slot(self, i):
        assert 0 <= i < self.n
        return self.base + i * self.stride


class Target(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for n in ('init', '_lazy_init'):
            self.stack.enter_context(patch.object(torch.cuda, n, side_effect=AssertionError('GPU initialization forbidden')))
        self.stack.enter_context(patch.object(hip, 'hip', side_effect=AssertionError('HIP access forbidden')))
        self.stack.enter_context(patch.object(torch.cuda, 'device', side_effect=lambda d: contextlib.nullcontext()))
        self.stack.enter_context(patch.object(torch.cuda, 'set_device'))
        self.stack.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
        self.stack.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=NS(multi_processor_count=42)))
        self.stack.enter_context(patch.object(torch.cuda, 'Event', side_effect=lambda *a, **k: Mock()))
        self.peer = self.stack.enter_context(patch.object(q35_dense_tp, '_hip', return_value=Mock()))
        self.stack.enter_context(patch.object(q35_dense_tp, 'Slots', Slots))
        zeros, to = torch.zeros, torch.Tensor.to

        def cpu_zeros(*a, **kw):
            return zeros(*a, **{k: v for k, v in kw.items() if k not in ('device', 'pin_memory')})

        def cpu_to(t, *a, **kw):
            if a and str(a[0]).startswith('cuda'): return t
            return to(t, *a, **kw)

        self.zeros = self.stack.enter_context(patch.object(torch, 'zeros', side_effect=cpu_zeros))
        self.stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
        self.stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t))

    def engine(self, line='swift', **kw):
        options = dict(max_ctx=64, spec_T=4, vram_check=False)
        options.update(kw)
        m = Dense35TP(str(PATHS[line]), **options)
        self.addCleanup(m.F.close)
        return m

    def test_real_target_and_all_verify_widths_keep_physical_rows_and_rotation_order(self):
        for line in PATHS:
            with self.subTest(line=line):
                m = self.engine(line, q8=True)
                pb = k3.kv_pos_bytes(m.nkv // 2 * m.hd, True)
                self.assertEqual(m.kv_rows, 67)
                self.assertEqual(Layout(m, T=4).pb, [pb, pb])
                self.assertEqual(m.kv_format_id, k3.Q35_KV_Q8_FORMAT_ID)
                self.assertEqual(m.vram_mib, Dense35TP.projection_mib(m.path, 64, spec_T=4, q8=True))
                for c in (0, 1):
                    for il, dd in enumerate(m.L):
                        if dd[c]['kind'] == 'attn':
                            self.assertEqual(dd[c]['cache'].dtype, torch.uint8)
                            self.assertEqual(dd[c]['cache'].numel(), 67 * pb)
                    self.check_tape(m.E[c])
                for T in range(1, 5):
                    v = Dense35VerifyTP(m, T, vram_check=False)
                    self.assertEqual(v.module, f'k3_q35_{line}_tp_ver_kvq8')
                    self.assertNotIn(v.module, k3.ATTN_PACK_MODULES)
                    for E in v.E: self.check_tape(E)
                with self.assertRaises(ValueError):
                    Dense35VerifyTP(m, 4, module=f'k3_q35_{line}_tp_ver', vram_check=False)

    def check_tape(self, E):
        self.assertFalse(E.unsupported())
        self.assertEqual(sum(s.op == k3.OP_HADAMARD for s in E.segs), 4)
        app = next(s for s in E.segs if s.op == k3.OP_KV_APPEND)
        gate = next(s for s in E.segs if s.op == k3.OP_GATE_MUL16)
        self.assertEqual(E.segs[app.dep].op, k3.OP_HADAMARD)
        self.assertEqual(E.segs[app.dep2p1 - 1].op, k3.OP_HADAMARD)
        self.assertEqual(E.segs[gate.dep].op, k3.OP_HADAMARD)
        self.assertTrue(all(s.dep < i and s.dep2p1 - 1 < i for i, s in enumerate(E.segs)))

    def test_f16_allocation_unchanged_and_memory_saving_is_exact(self):
        for line in PATHS:
            with self.subTest(line=line):
                a, b = self.engine(line), self.engine(line, q8=True)
                n = sum(d[0]['kind'] == 'attn' for d in a.L)
                row_delta = k3.kv_pos_bytes(a.nkv // 2 * a.hd, False) - k3.kv_pos_bytes(a.nkv // 2 * a.hd, True)
                for c in (0, 1):
                    self.assertAlmostEqual((a.vram_mib[c] - b.vram_mib[c]) * 2 ** 20, n * 67 * row_delta - 4)
                    self.assertFalse(any(s.op == k3.OP_HADAMARD for s in a.E[c].segs))
                self.assertFalse(hasattr(a, 'kv_format_id'))

    def test_admission_and_capacity_refuse_before_peer_access_or_tensor_allocation(self):
        for flag in (1, None, 'q8'):
            with self.subTest(flag=flag), self.assertRaises(ValueError): self.engine(q8=flag)
        with self.assertRaises(ValueError): self.engine(module='k3_q35_swift_tp_kvq8')
        self.peer.assert_not_called()
        self.zeros.assert_not_called()
        with patch.object(q35_dense_tp.capacity, 'check', side_effect=ValueError('synthetic no headroom')):
            with self.assertRaisesRegex(ValueError, 'no headroom'): self.engine(q8=True, vram_check=True)
        self.peer.assert_not_called()
        self.zeros.assert_not_called()

    def test_decode_error_on_either_card_precedes_bounds_timeout_and_survives_reset(self):
        for card in (0, 1):
            with self.subTest(card=card):
                m = self.engine(q8=True)
                m._card_ev = None
                m.h_st[0][1] = 1
                m.E[card].kv_error[0] = 2
                with self.assertRaises(k3.InvalidKVEncoding): m._pass_wait(0)
                for e in m.ev_done: e.synchronize.assert_called_once()
                self.assertEqual(int(m.E[card].kv_error[0]), 2)
                m.reset()
                with self.assertRaises(k3.InvalidKVEncoding): m._check([1], 0, 0, 'after reset')

    def test_verify_error_on_second_card_precedes_first_card_timeout(self):
        m = self.engine(q8=True)
        v = Dense35VerifyTP(m, 4, vram_check=False)
        for E in v.E: E.run = Mock()
        v.E[0].sync[2 * k3.MAXSEG + 1] = 1
        v.E[1].kv_error[0] = 2
        with self.assertRaises(k3.InvalidKVEncoding): v._pass(0, 0, False)
        self.assertEqual(int(v.E[0].sync[2 * k3.MAXSEG + 1]), 1)
        self.assertEqual(int(v.E[1].kv_error[0]), 2)
        for e in v.ev_done: e.synchronize.assert_called_once()
        m.reset()
        with self.assertRaises(k3.InvalidKVEncoding): v.run([1] * 4, 0, 0, argmax=False)

    def test_real_checkpoint_restores_every_byte_of_both_cards_tail_rows(self):
        m = self.engine(q8=True)
        layout = Layout(m, T=4)
        pc = PromptCache.__new__(PromptCache)
        pc.m, pc.layout, pc.T, pc.df, pc.draft = m, layout, 4, None, None
        pc._sync = lambda: None
        expected = {}
        for key, value in m.state.items():
            if key[-1] == 'kv':
                raw = value.view(torch.uint8).view(m.kv_rows, layout.pb[key[0]])
                raw[:] = torch.arange(m.kv_rows, dtype=torch.uint8)[:, None]
                want = torch.zeros_like(raw)
                want[28:32] = raw[28:32]
                expected[key] = want.reshape(-1)
            else:
                value.fill_(7)
                expected[key] = value.view(torch.uint8).reshape(-1).clone()
        buffers = [torch.zeros(n, dtype=torch.uint8) for n in layout.nbytes]
        pc._save(buffers, 32, 4)
        for value in m.state.values(): value.zero_()
        pc._load(buffers, NS(pos=32, commit=4, fresh=False, last_T=4, pool_ok=0, df=None), 0)
        for key, value in m.state.items():
            self.assertTrue(torch.equal(value.view(torch.uint8).reshape(-1), expected[key]), key)
        self.assertEqual(m.last_T, 4)

    def test_q8_verify_refuses_pull_before_tape_changes_and_rechecks_raw_segments(self):
        partial, out, epoch = torch.zeros(4), torch.zeros(4), torch.zeros(1, dtype=torch.int32)
        for line in PATHS:
            m = self.engine(line, q8=True)
            v = Dense35VerifyTP(m, 4, vram_check=False)
            for E in v.E:
                counts = (len(E.segs), len(E.keep))
                for mode in (1, 2):
                    with self.subTest(line=line, mode=mode), self.assertRaisesRegex(ValueError, 'push-only'):
                        E.xchg(partial, out, 4096, 8192, epoch, 0, 0, pull=12288, pull_mode=mode)
                    self.assertEqual((len(E.segs), len(E.keep)), counts)
                sg = next(s for s in E.segs if s.op == k3.OP_XCHG)
                for mode in (1, 2):
                    sg.pad2 = mode
                    with self.assertRaisesRegex(ValueError, 'push-only'): E.build()
                sg.pad2 = 0
                E.build()
            # Existing F16 verify modules retain their pull-capable exchange.
            f = self.engine(line)
            F = Dense35VerifyTP(f, 4, vram_check=False).E[0]
            F.xchg(partial, out, 4096, 8192, epoch, 0, 0, pull=12288, pull_mode=1)
            self.assertEqual(F.segs[-1].pad2, 1)
            self.assertEqual(F.segs[-1].aux, 12288)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); args = ap.parse_args()
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
    PATHS = {line: scratch / (line + '.gguf') for line in ('swift', 'tcap')}
    for line, path in PATHS.items(): tiny_q35.write(path, variant=line, tp=True)
    r = unittest.main(argv=[sys.argv[0]], exit=False).result
    print('RESULT ' + ('PASS' if r.wasSuccessful() else 'FAIL'))
    raise SystemExit(not r.wasSuccessful())
