"""Full-model gate state coverage and reference-format refusals, without GPU work.

These tests cover the byte snapshots used by the real geometry gate. They do
not establish device arithmetic, model quality, launch latency or memory fit.
"""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch


with patch('torch.cuda.init', side_effect=AssertionError('GPU initialization forbidden')), \
     patch('torch.cuda._lazy_init', side_effect=AssertionError('GPU initialization forbidden')):
    spec = importlib.util.spec_from_file_location('q35_prefill_model_gate',
                                                 Path(__file__).with_name('test_q35_prefill.py'))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)


def engine(q8):
    # Seven physical rows, four usable positions: three verify-tail rows must
    # stay outside the committed-prefix geometry comparison.
    width, physical = 2 * 256, 7
    row_bytes = 2 * (width // 32) * 34 if q8 else 2 * width * 2
    raw = torch.arange(physical * row_bytes, dtype=torch.int64).to(torch.uint8)
    cache = raw if q8 else raw.view(torch.float16)
    recurrent = dict(kind='gdn', conv_st=torch.arange(12, dtype=torch.float32),
                     gdn=torch.arange(11, dtype=torch.float32))
    return NS(q8=q8, nkv=2, hd=256, max_ctx=4, NV=2, S=2,
              L=[dict(kind='attn', cache=cache), recurrent]), row_bytes


class FullModelState(unittest.TestCase):
    def test_complete_physical_rows_including_final_q8_scales_and_v_bytes(self):
        for q8 in (False, True):
            with self.subTest(q8=q8):
                eng, rb = engine(q8)
                snapshot = gate.state_bytes(eng, 3)
                self.assertEqual(snapshot[0].dtype, torch.uint8)
                self.assertEqual(snapshot[0].numel(), 3 * rb)
                raw = eng.L[0]['cache'].view(torch.uint8)
                self.assertTrue(torch.equal(snapshot[0], raw[:3 * rb]))
                raw[3 * rb - 1] ^= 1
                self.assertFalse(gate.same_state(snapshot, gate.state_bytes(eng, 3)))

    def test_future_and_verify_tail_excluded_and_snapshots_own_their_bytes(self):
        for q8 in (False, True):
            eng, rb = engine(q8)
            snapshot = gate.state_bytes(eng, 3)
            raw = eng.L[0]['cache'].view(torch.uint8)
            raw[3 * rb:] ^= 1
            self.assertTrue(gate.same_state(snapshot, gate.state_bytes(eng, 3)))
            original = int(snapshot[0][0])
            raw[0] ^= 1
            self.assertEqual(int(snapshot[0][0]), original)
            self.assertFalse(gate.same_state(snapshot, gate.state_bytes(eng, 3)))

    def test_recurrent_state_bytes_preserve_nan_payloads_and_signed_zero(self):
        eng, _ = engine(True)
        conv = eng.L[1]['conv_st'].view(torch.int32)
        conv[0], conv[1] = 0x7fc00001, -2147483648
        before = gate.state_bytes(eng, 2)
        self.assertTrue(gate.same_state(before, gate.state_bytes(eng, 2)))
        conv[0] = 0x7fc00002
        self.assertFalse(gate.same_state(before, gate.state_bytes(eng, 2)))
        conv[0], conv[1] = 0x7fc00001, 0
        self.assertFalse(gate.same_state(before, gate.state_bytes(eng, 2)))

    def test_budget_inputs_count_exact_prefix_and_live_recurrent_bytes(self):
        for q8 in (False, True):
            eng, rb = engine(q8)
            self.assertEqual(sum(v.numel() for v in gate.state_views(eng, 3)), 3 * rb + 12 * 4 + 8 * 4)
            before = gate.state_bytes(eng, 3)
            eng.L[1]['gdn'][8:] += 1
            self.assertTrue(gate.same_state(before, gate.state_bytes(eng, 3)))
            eng.L[1]['gdn'][7] += 1
            self.assertFalse(gate.same_state(before, gate.state_bytes(eng, 3)))

    def test_missing_or_extra_state_cannot_pass_via_zip_truncation(self):
        eng, _ = engine(True)
        state = gate.state_bytes(eng, 2)
        self.assertFalse(gate.same_state(state, state[:-1]))
        self.assertFalse(gate.same_state(state[:-1], state))

    def test_invalid_prefix_dtype_and_partial_rows_are_refused(self):
        for q8 in (False, True):
            eng, _ = engine(q8)
            for prefix in (-1, 5, True, 1.5):
                with self.assertRaises(ValueError):
                    gate.state_views(eng, prefix)
            cache = eng.L[0]['cache']
            eng.L[0]['cache'] = cache[:-1]
            with self.assertRaisesRegex(ValueError, 'whole physical rows'):
                gate.state_views(eng, 2)
            eng.L[0]['cache'] = cache.to(torch.int32)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                gate.state_views(eng, 2)

    def test_recorded_llama_reference_requires_matching_declared_cache_format(self):
        base = ['model.gguf', '--ref', 'ref.npz']
        self.assertEqual(gate.parse_args(base).kv, 'f16')
        self.assertEqual(gate.parse_args(base + ['--kv', 'q8']).kv, 'q8')
        self.assertEqual(gate.parse_args(base + ['--llama', 'll.bin']).llama_kv, 'f16')
        for candidate, reference in [('q8', 'f16'), ('f16', 'q8')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gate.parse_args(base + ['--kv', candidate, '--llama', 'll.bin', '--llama-kv', reference])
            self.assertEqual(raised.exception.code, 2)
        args = gate.parse_args(base + ['--kv', 'q8', '--llama', 'll.bin', '--llama-kv', 'q8'])
        self.assertEqual(args.kv, args.llama_kv)

    def test_invalid_position_and_timing_arguments_refused_before_loading_model(self):
        for flag, value in [('--from', '0'), ('--from', '-1'), ('--speed', '-1')]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                gate.parse_args(['model.gguf', '--ref', 'ref.npz', flag, value])
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    raise SystemExit(0 if result.wasSuccessful() else 1)
