"""Q8 prefill admission and exact launch ABI, using CPU views and no HIP calls."""
import contextlib
import ctypes
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from caminus.device import bufcheck
from caminus.engine.k3 import ContextBoundsError
from caminus.prefill import q35_kvq8 as K


class Recorder:
    def __init__(self):
        self.calls = []

    def q35p_kv_append_q8(self, *args):
        self.calls.append(('append', args))

    def q35p_attn_wm_q8(self, *args):
        self.calls.append(('attn', args))


class Host(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name in ('init', '_lazy_init'):
            stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError('GPU initialization forbidden')))
        self.context = stack.enter_context(patch.object(torch.cuda, 'device', side_effect=lambda d: contextlib.nullcontext()))
        stack.enter_context(patch.object(bufcheck, '_host_mapped', return_value=True))
        self.rec = Recorder()
        self.load = stack.enter_context(patch.object(K, 'Module', return_value=self.rec))
        self.ops = K.Ops('cuda:0')
        self.a = dict(k=torch.zeros(160), ldk=96, v=torch.zeros(144), ldv=80,
                      cache=torch.zeros(5 * 136, dtype=torch.uint8), M=64, pos0=1, n=2, rows=4,
                      err=torch.zeros(1, dtype=torch.int32))
        self.w = dict(q=torch.zeros(10 * 6 * 256), cache=torch.zeros(20 * 544, dtype=torch.uint8),
                      pos0=7, n=10, heads=6, nkv=1, D=256, scale=1 / 16,
                      out=torch.zeros(10 * 6 * 256), rows=18)
        self.context.reset_mock()

    def append(self, **changes):
        self.ops.append(**dict(self.a, **changes))

    def attend(self, **changes):
        self.ops.attn_wm(**dict(self.w, **changes))

    def refused(self, method, **changes):
        self.rec.calls.clear()
        self.context.reset_mock()
        with self.assertRaises((ValueError, ContextBoundsError)):
            method(**changes)
        self.assertEqual(self.rec.calls, [])
        self.context.assert_not_called()

    def test_constructor_validates_before_loading_and_loads_once(self):
        self.load.assert_called_once_with('q35_prefill_kvq8')
        self.append()
        self.attend()
        self.load.assert_called_once_with('q35_prefill_kvq8')
        self.load.reset_mock()
        for dev in ('cpu', 'cuda', True, 0, None):
            with self.subTest(dev=dev), self.assertRaises(ValueError):
                K.Ops(dev)
        self.load.assert_not_called()

    def test_append_exact_abi_padded_strides_and_physical_tail(self):
        self.append()
        name, a = self.rec.calls[0]
        self.assertEqual(name, 'append')
        self.assertEqual(a[:2], ((2,), (256,)))
        for i, key in ((2, 'k'), (4, 'v'), (6, 'cache'), (10, 'err')):
            self.assertIs(a[i], self.a[key])
        self.assertEqual([a[i].value for i in (3, 5, 7, 8, 9)], [96, 80, 64, 1, 4])
        self.context.assert_called_once_with(torch.device('cuda:0'))

    def test_attention_exact_abi_absolute_block_range_and_force(self):
        self.attend(force_fp32=True)
        name, a = self.rec.calls[0]
        self.assertEqual(name, 'attn')
        self.assertEqual(a[:2], ((3, 1), (256,)))
        for i, key in ((2, 'q'), (3, 'cache'), (10, 'out')):
            self.assertIs(a[i], self.w[key])
        self.assertEqual([a[i].value for i in (4, 5, 6, 7, 8, 11)], [7, 10, 0, 6, 1, 1])
        self.assertEqual(a[9].value, ctypes.c_float(self.w['scale'] * 1.4426950408889634).value)
        self.attend(b0=1, b1=2)
        a = self.rec.calls[1][1]
        self.assertEqual(a[:2], ((1, 1), (256,)))
        self.assertEqual(a[6].value, 1)
        self.assertEqual(a[11].value, 0)

    def test_gqa_extremes_and_multiple_kv_heads(self):
        for heads, nkv in ((1, 1), (8, 1), (12, 2)):
            with self.subTest(heads=heads, nkv=nkv):
                self.attend(heads=heads, nkv=nkv, q=torch.zeros(10 * heads * 256),
                            out=torch.zeros(10 * heads * 256), cache=torch.zeros(20 * nkv * 544, dtype=torch.uint8))
                self.assertEqual(self.rec.calls[-1][1][0], (3, nkv))

    def test_cache_needs_only_two_byte_alignment(self):
        self.append(cache=torch.zeros(5 * 136 + 2, dtype=torch.uint8)[2:])
        self.attend(cache=torch.zeros(20 * 544 + 2, dtype=torch.uint8)[2:])
        self.assertEqual(len(self.rec.calls), 2)

    def test_read_only_inputs_may_alias(self):
        self.append(v=self.a['k'])
        self.assertEqual(len(self.rec.calls), 1)

    def test_counts_are_strict_int32_in_both_entry_points(self):
        bad = (True, False, np.bool_(True), 1.0, '1', torch.tensor(1), -1, 2 ** 31)
        for method, names in ((self.append, ('ldk', 'ldv', 'M', 'pos0', 'n', 'rows')),
                              (self.attend, ('pos0', 'n', 'heads', 'nkv', 'D', 'rows', 'b0', 'b1'))):
            for key in names:
                for value in bad:
                    with self.subTest(method=method.__name__, key=key, value=value):
                        self.refused(method, **{key: value})

    def test_numpy_integer_counts_are_supported(self):
        self.append(M=np.int64(64), n=np.int32(2))
        self.attend(heads=np.int32(6), b0=np.int64(1), b1=np.int64(2))
        self.assertEqual(len(self.rec.calls), 2)

    def test_append_geometry_spans_and_exact_view_extents(self):
        for changed in (dict(M=0), dict(M=33), dict(ldk=63), dict(ldv=63), dict(n=0),
                        dict(rows=0), dict(rows=6), dict(rows=2), dict(pos0=4),
                        dict(k=self.a['k'][:-1]), dict(v=self.a['v'][:-1]),
                        dict(cache=self.a['cache'][:-1]), dict(cache=self.a['cache'][:0]),
                        dict(err=self.a['err'][:0])):
            with self.subTest(changed=tuple(changed)):
                self.refused(self.append, **changed)

    def test_attention_geometry_spans_and_block_limits(self):
        for changed in (dict(D=128), dict(heads=0), dict(nkv=0), dict(heads=9), dict(heads=5, nkv=2),
                        dict(n=0), dict(rows=0), dict(rows=21), dict(rows=16), dict(pos0=18),
                        dict(b0=3), dict(b0=2, b1=2), dict(b1=4), dict(b1=0),
                        dict(q=self.w['q'][:-1]), dict(out=self.w['out'][:-1]),
                        dict(cache=self.w['cache'][:-1]), dict(cache=self.w['cache'][:0])):
            with self.subTest(changed=tuple(changed)):
                self.refused(self.attend, **changed)
        # A tiny CPU view cannot represent this logical cache; mock only the
        # row admission to exercise overflow refusal before buffers or dispatch.
        with patch.object(K, '_cache', return_value=(K.INT_MAX, 544)):
            self.refused(self.attend, pos0=K.INT_MAX - 1, n=1)
            # Rounded-eight admission alone permits this endpoint, but the
            # device's final c0 += 32 would overflow after its last tile.
            self.refused(self.attend, pos0=K.INT_MAX - 8, n=1)
        self.refused(self.attend, nkv=2 ** 23, heads=2 ** 23)

    def test_finite_fp32_scale_and_explicit_bool_force(self):
        for scale in (True, float('nan'), float('inf'), -float('inf'), 1e39, 10 ** 1000, '1', 1j):
            with self.subTest(scale=scale):
                self.refused(self.attend, scale=scale)
        for force in (0, 1, 'yes', None, np.bool_(True)):
            with self.subTest(force=force):
                self.refused(self.attend, force_fp32=force)
        self.attend(scale=0.0)
        self.attend(scale=-0.5)

    def test_dtypes_contiguity_and_missing_buffers(self):
        for method, args, names in ((self.append, self.a, ('k', 'v', 'cache', 'err')),
                                     (self.attend, self.w, ('q', 'cache', 'out'))):
            for key in names:
                with self.subTest(method=method.__name__, key=key):
                    self.refused(method, **{key: None})
                    self.refused(method, **{key: args[key].to(torch.float64)})
                    noncontig = torch.zeros(2 * max(2, args[key].numel()), dtype=args[key].dtype)[::2]
                    self.refused(method, **{key: noncontig})

    def test_pointer_alignment(self):
        for method, args in ((self.append, self.a), (self.attend, self.w)):
            raw = torch.zeros(args['cache'].numel() + 1, dtype=torch.uint8)
            self.refused(method, cache=raw[1:])
        for name in ('k', 'v', 'err'):
            t = self.a[name]
            raw = bytearray(4 * t.numel() + 1)
            self.refused(self.append, **{name: torch.frombuffer(raw, dtype=t.dtype, count=t.numel(), offset=1)})
        self.refused(self.attend, q=torch.zeros(self.w['q'].numel() + 1)[1:])
        raw = bytearray(4 * self.w['out'].numel() + 1)
        self.refused(self.attend, out=torch.frombuffer(raw, dtype=torch.float32, count=self.w['out'].numel(), offset=1))

    def test_written_buffers_must_not_alias_any_operand(self):
        base = torch.zeros(200)
        self.refused(self.append, k=base[:160], cache=base.view(torch.uint8)[:680])
        self.refused(self.append, err=self.a['k'].view(torch.int32)[:1])
        self.refused(self.append, err=self.a['v'].view(torch.int32)[:1])
        self.refused(self.append, err=self.a['cache'][:4].view(torch.int32))
        self.refused(self.attend, out=self.w['q'])
        base = torch.zeros(self.w['out'].numel() + 4)
        self.refused(self.attend, q=base[:-4], out=base[4:])
        base = torch.zeros(self.w['out'].numel())
        self.refused(self.attend, out=base, cache=base.view(torch.uint8)[:20 * 544])

    def test_unregistered_cpu_buffers_are_refused(self):
        with patch.object(bufcheck, '_host_mapped', return_value=False):
            self.refused(self.append)
            self.refused(self.attend)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
