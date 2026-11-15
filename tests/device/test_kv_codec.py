"""Own-Q8 launch admission and sticky error handling on real CPU tensor views.

Module, device-context and host-mapped plumbing are mocked; no HIP/GPU calls.
python -I tests/device/test_kv_codec.py
"""
import contextlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from caminus.device import bufcheck, kv_codec as K
from caminus.engine.k3 import ContextBoundsError


class Recorder:
    def __init__(self):
        self.calls = []

    def own_kv_q8_append(self, *args):
        self.calls.append(args)


class Codec(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(torch.cuda, 'init', side_effect=AssertionError('GPU initialization forbidden')))
        self.context = self.stack.enter_context(patch.object(torch.cuda, 'device', side_effect=lambda d: contextlib.nullcontext()))
        self.sync = self.stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        self.stack.enter_context(patch.object(bufcheck, '_host_mapped', return_value=True))
        self.rec = Recorder()
        self.load = self.stack.enter_context(patch.object(K, 'Module', return_value=self.rec))
        self.codec = K.Q8KVCodec('cuda:0', module=self.rec)
        self.args = dict(k=torch.zeros(160), ldk=96, v=torch.zeros(144), ldv=80, M=64,
                         cache=torch.zeros(4 * 136, dtype=torch.uint8), pos0=1, T=2,
                         err=torch.zeros(1, dtype=torch.int32))

    def append(self, **changes):
        self.codec.append(**dict(self.args, **changes))

    def refused(self, **changes):
        with self.assertRaises((ValueError, ContextBoundsError)):
            self.append(**changes)
        self.assertEqual(self.rec.calls, [])
        self.load.assert_not_called()
        self.context.assert_not_called()

    def test_constructor_is_lazy_and_requires_an_explicit_gpu_index(self):
        K.Q8KVCodec(torch.device('cuda:1'))
        self.load.assert_not_called()
        self.context.assert_not_called()
        for dev in ('cpu', 'cuda', 0):
            with self.subTest(dev=dev), self.assertRaises(ValueError):
                K.Q8KVCodec(dev)

    def test_padded_input_strides_row_bytes_and_launch_abi(self):
        self.append()
        args = self.rec.calls[0]
        self.assertEqual(args[:2], ((2,), (256,)))
        self.assertIs(args[2], self.args['k'])
        self.assertIs(args[4], self.args['v'])
        self.assertIs(args[7], self.args['cache'])
        self.assertIs(args[10], self.args['err'])
        self.assertEqual([args[i].value for i in (3, 5, 6, 8, 9)], [96, 80, 64, 1, 4])
        self.context.assert_called_once_with(torch.device('cuda:0'))
        self.load.assert_not_called()

    def test_lazy_module_load_occurs_once_after_validation(self):
        self.codec = K.Q8KVCodec('cuda:0')
        self.refused(k=self.args['k'][:-1])
        self.append()
        self.append()
        self.load.assert_called_once_with('kv_codec')
        self.assertEqual(len(self.rec.calls), 2)

    def test_logical_rows_may_limit_the_physical_cache(self):
        self.append(rows=3)
        self.assertEqual(self.rec.calls[0][9].value, 3)

    def test_invalid_spans_and_rows_never_reach_a_module(self):
        for changed in (dict(rows=5), dict(rows=2), dict(pos0=4), dict(pos0=3), dict(T=5), dict(pos0=-1), dict(rows=0)):
            with self.subTest(changed=changed):
                self.refused(**changed)

    def test_scalar_inputs_are_strict_integers_with_int32_bounds(self):
        for name in ('M', 'T', 'ldk', 'ldv', 'pos0', 'rows'):
            for value in (True, False, np.bool_(True), 1.0, '1', torch.tensor(1), -1, 2 ** 31):
                with self.subTest(name=name, value=value):
                    self.refused(**{name: value})

    def test_numpy_integer_scalars_are_admitted(self):
        self.append(M=np.int64(64), T=np.int32(2), pos0=np.int64(1), rows=np.int64(4))
        self.assertEqual(len(self.rec.calls), 1)

    def test_block_width_and_input_stride_layout(self):
        for changed in (dict(M=0), dict(M=31), dict(M=33), dict(ldk=63), dict(ldv=63), dict(T=0)):
            with self.subTest(changed=changed):
                self.refused(**changed)

    def test_short_views_are_refused_even_inside_larger_storage(self):
        self.refused(k=self.args['k'][:-1])
        self.refused(v=self.args['v'][:-1])
        self.refused(err=self.args['err'][:0])
        self.refused(cache=self.args['cache'][:-1])
        self.refused(cache=self.args['cache'][:0])

    def test_dtypes_contiguity_and_tensor_arguments(self):
        for name, dtype in (('k', torch.float16), ('v', torch.float64), ('cache', torch.int8), ('err', torch.int64)):
            with self.subTest(name=name):
                self.refused(**{name: self.args[name].to(dtype)})
        for name in ('k', 'v', 'cache', 'err'):
            with self.subTest(name=name):
                self.refused(**{name: None})
        self.refused(k=torch.zeros(320)[::2])
        self.refused(v=torch.zeros(288)[::2])
        self.refused(cache=torch.zeros(1088, dtype=torch.uint8)[::2])
        self.refused(err=torch.zeros(4, dtype=torch.int32)[::2])

    def test_pointer_alignment(self):
        self.refused(cache=torch.zeros(545, dtype=torch.uint8)[1:])
        for name, count, dtype in (('k', 160, torch.float32), ('v', 144, torch.float32), ('err', 1, torch.int32)):
            with self.subTest(name=name):
                raw = bytearray(4 * count + 1)
                self.refused(**{name: torch.frombuffer(raw, dtype=dtype, count=count, offset=1)})

    def test_written_buffers_cannot_alias_inputs_or_each_other(self):
        self.refused(cache=self.args['k'].view(torch.uint8)[:544])
        self.refused(err=self.args['k'].view(torch.int32)[:1])
        self.refused(err=self.args['v'].view(torch.int32)[:1])
        self.refused(err=self.args['cache'][:4].view(torch.int32))

    def test_read_only_inputs_may_alias(self):
        self.append(v=self.args['k'])
        self.assertEqual(len(self.rec.calls), 1)

    def test_ordinary_unmapped_cpu_buffers_are_refused(self):
        with patch.object(bufcheck, '_host_mapped', return_value=False):
            self.refused()

    def test_error_bits_are_sticky_and_every_nonzero_result_is_unusable(self):
        err = self.args['err']
        self.assertIsNone(self.codec.check(err))
        for flag in (K.ERR_SPAN, K.ERR_NUMERICAL, K.ERR_LAYOUT, 7, 8, -1):
            with self.subTest(flag=flag):
                err.fill_(flag)
                with self.assertRaisesRegex(RuntimeError, 'whole append result is unusable'):
                    self.codec.check(err)
                self.assertEqual(err.item(), flag)
        self.load.assert_not_called()
        self.context.assert_not_called()

    def test_check_uses_the_first_word_and_validates_its_buffer(self):
        self.assertIsNone(self.codec.check(torch.tensor([0, 7], dtype=torch.int32)))
        self.sync.assert_called_once_with(torch.device('cuda:0'))
        self.sync.reset_mock()
        for err in (None, torch.zeros(0, dtype=torch.int32), torch.zeros(1), torch.zeros(4, dtype=torch.int32)[::2]):
            with self.subTest(err=err), self.assertRaises(ValueError):
                self.codec.check(err)
        self.sync.assert_not_called()

    def test_error_check_waits_for_device_before_reading_a_mapped_word(self):
        err = self.args['err']
        self.sync.side_effect = lambda dev: err.fill_(K.ERR_NUMERICAL)
        with self.assertRaisesRegex(RuntimeError, 'numerical domain'):
            self.codec.check(err)
        self.sync.assert_called_once_with(torch.device('cuda:0'))


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
