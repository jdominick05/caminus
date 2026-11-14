"""Dense Q8 module admission, tape ABI and sticky numerical status (CPU only).

  python -I tests/engine/test_dense_q8_tape.py

Real Engine methods build tapes on CPU tensor stand-ins. HIP module lookup and
launch are recorders; GPU initialization is forbidden. No compiled-object,
quantizer, attention or model-numerics claim follows from these tests.
"""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from caminus import errors
from caminus.device import hip
from caminus.engine import k3


class DenseQ8Tape(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name in ('init', '_lazy_init'):
            self.stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError('GPU initialization forbidden')))
        self.stack.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
        self.stack.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=NS(multi_processor_count=2)))
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, 'current_stream', return_value=NS(cuda_stream=0)))
        self.wait = self.stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        zeros = torch.zeros
        self.stack.enter_context(patch.object(torch, 'zeros', side_effect=lambda *a, **kw:
            zeros(*a, **{key: value for key, value in kw.items() if key not in ('device', 'pin_memory')})))
        self.stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda t: t))
        self.launch = Mock(return_value=0)
        self.stack.enter_context(patch.object(hip, 'hip', return_value=NS(hipModuleLaunchKernel=self.launch)))
        self.module = self.stack.enter_context(patch.object(k3, '_m', return_value=NS(k3_run=NS(fn=123, lds_bytes=lambda: 1024))))

    def engine(self, module='k3_q35_swift_kvq8', T=1):
        return k3.Engine(module=module, T=T, grid=2)

    def buffers(self, E, M=256, rows=37):
        return dict(k=torch.zeros(E.T * M), v=torch.zeros(E.T * M),
                    cache=torch.zeros(rows * k3.kv_pos_bytes(M, True), dtype=torch.uint8),
                    pos=torch.zeros(1, dtype=torch.int32))

    def append(self, E, b):
        return E.kv_append(**b, q8=True)

    def test_seven_specializations_keep_their_weight_and_execution_families(self):
        bases = ('k3_q35_swift', 'k3_q35_swift_ver', 'k3_q35_swift_tp',
                 'k3_q35_tcap', 'k3_q35_tcap_ver', 'k3_q35_tcap_tp', 'k3_q35_gsq')
        self.assertEqual(k3.Q35_KV_Q8_BASE, {base + '_kvq8': base for base in bases})
        for new, base in k3.Q35_KV_Q8_BASE.items():
            with self.subTest(module=new):
                self.assertEqual(k3.Q35_KV_Q8_OF[base], new)
                self.assertIn(new, k3.Q35_MODULES)
                self.assertIn(new, k3.Q35_SPLIT_MODULES)
                self.assertIn(new, k3.ATTN_SPLIT_MODULES)
                self.assertNotIn(new, k3.ATTN_PACK_MODULES)
                self.assertEqual(k3.ATTN_G_OF[new], 6)
                self.assertEqual(k3.ATTN_SS_OF[new], 32)
                self.assertEqual(k3.MODULE_CAPS[new], k3.MODULE_CAPS[base])
                for family in (k3.Q35S_MODULES, k3.Q35T_MODULES, k3.Q35R_MODULES, k3.Q35V_MODULES):
                    self.assertEqual(new in family, base in family)
        for line in ('swift', 'tcap'):
            one = f'k3_q35_{line}_kvq8'
            tp = f'k3_q35_{line}_tp_kvq8'
            self.assertEqual(k3.Q35_TP_OF[one], tp)
            self.assertIn(tp, k3.Q35_TP_MODULES)
            self.assertNotIn(tp, k3.Q35_TPV_OF)  # no Q8 TP verify specialization yet
        self.assertNotIn('k3_q35_gsq_kvq8', k3.Q35_TP_OF)

    def test_unsupported_token_width_and_teams_are_refused_before_allocation(self):
        for module in k3.Q35_KV_Q8_MODULES:
            allowed = (1, 2, 3, 4) if module in k3.Q35V_MODULES else (1,)
            for T in (0, True, 1.0, 2, 3, 4, 5):
                if type(T) is int and T in allowed:
                    continue
                with self.subTest(module=module, T=T), patch.object(torch, 'zeros') as alloc:
                    with self.assertRaises(ValueError):
                        self.engine(module, T)
                    alloc.assert_not_called()
        with self.assertRaises(ValueError):
            k3.Engine(module='k3_q35_swift_kvq8', grid=2, teams=[2])

    def test_extra_error_word_does_not_move_any_sync_status_field(self):
        for module in k3.Q35_KV_Q8_MODULES:
            E = self.engine(module)
            self.assertEqual(E.sync.numel(), k3.SYNC_WORDS + 1)
            self.assertEqual(E.kv_error.data_ptr(), E.sync.data_ptr() + 4 * k3.SYNC_WORDS)
            E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS] = torch.arange(1, 9, dtype=torch.int32)
            E.kv_error.fill_(2)
            self.assertEqual(E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + 8].tolist(), list(range(1, 9)))
        legacy = self.engine('k3_q35_swift')
        self.assertEqual(legacy.sync.numel(), k3.SYNC_WORDS)
        self.assertIsNone(legacy.kv_error)

    def test_append_keeps_a_separate_error_pointer_and_physical_row_bound(self):
        for module in k3.Q35_KV_Q8_MODULES:
            for T in ((1, 4) if module in k3.Q35V_MODULES else (1,)):
                with self.subTest(module=module, T=T):
                    E = self.engine(module, T)
                    b = self.buffers(E, rows=33 + T - 1)
                    i = self.append(E, b)
                    s = E.segs[i]
                    self.assertEqual((s.op, s.ntask, s.N, s.work), (k3.OP_KV_APPEND, T, 256, 1))
                    self.assertEqual(s.gs, E.kv_error.data_ptr())
                    self.assertNotEqual(s.gs, E.sync.data_ptr() + 4 * (2 * k3.MAXSEG + 1))
                    self.assertTrue(any(t is E.kv_error for t in E.keep))
                    self.assertEqual(E.guard_caps[b['pos'].data_ptr()][1], 33 + T - 1)
                    E.build()
                    self.assertEqual(E.tape.numel(), len(E.segs) * k3.ctypes.sizeof(k3.Seg))

    def test_q8_verify_attention_is_unpacked_for_every_pass_row(self):
        for module in ('k3_q35_swift_ver_kvq8', 'k3_q35_tcap_ver_kvq8'):
            for T in (1, 2, 3, 4):
                with self.subTest(module=module, T=T):
                    E = self.engine(module, T)
                    b = self.buffers(E, rows=65)
                    ap = self.append(E, b)
                    q, out = torch.zeros(T * 6 * 256), torch.zeros(T * 6 * 256)
                    comb = E.attn_dec(q, b['cache'], b['pos'], out, 6, 1, 256, 1 / 16, q8=True, dep=ap)
                    part = E.segs[comb - 1]
                    self.assertEqual((part.ntask, part.work, part.pad2, part.N), (3 * T, 1, 3, T * 6))
                    self.assertEqual(E.segs[comb].ntask, T * 6)
                    self.assertEqual(part.dep, ap)
                    E.build()
                    if T > 1:
                        self.assertTrue(all(s.team == T for s in E.segs))

    def test_format_mismatches_are_refused_before_retention_and_at_build(self):
        for new, base in k3.Q35_KV_Q8_BASE.items():
            for module, wrong in ((new, False), (base, True)):
                with self.subTest(module=module):
                    E = self.engine(module)
                    b = self.buffers(E)
                    with self.assertRaises(ValueError):
                        E.kv_append(**b, q8=wrong)
                    with self.assertRaises(ValueError):
                        E.attn_dec(torch.zeros(1536), b['cache'], b['pos'], torch.zeros(1536),
                                   6, 1, 256, 1 / 16, q8=wrong)
                    self.assertEqual(E.keep, [])
                    self.assertEqual(E.segs, [])
                    self.assertEqual(E.guard_caps, {})
                    # Raw segments must not bypass the constructor admission.
                    for op in (k3.OP_KV_APPEND, k3.OP_ATTN_PART, k3.OP_ATTN_COMB):
                        E.segs = [k3.Seg(op=op, work=int(wrong))]
                        with self.assertRaises(ValueError):
                            E.build()

    def test_malformed_rows_views_and_aliases_never_reach_tape(self):
        E = self.engine()
        good = self.buffers(E)
        changes = [dict(k=torch.zeros(31)), dict(k=torch.zeros(0)), dict(k=good['k'].half()),
                   dict(v=good['v'][:-1]), dict(pos=torch.zeros(1)),
                   dict(cache=good['cache'].view(torch.float16)), dict(cache=good['cache'][:-1]),
                   dict(cache=torch.zeros(good['cache'].numel() + 1, dtype=torch.uint8)[1:]),
                   dict(cache=torch.zeros(2 * good['cache'].numel(), dtype=torch.uint8)[::2])]
        # A whole cache view that aliases K is also invalid, independently of size.
        shared = torch.zeros(good['cache'].numel(), dtype=torch.uint8)
        changes.append(dict(k=shared[:1024].view(torch.float32), cache=shared))
        for changed in changes:
            with self.subTest(changed=list(changed)), self.assertRaises(ValueError):
                self.append(E, dict(good, **changed))
            self.assertEqual(E.segs, [])
            self.assertEqual(E.keep, [])
            self.assertEqual(E.guard_caps, {})
        E4 = self.engine('k3_q35_swift_ver_kvq8', 4)
        with self.assertRaises(ValueError):
            self.append(E4, dict(self.buffers(E4), k=torch.zeros(1025)))
        for M in (2 ** 31, 2 ** 32):
            with self.subTest(M=M), self.assertRaisesRegex(ValueError, 'int32'):
                E._dense_q8_rows(good['cache'], M)

    def test_attention_alignment_is_checked_for_the_vector_query_load(self):
        E = self.engine()
        b = self.buffers(E)
        q = torch.zeros(1537)[1:]
        with self.assertRaisesRegex(ValueError, '16-byte aligned'):
            E.attn_dec(q, b['cache'], b['pos'], torch.zeros(1536), 6, 1, 256, 1 / 16, q8=True)
        self.assertEqual(E.segs, [])
        self.assertEqual(E.guard_caps, {})

    def test_hadamard_is_whitelisted_only_on_q8_dense_objects(self):
        for new, base in k3.Q35_KV_Q8_BASE.items():
            E = self.engine(new)
            E.hadamard(torch.zeros(256), 256)
            E.build()
            old = self.engine(base)
            old.hadamard(torch.zeros(256), 256)
            with self.assertRaises(ValueError):
                old.build()

    def test_q8_hadamard_refuses_non_fp32_or_noncontiguous_storage(self):
        E = self.engine()
        for x in (torch.zeros(256, dtype=torch.float16), torch.zeros(512)[::2], torch.zeros(0)):
            with self.subTest(dtype=x.dtype, size=x.numel()), self.assertRaises(ValueError):
                E.hadamard(x, 256)
        with self.assertRaises(ValueError):
            E.hadamard(torch.zeros(256), 256, rows_per_task=0)
        self.assertEqual(E.keep, [])
        self.assertEqual(E.segs, [])

    def test_forged_append_error_pointer_is_refused(self):
        E = self.engine()
        self.append(E, self.buffers(E))
        E.segs[0].gs = E.sync.data_ptr() + 4 * (2 * k3.MAXSEG + 1)
        with self.assertRaisesRegex(ValueError, 'dedicated numerical error pointer'):
            E.build()

    def test_numeric_errors_precede_timeout_handling_and_remain_sticky(self):
        for code in (2, 4, -1):
            with self.subTest(code=code):
                E = self.engine()
                E.nop(1, 0)
                E.build()
                E.sync[2 * k3.MAXSEG + 1] = 1  # simultaneous wait failure
                E.kv_error.fill_(code)
                before = E.sync.clone()
                with patch.object(E, 'stall_report', side_effect=AssertionError('timeout handling must not run')):
                    with self.assertRaises(errors.InvalidKVEncoding):
                        E.run(check=True)
                self.assertTrue(torch.equal(E.sync, before))
                with self.assertRaises(errors.InvalidKVEncoding):
                    E.check_kv_error()
        self.assertEqual(self.launch.call_count, 3)
        self.assertEqual(self.wait.call_count, 3)

    def test_check_false_remains_async_and_caller_checks_after_wait(self):
        E = self.engine()
        E.nop(1, 0)
        E.build()
        E.run(check=False)
        self.wait.assert_not_called()
        self.wait.side_effect = lambda: E.kv_error.fill_(2)
        torch.cuda.synchronize()
        with self.assertRaises(errors.InvalidKVEncoding):
            E.check_kv_error()
        self.assertEqual(E.kv_error.item(), 2)

    def test_clean_error_word_preserves_ordinary_timeout_refusal(self):
        E = self.engine()
        E.nop(1, 0)
        E.build()
        E.sync[2 * k3.MAXSEG + 1] = 1
        with self.assertRaisesRegex(RuntimeError, 'a wait gave up'):
            E.run(check=True)
        self.assertFalse(E.sync.any())
        self.assertIsNone(self.engine('k3_q35_swift').check_kv_error())

    def test_residency_uses_q8_compiled_metadata_and_refuses_excess_grid(self):
        module = 'k3_q35_tcap_ver_kvq8'
        lds = 20 * 1024
        self.module.return_value = NS(k3_run=NS(lds_bytes=lambda: lds))
        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(k3, '_RESIDENT_OK', set()):
            self.assertEqual(k3.resident_grid(module, 6), 12)
            with self.assertRaises(ValueError):
                k3.resident_grid(module, 7)
            k3.check_resident(module, 12)
            with self.assertRaises(ValueError):
                k3.check_resident(module, 13)
        self.assertTrue(all(call.args[0] == module for call in self.module.call_args_list))


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
