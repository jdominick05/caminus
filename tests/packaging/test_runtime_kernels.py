"""Runtime membership from real model selectors and prefill constructors (CPU only).

  python -I tests/packaging/test_runtime_kernels.py

No model files, HIP loads or kernel launches. Tiny metadata stands in for GGUF;
prefill workspaces are small CPU tensors. This checks selection and packaging
membership, not installed-object symbols, numerics or actual model execution.
"""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from caminus.device import hip, kernels, topk
from caminus.engine import k3, q3moe_one, q35_dense, q35_dense_verify, q35_dense_verify_tp, q35_mtp, q35_dflash
from caminus.prefill import attn, gdn, glue, q3m, q3m_ops, q35own, q35own_tp, xchg


# Frozen pre-fix registry: a negative control independent of the current registry.
OLD_RUNTIME = (
    'k3_gguf_gsq', 'k3_gguf_gsq_mt', 'k3_gguf_gsq_qsa', 'k3_gguf_gsq_qsa_mt',
    'k3_gguf_gsq_mtp', 'k3_gguf_gsq_mtp_mt', 'k3_gguf_draft', 'k3_gguf', 'k3_gguf_mt', 'k3_guard',
    'k3_gguf_q35h', 'k3_gguf_q35g', 'k3_gguf_q35d', 'k3_gguf_q3n',
    'prefill_attn', 'prefill_feed', 'prefill_gdn', 'prefill_glue', 'prefill_tp', 'prefill_dense_q6_k', 'gemm',
    'prefill_glue_q35', 'prefill_glue_q35d', 'prefill_gdn_q35', 'prefill_attn_q35', 'prefill_attn_q35d',
    'prefill_mmq_iq2_s', 'prefill_mmq_iq3_s', 'prefill_mmq_iq3_xxs', 'prefill_mmq_iq4_nl',
    'prefill_mmq_iq4_xs', 'prefill_mmq_q2_0', 'prefill_mmq_q3_k', 'prefill_mmq_q4_k',
    'prefill_mmq_q5_k', 'prefill_mmq_q6_k', 'prefill_mmq_q8_0', 'prefill_mmq_glue', 'prefill_mmq_glue8',
    'prefill_moe_iq2_s', 'prefill_moe_iq3_s', 'prefill_moe_iq3_xxs', 'prefill_moe_iq4_xs',
    'prefill_moe_iq4_nl', 'prefill_moe_q2_0', 'prefill_moe_glue', 'own_topk', 'smoke',
)


class Metadata:
    """Only the tensor/header fields module_of actually reads; no tensor bytes."""
    def __init__(self, formats, embedding='Q4_K', extra=()):
        self.tensors = [NS(name='token_embd.weight', type=embedding, shape=(5120, 64))]
        self.tensors += [NS(name=f'blk.0.matrix{i}.weight', type=f, shape=(256, 256))
                         for i, f in enumerate(formats)]
        self.tensors += list(extra)
        self.T = {t.name: t for t in self.tensors}

    def hp(self, key):
        return {'block_count': 2, 'nextn_predict_layers': 1}[key]


def q3_engine(iq=(), rs=(), **flags):
    m = q3moe_one.Q3MoEOne.__new__(q3moe_one.Q3MoEOne)
    m.module = None
    m.fused = m.gu = m.attn_one = m.attn_split = True
    m.iq, m.rs = set(iq), set(rs)
    for key, value in flags.items():
        setattr(m, key, value)
    return m


class RuntimeKernels(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.loads = []
        self.stack.enter_context(patch.object(torch.cuda, 'init', side_effect=AssertionError('GPU initialization forbidden')))
        self.stack.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU initialization forbidden')))
        self.stack.enter_context(patch.object(hip, 'hip', side_effect=AssertionError('HIP access forbidden')))
        self.stack.enter_context(patch.object(torch.cuda, 'device', side_effect=lambda d: contextlib.nullcontext()))
        self.stack.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=NS(multi_processor_count=42)))
        self.stack.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
        self.stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, 'Event', return_value=NS()))
        self.stack.enter_context(patch.object(torch.cuda, 'current_stream', side_effect=self.stream))
        self.stack.enter_context(patch.object(torch.cuda, 'Stream', side_effect=lambda device: self.stream(device)))
        self.stack.enter_context(patch.object(xchg, '_peer_copy'))
        zeros = torch.zeros

        def cpu_zeros(*args, **kw):
            kw.pop('device', None)
            kw.pop('pin_memory', None)
            return zeros(*args, **kw)

        self.stack.enter_context(patch.object(torch, 'zeros', side_effect=cpu_zeros))
        for module in (hip, k3, attn, gdn, glue, xchg):
            self.stack.enter_context(patch.object(module, 'Module', side_effect=self.load))

    @staticmethod
    def stream(device):
        d = device if isinstance(device, torch.device) else torch.device('cuda', int(device))
        return NS(device=d, synchronize=lambda: None)

    def load(self, name):
        self.loads.append(name)
        # A constructor may retain a module; touching a kernel would fail here.
        return NS(name=name)

    def require_runtime(self, names):
        missing = set(names) - set(kernels.RUNTIME_SET)
        self.assertFalse(missing, f'selected runtime objects missing from package: {sorted(missing)}')

    def dense_prefill(self, module, ratio=6, tp=False, feature_card=0):
        tcap = module in k3.Q35T_MODULES
        fmt = 'IQ2_XS' if module in k3.Q35R_MODULES else 'IQ3_S' if tcap else 'Q4_K'
        layer = dict(kind='attn', **{key: dict(fmt=fmt) for key in ('q', 'k', 'v', 'o', 'fg', 'fu', 'fd')})
        m = NS(module=module, tcap=tcap, dev=0, devs=[0, 1], nl=1, nl_build=1, local_only=False,
               D=5120, nh=ratio * (2 if tp else 1), nkv=2 if tp else 1, hd=256, ffn=256, f0=128,
               S=128, NK=1, NV=1, gsplit=[(None, 1, [0]), (None, 1, [0])],
               vocab=64, emb=NS(type='IQ2_S' if tcap else 'Q4_K'), feature_card=feature_card)
        m.L = [[layer, layer]] if tp else [layer]
        start = len(self.loads)
        if tp:
            q35own_tp.Q35OwnPrefillTP(m, chunk=16, check=False)
        else:
            q35own.Q35OwnPrefill(m, chunk=16, check=False)
        return self.loads[start:]

    def q3_prefill(self, fmt):
        m = NS(dev=0, D=256, nh=8, nkv=1, hd=128, k_used=1, n_exp=2, ffn=256,
               L=[{key: (fmt,) for key in q3m.MM_KEYS}], emb=NS(shape=(256, 64), rows=lambda: (64, 144)))
        start = len(self.loads)
        q3m.Q3MPrefill(m, chunk=16, check=False)
        return self.loads[start:]

    def test_dense_format_selectors_ignore_non_matrix_and_mtp_extras(self):
        cases = [('swift', ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K', 'F32'), 'Q4_K', q35_dense.MODULE),
                 ('tcap', ('IQ3_S', 'Q8_0', 'Q6_K'), 'IQ2_S', q35_dense.MODULE_TCAP)]
        cases += [(fmt, (fmt, 'IQ3_S'), 'IQ2_S', q35_dense.MODULE_GSQ) for fmt in q35_dense.GSQ_ONLY_FMT]
        for label, formats, emb, want in cases:
            with self.subTest(label=label):
                extra = [NS(name='blk.1.nextn.eh_proj.weight', type='UNSUPPORTED', shape=(256, 256)),
                         NS(name='blk.0.norm.weight', type='UNSUPPORTED', shape=(256,))]
                selected = q35_dense.module_of(Metadata(formats, emb, extra))
                self.assertEqual(selected, want)
                self.require_runtime([selected])
        # Embedding format can select ThinkingCap even with otherwise Swift formats.
        selected = q35_dense.module_of(Metadata(('Q4_K',), embedding='IQ2_S'))
        self.assertEqual(selected, q35_dense.MODULE_TCAP)
        self.require_runtime([selected])

    def test_dense_tp_maps_and_drafter_constants_are_shipped(self):
        for one in (q35_dense.MODULE, q35_dense.MODULE_TCAP):
            self.require_runtime([k3.Q35_TP_OF[one]])
        self.require_runtime([q35_dense_verify.MODULE, q35_dense_verify.MODULE_TCAP,
                              q35_mtp.MODULE, q35_mtp.MODULE_TCAP, q35_dflash.MODULE, q35_dflash.MODULE_TCAP])
        self.assertNotIn(q35_dense.MODULE_GSQ, k3.Q35_TP_OF)

    def test_real_tp_verify_constructor_selects_shipped_objects_on_both_cards(self):
        # Build actual verify buffers/Engine objects, with only the tensor tape
        # omitted: its full operation/weight coverage lives in q35_tp_ver.
        # CUDA/HIP access is forbidden by setUp; every allocation stays on CPU.
        cases = [('k3_q35_swift_tp', 'k3_q35_swift_tp_ver'),
                 ('k3_q35_tcap_tp', 'k3_q35_tcap_tp_ver')]
        old_runtime = tuple(n for n in kernels.RUNTIME_SET if n not in {want for _, want in cases})
        for module, want in cases:
            for T in (1, 4):
                with self.subTest(module=module, T=T):
                    eng = NS(module=module, spec_T=4, nl=1, nl_build=1, D=5120, hd=256, nh=12, nkv=2,
                             S=128, f0=128, ffn=256, vh=32, vocab=64, max_ctx=64, devs=[0, 1],
                             per_wgp_c=[4, 4], limit_us=200000,
                             L=[[dict(kind='attn'), dict(kind='attn')]])
                    with patch.object(q35_dense_verify_tp.Dense35VerifyTP, '_tape'):
                        ver = q35_dense_verify_tp.Dense35VerifyTP(eng, T, vram_check=False)
                    selected = [ver.module] + [e.module for e in ver.E]
                    self.assertEqual(selected, [want] * 3)
                    self.assertEqual([e.T for e in ver.E], [T, T])
                    self.require_runtime(selected)
                    with patch.object(kernels, 'RUNTIME_SET', old_runtime), \
                            self.assertRaisesRegex(AssertionError, want):
                        self.require_runtime(selected)

    def test_real_dense_prefill_constructors_select_shipped_helpers(self):
        for module in (q35_dense.MODULE, q35_dense.MODULE_TCAP, q35_dense.MODULE_GSQ):
            for ratio in (6, 8):
                with self.subTest(module=module, ratio=ratio):
                    names = self.dense_prefill(module, ratio)
                    self.assertIn('prefill_attn_q35d' if ratio == 6 else 'prefill_attn_q35', names)
                    self.assertIn(q35own.MODULE_GSQ if module == q35_dense.MODULE_GSQ else
                                  q35own.MODULE_TCAP if module == q35_dense.MODULE_TCAP else q35own.MODULE, names)
                    self.require_runtime(names)
        for one in (q35_dense.MODULE, q35_dense.MODULE_TCAP):
            for feature_card in (0, 1):
                with self.subTest(tp=one, feature_card=feature_card):
                    names = self.dense_prefill(k3.Q35_TP_OF[one], tp=True, feature_card=feature_card)
                    self.assertEqual(names.count('prefill_tp'), 2)
                    self.require_runtime(names)

    def test_real_q3_module_property_selects_lean_iq_and_row_scale(self):
        self.require_runtime([q3_engine().module])
        self.assertEqual(q3_engine().module, q3moe_one.MODULE_LEAN)
        for fmt in k3.OWN_IQ_FMT:
            with self.subTest(iq=fmt):
                selected = q3_engine(iq=(fmt,)).module
                self.assertEqual(selected, q3moe_one.MODULE_IQ)
                self.require_runtime([selected])
        for fmt, (want, _) in q3moe_one.MODULE_RS.items():
            with self.subTest(rs=fmt):
                selected = q3_engine(rs=(fmt,)).module
                self.assertEqual(selected, want)
                self.require_runtime([selected])

    def test_q3_experimental_or_invalid_tapes_are_not_promoted(self):
        for flag in ('fused', 'gu', 'attn_one', 'attn_split'):
            with self.subTest(flag=flag):
                self.assertEqual(q3_engine(**{flag: False}).module, q3moe_one.MODULE)
                self.assertNotIn(q3moe_one.MODULE, kernels.RUNTIME_SET)
                with self.assertRaises(ValueError):
                    q3_engine(iq=('IQ3_S',), **{flag: False}).module
        for iq, rs in ((('IQ3_S',), ('Q4_0',)), ((), ('Q4_0', 'Q2_0'))):
            with self.subTest(iq=iq, rs=rs), self.assertRaises(ValueError):
                q3_engine(iq=iq, rs=rs).module

    def test_real_q3_prefill_selects_base_or_iq_and_refuses_rs(self):
        for fmt in ('Q4_K', 'Q6_K') + tuple(q3m_ops.IQ_KERNEL):
            with self.subTest(fmt=fmt):
                names = self.q3_prefill(fmt)
                self.assertEqual(names, [q3m_ops.IQ_MODULE if fmt in q3m_ops.IQ_KERNEL else 'q3m_prefill'])
                self.require_runtime(names)
        for fmt in q3moe_one.MODULE_RS:
            with self.subTest(fmt=fmt), self.assertRaises(ValueError):
                self.q3_prefill(fmt)

    def test_lazy_sampler_and_guard_are_included(self):
        # Exercise the lazy sampler module selection, without allocating candidates.
        with patch.dict(topk._mods, {}, clear=True):
            mod = NS(own_topk_rows=object())
            with patch.object(hip, 'Module', side_effect=lambda name: (self.loads.append(name), mod)[1]):
                topk._kernel(0)
        # Exercise guard module selection with CPU pointer arguments; the HIP
        # primitive records no work and never dereferences those addresses.
        E = k3.Engine.__new__(k3.Engine)
        pos = torch.zeros(1, dtype=torch.int32)
        E.guard_caps = {pos.data_ptr(): (pos, 4)}
        E.guard_commit, E.guard_T, E.T, E._glc = False, 1, 1, None
        E.params = torch.zeros(1, dtype=torch.int32)
        E.sync = torch.zeros(2 * k3.MAXSEG + k3.STATUS_WORDS, dtype=torch.int32)

        def guard_module(name):
            self.loads.append(name)
            return NS(k3_guard=NS(fn=123))

        with patch.object(torch.cuda, 'current_device', return_value=0), patch.object(k3, '_m', side_effect=guard_module), \
                patch.object(hip, 'hip', return_value=NS(hipModuleLaunchKernel=lambda *args: 0)):
            E._guard(0)
        self.assertIn('own_topk', self.loads)
        self.assertIn('k3_guard', self.loads)
        self.require_runtime(self.loads)

    def test_frozen_old_registry_reproduces_missing_runtime_selection(self):
        selected = q35_dense.module_of(Metadata(('Q4_K',)))
        names = [selected, k3.Q35_TP_OF[selected], q35_mtp.MODULE, q35_dense_verify.MODULE,
                 q35_dflash.MODULE, q3_engine().module] + self.dense_prefill(selected) + self.q3_prefill('Q4_K')
        self.require_runtime(names)
        with patch.object(kernels, 'RUNTIME_SET', OLD_RUNTIME):
            with self.assertRaisesRegex(AssertionError, 'selected runtime objects missing'):
                self.require_runtime(names)

    def test_experimental_objects_remain_excluded_and_names_are_unique(self):
        names = kernels.RUNTIME_SET
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(any(n.endswith(('_trace', '_swiglu')) or '_abl' in n for n in names))
        self.assertTrue(set(k3.Q3M_VER_MODULES).isdisjoint(names))
        self.assertNotIn('kv_codec', names)
        self.assertNotIn('gg_mv_ref', names)
        self.assertNotIn('gg_mmvf_ref', names)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
