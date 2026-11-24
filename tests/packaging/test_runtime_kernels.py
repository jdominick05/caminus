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
from caminus.engine import k3, q3moe_one, q3moe_tp, q35_dense, q35_dense_verify, q35_dense_verify_tp, q35_mtp, q35_dflash
from caminus.prefill import attn, gdn, glue, q3m, q3m_ops, q3m_tp, q35own, q35own_tp, xchg


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
        for module in (hip, k3, attn, gdn, glue, xchg, q3m_tp):
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

    def q3_prefill(self, fmt, q8=False):
        m = NS(dev=0, D=256, nh=8, nkv=1, hd=128, k_used=1, n_exp=2, ffn=256, q8=q8,
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

    def test_real_gsq_dflash_selector_is_shipped(self):
        # Exercise the constructor's actual selector, stopping at file opening.
        # Complete verify/MTP tapes are exercised by test_q35_tape_cpu.py.
        class SelectionOnly(Exception):
            pass

        eng = NS(module=q35_dense.MODULE_GSQ, tcap=True,
                 head=dict(fmt='Q4_K'), emb=NS(type='IQ2_S'))
        drafter = q35_dflash.Dflash35.__new__(q35_dflash.Dflash35)
        with patch.object(q35_dflash.ggufio, 'GGUFFile', side_effect=SelectionOnly), self.assertRaises(SelectionOnly):
            q35_dflash.Dflash35.__init__(drafter, 'unused-selector-fixture.gguf', eng, vram_check=False)
        self.assertEqual(drafter.module, 'k3_q35_gsq_dflash')
        self.require_runtime([drafter.module])
        without = tuple(n for n in kernels.RUNTIME_SET if n != drafter.module)
        with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
            self.require_runtime([drafter.module])

    def test_real_tp_verify_constructor_selects_shipped_tp_verify_objects(self):
        # Two-card drafting is served (Swift 1.5, ThinkingCap), so the real constructor's two verify objects are in the
        # installed runtime; without them a wheel would refuse the default two-card serve.
        # CUDA/HIP access is forbidden by setUp; only the tensor tape is omitted (test_q35_verify_tp_cpu.py covers it).
        cases = [('k3_q35_swift_tp', 'k3_q35_swift_tp_ver'), ('k3_q35_tcap_tp', 'k3_q35_tcap_tp_ver')]
        for module, want in cases:
            for T in (1, 4):
                with self.subTest(module=module, T=T):
                    eng = NS(module=module, spec_T=4, nl=1, nl_build=1, D=5120, hd=256, nh=12, nkv=2,
                             S=128, f0=128, ffn=256, vh=32, vocab=64, max_ctx=64, devs=[0, 1],
                             per_wgp_c=[4, 4], limit_us=200000, feature_card=0,
                             L=[[dict(kind='attn'), dict(kind='attn')]])
                    with patch.object(q35_dense_verify_tp.Dense35VerifyTP, '_tape'):
                        ver = q35_dense_verify_tp.Dense35VerifyTP(eng, T, vram_check=False)
                    self.assertEqual([ver.module] + [e.module for e in ver.E], [want] * 3)
                    self.assertEqual([e.T for e in ver.E], [T, T])
                    self.require_runtime([ver.module])
                    without = tuple(n for n in kernels.RUNTIME_SET if n != want)
                    with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
                        self.require_runtime([ver.module])

    def test_two_card_drafters_select_the_shipped_one_card_objects(self):
        # The two-card drafted serve runs the one-card MTP / DFlash2 drafters on a TP trunk's view (DraftTarget35TP:
        # module k3_q35_*_tp, tcap as the trunk's). Exercise the real constructors' selectors, stopping at file opening.
        class SelectionOnly(Exception):
            pass

        for tp_module, tcap, mtp_want, df_want in (('k3_q35_swift_tp', False, q35_mtp.MODULE, q35_dflash.MODULE),
                                                   ('k3_q35_tcap_tp', True, q35_mtp.MODULE_TCAP, q35_dflash.MODULE_TCAP)):
            with self.subTest(module=tp_module):
                eng = NS(module=tp_module, tcap=tcap, head=dict(fmt='Q8_0' if tcap else 'Q6_K'),
                         emb=NS(type='IQ2_S' if tcap else 'Q4_K'))
                drafter = q35_dflash.Dflash35.__new__(q35_dflash.Dflash35)
                with patch.object(q35_dflash.ggufio, 'GGUFFile', side_effect=SelectionOnly), self.assertRaises(SelectionOnly):
                    q35_dflash.Dflash35.__init__(drafter, 'unused-selector-fixture.gguf', eng, vram_check=False)
                self.assertEqual(drafter.module, df_want)
                # (the MTP block's tensors in the line's format; selection ends where the projection starts)
                block = 'Q6_K' if tcap else 'Q4_0'
                eng.F = NS(T={f'blk.1.{n}.weight': NS(type=block) for n in q35_mtp.NEXTN})
                eng.path = 'unused-selector-fixture.gguf'
                mtp = q35_mtp.Mtp35.__new__(q35_mtp.Mtp35)
                with patch.object(q35_mtp, 'block_of', return_value=1), \
                        patch.object(q35_mtp.Mtp35, 'need_mib', side_effect=SelectionOnly), self.assertRaises(SelectionOnly):
                    q35_mtp.Mtp35.__init__(mtp, eng)
                self.assertEqual(mtp.module, mtp_want)
                self.require_runtime([drafter.module, mtp.module])

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
            for feature_card in (0, 1):   # (the drafter's card: its feature rows and launchers; same objects)
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
                # (q3m_prefill's wide GEMM tiles are their own object: the IQ study object runs the narrow tiles)
                self.assertEqual(names, [q3m_ops.IQ_MODULE] if fmt in q3m_ops.IQ_KERNEL else ['q3m_prefill', q3m_ops.WIDE_MODULE])
                self.require_runtime(names)
        without = tuple(n for n in kernels.RUNTIME_SET if n != q3m_ops.WIDE_MODULE)
        with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
            self.require_runtime(self.q3_prefill('Q4_K'))
        for fmt in q3moe_one.MODULE_RS:
            with self.subTest(fmt=fmt), self.assertRaises(ValueError):
                self.q3_prefill(fmt)

    def q3_prefill_tp(self, q8=False):
        # a stand-in Q3MoETP with two-card shapes (2 kv heads and their 16 query heads per card, expert rows 512 / 256):
        # the real Q3MPrefillTP constructor, its code objects recorded at load
        mat = lambda rows, K: dict(fmt='Q4_K', rows=rows, K=K, w=None)
        def card(r): return dict(q=mat(2048, 256), k=mat(256, 256), v=mat(256, 256), o=mat(256, 2048), gate=mat(8 * r, 256),
                                 up=mat(8 * r, 256), down=dict(mat(8 * 256, r), fmt='Q6_K'), attn_norm=None, ffn_norm=None,
                                 qk_par=None, router=dict(w=None), cache=None)
        m = NS(devs=[0, 1], nl=1, D=256, nh=32, nkv=4, hd=128, k_used=2, n_exp=8, ffn=768, kvc=[2, 2], exp_rows=[512, 256],
               local_only=False, broken=None, L=[[card(512), card(256)]], q8=q8,
               emb=NS(type='Q4_K', shape=(256, 64), rows=lambda: (64, 144)))
        start = len(self.loads)
        q3m_tp.Q3MPrefillTP(m, chunk=16, check=False)
        return self.loads[start:]

    def test_q3_two_card_serve_selects_shipped_objects(self):
        # caminus serve --cards 2 for Qwen3-30B-A3B (arch.Q3MoETP): the engine's decode module is the lean module's
        # two-card build, the prefill loads its GEMM object, the slot sum's object (on both cards) and the exchange's add
        tp = k3.Q3M_TP_OF[q3moe_one.MODULE_LEAN]
        self.assertEqual(tp, 'k3_q3m_lean_tp')
        self.assertEqual(q3moe_tp.MODULE, tp)
        names = self.q3_prefill_tp()
        self.assertEqual(sorted(set(names)), ['prefill_tp', 'q3m_prefill', 'q3m_prefill_tp'])
        self.assertEqual(names.count('q3m_prefill_tp'), 2)
        self.require_runtime([tp] + names)
        self.assertNotIn(k3.Q3M_TP_OF['k3_q3m_lean_trace'], kernels.RUNTIME_SET)
        for name in (tp, 'q3m_prefill_tp'):
            without = tuple(n for n in kernels.RUNTIME_SET if n != name)
            with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
                self.require_runtime([tp] + names)

    def test_q3_q8_kv_serve_selects_shipped_objects(self):
        # caminus serve --kv q8 for Qwen3-30B-A3B (arch.Q3MoE / Q3MoETP with opt['kv'] = 'q8'): one card decodes on the real
        # module property's Q8 module, two cards on its Q3M_TP_OF build (as Q3MoETP(q8=True) picks it); both prefills'
        # constructors load q3m_prefill_kvq8 beside their F16 objects (the rows and attention). All are in the runtime set,
        # and each one's removal is caught (a wheel without it would refuse the served --kv q8)
        one = q3_engine(q8=True).module
        self.assertEqual(one, 'k3_q3m_lean_kvq8')
        tp = k3.Q3M_TP_OF[one]
        self.assertEqual(tp, 'k3_q3m_lean_tp_kvq8')
        self.assertEqual(q3_engine().module, q3moe_one.MODULE_LEAN)   # (the F16 default unchanged)
        names1 = self.q3_prefill('Q4_K', q8=True)
        self.assertEqual(sorted(set(names1)), ['q3m_prefill', 'q3m_prefill_kvq8', 'q3m_prefill_mm'])
        names2 = self.q3_prefill_tp(q8=True)
        self.assertEqual(sorted(set(names2)), ['prefill_tp', 'q3m_prefill', 'q3m_prefill_kvq8', 'q3m_prefill_tp'])
        self.assertEqual(names2.count('q3m_prefill_kvq8'), 2)   # (one per card)
        self.assertNotIn('q3m_prefill_kvq8', self.q3_prefill('Q4_K') + self.q3_prefill_tp())   # (F16 prefills load no Q8 object)
        want = [one, tp] + names1 + names2
        self.require_runtime(want)
        for name in (one, tp, 'q3m_prefill_kvq8'):
            without = tuple(n for n in kernels.RUNTIME_SET if n != name)
            with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
                self.require_runtime(want)

    def test_q3_long_context_mode_selects_shipped_objects(self):
        # caminus serve --long-context for Qwen3-30B-A3B (YaRN): the real module property picks each decode module's YaRN
        # build (F16 / Q8, one card; the two-card ones through k3.Q3M_TP_OF), the prefills load the YaRN row kernel's object
        # beside their own (and the Q8 attention's object with Q8 rows); every one is in the installed runtime set
        from caminus import rope
        Y = rope.YarnMode(4.0, 32768)
        one = q3_engine(rope=Y).module
        q8 = q3_engine(rope=Y, q8=True).module
        self.assertEqual((one, q8), ('k3_q3m_lean_yarn', 'k3_q3m_lean_kvq8_yarn'))
        tps = [k3.Q3M_TP_OF[one], k3.Q3M_TP_OF[q8]]
        self.assertEqual(tps, ['k3_q3m_lean_tp_yarn', 'k3_q3m_lean_tp_kvq8_yarn'])
        for q8_ in (False, True):
            with self.subTest(q8=q8_):
                m = NS(dev=0, D=256, nh=8, nkv=1, hd=128, k_used=1, n_exp=2, ffn=256, q8=q8_, rope=Y, rope_key=Y.key(),
                       L=[{key: ('Q4_K',) for key in q3m.MM_KEYS}], emb=NS(shape=(256, 64), rows=lambda: (64, 144)))
                start = len(self.loads)
                q3m.Q3MPrefill(m, chunk=16, check=False)
                names = self.loads[start:]
                want = (['q3m_prefill', 'q3m_prefill_mm', 'q3m_prefill_kvq8', 'q3m_prefill_kvq8_yarn'] if q8_
                        else ['q3m_prefill', 'q3m_prefill_mm', 'q3m_prefill_yarn'])
                self.assertEqual(names, want)
                self.require_runtime(names)
        self.require_runtime([one, q8] + tps)
        for name in (one, 'q3m_prefill_yarn'):
            without = tuple(n for n in kernels.RUNTIME_SET if n != name)
            with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
                self.require_runtime([one, 'q3m_prefill', 'q3m_prefill_yarn'])

    def test_q35moe_serve_selects_shipped_objects(self):
        # caminus serve for Qwen3.6-35B-A3B (arch.Q35MoE): the decode module from the file (module_of / kv_module: F16 and
        # --kv q8) and the batched prefill's constructor loads in every mode (its wide expert tiles always; the Q8 rows and attention object only with a
        # Q8 engine, the YaRN attention object only in the long-context mode); all are in the installed runtime set and
        # each one's removal is caught
        from caminus.engine import q35_moe
        from caminus.prefill import q35moe
        f16 = q35_moe.kv_module(None, module=q35_moe.module_of(None))
        q8 = q35_moe.kv_module(None, module=q35_moe.module_of(None), q8=True)
        self.assertEqual((f16, q8), ('k3_q35_moe', 'k3_q35_moe_kvq8'))
        mm = lambda f: dict(fmt=f)
        layer = lambda kind: dict(kind=kind, **{k: mm('Q8_0') for k in q35moe.Q35MoEPrefill.MM_KEYS[kind] + q35moe.Q35MoEPrefill.SH_KEYS},
                                  **{k: mm('Q2_0') for k in q35moe.Q35MoEPrefill.EXP_KEYS}, al=mm('F32'), be=mm('F32'))
        names = {}
        tensor = torch.tensor
        self.stack.enter_context(patch.object(torch, 'tensor', side_effect=lambda *a, **kw: tensor(*a, **{k: v for k, v in kw.items() if k != 'device'})))
        for q8_, yarn in ((False, None), (True, None), (False, 4.0), (True, 4.0)):
            m = NS(dev=0, q8=q8_, yarn_factor=yarn, yarn=([1.0] * 32, 1.1), rope_key='yarn4', kv_format_id='q8', nl=2, nl_build=2, D=2048, nh=16, nkv=2, hd=256, k_used=8, n_exp=256,
                   ffn=512, ffn_shexp=512, S=128, NK=16, NV=32, vocab=64, emb=NS(type='BF16'), L=[layer('gdn'), layer('attn')])
            start = len(self.loads)
            pf = q35moe.Q35MoEPrefill(m, chunk=16, check=False)
            names[q8_, yarn] = self.loads[start:]
            self.assertEqual('rope' in pf.numerics(), yarn is not None)
        base = names[False, None]
        self.assertIn('q35m_prefill_mm', base)   # (the experts' wide tiles and fused gate | up: every mode)
        self.assertNotIn('q35m_prefill_kvq8', base); self.assertNotIn('prefill_attn_q35_yarn', base)
        self.assertEqual(sorted(set(names[True, None]) - set(base)), ['q35m_prefill_kvq8'])
        self.assertEqual(sorted(set(names[False, 4.0]) - set(base)), ['prefill_attn_q35_yarn'])
        self.assertNotIn('prefill_attn_q35', names[False, 4.0])   # (the YaRN object replaces the native one)
        self.assertEqual(sorted(set(names[True, 4.0]) - set(base)), ['prefill_attn_q35_yarn', 'q35m_prefill_kvq8'])
        want = [f16, q8] + [n for v in names.values() for n in v]
        self.require_runtime(want)
        for name in (f16, q8, 'q35m_prefill', 'q35m_prefill_mm', 'q35m_prefill_kvq8', 'prefill_attn_q35_yarn'):
            without = tuple(n for n in kernels.RUNTIME_SET if n != name)
            with patch.object(kernels, 'RUNTIME_SET', without), self.assertRaisesRegex(AssertionError, 'missing from package'):
                self.require_runtime(want)

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
        self.assertFalse(any(n.endswith('_trace') for n in names))
        self.assertTrue(set(k3.Q35TPV_MODULES) <= set(names))
        self.assertTrue(set(k3.Q3M_VER_MODULES).isdisjoint(names))
        self.assertNotIn('kv_codec', names)
        self.assertNotIn('gg_mv_ref', names)
        self.assertNotIn('gg_mmvf_ref', names)



class CardObjects(unittest.TestCase):
    """one register cap per card (kernels/k3.hip K3_PER_WGP): the XTX's <name>_w4 objects ship and load by card"""

    def test_every_own_decode_module_ships_its_xtx_object(self):
        own = [n for n in kernels.RUNTIME_SET if n.startswith('k3_q3m_') and not n.endswith('_w4')]
        self.assertTrue(own)
        self.assertEqual(sorted(n + '_w4' for n in own), sorted(kernels.CARD_W4))

    def test_card_object_follows_the_grid_and_the_files(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, 'k3_q3m_lean_w4.hsaco'), 'wb').close()
            with patch.object(hip, 'BUILD', d), patch.object(k3.torch.cuda, 'is_available', lambda: True):
                for per, name, want in ((4, 'k3_q3m_lean', 'k3_q3m_lean_w4'), (5, 'k3_q3m_lean', 'k3_q3m_lean'),
                                        (4, 'k3_q3m_iq', 'k3_q3m_iq')):   # (no k3_q3m_iq_w4 file: the module itself)
                    with patch.dict(os.environ, {'K3_CARD_PER_WGP': str(per)}):
                        self.assertEqual(k3.card_object(name), want, (per, name))
                with patch.dict(os.environ, {'K3_CARD_PER_WGP': '4', 'K3_CARD_OBJECTS': '0'}):
                    self.assertEqual(k3.card_object('k3_q3m_lean'), 'k3_q3m_lean')

if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
