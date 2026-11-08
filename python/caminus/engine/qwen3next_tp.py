"""Qwen3-Coder-Next (llama.cpp arch qwen3next: 48 layers, D 2048, every 4th layer gated attention, the rest Gated
DeltaNet, MoE 512 experts top-10 + one gated shared expert) on two cards with tensor parallelism in the K3 engine.

It reuses FlashNextTP's pass machinery (staging, launch / wait / bounded retry, verify, lazy commit, the head halves, the
exchange slots) and its memoized weight loaders; only the metadata and the tape differ. Flash-Next's hyper-connection
streams, PLE n-gram memory and QSA indexer do not exist here: one residual stream (hc = 1), no PLE layers, dense
attention up to max_ctx (qsa_limit None).

The split, as FlashNextTP's but grouped where qwen3next groups:
- GDN: card c takes key heads ks .. ks + kc - 1 and their value heads 2 ks .. 2 (ks + kc) - 1 (qwen3next repeats q / k
  INTERLEAVED: value head h reads key head h / 2, llama.cpp qwen3next.cpp's repeat_4d of [hd, 1, nk] -> [hd, 2, nk]; the
  module k3_gguf_q3n's OP_GDN_STEP maps kh = h / (H / nk), Flash-Next's tiles kh = h % nk). Its slices of attn_qkv / conv
  (q, k rows of its key heads, v rows of its value heads), attn_gate (z), ssm_ba (de-interleaved: per key head k the rows
  [b(2) | a(2)] = 4k + g for beta, 4k + 2 + g for alpha, value head 2k + g), dt / a, its heads' state, and ssm_out's
  columns of its value heads (contiguous: a partial sum).
- Attention: card c takes kv head c and its 8 query heads ([q | gate] rows per head), attn_output's columns for them.
- MoE: every expert's 512-wide intermediate split by moe_blocks of 32 (8 / 8: the K-split down projection needs block
  boundaries of its format, IQ3_S / Q6_K 256), the shared expert the same way; each card's FFN output is a partial sum.
- Replicated on both cards: the norms, the residual, the router, the shared-expert gate.
- Head: the vocabulary rows split by vocab_share.
OP_XCHG sums card 0's partial first on both cards, so the replicated state stays bitwise equal (check_replicas)."""
import ctypes
import os
import re
import numpy as np
import torch
from . import k3
from .flashnext import EPS, DirectReader
from .flashnext_tp import FlashNextTP, Slots, _hip
from gguf.constants import GGML_QUANT_SIZES          # noqa: E402  (gguf-py is on the path through caminus.engine.flashnext)

MODULE = 'k3_gguf_q3n'
# the header keys the host projection reads (under the qwen3next. prefix; caminus.arch.meta keeps the same ones)
PROJECTION_KEYS = ('context_length', 'block_count', 'full_attention_interval', 'attention.key_length', 'ssm.time_step_rank',
                   'ssm.state_size')


def projection(hp, weights):
    """Qwen3NextTP's host projection per card (no measured capacity entry yet), from its header (hp: {key: value} of
    PROJECTION_KEYS) and weights (the bytes of every tensor but token_embd, which stays on the host): 55 % of the weights (the
    default split is even: 8 / 16 GDN key heads, 8 / 16 expert blocks, half the vocabulary), the card's q8_0 KV (one kv
    head of each attention layer), 768 MiB of scratch (the tape, the cache's two slots). dict(n_ctx, kv_pos, state,
    need_mib(c)); the engine's own check (vram_check) and caminus serve's max_ctx auto (caminus.arch) both read it"""
    nl, fai, hd = int(hp['block_count']), int(hp['full_attention_interval']), int(hp['attention.key_length'])
    nv, S = int(hp['ssm.time_step_rank']), int(hp['ssm.state_size'])
    kv_pos = (nl // fai) * (2 * hd // 32 * 34)            # per card: one kv head of each attention layer
    return dict(n_ctx=int(hp.get('context_length', 262144)), kv_pos=kv_pos,
                state=(nl - nl // fai) * nv * S * S * 4 // 2,   # per card: half the GDN heads' states
                need_mib=lambda c: (0.55 * weights + kv_pos * int(c)) / 2 ** 20 + 768.0)


class Qwen3NextTP(FlashNextTP):
    def __init__(self, path, devices=(0, 1), max_ctx=4096, q8=True, module=None, local_only=False, gdn_keys=8, moe_blocks=8,
                 vocab_share=0.5, ntok=1, feature_layers=(), limit_us=200_000, max_retries=8, tape_only=False, feature_card=0,
                 vram_check=True, debug_layers=None):
        """As FlashNextTP (its docstring), for a qwen3next GGUF. Card 0's share: gdn_keys of the 16 GDN key heads,
        moe_blocks of each expert's 16 intermediate blocks of 32, vocab_share of the output rows. vram_check: before
        anything is read, refuse (capacity.CapacityError) a max_ctx whose host projection (projection(): no measured
        capacity entry for this model yet, a conservative one) passes a card's cap; self.vram_mib. max_ctx defaults to
        4096.
        debug_layers: TEST-ONLY, build only the first n layers (then the output norm and head as usual): after a pass the
        replicated residual self.res[c] is llama.cpp's l_out of layer n - 1 (tests/engine/test_q3n_e2e.py layers)."""
        from gguf import GGUFReader
        self.rd = GGUFReader(path, 'r'); self.T = {t.name: t for t in self.rd.tensors}; self.path = path
        f = {x.name: x for x in self.rd.fields.values()}
        arch = f['general.architecture'].contents()
        if arch != 'qwen3next': raise ValueError(f'Qwen3NextTP: {path} is a {arch!r} GGUF, not qwen3next')
        hp = lambda k, d=None: f[f'qwen3next.{k}'].contents() if f'qwen3next.{k}' in f else d
        self.nl, self.D, self.hc = int(hp('block_count')), int(hp('embedding_length')), 1
        self.fai = int(hp('full_attention_interval', 4))
        self.k_used, self.ple_layers = int(hp('expert_used_count')), set()
        self.ffn, self.n_exp = int(hp('expert_feed_forward_length')), int(hp('expert_count'))
        self.ffn_sh = int(hp('expert_shared_feed_forward_length', self.ffn))
        self.nh, self.nkv, self.hd, self.nrot = (int(hp('attention.head_count')), int(hp('attention.head_count_kv')),
                                                 int(hp('attention.key_length')), int(hp('rope.dimension_count')))
        self.ssm = dict(d_conv=int(hp('ssm.conv_kernel')), d_inner=int(hp('ssm.inner_size')), S=int(hp('ssm.state_size')),
                        nv=int(hp('ssm.time_step_rank')), nk=int(hp('ssm.group_count')))
        eps = float(hp('attention.layer_norm_rms_epsilon'))
        if np.float32(eps) != np.float32(EPS): raise ValueError(f'Qwen3NextTP: rms eps {eps} != the engine EPS {EPS}')
        s = self.ssm
        # the shapes the tape and the module's ops are written for (each one checked, never assumed)
        assert (s['d_conv'], s['S'], s['nk'], s['nv'], s['d_inner']) == (4, 128, 16, 32, 4096), s
        assert (self.nh, self.nkv, self.hd) == (16, 2, 256) and self.nrot <= self.hd, (self.nh, self.nkv, self.hd, self.nrot)
        assert self.ffn % 32 == 0 and self.ffn_sh == self.ffn, (self.ffn, self.ffn_sh)
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.nrot)
        self.extra, self.ple, self.dr = {}, None, DirectReader(path)
        self.n_ctx_train = int(hp('context_length')) if hp('context_length') is not None else None
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or max_ctx < 1:
            raise ValueError(f'max_ctx {max_ctx!r}: a positive int (the KV cache positions)')
        if self.n_ctx_train is not None and max_ctx > self.n_ctx_train:
            raise ValueError(f'max_ctx {max_ctx} > the model context_length {self.n_ctx_train}')
        if isinstance(ntok, bool) or not isinstance(ntok, int) or not 1 <= ntok <= max_ctx:
            raise ValueError(f'ntok {ntok!r}: an int in [1, max_ctx {max_ctx}]')
        self.q8, self.max_ctx, self.devs, self.split, self.module = q8, int(max_ctx), devices, None, module or MODULE
        # the VRAM this engine allocates per card, checked against the guard's line before anything is read (projection())
        self._proj = projection({k: hp(k) for k in PROJECTION_KEYS if hp(k) is not None},
                                sum(int(t.n_bytes) for t in self.rd.tensors if t.name != 'token_embd.weight'))
        self.vram_mib = self.projection_mib()
        if vram_check:
            from ..device import capacity
            capacity.check(list(devices), [self.vram_mib] * len(devices), f'Qwen3NextTP {os.path.basename(path)} (max_ctx {self.max_ctx}: '
                           f'~{self.vram_mib:.0f} MiB per card)')
        self.limit_us, self.max_retries, self.bails = limit_us, max_retries, []
        self.grid, self.fuse_post = None, True
        self.layer_seg, self.fresh, self.poisoned = [], True, None
        self.engines, self.res, self.pos, self.inputs, self.logits = [], [], [], [], None
        # FlashNextTP's state without the Flash-Next parts: no QSA (dense attention to max_ctx), no indexer caches, no PLE
        self.idx_cache, self.qsa, self.qsa_limit, self.dual, self._pool_ok = False, False, None, False, 0
        self.qsa_from = None              # (the status' attention kind: dense at every depth; absent, the session's QSA_EXACT default
                                          # reported 'qsa' past 2051 for a model with no QSA)
        self.model_tag = 'qwen3next'
        self.geom = dict(n_layer=self.nl, n_embd=self.D, n_attn=sum(1 for il in range(self.nl) if il % self.fai == self.fai - 1),
                         n_head=self.nh, n_head_kv=self.nkv, head_dim=self.hd, n_expert=self.n_exp, n_ff_exp=self.ffn, **s)
        self.local_only, self.gdn_keys, self.moe_blocks, self.vocab_share, self.ntok = local_only, gdn_keys, moe_blocks, vocab_share, ntok
        assert 0.0 < self.vocab_share < 1.0, self.vocab_share
        self.vram_plan = None
        self.tape_only = tape_only
        self._tok, self._emb = False, {}
        self.ple_guess, self._during = 0, None
        self.feature_layers = list(feature_layers)
        assert feature_card in (0, 1), feature_card
        self.feature_card = feature_card
        hip = _hip()
        for a, b in ((0, 1), (1, 0)):
            hip.hipSetDevice(devices[a]); hip.hipDeviceEnablePeerAccess(devices[b], 0)
            hip.hipGetLastError()
        torch.cuda.set_device(devices[0])
        self.n_ex = 2 * self.nl
        self.token, self.seq = 0, 0                   # (FlashNextTP's counters; staged: its property)
        self._card_ev, self.card_ms = None, None
        self.shards, self.weights = {}, {}
        self.qsa_sel = [[], []]                       # (check_replicas compares these: none without QSA)
        self.nl_build = self.nl if debug_layers is None else int(debug_layers)
        assert 1 <= self.nl_build <= self.nl, debug_layers
        self._build_tp()


    def projection_mib(self, max_ctx=None):
        """MiB per card this engine allocates at max_ctx (default: its own): projection()'s need_mib"""
        return self._proj['need_mib'](self.max_ctx if max_ctx is None else int(max_ctx))

    def prefetch(self, tokens, positions):
        """no PLE rows to read ahead (the speculative loop's hook)"""

    def _build_tapes(self):
        D, hd, S_ = self.D, self.hd, self.ssm['S']
        nh, nkv = self.nh // 2, self.nkv // 2                       # per card: 8 query heads, 1 kv head
        NK, NV = self.ssm['nk'], self.ssm['nv']; R = NV // NK          # 16 key heads, 32 value heads, 2 value heads per key head
        k0 = self.gdn_keys; assert 0 < k0 < NK, k0
        kc, ks = [k0, NK - k0], [0, k0]                             # GDN key heads per card, first key head
        VH = [list(range(R * ks[c], R * (ks[c] + kc[c]))) for c in (0, 1)]   # its value heads (grouped: h // R = key head)
        nbt = self.ffn // 32; mb = [(0, self.moe_blocks), (self.moe_blocks, nbt)]   # MoE intermediate, in 32s
        assert 0 < self.moe_blocks < nbt, (self.moe_blocks, nbt)
        ffc = [32 * (b1 - b0) for b0, b1 in mb]
        CH = [2 * S_ * kc[c] + S_ * R * kc[c] for c in (0, 1)]       # qkv channels: q, k (128 each per key head), 2 v heads
        KD, VD = NK * S_, NV * S_                                     # attn_qkv rows: q [KD] | k [KD] | v [VD]
        Tn = self.ntok
        RPT = dict(router=(4, 4), attn_qkv=(16, 16), ssm_out=(8, 8), attn_output=(16, 16), down_shexp=(16, 16)) if Tn == 1 else \
            dict(router=(4, 4), attn_qkv=(16, 16), ssm_out=(8, 8), attn_output=(8, 8), down_shexp=(64, 64))
        E, S, res, pos, epoch, slots = [], [], [], [], [], []
        # the per-pass inputs, one contiguous block per card (as FlashNextTP: 64 words of scalars, then the residual [Tn, D];
        # no PLE input)
        W0, nres = 64, Tn * D
        self.stage_words = (W0, nres, 0); self.stage = []
        self.hstage = torch.zeros(W0 + nres, dtype=torch.float32, pin_memory=True)
        hs = self.hstage.numpy(); self.h_i32 = hs[:W0].view(np.int32)
        self.h_res = hs[W0:W0 + nres].reshape(Tn, 1, D); self.h_ple = None
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                B = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
                q8b = lambda n: B((n + 31) // 32 * 36, dt=torch.uint8)
                E.append(k3.Engine(module=self.module, grid=k3.resident_grid(self.module), T=Tn, limit_us=self.limit_us))
                blk = B(W0 + nres); self.stage.append(blk)
                E[c].params = blk[0:16].view(torch.int32)
                pos.append(blk[16:17].view(torch.int32)); epoch.append(blk[17:18].view(torch.int32))
                res.append(blk[W0:W0 + nres].view(Tn, D))
                assert blk.data_ptr() % 256 == 0 and res[c].data_ptr() == blk.data_ptr() + 4 * W0
                assert E[c].params.data_ptr() == blk.data_ptr() and pos[c].data_ptr() == blk.data_ptr() + 64 and epoch[c].data_ptr() == blk.data_ptr() + 68
                av_ = max(nh * hd, S_ * R * kc[c])
                S.append(dict(mixed=B(Tn, D), mq=q8b(Tn * D), blkp=B(Tn * D), blk=B(Tn * D), zero=B(Tn * D),
                              qkv=B(Tn * CH[c]), z=B(Tn * S_ * R * kc[c]), al=B(Tn * R * kc[c]), be=B(Tn * R * kc[c]),
                              gates=B(Tn * 2 * R * kc[c]), conv=B(Tn * CH[c]), qk=B(Tn * 2 * kc[c] * S_), att=B(Tn * av_),
                              fin=B(Tn * S_ * R * kc[c]), fq=q8b(Tn * av_),
                              qf=B(Tn * nh * 2 * hd), kr=B(Tn * nkv * hd), vr=B(Tn * nkv * hd), qo=B(Tn * nh * hd), ko=B(Tn * nkv * hd),
                              gat=B(Tn * nh * hd), logit=B(Tn * self.n_exp), ids=B(Tn * self.k_used, dt=torch.int32), wts=B(Tn * self.k_used),
                              h=B(Tn * self.k_used * ffc[c]), hq=q8b(Tn * self.k_used * ffc[c]), moe=B(Tn * D), shh=B(Tn * ffc[c]),
                              shu=B(Tn * ffc[c]), shq=q8b(Tn * ffc[c]), shx=B(Tn * D), sgl=B(Tn), ffp=B(Tn * D), ffo=B(Tn * D)))
            slots.append(Slots(self.devs[c], self.n_ex, Tn * D))
        pull = Slots(self.devs[1], self.n_ex, Tn * D) if os.environ.get('K3_XPULL', '1') == '1' and not self.local_only and Tn == 1 else None
        self._pull = pull
        self.engines, self.S, self.res, self.pos, self.inputs, self.epoch, self.slots = E, S, res, pos, [None, None], epoch, slots
        self.state = {}; self.layer_seg = [[], []]; self.seg_names = [{}, {}]
        ON = lambda c: torch.cuda.device(self.devs[c])
        self.qsa_score, self.qsa_sel = [], [[], []]
        # split attention (kernels/k3_attn_split.h, ATTN_G 8 in this module): every position up to max_ctx is a cell
        self.attn_cells = self.max_ctx
        self._attn_part = [None, None]
        def akw(c):
            if not E[c].split_attn(): return {}
            if self._attn_part[c] is None:
                ss = k3.ATTN_SS_OF.get(E[c].module, k3.ATTN_SS)
                ns = (self.attn_cells + ss - 1) // ss
                self._attn_part[c] = torch.zeros(Tn * nh * ns * k3.ATTN_RS, device='cuda')
            return dict(max_cells=self.attn_cells, part=self._attn_part[c])
        st_ = lambda key, make: self.state[key] if key in self.state else self.state.setdefault(key, make())
        pf_cap = [(re.compile(k), int(v)) for k, v in (e.rsplit(':', 1) for e in os.environ.get('K3_PF_CAP', '.:16384' if Tn == 1 else '.:0').split(',') if e)]
        gg_rpt = [(re.compile(k), [int(u) for u in (v.split('/') * 2)[:2]])
                  for k, v in [e.rsplit(':', 1) for e in os.environ.get('K3_GG_RPT', '').split(',') if e]]
        ggr = lambda name, rpt, c: next((v[c] for k, v in gg_rpt if k.search(name)), rpt)

        def pf_set(c, si, name):
            cap = next((v for k, v in pf_cap if k.search(name)), None)
            if cap is not None: E[c].segs[si].aux = cap + 1

        def mv(c, W, x, out, dep, rpt=16):
            si = E[c].gg_matvec(W['w'], W['fmt'], W['rows'], W['cols'], W['rb'], x, out, rows_per_task=ggr(W['name'], rpt, c), dep=dep)
            pf_set(c, si, W['name']); self.seg_names[c][si] = W['name']; return si

        def f32mv(c, w, x, out, rpt, dep, name):
            si = E[c].f32_mv(w, x, out, rows_per_task=rpt, dep=dep); pf_set(c, si, name); self.seg_names[c][si] = name; return si

        def xchg(c, partial, out, x, dep):
            if pull is not None:
                return E[c].xchg(partial, out, slots[1 - c].slot(x), slots[c].slot(x), epoch[c], x, c, local_only=self.local_only, dep=dep,
                                 pull=pull.slot(x) + 64, pull_mode=1 if c == 1 else 2)
            return E[c].xchg(partial, out, slots[1 - c].slot(x), slots[c].slot(x), epoch[c], x, c, local_only=self.local_only, dep=dep)

        self.features = torch.zeros(len(self.feature_layers), Tn, D, device=self._dev(self.feature_card))
        if self.feature_layers: raise NotImplementedError('Qwen3NextTP: the DFlash feature export comes with M3')

        def norm(name, dep):
            """mixed = rms_norm(res) * w (fp32, for the F32 router / shared-expert gate) and its q8_1 copy mq (the GG
            matvecs' activation), on both cards"""
            w = self._rep_f32(name, (1, D)); out = []
            for c in (0, 1):
                with ON(c): out.append(E[c].rms_mul(res[c], w[c], S[c]['mixed'], EPS, dep=dep[c], q8=S[c]['mq']))
            return out

        self._tapes, self._tape_meta = {}, {}
        dep = [-1, -1]
        for il in range(self.nl_build):
            p = f'blk.{il}'
            for c in (0, 1): self.layer_seg[c].append((il, len(E[c].segs)))
            M = norm(f'{p}.attn_norm.weight', dep)
            BLK = []
            if il % self.fai != self.fai - 1:                  # GDN, split by key heads (grouped value heads)
                qkv_r = [[(S_ * ks[c], S_ * (ks[c] + kc[c])), (KD + S_ * ks[c], KD + S_ * (ks[c] + kc[c])),
                          (2 * KD + S_ * VH[c][0], 2 * KD + S_ * (VH[c][-1] + 1))] for c in (0, 1)]
                qkv = self._rows(f'{p}.attn_qkv.weight', qkv_r)
                assert int(self.T[f'{p}.attn_qkv.weight'].shape[1]) == 2 * KD + VD
                assert self.T[f'{p}.ssm_conv1d.weight'].tensor_type.name == 'F32'
                cw = self._full(f'{p}.ssm_conv1d.weight').view(torch.float32).view(-1, 4)
                assert cw.shape[0] == 2 * KD + VD, cw.shape
                conv_w = [torch.cat([cw[r0:r1] for r0, r1 in qkv_r[c]]).contiguous() for c in (0, 1)]; conv_w[1] = conv_w[1].to(self._dev(1))
                conv_w = self.weights.setdefault(('conv', f'{p}.ssm_conv1d.weight'), conv_w)
                z = self._rows(f'{p}.attn_gate.weight', [[(S_ * VH[c][0], S_ * (VH[c][-1] + 1))] for c in (0, 1)])
                # ssm_ba: per key head k the rows [beta(R) | alpha(R)]: value head h = R k + g -> beta row 2R k + g, alpha 2R k + R + g
                # (both are row slices of the one ssm_ba tensor: FlashNextTP.shards keeps the last one by name; the batched
                # prefill, M2, must take the two slices from here instead)
                be = self._rows(f'{p}.ssm_ba.weight', [[(2 * R * (h // R) + h % R, 2 * R * (h // R) + h % R + 1) for h in VH[c]] for c in (0, 1)])
                al = self._rows(f'{p}.ssm_ba.weight', [[(2 * R * (h // R) + R + h % R, 2 * R * (h // R) + R + h % R + 1) for h in VH[c]] for c in (0, 1)])
                dt = self._f32_idx(f'{p}.ssm_dt.bias', VH); sa = self._f32_idx(f'{p}.ssm_a', VH)
                nw = self._rep_f32(f'{p}.ssm_norm.weight')
                bs = GGML_QUANT_SIZES[self.T[f'{p}.ssm_out.weight'].tensor_type][0]
                assert all(S_ * VH[c][0] % bs == 0 and S_ * (VH[c][-1] + 1) % bs == 0 for c in (0, 1)), f'ssm_out {bs}-blocks vs the value-head split'
                so = self._cols(f'{p}.ssm_out.weight', [[(S_ * VH[c][0] // bs, S_ * (VH[c][-1] + 1) // bs)] for c in (0, 1)])
                for c in (0, 1):
                    with ON(c):
                        s = S[c]; H = R * kc[c]
                        conv_st = st_((c, il, 'conv'), lambda: torch.zeros(CH[c], 3, device='cuda'))
                        cxp = st_((c, il, 'convx'), lambda: torch.zeros(Tn * CH[c], device='cuda'))
                        gblk = st_((c, il, 'gdn'), lambda: torch.zeros(k3.gdn_block_size(H, S_, kc[c], Tn), device='cuda'))
                        a_ = mv(c, al[c], s['mq'], s['al'], M[c]); b_ = mv(c, be[c], s['mq'], s['be'], M[c])
                        gt = E[c].also(E[c].gdn_gates(s['al'], s['be'], dt[c], sa[c], s['gates'], dep=a_), b_)
                        q_ = mv(c, qkv[c], s['mq'], s['qkv'], M[c], RPT['attn_qkv'][c])
                        c_ = E[c].gdn_conv(s['qkv'], conv_w[c], conv_st, s['conv'], cxp, dep=q_)
                        l_ = E[c].gdn_l2(s['conv'], s['qk'], 2 * kc[c], S_, EPS, x_stride=CH[c], dep=c_)
                        z_ = mv(c, z[c], s['mq'], s['z'], M[c])
                        nv = Tn * S_ * H; fqv = s['fq'][:nv // 32 * 36]
                        st = E[c].also(E[c].gdn_step(s['qk'], s['conv'][2 * S_ * kc[c]:], s['gates'], gblk, s['att'][:nv], H, S_, kc[c], CH[c], dep=l_), gt)
                        E[c].gdn_save(s['qk'], s['conv'][2 * S_ * kc[c]:], s['gates'], gblk, H, S_, kc[c], CH[c], dep=st)
                        nm = E[c].also(E[c].gdn_norm(s['att'][:nv], nw[c], s['z'], s['fin'], H, S_, EPS, dep=st, q8=fqv), z_)
                        BLK.append(mv(c, so[c], fqv, s['blkp'], nm, RPT['ssm_out'][c]))
            else:                                                # gated attention, split by kv head
                aq = self._rows(f'{p}.attn_q.weight', [[(nh * 2 * hd * c, nh * 2 * hd * (c + 1))] for c in (0, 1)])
                ak = self._rows(f'{p}.attn_k.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
                av = self._rows(f'{p}.attn_v.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
                qn_, kn_ = self._rep_f32(f'{p}.attn_q_norm.weight'), self._rep_f32(f'{p}.attn_k_norm.weight')
                bph = hd // GGML_QUANT_SIZES[self.T[f'{p}.attn_output.weight'].tensor_type][0]   # blocks per head
                assert bph * GGML_QUANT_SIZES[self.T[f'{p}.attn_output.weight'].tensor_type][0] == hd
                ao = self._cols(f'{p}.attn_output.weight', [[(nh * bph * c, nh * bph * (c + 1))] for c in (0, 1)])
                pb = (2 * nkv * hd // 32 * 34) if self.q8 else 4 * nkv * hd
                for c in (0, 1):
                    with ON(c):
                        s = S[c]; cache = st_((c, il, 'kv'), lambda: torch.zeros(self.max_ctx * pb, dtype=torch.uint8, device='cuda'))
                        k_ = mv(c, ak[c], s['mq'], s['kr'], M[c]); v_ = mv(c, av[c], s['mq'], s['vr'], M[c])
                        kd = E[c].qk_nr(s['kr'], hd, nkv, hd, kn_[c], pos[c], s['ko'], self.nrot, self.theta_scale, EPS, dep=k_)
                        vd = v_
                        if self.q8: kd = E[c].hadamard(s['ko'], hd, dep=kd); vd = E[c].hadamard(s['vr'], 64, dep=vd)
                        ap = E[c].also(E[c].kv_append(s['ko'], s['vr'], cache, pos[c], q8=self.q8, dep=kd), vd)
                        q_ = mv(c, aq[c], s['mq'], s['qf'], M[c])
                        qd = E[c].qk_nr(s['qf'], 2 * hd, nh, hd, qn_[c], pos[c], s['qo'], self.nrot, self.theta_scale, EPS, dep=q_)
                        if self.q8: qd = E[c].hadamard(s['qo'], hd, dep=qd)
                        na = Tn * nh * hd
                        a_ = E[c].also(E[c].attn_dec(s['qo'], cache, pos[c], s['att'][:na], nh, nkv, hd, 1.0 / np.sqrt(hd), q8=self.q8, dep=ap, **akw(c)), qd)
                        if self.q8: a_ = E[c].hadamard(s['att'][:na], 64, rows_per_task=4, dep=a_)
                        fqa = s['fq'][:na // 32 * 36]
                        a_ = E[c].gate_mul(s['att'][:na], s['qf'], 2 * hd, hd, s['gat'], hd, dep=a_, q8=fqa)
                        BLK.append(mv(c, ao[c], fqa, s['blkp'], a_, RPT['attn_output'][c]))
            # the block's partial sums -> both cards; res += block (llama.cpp's attn_residual), then the post-attention norm
            RB = []
            for c in (0, 1):
                with ON(c):
                    x_ = xchg(c, S[c]['blkp'], S[c]['blk'], 2 * il, BLK[c])
                    RB.append(E[c].add_sum(S[c]['blk'], S[c]['zero'], res[c].reshape(-1), dep=x_))
            M = norm(f'{p}.post_attention_norm.weight', RB)
            # ffn: routed (router replicated, experts split by intermediate rows) beside the shared expert (split the same way)
            rw = self._rep_f32(f'{p}.ffn_gate_inp.weight', (self.n_exp, D)); sw = self._rep_f32(f'{p}.ffn_gate_inp_shexp.weight', (1, D))
            rr = [[(32 * b0, 32 * b1)] for b0, b1 in mb]
            sgu, suu = self._rows(f'{p}.ffn_gate_shexp.weight', rr), self._rows(f'{p}.ffn_up_shexp.weight', rr)
            mbt = {}
            for n in ('ffn_down_shexp', 'ffn_down_exps'):
                qk = GGML_QUANT_SIZES[self.T[f'{p}.{n}.weight'].tensor_type][0]
                assert all(32 * v % qk == 0 for r in mb for v in r), f'{p}.{n}: {qk}-blocks vs the intermediate split {mb} (in 32s)'
                mbt[n] = [(32 * b0 // qk, 32 * b1 // qk) for b0, b1 in mb]
            sdd = self._cols(f'{p}.ffn_down_shexp.weight', [[b] for b in mbt['ffn_down_shexp']])
            g = self._exp_rows(f'{p}.ffn_gate_exps.weight', [r[0] for r in rr]); u = self._exp_rows(f'{p}.ffn_up_exps.weight', [r[0] for r in rr])
            d = self._exp_cols(f'{p}.ffn_down_exps.weight', mbt['ffn_down_exps'])
            FF = []
            for c in (0, 1):
                with ON(c):
                    s = S[c]
                    r_ = f32mv(c, rw[c], s['mixed'], s['logit'], RPT['router'][c], M[c], 'f32_router_logits')
                    r_ = E[c].router(s['logit'], self.k_used, s['ids'], s['wts'], dep=r_)
                    sl = f32mv(c, sw[c], s['mixed'], s['sgl'], 8, M[c], 'f32_shexp_gate')
                    sg_ = mv(c, sgu[c], s['mq'], s['shh'], M[c]); su_ = mv(c, suu[c], s['mq'], s['shu'], M[c])
                    s_ = E[c].also(E[c].gg_q8_1(s['shh'], s['shq'], dep=sg_, mul=s['shu']), su_)
                    s_ = E[c].also(mv(c, sdd[c], s['shq'], s['shx'], s_, RPT['down_shexp'][c]), sl)
                    r_ = E[c].gg_moe_gu(g[c]['w'], u[c]['w'], g[c]['fmt'], g[c]['slab'], ffc[c], D, s['ids'], s['mq'], s['h'], rows_per_task=8, dep=r_, up_fmt=u[c]['fmt'])
                    r_ = E[c].gg_q8_1(s['h'], s['hq'], dep=r_)
                    r_ = E[c].gg_moe_down(d[c]['w'], d[c]['fmt'], d[c]['slab'], ffc[c], D, s['ids'], s['wts'], s['hq'], s['moe'], rows_per_task=8, dep=r_)
                    f_ = E[c].also(E[c].shexp_add(s['moe'], s['shx'], s['sgl'], s['ffp'], dep=r_), s_)
                    x_ = xchg(c, s['ffp'], s['ffo'], 2 * il + 1, f_)
                    FF.append(E[c].add_sum(s['ffo'], s['zero'], res[c].reshape(-1), dep=x_))
            dep = FF
        for c in (0, 1): self.layer_seg[c].append(('head', len(E[c].segs)))
        M = norm('output_norm.weight', dep)
        nvoc = int(self.T['output.weight'].shape[1]); vh = int(round(nvoc * self.vocab_share))
        out = self._rows('output.weight', [[(0, vh)], [(vh, nvoc)]])
        self.vh = vh; self.head_w = out
        self.logits_half, self.d_max, self.h_max, self.h_st, self.ev_done = [], [], [], [], []
        for c in (0, 1):
            with ON(c):
                self.d_max.append((torch.zeros(Tn, device='cuda'), torch.zeros(Tn, dtype=torch.int64, device='cuda')))
                self.ev_done.append(torch.cuda.Event())
                self.logits_half.append(torch.zeros(Tn, out[c]['rows'], device='cuda'))
            self.h_max.append((torch.zeros(Tn, pin_memory=True), torch.zeros(Tn, dtype=torch.int64, pin_memory=True)))
            self.h_st.append(torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True))
        for c in (0, 1):
            with ON(c):
                mv(c, out[c], S[c]['mq'], self.logits_half[c], M[c], 64)
                E[c].build(strict=not self.tape_only)
        g_ = [(sorted(cap for _, cap in E[j].guard_caps.values()), E[j].guard_commit, E[j].T) for j in (0, 1)]
        assert g_[0] == g_[1], f'k3 TP: the cards guard different bounds {g_}'
        self._tapes['dense'] = list(E); self._tape_meta['dense'] = (self.layer_seg, self.seg_names)
        torch.cuda.set_device(self.devs[0])
