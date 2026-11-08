"""Qwen3.6-35B-A3B (llama.cpp arch qwen35moe: 40 layers, D 2048, every 4th layer gated attention, the rest Gated DeltaNet,
MoE 256 experts top-8 + one gated shared expert) on ONE card in the K3 engine.

Not FlashNextTP: that class is two-card throughout (its loaders copy to a second device, its passes launch, wait, merge
logits and compare replicas over two cards, its tape exchanges partial sums). This one has the same public calls
(step / step_token / verify / verify_tokens / reset / check_replicas / res / bails / poisoned / fresh / seq) over one
device and one tape, and the same pass order (stage copy, run, read-backs, status copy, event), so the two could fold
into one N-card class later once gates prove both paths equal. No exchange: every head, every expert's full
intermediate and the whole vocabulary are on the card; a block's output goes straight into the residual.

The tape is Qwen3NextTP's op sequence with qwen35moe's choices (tests/engine/test_q36_golden_cpu.py checked each against
llama.cpp's intermediates): GDN value head h reads key head h % 16 (tiled: the module's default mapping), ssm_alpha and
ssm_beta are separate matrices (GG formats, or F32 through OP_F32_MV), the gated RMSNorm's gate is SiLU (K3_GDN_SILU),
attention q rows are [q | gate] per head, IMRoPE with equal text positions is NEOX on the first 64 dims, softmax top-8
renormalized, the shared expert gated by sigmoid of an F32 row. Module: k3_gguf_q35h (HauhauCS Q3_K_P's formats);
another model's formats need their own module (k3.MODULE_CAPS refuses the tape otherwise)."""
import ctypes
import os
import re
import sys
import numpy as np
import torch
from . import k3
from .flashnext import EPS, DirectReader
from gguf import quants                                # (gguf-py on the path through caminus.engine.flashnext)

MODULE = 'k3_gguf_q35h'
DENSE_MODULE = 'k3_gguf_q35d'   # the dense line (llama.cpp arch qwen35: AREX-2, a Qwen3.8-27B finetune)


class Qwen35One:
    def __init__(self, path, device=0, max_ctx=4096, q8=True, module=None, ntok=1, limit_us=200_000, max_retries=8,
                 tape_only=False, debug_layers=None, vram_check=True, vram_extra_mib=0.0, shape_only=False):
        """path: a qwen35moe GGUF. device: the torch / HIP device index (with HIP_VISIBLE_DEVICES=1 the XT is device 0).
        max_ctx: KV cache positions (no measured capacity entry yet: callers keep it small until one exists). ntok: tokens
        per pass (T > 1 runs module_for's _mt module). tape_only: build even if the module lacks an (op, format) pair
        (k3.Engine.build(strict=False); runs refuse). debug_layers: TEST-ONLY, build only the first n layers, then the output
        norm and head; after a pass res[0] is llama.cpp's l_out of layer n - 1 (tests/engine/test_q3n_e2e.py layers)."""
        from gguf import GGUFReader
        self.rd = GGUFReader(path, 'r'); self.T = {t.name: t for t in self.rd.tensors}; self.path = path
        f = {x.name: x for x in self.rd.fields.values()}
        arch = f['general.architecture'].contents()
        if arch not in ('qwen35moe', 'qwen35'):
            raise ValueError(f'Qwen35One: {path} is a {arch!r} GGUF, not qwen35moe (Qwen3.6-35B-A3B) or qwen35 (dense: AREX-2)')
        hp = lambda k, d=None: f[f'{arch}.{k}'].contents() if f'{arch}.{k}' in f else d
        self.arch, self.dense = arch, arch == 'qwen35'      # dense: one FFN per layer (gate / up / down), no router or experts
        self.nl, self.D = int(hp('block_count')), int(hp('embedding_length'))
        self.fai = int(hp('full_attention_interval', 4))
        if self.dense:
            self.k_used = self.n_exp = 0; self.ffn = int(hp('feed_forward_length')); self.ffn_sh = self.ffn
        else:
            self.k_used, self.n_exp = int(hp('expert_used_count')), int(hp('expert_count'))
            self.ffn, self.ffn_sh = int(hp('expert_feed_forward_length')), int(hp('expert_shared_feed_forward_length'))
        self.nh, self.nkv, self.hd, self.nrot = (int(hp('attention.head_count')), int(hp('attention.head_count_kv')),
                                                 int(hp('attention.key_length')), int(hp('rope.dimension_count')))
        self.ssm = dict(d_conv=int(hp('ssm.conv_kernel')), d_inner=int(hp('ssm.inner_size')), S=int(hp('ssm.state_size')),
                        nv=int(hp('ssm.time_step_rank')), nk=int(hp('ssm.group_count')))
        eps = float(hp('attention.layer_norm_rms_epsilon'))
        if np.float32(eps) != np.float32(EPS): raise ValueError(f'Qwen35One: rms eps {eps} != the engine EPS {EPS}')
        s = self.ssm
        # the shapes the tape and the modules' ops are written for (checked, never assumed): Qwen3.6-35B-A3B (32 value / 16
        # key heads, 16 q / 2 kv heads) and the dense qwen35 line (AREX-2: 48 / 16, 24 q / 4 kv heads)
        assert (s['d_conv'], s['S']) == (4, 128) and s['nv'] % s['nk'] == 0 and s['d_inner'] == s['nv'] * s['S'], s
        assert self.hd == 256 and self.nh % self.nkv == 0 and self.nrot <= self.hd, (self.nh, self.nkv, self.hd, self.nrot)
        assert self.D % 256 == 0 and self.ffn % 32 == 0, (self.D, self.ffn)
        if not self.dense: assert self.ffn == self.ffn_sh and self.k_used <= 64 and self.n_exp <= 1024, (self.ffn, self.ffn_sh)
        sec = hp('rope.dimension_sections')
        # IMRoPE sections with text-only positions (all components equal) are plain NEOX rope over n_rot dims (golden check)
        assert sec is None or sum(int(v) for v in sec) * 2 == self.nrot, (sec, self.nrot)
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        self.theta_scale = libm.powf(float(hp('rope.freq_base')), -2.0 / self.nrot)
        self.n_ctx_train = int(hp('context_length')) if hp('context_length') is not None else None
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or max_ctx < 1:
            raise ValueError(f'max_ctx {max_ctx!r}: a positive int (the KV cache positions)')
        if self.n_ctx_train is not None and max_ctx > self.n_ctx_train:
            raise ValueError(f'max_ctx {max_ctx} > the model context_length {self.n_ctx_train}')
        if isinstance(ntok, bool) or not isinstance(ntok, int) or not 1 <= ntok <= max_ctx:
            raise ValueError(f'ntok {ntok!r}: an int in [1, max_ctx {max_ctx}]')
        self.dev, self.max_ctx, self.q8, self.ntok = int(device), int(max_ctx), q8, ntok
        if shape_only: return              # (the shapes and projection_mib only, nothing on the card: caminus serve's max_ctx auto)
        # the VRAM this engine allocates, checked against the guard's line before anything is read (FlashNextTP._check_vram's
        # rule: refuse on the host; a load past the line on the XTX would be killed by the guard mid-way, on the monitor card).
        # vram_extra_mib: what the caller will add right after (Prefill35's buffers, ...), counted in the same check
        self.vram_mib = self.projection_mib()
        if vram_check:
            from ..device import capacity
            capacity.check([self.dev], [self.vram_mib + float(vram_extra_mib)],
                           f'Qwen35One {os.path.basename(path)} (max_ctx {self.max_ctx}, T {ntok}: {self.vram_mib:.0f} MiB + {float(vram_extra_mib):.0f} MiB after it)')
        self.module = module or self.pick_module()
        if k3.ATTN_G_OF.get(self.module, k3.ATTN_G) != self.nh // self.nkv:
            raise ValueError(f'Qwen35One: module {self.module} has {k3.ATTN_G_OF.get(self.module, k3.ATTN_G)} query heads per kv head, '
                             f'the model {self.nh // self.nkv}')
        self.limit_us, self.max_retries, self.bails, self.tape_only = limit_us, max_retries, [], tape_only
        self.fresh, self.poisoned, self.seq, self.token = True, None, 0, 0
        self.dr = DirectReader(path)
        self._emb = {}
        self.nl_build = self.nl if debug_layers is None else int(debug_layers)
        # the server's engine contract (FlashNextTP's names; caminus.serve.k3_session / promptcache / capacity read them):
        # one card, no TP split, no QSA (attention is dense at every depth: no pass limit below max_ctx), no DFlash features
        self.devs = [self.dev]
        self.qsa, self.qsa_limit, self.idx_cache, self.dual, self.local_only = False, None, False, False, True
        self.qsa_from = None                                  # (the status' attention kind: dense everywhere)
        self.feature_layers, self.feature_card, self.features, self.ple_layers = [], 0, None, None
        self.gdn_keys = self.moe_blocks = 0; self.vocab_share = 1.0
        self._during = None
        assert 1 <= self.nl_build <= self.nl, debug_layers
        self.model_tag = 'qwen35moe'
        torch.cuda.set_device(self.dev)
        with torch.cuda.device(self.dev): self._build_tape()

    def projection_parts(self):
        """projection_mib's terms in bytes: weights (every tensor read onto the card), kv_pos (the q8_0 KV of every attention
        layer per position), rec_split (the split attention's records per 32-cell split), gdn (the GDN blocks and conv states)"""
        w = sum(int(t.n_bytes) for t in self.rd.tensors if t.name != 'token_embd.weight')
        n_attn = sum(1 for i in range(self.nl) if i % self.fai == self.fai - 1)
        pb = (2 * self.nkv * self.hd // 32 * 34) if self.q8 else 4 * self.nkv * self.hd
        S, NK, NV = self.ssm['S'], self.ssm['nk'], self.ssm['nv']
        CH = 2 * S * NK + S * NV
        gdn = (self.nl - n_attn) * (k3.gdn_block_size(NV, S, NK, self.ntok) + CH * 3 + self.ntok * CH) * 4
        return dict(weights=w, kv_pos=n_attn * pb, rec_split=self.ntok * self.nh * 260 * 4, gdn=gdn)

    def projection_mib(self, max_ctx=None):
        """MiB this engine allocates on its card at max_ctx (default: its own): every tensor it reads onto the card (all but
        token_embd, read on the host), the q8_0 KV of every attention layer, the split attention's records, the GDN blocks
        and conv states, the pass buffers and a fixed scratch allowance (the tape, the status words, torch's rounding: 256
        MiB, measured 11-07 as the build's allocated total minus the tensors: ~150 MiB at max_ctx 4096)"""
        MIB = 1 << 20
        c = self.max_ctx if max_ctx is None else int(max_ctx)
        p = self.projection_parts()
        return (p['weights'] + p['kv_pos'] * c + p['rec_split'] * ((c + 31) // 32) + p['gdn']) / MIB + 256.0

    def pick_module(self):
        """the code object for this checkpoint's formats, from its routed experts' format: Q2_0 (the 2-bit GSQ checkpoint)
        -> k3_gguf_q35g, Q3_K / Q5_K (HauhauCS Q3_K_P) -> k3_gguf_q35h; anything else: refused (no module lists it)"""
        if self.dense: return DENSE_MODULE
        f = {self.T[f'blk.{i}.ffn_gate_exps.weight'].tensor_type.name for i in range(self.nl)}
        if f == {'Q2_0'}: return 'k3_gguf_q35g'
        if f <= {'Q3_K', 'Q5_K'}: return MODULE
        raise ValueError(f'Qwen35One: no module for routed-expert formats {sorted(f)} (k3_gguf_q35h: Q3_K / Q5_K, k3_gguf_q35g: Q2_0)')

    # -- weights, read straight onto the card --------------------------------------------------------------------------
    def _devs(self):
        return f'cuda:{self.dev}'

    def _full(self, name):
        t = self.T[name]
        return self.dr.read(int(t.data_offset), int(t.n_bytes), device=self._devs())

    def _mat(self, name):
        """a whole GG matrix: dict(w, fmt, rows, cols, rb); a 1-row matrix may be stored 1-D"""
        t = self.T[name]; w = self._full(name)
        rows, cols = (int(t.shape[1]) if len(t.shape) > 1 else 1), int(t.shape[0])
        return dict(name=name, w=w.reshape(-1), fmt=t.tensor_type.name, rows=rows, cols=cols, rb=int(t.n_bytes) // rows)

    def _f32(self, name, shape=None):
        t = self.T[name]; assert t.tensor_type.name == 'F32', (name, t.tensor_type.name)
        w = self._full(name).view(torch.float32)
        return w.reshape(shape) if shape else w

    def _exp(self, name):
        """an expert tensor [E, rows, row bytes] whole: dict(w, fmt, rows, cols, rb, slab = one expert's bytes)"""
        t = self.T[name]; E, R, rb = t.data.shape; w = self._full(name)
        return dict(name=name, w=w.reshape(-1), fmt=t.tensor_type.name, rows=R, cols=int(t.shape[0]), rb=rb, slab=R * rb)

    def _build_tape(self):
        D, hd, S_ = self.D, self.hd, self.ssm['S']
        nh, nkv = self.nh, self.nkv
        NK, NV = self.ssm['nk'], self.ssm['nv']
        CH = 2 * S_ * NK + S_ * NV                                   # conv channels: q | k (key heads) | v (value heads)
        KD, VD = NK * S_, NV * S_
        Tn, ffn = self.ntok, self.ffn
        RPT = dict(router=4, attn_qkv=16, ssm_out=8, attn_output=16, down_shexp=16, head=64) if Tn == 1 else \
            dict(router=4, attn_qkv=16, ssm_out=8, attn_output=8, down_shexp=64, head=64)
        RPT.update(ffn_gu=16, ffn_down=16)                     # (the dense FFN: not measured yet)
        B = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        q8b = lambda n: B((n + 31) // 32 * 36, dt=torch.uint8)
        E = k3.Engine(module=self.module, grid=k3.resident_grid(self.module), T=Tn, limit_us=self.limit_us)
        # the per-pass inputs, one contiguous block (FlashNextTP's layout: 64 words of scalars, then the residual [Tn, D])
        W0, nres = 64, Tn * D
        self.stage_words = (W0, nres, 0)
        self.hstage = torch.zeros(W0 + nres, dtype=torch.float32, pin_memory=True)
        hs = self.hstage.numpy(); self.h_i32 = hs[:W0].view(np.int32); self.h_res = hs[W0:W0 + nres].reshape(Tn, D)
        blk = B(W0 + nres); self.stage = blk
        E.params = blk[0:16].view(torch.int32)
        pos, self.epoch = blk[16:17].view(torch.int32), blk[17:18].view(torch.int32)
        res = blk[W0:W0 + nres].view(Tn, D)
        assert blk.data_ptr() % 256 == 0 and res.data_ptr() == blk.data_ptr() + 4 * W0 and pos.data_ptr() == blk.data_ptr() + 64
        av_ = max(nh * hd, VD)
        s = dict(mixed=B(Tn, D), mq=q8b(Tn * D), blk=B(Tn * D), zero=B(Tn * D),
                 qkv=B(Tn * CH), z=B(Tn * VD), al=B(Tn * NV), be=B(Tn * NV), gates=B(Tn * 2 * NV), conv=B(Tn * CH),
                 qk=B(Tn * 2 * KD), att=B(Tn * av_), fin=B(Tn * VD), fq=q8b(Tn * av_),
                 qf=B(Tn * nh * 2 * hd), kr=B(Tn * nkv * hd), vr=B(Tn * nkv * hd), qo=B(Tn * nh * hd), ko=B(Tn * nkv * hd),
                 gat=B(Tn * nh * hd), logit=B(Tn * max(1, self.n_exp)), ids=B(Tn * max(1, self.k_used), dt=torch.int32), wts=B(Tn * max(1, self.k_used)),
                 h=B(Tn * max(1, self.k_used) * ffn), hq=q8b(Tn * max(1, self.k_used) * ffn), moe=B(Tn * D), shh=B(Tn * ffn), shu=B(Tn * ffn),
                 shq=q8b(Tn * ffn), shx=B(Tn * D), sgl=B(Tn), ffp=B(Tn * D))
        self.E, self.S, self.res, self.pos = E, s, [res], pos
        self.state, self.layer_seg, self.seg_names = {}, [], {}
        self.attn_cells = self.max_ctx
        self._attn_part = None
        def akw():
            if not E.split_attn(): return {}
            if self._attn_part is None:
                ss = k3.ATTN_SS_OF.get(E.module, k3.ATTN_SS)
                ns = (self.attn_cells + ss - 1) // ss
                self._attn_part = torch.zeros(Tn * nh * ns * k3.ATTN_RS, device='cuda')
            return dict(max_cells=self.attn_cells, part=self._attn_part)
        # (keys (card, layer, kind): card 0, the one card; caminus.cache.prompt.Layout reads the card from the key)
        st_ = lambda key, make: self.state[key] if key in self.state else self.state.setdefault(key, make())
        pf_cap = [(re.compile(k), int(v)) for k, v in (e.rsplit(':', 1) for e in os.environ.get('K3_PF_CAP', '.:16384' if Tn == 1 else '.:0').split(',') if e)]

        def named(si, name):
            cap = next((v for k, v in pf_cap if k.search(name)), None)
            if cap is not None: E.segs[si].aux = cap + 1
            self.seg_names[si] = name; return si

        def mv(W, x_q8, x_f32, out, dep, rpt=16):
            """a matvec of W (a GG matrix, from the q8_1 activation; an F32 one through OP_F32_MV from the fp32 one)"""
            if W['fmt'] == 'BF16':   # OP_BF16_MV from the fp32 activation (the GSQ checkpoint's ssm_alpha / ssm_beta)
                return named(E.bf16_mv(W['w'], W['rows'], W['cols'], x_f32, out, rows_per_task=4, dep=dep), W['name'])
            if W['fmt'] == 'F32':
                w = W['w'].view(torch.float32).view(W['rows'], W['cols'])
                return named(E.f32_mv(w, x_f32, out, rows_per_task=4, dep=dep), W['name'])
            return named(E.gg_matvec(W['w'], W['fmt'], W['rows'], W['cols'], W['rb'], x_q8, out, rows_per_task=rpt, dep=dep), W['name'])

        def norm(name, dep):
            w = self._f32(name, (1, D)); self.norm_w[name] = w
            return E.rms_mul(res, w, s['mixed'], EPS, dep=dep, q8=s['mq'])

        # the tensors the tape holds, by layer, for the batched prefill (caminus.prefill.q35: references, nothing copied)
        self.norm_w, self.pw = {}, []

        self.features = None
        dep = -1
        for il in range(self.nl_build):
            p = f'blk.{il}'
            self.layer_seg.append((il, len(E.segs)))
            M = norm(f'{p}.attn_norm.weight', dep)
            if il % self.fai != self.fai - 1:                  # Gated DeltaNet (value head h reads key head h % 16)
                qkv = self._mat(f'{p}.attn_qkv.weight'); assert qkv['rows'] == CH, (qkv['rows'], CH)
                assert self.T[f'{p}.ssm_conv1d.weight'].tensor_type.name == 'F32'
                conv_w = self.state_w(f'{p}.ssm_conv1d.weight', lambda: self._full(f'{p}.ssm_conv1d.weight').view(torch.float32).view(-1, 4).contiguous())
                assert conv_w.shape[0] == CH, conv_w.shape
                z = self._mat(f'{p}.attn_gate.weight'); al = self._mat(f'{p}.ssm_alpha.weight'); be = self._mat(f'{p}.ssm_beta.weight')
                assert z['rows'] == VD and al['rows'] == NV and be['rows'] == NV, (z['rows'], al['rows'], be['rows'])
                dt, sa = self._f32(f'{p}.ssm_dt.bias'), self._f32(f'{p}.ssm_a')
                nw = self._f32(f'{p}.ssm_norm.weight'); so = self._mat(f'{p}.ssm_out.weight')
                H = NV
                conv_st = st_((0, il, 'conv'), lambda: torch.zeros(CH, 3, device='cuda'))
                cxp = st_((0, il, 'convx'), lambda: torch.zeros(Tn * CH, device='cuda'))
                gblk = st_((0, il, 'gdn'), lambda: torch.zeros(k3.gdn_block_size(H, S_, NK, Tn), device='cuda'))
                a_ = mv(al, s['mq'], s['mixed'], s['al'], M); b_ = mv(be, s['mq'], s['mixed'], s['be'], M)
                gt = E.also(E.gdn_gates(s['al'], s['be'], dt, sa, s['gates'], dep=a_), b_)
                q_ = mv(qkv, s['mq'], s['mixed'], s['qkv'], M, RPT['attn_qkv'])
                c_ = E.gdn_conv(s['qkv'], conv_w, conv_st, s['conv'], cxp, dep=q_)
                l_ = E.gdn_l2(s['conv'], s['qk'], 2 * NK, S_, EPS, x_stride=CH, dep=c_)
                z_ = mv(z, s['mq'], s['mixed'], s['z'], M)
                nv = Tn * S_ * H; fqv = s['fq'][:nv // 32 * 36]
                st = E.also(E.gdn_step(s['qk'], s['conv'][2 * S_ * NK:], s['gates'], gblk, s['att'][:nv], H, S_, NK, CH, dep=l_), gt)
                E.gdn_save(s['qk'], s['conv'][2 * S_ * NK:], s['gates'], gblk, H, S_, NK, CH, dep=st)
                nm = E.also(E.gdn_norm(s['att'][:nv], nw, s['z'], s['fin'], H, S_, EPS, dep=st, q8=fqv), z_)
                blk_ = mv(so, fqv, None, s['blk'], nm, RPT['ssm_out'])
                L = dict(kind='gdn', qkv=qkv, conv_w=conv_w, z=z, al=al, be=be, dt=dt, a=sa, nw=nw, so=so, conv_st=conv_st, gdn=gblk)
            else:                                                # gated attention, all heads
                aq, ak, av = self._mat(f'{p}.attn_q.weight'), self._mat(f'{p}.attn_k.weight'), self._mat(f'{p}.attn_v.weight')
                assert aq['rows'] == nh * 2 * hd and ak['rows'] == nkv * hd and av['rows'] == nkv * hd
                qn_, kn_ = self._f32(f'{p}.attn_q_norm.weight'), self._f32(f'{p}.attn_k_norm.weight')
                ao = self._mat(f'{p}.attn_output.weight')
                pb = (2 * nkv * hd // 32 * 34) if self.q8 else 4 * nkv * hd
                cache = st_((0, il, 'kv'), lambda: torch.zeros(self.max_ctx * pb, dtype=torch.uint8, device='cuda'))
                k_ = mv(ak, s['mq'], s['mixed'], s['kr'], M); v_ = mv(av, s['mq'], s['mixed'], s['vr'], M)
                kd = E.qk_nr(s['kr'], hd, nkv, hd, kn_, pos, s['ko'], self.nrot, self.theta_scale, EPS, dep=k_)
                vd = v_
                if self.q8: kd = E.hadamard(s['ko'], hd, dep=kd); vd = E.hadamard(s['vr'], 64, dep=vd)
                ap = E.also(E.kv_append(s['ko'], s['vr'], cache, pos, q8=self.q8, dep=kd), vd)
                q_ = mv(aq, s['mq'], s['mixed'], s['qf'], M)
                qd = E.qk_nr(s['qf'], 2 * hd, nh, hd, qn_, pos, s['qo'], self.nrot, self.theta_scale, EPS, dep=q_)
                if self.q8: qd = E.hadamard(s['qo'], hd, dep=qd)
                na = Tn * nh * hd
                a_ = E.also(E.attn_dec(s['qo'], cache, pos, s['att'][:na], nh, nkv, hd, 1.0 / np.sqrt(hd), q8=self.q8, dep=ap, **akw()), qd)
                if self.q8: a_ = E.hadamard(s['att'][:na], 64, rows_per_task=4, dep=a_)
                fqa = s['fq'][:na // 32 * 36]
                a_ = E.gate_mul(s['att'][:na], s['qf'], 2 * hd, hd, s['gat'], hd, dep=a_, q8=fqa)
                blk_ = mv(ao, fqa, None, s['blk'], a_, RPT['attn_output'])
                L = dict(kind='attn', q=aq, k=ak, v=av, qn=qn_, kn=kn_, o=ao, kv=cache)
            # res += block (llama.cpp's attn_residual; no exchange on one card), then the post-attention norm
            r_ = E.add_sum(s['blk'], s['zero'], res.reshape(-1), dep=blk_)
            M = norm(f'{p}.post_attention_norm.weight', r_)
            if self.dense:                                       # one dense FFN: gate / up matvecs, the fused swiglu q8_1, down
                fg, fu, fd = self._mat(f'{p}.ffn_gate.weight'), self._mat(f'{p}.ffn_up.weight'), self._mat(f'{p}.ffn_down.weight')
                assert fg['rows'] == ffn and fu['rows'] == ffn and fd['rows'] == D and fd['cols'] == ffn, (fg['rows'], fd['rows'], fd['cols'])
                g_ = mv(fg, s['mq'], s['mixed'], s['shh'], M, RPT['ffn_gu']); u_ = mv(fu, s['mq'], s['mixed'], s['shu'], M, RPT['ffn_gu'])
                s_ = E.also(E.gg_q8_1(s['shh'], s['shq'], dep=g_, mul=s['shu']), u_)
                f_ = mv(fd, s['shq'], None, s['ffp'], s_, RPT['ffn_down'])
                dep = E.add_sum(s['ffp'], s['zero'], res.reshape(-1), dep=f_)
                L.update(attn_norm=self.norm_w[f'{p}.attn_norm.weight'], post_norm=self.norm_w[f'{p}.post_attention_norm.weight'],
                         fg=fg, fu=fu, fd=fd)
                self.pw.append(L)
                continue
            rw = self._f32(f'{p}.ffn_gate_inp.weight', (self.n_exp, D)); sw = self._f32(f'{p}.ffn_gate_inp_shexp.weight', (1, D))
            sg, su, sd = self._mat(f'{p}.ffn_gate_shexp.weight'), self._mat(f'{p}.ffn_up_shexp.weight'), self._mat(f'{p}.ffn_down_shexp.weight')
            g, u, d = self._exp(f'{p}.ffn_gate_exps.weight'), self._exp(f'{p}.ffn_up_exps.weight'), self._exp(f'{p}.ffn_down_exps.weight')
            assert g['rows'] == ffn and u['rows'] == ffn and d['cols'] == ffn and d['rows'] == D, (g['rows'], d['rows'], d['cols'])
            r1 = named(E.f32_mv(rw, s['mixed'], s['logit'], rows_per_task=RPT['router'], dep=M), 'f32_router_logits')
            r1 = E.router(s['logit'], self.k_used, s['ids'], s['wts'], dep=r1)
            sl = named(E.f32_mv(sw, s['mixed'], s['sgl'], rows_per_task=8, dep=M), 'f32_shexp_gate')
            sg_ = mv(sg, s['mq'], s['mixed'], s['shh'], M); su_ = mv(su, s['mq'], s['mixed'], s['shu'], M)
            s_ = E.also(E.gg_q8_1(s['shh'], s['shq'], dep=sg_, mul=s['shu']), su_)
            s_ = E.also(mv(sd, s['shq'], None, s['shx'], s_, RPT['down_shexp']), sl)
            r1 = E.gg_moe_gu(g['w'], u['w'], g['fmt'], g['slab'], ffn, D, s['ids'], s['mq'], s['h'], rows_per_task=8, dep=r1, up_fmt=u['fmt'])
            r1 = E.gg_q8_1(s['h'], s['hq'], dep=r1)
            r1 = E.gg_moe_down(d['w'], d['fmt'], d['slab'], ffn, D, s['ids'], s['wts'], s['hq'], s['moe'], rows_per_task=8, dep=r1)
            f_ = E.also(E.shexp_add(s['moe'], s['shx'], s['sgl'], s['ffp'], dep=r1), s_)
            dep = E.add_sum(s['ffp'], s['zero'], res.reshape(-1), dep=f_)
            L.update(attn_norm=self.norm_w[f'{p}.attn_norm.weight'], post_norm=self.norm_w[f'{p}.post_attention_norm.weight'],
                     router=rw, sgate=sw, sg=sg, su=su, sd=sd, g=g, u=u, d=d)
            self.pw.append(L)
        self.layer_seg.append(('head', len(E.segs)))
        M = norm('output_norm.weight', dep)
        out = self._mat('output.weight')
        self.vocab = out['rows']
        self.logits = B(Tn, out['rows'])
        mv(out, s['mq'], s['mixed'], self.logits, M, RPT['head'])
        E.build(strict=not self.tape_only)
        self.d_max = (B(Tn), B(Tn, dt=torch.int64))
        self.h_max = (torch.zeros(Tn, pin_memory=True), torch.zeros(Tn, dtype=torch.int64, pin_memory=True))
        self.h_st = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        self.ev_done = torch.cuda.Event()
        self.engines = [E]
        # the device sampler's two vocabulary shards (K3Session: each shard's top-k, then candsample's exact merge): column
        # views of this card's one logits row, on this card (shard_devs); m.devs stays one card for everything else
        self.vh = self.vocab // 2
        self.logits_half = [self.logits[:, :self.vh], self.logits[:, self.vh:]]
        self.shard_devs = [self.dev, self.dev]

    def state_w(self, key, make):
        """a weight-derived tensor kept with the states' dict but never zeroed by reset (the conv weights)"""
        self._wts = getattr(self, '_wts', {})
        if key not in self._wts: self._wts[key] = make()
        return self._wts[key]

    # -- passes -----------------------------------------------------------------------------------------------------------
    @property
    def staged(self):
        """the position the staging buffer holds for the next launch (h_i32[16], which the device reads; _pass refuses a launch
        whose i0 differs). With seq and token one of the three counters the prompt cache and the session read"""
        return int(self.h_i32[16])

    def reset(self):
        """a new sequence: zero every recurrent state and KV cache use (positions restart at 0); clears poisoned"""
        for t in self.state.values(): t.zero_()
        self.seq += 1; self.fresh = True; self.poisoned = None

    def prefetch(self, tokens, positions):
        """nothing to read ahead (no PLE rows): the speculative loop's hook"""

    def _emb_row(self, tok):
        """token_embd row tok, dequantized by gguf-py to float32 (cached; checked against llama.cpp's first norm-0 at 8e-8)"""
        r = self._emb.get(tok)
        if r is None:
            te = self.T['token_embd.weight']
            assert 0 <= tok < int(te.shape[1]), f'token {tok} out of range'
            r = quants.dequantize(np.ascontiguousarray(np.asarray(te.data)[[tok]]), te.tensor_type).astype(np.float32).reshape(-1)
            if len(self._emb) >= 16384: self._emb.clear()
            self._emb[tok] = r
        return r

    def _check(self, tokens, i0, commit, what):
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, self.ntok, self.max_ctx, commit=commit, ntok=self.ntok, what=what)
        if tokens is not None and len(tokens) < i0 + self.ntok:
            raise k3.ContextBoundsError(f'{what}: positions {i0} .. {i0 + self.ntok - 1} but {len(tokens)} tokens')

    def step(self, tokens, i):
        """position i (ntok == 1; commits the previous token); returns the logits [vocab] on the card"""
        assert self.ntok == 1
        return self.verify(tokens, i, 0 if self.fresh else 1)[0]

    def step_token(self, tokens, i):
        """step(), but returns only the greedy next token (argmax on the card, the first index on ties)"""
        assert self.ntok == 1
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """positions i0 .. i0 + ntok - 1 in one pass, after committing the previous pass's first `commit` tokens into the
        recurrent states (lazy commit, FlashNextTP.verify's contract, bail retries included); returns logits [ntok, vocab]"""
        self._run(tokens, i0, commit, False)
        return self.logits.clone()   # a fresh tensor per pass, as FlashNextTP.verify's torch.cat (callers may hold it)

    def verify_tokens(self, tokens, i0, commit):
        return self._run(tokens, i0, commit, True)

    def _run(self, tokens, i0, commit, tok):
        Tn = self.ntok
        self._check(tokens, i0, commit, 'Qwen35One pass')
        for t in range(Tn): self.h_res[t] = self._emb_row(int(tokens[i0 + t]))
        self.h_i32[16] = i0; self.fresh = False; self._tok = tok
        for attempt in range(self.max_retries + 1):
            bad = self._pass(i0, commit if attempt == 0 else 0)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 one-card: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 one-card: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        if not tok: return None
        return [int(v) for v in self.h_max[1].numpy()[:Tn]]

    def _pass(self, i0, commit):
        """one launch from the staged host block, then the wait; returns the bail reports (empty: every wait finished)"""
        self._pass_launch(i0, commit)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit):
        """_pass's first half: the launch (and the logits' max / argmax read back when self._tok); FlashNextTP's name and
        signature (K3Session's K3_TEST_BAIL hook wraps it)"""
        tok = self._tok
        self._check(None, i0, commit, 'Qwen35One._pass')
        if self.h_i32[16] != i0: raise k3.ContextBoundsError(f'Qwen35One._pass: staged position {self.h_i32[16]} != {i0}')
        self.token += 1
        self.h_i32[1] = commit; self.h_i32[17] = 0
        with torch.cuda.device(self.dev):
            self.stage.copy_(self.hstage, non_blocking=True)    # the pinned block is rewritten only after the sync below
            self.E.run(check=False)
            if tok:
                torch.max(self.logits, dim=1, out=self.d_max)
                self.h_max[0].copy_(self.d_max[0], non_blocking=True); self.h_max[1].copy_(self.d_max[1], non_blocking=True)
            self.h_st.copy_(self.E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
            self.ev_done.record()

    def _pass_wait(self, i0):
        """_pass's second half: wait for the launch, read its status (again, unchanged, when called twice); the bounds guard
        poisons and raises, a wait that gave up is returned as its report"""
        self.ev_done.synchronize()
        st = self.h_st.tolist()
        b = k3.bounds_status(st)
        if b:
            with torch.cuda.device(self.dev): torch.cuda.synchronize(); self.E.sync.zero_()
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: {b}'
            raise k3.ContextBoundsError(f'k3 one-card: {self.poisoned}')
        if st[1]:
            with torch.cuda.device(self.dev):
                torch.cuda.synchronize()
                names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                rep = f'segment {st[2]} = {names.get(self.E.segs[st[2]].op)} (seen {st[3]}); {self.E.stall_report()}'
                self.E.sync.zero_()
            return [rep]
        return []

    # -- the server's engine contract (caminus.serve.k3_session, caminus.cache.prompt: FlashNextTP's names, one card) --------
    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting and without reading anything back (K3Session's device sampler queues its
        top-k behind it on the same stream); then finish_pass(i0, commit). The same staging and launch as _run"""
        Tn = self.ntok
        self._check(tokens, i0, commit, 'Qwen35One pass')
        for t in range(Tn): self.h_res[t] = self._emb_row(int(tokens[i0 + t]))
        self.h_i32[16] = i0; self._tok = False; self.fresh = False
        self._pass_launch(i0, commit)

    def finish_pass(self, i0, commit):
        """wait for launch_tokens' pass; a bailed pass is run again (commit 0, as _run) before returning. Returns True when it
        was rerun (anything queued behind the first launch read a bailed pass's outputs and must run again). At most
        max_retries + 1 attempts in all, as _run"""
        bad = self._pass_wait(i0)
        if not bad: return False
        for attempt in range(1, self.max_retries + 1):
            self.bails.append((i0, attempt - 1, bad))
            print(f'k3 one-card: a wait gave up at position {i0} (attempt {attempt}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
            bad = self._pass(i0, 0)
            if not bad: return True
        self.bails.append((i0, self.max_retries, bad))
        raise RuntimeError(f'k3 one-card: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))

    def physics_bytes(self):
        """per card (one): the weight bytes one decode pass reads (every matrix and norm the tape holds once, the routed
        experts at k_used / n_exp of their bytes: the average token), for the server's physics floor"""
        tot = 0
        for L in self.pw:
            for k, W in L.items():
                if isinstance(W, dict) and 'w' in W:
                    nb = W['w'].numel() * W['w'].element_size()
                    tot += nb * self.k_used / self.n_exp if k in ('g', 'u', 'd') and self.n_exp else nb
                elif isinstance(W, torch.Tensor) and k not in ('kv', 'gdn', 'conv_st'): tot += W.numel() * W.element_size()
        tot += self.T['output.weight'].n_bytes
        return (float(tot),)

    def check_replicas(self):
        """one card: no replicas to compare (FlashNextTP's call, kept for the shared tests): always (True, True)"""
        return True, True
