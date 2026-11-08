"""Flash-Next (qwen4exp) on two cards with tensor parallelism in the K3 engine: both cards run every layer on their
share of it at the same time, and exchange two partial sums per layer (OP_XCHG).

The split follows the model's structure, so every op keeps its existing indexing:
- GDN: card c takes key heads 8c..8c+7 and the 24 value heads that read them (h % 16 in that range): its slices of
  qkv / conv / z / alpha / beta / dt / a, its heads' state, and ssm_out's columns for those heads (a partial sum).
- Attention: card c takes kv head c and its 12 query heads (q / gate, k, v rows, its own KV cache), and attn_output's
  columns for those heads.
- MoE: every expert's 640-wide intermediate is split 320 / 320: gate / up rows, down columns; the shared expert the
  same way. Each card's FFN output (routed + gated shared part) is a partial sum.
- Replicated on both cards: the hyper-connection ops (norm / down / inject / up / mean / residual), the router, the
  shared-expert gate, PLE. Both cards must hold bitwise-equal replicated state (else their routers could pick
  different experts): OP_XCHG sums card 0's partial first on both cards, and the ops' results do not depend on the
  grid, so the replicated state is the same bits (checked by check_replicas).
- Head: the vocabulary rows split in halves.
"""
import ctypes
import os
import re
import sys
import numpy as np
import torch
from . import k3
from ..device import capacity
from .flashnext import FlashNextK3, EPS, quants      # (puts gguf-py on the path)
from gguf.constants import GGML_QUANT_SIZES          # noqa: E402


QSA_LAYER_HOOKS = 'qsa-scores'   # K3_TEST_HOOKS value: every QSA layer its own SCORE / indexer-query rows (test-only, tools/eval/qsa_depth.py)


def _hip():
    return ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libamdhip64.so'))


class Slots:
    """receive slots for the exchanges, in uncached VRAM of one card (a peer's PCIe writes are only seen there):
    n slots of 16 header words (word 0 = the epoch flag) + N floats"""
    def __init__(self, dev, n, N):
        hip = _hip(); self.stride = (16 + N) * 4; size = n * self.stride; p = ctypes.c_void_p()
        hip.hipSetDevice(dev)
        assert hip.hipExtMallocWithFlags(ctypes.byref(p), ctypes.c_size_t(size), ctypes.c_uint(3)) == 0   # Uncached
        assert hip.hipMemset(p, 0, ctypes.c_size_t(size)) == 0 and hip.hipDeviceSynchronize() == 0
        self.base, self.size, self.dev = p.value, size, dev

    def slot(self, i):
        return self.base + i * self.stride

    def flags(self):
        """the epoch words of every slot (host copy, for debugging)"""
        hip = _hip(); buf = (ctypes.c_uint32 * (self.size // 4))()
        hip.hipSetDevice(self.dev); hip.hipMemcpy(buf, ctypes.c_void_p(self.base), ctypes.c_size_t(self.size), 2)
        return np.frombuffer(buf, dtype=np.uint32)[::self.stride // 4]


QSA_SCORE_NTASK = 384   # the indexer score's tasks: 64 left the 12 layers' scoring latency-bound at depth (T = 4 pass at 131K
                        # 27.50 -> 26.38 ms with 384, logits bitwise: each pool's arithmetic does not depend on its task)


def qsa_score_ntask(prow):
    """OP_QSA_SCORE's task count for a tape of prow pool rows: env K3_QSA_NTASK (A/B), else QSA_SCORE_NTASK; capped at
    its 32-pool chunks"""
    return max(1, min(int(os.environ.get('K3_QSA_NTASK') or QSA_SCORE_NTASK), (prow + 31) // 32))


def qsa_psel():
    """the pool selection as OP_QSA_PSEL (Engine.qsa_select_par: the same SEL rows bit for bit, spread over slices) unless
    env K3_QSA_PSEL=0 (the one-workgroup OP_QSA_SELECT: A/B)"""
    return os.environ.get('K3_QSA_PSEL', '1') != '0'


class FlashNextTP(FlashNextK3):
    def __init__(self, path, devices=(0, 1), max_ctx=2048, q8=True, module=None, local_only=False, gdn_keys=10, moe_blocks=12, vocab_share=None, ntok=1, feature_layers=(), limit_us=200_000, max_retries=8, tape_only=False, feature_card=0,
                 idx_cache=True, dense_beyond_qsa=False, vram_check=True, vram_reserve_mib=(0, 0), qsa=None, qsa_tape=None, vram_drafters=(),
                 vram_prefill=None, kv_host=False, ctx_extend=False):
        """local_only: the exchanges return each card's own partial (no peer): each card's time alone, the ceiling.
        kv_host (the opt-in long-context mode): every attention layer's KV cache in pinned host memory that both cards
        address directly (caminus.device.hip.host_mapped; one buffer per card), not in VRAM: the same bytes at the same
        offsets, so the same results, read over PCIe. Checked before any allocation against the host's MemAvailable and
        the kernel's GTT page limit (ttm pages_limit), and left out of the VRAM projection.
        ctx_extend: allow max_ctx past the model's context_length (no position scaling: the model's own RoPE).
        Card 0's share: gdn_keys of the 16 GDN key heads (even), moe_blocks of each expert's intermediate in 32s,
        vocab_share of the output rows (attention is always one kv head each; None: 0.6, the GSQ-RCO Coder 0.5). The XT's fixed cost per segment is higher,
        so the XTX takes more: each card alone ~11.6-12.5 ms (even split: XTX 10.9, XT 13.2). End to end the split
        matters little (68.6 even, 69.1 at the defaults, 67.4 at 12 / 12 / 0.7): the cards meet 96 times per token.
        ntok: tokens per pass (speculative verification, verify()); 1 = plain decoding (step()).
        feature_layers: export the stream mean of the residual entering these layers, per position, on card 0 into
        self.features [len, ntok, D] (a DFlash drafter's input; llama.cpp layer_inp). feature_card = 1: exported on card 1 (the
        XT) instead, beside the K3 drafter (caminus.decode.dflash) that reads it there (the residual is the same bits on both cards).
        tape_only: build the tapes even if their module lacks some (op, format) pair (k3.Engine.build(strict=False):
        self.engines[c].inventory() / unsupported() to inspect; a launch is refused); else such a tape raises at build.
        module: 'k3_gguf' (the REAP-308 cut's formats) or 'k3_gguf_gsq' (the GSQ-RCO Coder's whitelist, with OP_BF16_MV);
        None picks it from the tensor formats (flashnext.pick_module).
        A split GGUF is given by its first shard (the PLE table is read from its own shard with O_DIRECT).
        max_ctx: the KV cache's positions per attention layer (ntok <= max_ctx <= the model's context_length). Every pass's
        positions i0 .. i0 + ntok - 1 and its commit are checked against it on the host before anything is staged or
        launched (k3.check_span; k3.ContextBoundsError), and on the device k3_run's prologue skips the whole tape when a
        position leaves the cache's allocation or the commit leaves [0, ntok] (defense in depth, kernels/k3.hip: nothing
        written; the engine then refuses passes until reset()). Up to the model's context_length (262,144).
        QSA (docs/specs/spec-qsa.md): qsa=None (auto) wires the model's QSA attention into the 12 attention layers of the
        GSQ-RCO Coder when max_ctx > capacity.QSA_EXACT (2051): per layer the indexer (k_proj / q_proj BF16 matvecs on the
        block input, IDX_APPEND into IDXRAW / IDXPOOL from token 0, QK_NR of the indexer query, QSA_SCORE, QSA_SELECT) and
        ATTN_SEL (attention over the selected cells) instead of ATTN_DEC; the tapes then run on the QSA modules (k3.QSA_OF)
        and passes go up to max_ctx. qsa=True forces it at any max_ctx (Coder only; tests: at <= 2051 positions it equals
        the dense tape bit for bit), False keeps the dense tape. qsa_tape: 'dual' (default) builds both tapes over the same
        buffers and runs the dense one (trunk module, plus the IDXRAW writes) for every pass ending below QSA_EXACT, the QSA
        one from there (its pools of the dense positions backfilled from IDXRAW first): bitwise the QSA tape everywhere,
        at the trunk's speed below 2051; 'always': the QSA tape for every pass (tests); None: env K3_QSA_TAPE, else 'dual'. Without QSA (REAP-308, the Coder at max_ctx <= 2051, or
        qsa=False) the attention is full attention, equal to the model's QSA only up to capacity.QSA_EXACT positions: a pass
        past it is refused (k3.ContextBoundsError), unless dense_beyond_qsa=True: a TEST-ONLY override (the full-attention
        baseline at long context; its results are not the model's; it never wires QSA).
        idx_cache: allocate the QSA indexer caches (IDXRAW / IDXPOOL per card, spec-qsa.md section 3; required by QSA).
        vram_check: before anything is allocated, refuse (capacity.CapacityError) a configuration whose projected VRAM per
        card (the measured fixed part + the context-sized buffers + vram_reserve_mib, e.g. a drafter's, per card) would pass
        the card's cap (capacity.check: the guard's line less a margin, against the device's used memory now as the guard reads it).
        vram_drafters: the drafters the check also counts (capacity.drafter_need_mib at this max_ctx / ntok): 'mtp' and / or
        ('dflash', (layers, kv heads, head dim[, window])[, ring rows]) (ring None: capacity.dflash_ring, a fixed 4096);
        the smoke factories pass theirs, so a config over the caps is refused before the target loads (the drafters check
        again, against the use then, before they allocate).
        vram_prefill: the batched prefill this engine will be given, counted in the same check: its largest chunk (tokens:
        the default dense path) or dict(chunk=, q6k=, q6k_buffers=) as Prefill / ChunkStack will be built (the 'wmma' dense
        path's bf16 buffers). Its plan (self.prefill_plan, caminus.prefill.plan.PrefillPlan: this engine's split, tokens
        per pass, feature layers and card, QSA, the feed's shapes from this GGUF) is what is counted, and a Prefill /
        ChunkStack that would allocate more than it is refused before it allocates (plan.check_counted); None = no prefill
        planned (each allocation still checks the cards' caps: caminus.prefill.mmq.Workspace, feed.PrefillFeed(devices=...),
        glue.GlueWorkspace)."""
        super().__init__(path, devices=devices, max_ctx=max_ctx, q8=q8, module=module, build=False, ctx_extend=ctx_extend)   # metadata only
        self.kv_host = bool(kv_host); self._kvh = [None, None]
        if isinstance(ntok, bool) or not isinstance(ntok, int) or not 1 <= ntok <= self.max_ctx:
            raise ValueError(f'ntok {ntok!r}: an int in [1, max_ctx {self.max_ctx}]')
        self.idx_cache = bool(idx_cache)
        self.geom = capacity.geometry(self.rd)
        self.model_tag = 'coder' if 'gsq' in self.module else 'reap'
        self.qsa, self.qsa_limit = self.qsa_mode(self.model_tag, self.idx_cache, dense_beyond_qsa, self.max_ctx, qsa)
        qsa_tape = qsa_tape or os.environ.get('K3_QSA_TAPE', 'dual')
        if qsa_tape not in ('dual', 'always'): raise ValueError(f"qsa_tape {qsa_tape!r}: 'dual' or 'always'")
        self.dual = self.qsa and qsa_tape == 'dual'
        self._pool_ok = 0   # (two-tape QSA) IDXPOOL pools j < _pool_ok match IDXRAW on both cards (_pass_launch / _backfill)
        self.poisoned = None   # set when the device bound guard caught a position (states suspect): passes refused until reset()
        self.local_only, self.gdn_keys, self.moe_blocks, self.vocab_share, self.ntok = local_only, gdn_keys, moe_blocks, vocab_share, ntok
        self.prefill_plan = self.plan_prefill(vram_prefill, len(feature_layers), feature_card) if vram_prefill is not None else None
        self.vram_plan = self._check_vram(vram_reserve_mib, vram_drafters, self.prefill_plan, len(feature_layers), feature_card) if vram_check else None   # before any allocation
        # the Coder's head rows: 0.6 -> 0.5 on card 0 (the head splits by rows, so the logits stay bitwise the same): fn_bench 96
        # --tp 10.10 -> 10.03 ms/token, T=4 pass 18.29 -> 18.25 ms (interleaved x 8; 0.45 / 0.55 / 0.65 / 0.7 no better; its
        # gdn_keys 8 / 12 and moe_blocks 10 / 14 slower at T=1 and T=4: tmp/m1/s6/rs/split_s*.out)
        if self.vocab_share is None: self.vocab_share = 0.5 if 'gsq' in self.module else 0.6
        assert 0.0 < self.vocab_share < 1.0, self.vocab_share
        self.tape_only = tape_only
        self.fresh = True      # lazy commit: the first pass after a reset commits nothing
        self._tok, self._emb = False, {}   # _pass reads back the greedy tokens; token_embd rows dequantized (_emb_row)
        # plain decoding (step_token): while the pass runs, prefetch the next position's PLE rows for up to ple_guess likely
        # next tokens (_guesses); a right guess takes the row reads (~0.15 ms) off the token path. 0 = off (env K3_PLE_GUESS)
        self.ple_guess, self._during = int(os.environ.get('K3_PLE_GUESS', '8')), None
        # wait bound per kernel wait: KFD evicts all of the process's queues (every GPU at once) for 5-280 ms when the host
        # migrates or reclaims pages (memory compaction; tmp/ceiling/s11), and the waves resume to a steady counter that
        # kept running. 200 ms covers the common evictions; a longer one bails and verify() retries the pass
        self.limit_us, self.max_retries, self.bails = limit_us, max_retries, []
        self.feature_layers = list(feature_layers)
        assert feature_card in (0, 1), feature_card
        self.feature_card = feature_card
        hip = _hip()
        for a, b in ((0, 1), (1, 0)):
            hip.hipSetDevice(devices[a]); hip.hipDeviceEnablePeerAccess(devices[b], 0)   # 'already enabled' is fine...
            hip.hipGetLastError()                        # ...but it stays the sticky last error, and torch would raise it
        torch.cuda.set_device(devices[0])
        self.n_ex = 2 * self.nl
        # the engine's counters (the prompt cache and the session read them, never with a default): token grows at every pass
        # launch (a retry too), seq at every reset(), staged is the position staged for the next launch
        self.token, self.seq = 0, 0
        self._card_ev, self.card_ms = None, None   # per-card pass spans (enable_card_timing; off by default)
        self.shards = {}   # split tensors' per-card shards (_shard), for the batched prefill
        # every per-card weight the tapes read, by (loader, tensor name): the memoized loaders' results (_rep / _rep_f32 /
        # _router_w / _f32_idx: the replicated BF16 hc / router / alpha-beta matrices, norms, dt / a; _rows / _cols / _exp_*:
        # the same shards as self.shards) and the per-card conv weights sliced here ('conv': ssm_conv1d's qkv rows, 'ple_conv').
        # References to the tensors the tapes hold, nothing copied: the batched prefill (caminus.prefill.tp) reads K3's
        # own weights from here, never a second copy
        self.weights = {}
        self._build_tp()

    @staticmethod
    def qsa_mode(model_tag, idx_cache, dense_beyond_qsa, max_ctx, qsa=None):
        """(QSA wired, the last position + 1 a pass may reach: None = max_ctx): qsa None = auto (the Coder with the indexer
        caches at max_ctx > QSA_EXACT, never with dense_beyond_qsa); the dense tape stops at QSA_EXACT unless
        dense_beyond_qsa (the test override)"""
        if qsa is None: qsa = model_tag == 'coder' and idx_cache and not dense_beyond_qsa and max_ctx > capacity.QSA_EXACT
        if qsa and (model_tag != 'coder' or not idx_cache or dense_beyond_qsa):
            raise ValueError('FlashNextTP(qsa=True): the GSQ-RCO Coder only, with idx_cache and without dense_beyond_qsa')
        return bool(qsa), (None if (dense_beyond_qsa or qsa) else capacity.QSA_EXACT)

    # -- capacity ---------------------------------------------------------------------------------------------------------
    def plan_prefill(self, spec, n_feat=0, feature_card=0):
        """the plan of the batched prefill this engine will be given (vram_prefill: the chunk, or dict(chunk=, q6k=,
        q6k_buffers=)): caminus.prefill.plan.PrefillPlan at this engine's split, ntok, feature layers / card and QSA, the
        'wmma' bf16 buffers by prefill.tp.bf16_wanted, the feed's shapes from this engine's own reader (feed.feed_shape)"""
        from ..prefill.plan import PrefillPlan
        from ..prefill.tp import bf16_wanted
        from ..prefill.feed import feed_shape
        if isinstance(spec, bool) or not isinstance(spec, (int, dict)):
            raise ValueError(f'FlashNextTP: vram_prefill {spec!r} (a chunk, or dict(chunk=, q6k=, q6k_buffers=))')
        sp = dict(chunk=spec) if isinstance(spec, int) else dict(spec)
        if set(sp) - {'chunk', 'q6k', 'q6k_buffers'} or 'chunk' not in sp:
            raise ValueError(f'FlashNextTP: vram_prefill {spec!r} (a chunk, or dict(chunk=, q6k=, q6k_buffers=))')
        return PrefillPlan.of(self.geom, int(sp['chunk']), gdn_keys=self.gdn_keys, moe_blocks=self.moe_blocks, T_pass=self.ntok,
                              n_feat=n_feat, feature_card=feature_card, qsa_ctx=self.max_ctx if self.qsa else 0,
                              bf16_dense=bf16_wanted(sp.get('q6k'), sp.get('q6k_buffers')), feed=feed_shape(self.rd, self.path))

    def _check_vram(self, reserve_mib, drafters=(), prefill=None, n_feat=0, feature_card=0):
        """the projected VRAM per card (MiB) of this configuration against each card's cap, before any allocation
        (capacity.CapacityError); returns the plan rows. A model without a measured fixed part (capacity.FIXED_MIB /
        MODEL_BYTES: another cut or quantization behind the same module) is refused, never counted as 0"""
        wb = capacity.model_bytes(self.rd)
        sp = (self.gdn_keys, self.moe_blocks)     # another split than the measured one: its tensors move between the cards (D-292)
        fx = capacity.fixed_mib(self.model_tag, self.ntok, wb, split=sp,
                                tensors=None if sp == capacity.MEASURED_SPLIT else capacity.split_tensors(self.rd))
        if fx is None:
            raise capacity.CapacityError(f'FlashNextTP: no measured fixed VRAM for this model ({self.model_tag} module, '
                                         f'{wb} tensor bytes; measured: {capacity.MODEL_BYTES}): run tools/k3_capacity.py measure and add '
                                         f'it to capacity.FIXED_MIB / MODEL_BYTES, or build with vram_check=False')
        cb = capacity.tp_ctx_bytes(self.geom, self.max_ctx, self.idx_cache)
        if self.kv_host: capacity.check_host_kv(2 * cb.pop('kv'))   # (both cards' caches in host memory, not in VRAM)
        if self.qsa: cb.update(capacity.tp_qsa_bytes(self.geom, self.max_ctx, self.ntok))
        ctx = sum(cb.values()) / capacity.MIB
        dr = [0.0, 0.0]
        for d in drafters:
            kind, dg, ring = (d, None, None) if isinstance(d, str) else (tuple(d) + (None, None))[:3]
            x = capacity.drafter_need_mib(kind, self.geom, self.max_ctx, self.ntok, dflash_geom=dg, ring=ring)
            dr = [dr[c] + x[c] for c in (0, 1)]
        # prefill: the plan (plan_prefill), or a chunk (the default paths, the Coder's feed shapes: prefill.plan.CODER_FEED)
        from ..prefill.plan import PrefillPlan
        if prefill is not None and not isinstance(prefill, PrefillPlan):
            prefill = PrefillPlan.of(self.geom, int(prefill), gdn_keys=self.gdn_keys, moe_blocks=self.moe_blocks, T_pass=self.ntok,
                                     n_feat=n_feat, feature_card=feature_card, qsa_ctx=self.max_ctx if self.qsa else 0)
        pf = prefill.need_mib() if prefill is not None else (0.0, 0.0)
        need = [fx[c] + ctx + float(reserve_mib[c]) + dr[c] + pf[c] for c in (0, 1)]
        rows = capacity.check(self.devs, need, f'FlashNextTP(max_ctx={self.max_ctx}, ntok={self.ntok}' +
                              (f', drafters {[d if isinstance(d, str) else d[0] for d in drafters]}' if drafters else '') +
                              (f', prefill chunk {prefill.chunk}{", bf16 dense" if prefill.bf16_dense else ""})' if prefill is not None else ')'))
        return dict(fixed_mib=fx, ctx_mib=ctx, reserve_mib=tuple(reserve_mib), drafters_mib=tuple(dr), prefill_mib=tuple(pf), rows=rows)

    def _alloc_idx(self):
        """the QSA indexer caches per card, zeroed (spec-qsa.md section 3, the dense QSA layout, q8_0 as the KV): IDXRAW
        [attention layers][max_ctx][136 B] (every position's raw indexer key) and IDXPOOL [attention layers][max_ctx / 4]
        [136 B] (one pooled key per 4 positions). The QSA tapes write and read them (allocated before the tapes are built,
        which take their addresses); a QSA engine's dense tape writes IDXRAW only (its pools: _backfill), any other dense
        tape leaves them alone. In self.state (zeroed by reset(); never part of a
        snapshot: rows by position, like the KV)"""
        rb, n = capacity.q8_bytes(self.geom['idx_dim']), self.geom['n_attn']
        assert n == sum(1 for il in range(self.nl) if il % self.fai == self.fai - 1), 'indexer caches: one per attention layer'
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                self.state[(c, 'idxraw')] = torch.zeros(n, self.max_ctx, rb, dtype=torch.uint8, device='cuda')
                self.state[(c, 'idxpool')] = torch.zeros(n, (self.max_ctx + 3) // 4, rb, dtype=torch.uint8, device='cuda')
        torch.cuda.set_device(self.devs[0])

    # -- weights: read once onto card 0, sliced there; card 1's piece copied over the link -------------------------------
    def _dev(self, c):
        return f'cuda:{self.devs[c]}'

    def _full(self, name):
        t = self.T[name]
        return self.dr.read(int(t.data_offset), int(t.n_bytes), device=self._dev(0))

    def _info(self, name, w, rows, cols, rb, slab=None):
        t = self.T[name]
        return dict(name=name, w=w.reshape(-1), fmt=t.tensor_type.name, rows=rows, cols=cols, rb=rb, slab=slab)

    def _kv_host_rows(self, c, ai, pb):
        """kv_host: attention layer ai's KV cache rows of card c, a slice of the card's one host_mapped buffer"""
        from ..device import hip as _hip
        na = self.geom['n_attn']; n = self.max_ctx * pb
        if self._kvh[c] is None:
            with torch.cuda.device(self.devs[c]): self._kvh[c] = _hip.host_mapped(na * n)
        assert 0 <= ai < na, (ai, na)
        return self._kvh[c][ai * n:(ai + 1) * n]

    def _rep(self, name):
        """replicated GG matrix: [card 0, card 1]"""
        t = self.T[name]; w = self._full(name)
        rows, cols = (int(t.shape[1]) if len(t.shape) > 1 else 1), int(t.shape[0])   # a 1-row matrix may be stored 1-D
        rb = int(t.n_bytes) // rows
        return [self._info(name, w, rows, cols, rb), self._info(name, w.to(self._dev(1)), rows, cols, rb)]

    def _rep_f32(self, name, shape=None):
        t = self.T[name]; assert t.tensor_type.name == 'F32', (name, t.tensor_type.name)
        w = self._full(name).view(torch.float32); w = w.reshape(shape) if shape else w
        return [w, w.to(self._dev(1))]

    def _rep_bf16(self, name):
        """replicated BF16 matrix [rows, cols] (raw bytes, rb = 2 * cols): [card 0, card 1] infos for mv's OP_BF16_MV"""
        t = self.T[name]; assert t.tensor_type.name == 'BF16', (name, t.tensor_type.name)
        return self._rep(name)

    def _router_w(self, name, rows):
        """a small replicated matrix read by fp32 activations (router logits, shared-expert gate): F32 (the REAP-308 cut,
        OP_F32_MV) or BF16 (the GSQ-RCO Coder, OP_BF16_MV); per card a dict(fmt, w)"""
        f = self.T[name].tensor_type.name
        if f == 'F32': return [dict(fmt='F32', w=w) for w in self._rep_f32(name, (rows, self.D))]
        assert f == 'BF16', f'{name}: {f} (router / shared-expert gate: F32 or BF16 only)'
        return [dict(fmt='BF16', w=i['w'], rows=i['rows'], cols=i['cols']) for i in self._rep_bf16(name)]

    def _rows(self, name, ranges):
        """per card, the listed row ranges of a GG matrix, concatenated"""
        t = self.T[name]; rb = t.data.shape[-1]; full = self._full(name).view(-1, rb); out = []
        for c in (0, 1):
            w = torch.cat([full[r0:r1] for r0, r1 in ranges[c]]).contiguous()
            out.append(self._info(name, w if c == 0 else w.to(self._dev(1)), w.shape[0], int(t.shape[0]), rb))
        self._shard(name, 'rows', out, [[(int(a), int(b)) for a, b in ranges[c]] for c in (0, 1)])
        return out

    @staticmethod
    def ksplit_pad(t, K):
        """the zeroed tail (bytes) after a K-split shard of K values per row: what the prefill's MMQ reads past the last
        row (caminus.prefill.mmq.overread_bytes: it loads K in 256-value steps, so only a K that is not a multiple of 256 has one:
        the card-0 MoE down / shared-expert down shards, 384 wide: IQ4_NL 72 B, Q2_0 36 B, Q8_0 136 B). Not ggml's
        512-value rule: card 1's 256-wide IQ4_NL down shards are exactly 90 MiB, and its 144 B tail pushed each into a
        92 MiB allocator segment (+78 MiB on the XT, measured); card 0's 135 MiB shards had the slack. The decode never
        reads it"""
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        if 256 % bs: return 0
        nb, bpi = K // bs, 256 // bs
        return ((nb + bpi - 1) // bpi * bpi - nb) * ts

    def _padded(self, c, nbytes, pad):
        """a uint8 buffer of nbytes + pad on card c, its tail zeroed; returns its first nbytes (the shard, filled by the
        caller; the tail stays zero)"""
        b = torch.empty(nbytes + pad, dtype=torch.uint8, device=self._dev(c)); b[nbytes:].zero_()
        return b[:nbytes]

    def _cols(self, name, blocks):
        """per card, the listed block-column ranges of every row (a K split: partial dot products), each card's shard with
        a zeroed tail (ksplit_pad)"""
        t = self.T[name]; bs, ts = GGML_QUANT_SIZES[t.tensor_type]; rb = t.data.shape[-1]
        full = self._full(name).view(-1, rb // ts, ts); out = []; R = full.shape[0]
        for c in (0, 1):
            nb = sum(b1 - b0 for b0, b1 in blocks[c]); w = self._padded(c, R * nb * ts, self.ksplit_pad(t, nb * bs))
            if c == 0: torch.cat([full[:, b0:b1] for b0, b1 in blocks[c]], dim=1, out=w.view(R, nb, ts))
            else: w.view(R, nb, ts).copy_(torch.cat([full[:, b0:b1] for b0, b1 in blocks[c]], dim=1))
            out.append(self._info(name, w, R, nb * bs, nb * ts))
        self._shard(name, 'cols', out, [[(int(a) * bs, int(b) * bs) for a, b in blocks[c]] for c in (0, 1)])
        return out

    def _exp_rows(self, name, ranges):
        """experts [E, rows, rb]: card c gets rows ranges[c] = (r0, r1) of every expert"""
        t = self.T[name]; E, R, rb = t.data.shape; full = self._full(name).view(E, R, rb); out = []
        for c in (0, 1):
            r0, r1 = ranges[c]; w = full[:, r0:r1].contiguous()
            out.append(self._info(name, w if c == 0 else w.to(self._dev(1)), r1 - r0, int(t.shape[0]), rb, slab=(r1 - r0) * rb))
        self._shard(name, 'exp_rows', out, [[(int(ranges[c][0]), int(ranges[c][1]))] for c in (0, 1)])
        return out

    def _exp_cols(self, name, ranges):
        """experts [E, rows, blocks]: card c gets block columns ranges[c] = (b0, b1) of every row, with a zeroed tail
        (ksplit_pad)"""
        t = self.T[name]; bs, ts = GGML_QUANT_SIZES[t.tensor_type]; E, R, rb = t.data.shape
        full = self._full(name).view(E, R, rb // ts, ts); out = []
        for c in (0, 1):
            b0, b1 = ranges[c]; nb = b1 - b0; w = self._padded(c, E * R * nb * ts, self.ksplit_pad(t, nb * bs))
            w.view(E, R, nb, ts).copy_(full[:, :, b0:b1])
            out.append(self._info(name, w, R, nb * bs, nb * ts, slab=R * nb * ts))
        self._shard(name, 'exp_cols', out, [[(int(ranges[c][0]) * bs, int(ranges[c][1]) * bs)] for c in (0, 1)])
        return out

    def _shard(self, name, kind, infos, ranges):
        """records a split tensor's per-card shards for the batched prefill (caminus.prefill.mmq on the same weights): kind 'rows'
        / 'exp_rows' (column split: each card owns whole output rows) or 'cols' / 'exp_cols' (K split: partial sums, summed
        by the exchange), ranges per card in the original tensor (rows, or K values), the infos (w, fmt, rows, cols, rb)"""
        self.shards[name] = dict(kind=kind, infos=infos, ranges=ranges, fmt=self.T[name].tensor_type.name)

    def _f32_idx(self, name, idx):
        assert self.T[name].tensor_type.name == 'F32', (name, self.T[name].tensor_type.name)
        w = self._full(name).view(torch.float32); out = []
        for c in (0, 1):
            v = w[torch.tensor(idx[c], device=w.device)].contiguous()
            out.append(v if c == 0 else v.to(self._dev(1)))
        return out

    # -- the two tapes --------------------------------------------------------------------------------------------------
    def _build_tp(self):
        """_build_tapes with the weight loaders memoized (a second tape takes the first one's tensors: read and sliced once)"""
        memo, names = {}, ('_rep', '_rep_f32', '_rows', '_cols', '_router_w', '_exp_rows', '_exp_cols', '_f32_idx')
        def cached(f):
            def g(*a):
                k = (f.__name__, repr(a))
                if k not in memo: memo[k] = f(*a)
                self.weights.setdefault((f.__name__, a[0]), memo[k])
                return memo[k]
            return g
        for n in names: setattr(self, n, cached(getattr(self, n)))
        try: self._build_tapes()
        finally:
            for n in names: self.__dict__.pop(n, None)

    def _build_tapes(self):
        D, HC, hd = self.D, self.hc, self.hd
        nh, nkv = self.nh // 2, self.nkv // 2                       # per card: 12 query heads, 1 kv head
        assert self.nkv == 2 and self.nh % 2 == 0
        k0 = self.gdn_keys; assert k0 % 2 == 0 and 0 < k0 < 16
        kc, ks = [k0, 16 - k0], [0, k0]                             # GDN key heads per card, first key head
        KH = [list(range(ks[c], ks[c] + kc[c])) for c in (0, 1)]
        VH = [[h for h in range(48) if h % 16 in KH[c]] for c in (0, 1)]   # its value heads (local h % kc = key head)
        vruns = [[(16 * g + ks[c], 16 * g + ks[c] + kc[c]) for g in range(3)] for c in (0, 1)]   # value-head runs
        nbt = self.ffn // 32; mb = [(0, self.moe_blocks), (self.moe_blocks, nbt)]   # MoE intermediate, in 32s
        ffc = [32 * (b1 - b0) for b0, b1 in mb]
        CH = [640 * kc[c] for c in (0, 1)]                          # qkv channels: q, k (128 each per key head), 3 v heads
        Tn = self.ntok
        # rows per task, per card (XTX, XT), one-token passes only (T > 1 keeps the old values: not measured there). Swept
        # per half-layer on each card alone and on the TP wall; all clamp the last task, so any value is bitwise the same.
        # hc up (10240 rows): block half 107 tasks (XTX) / 160 (XT), FFN half 320 on both (best measured; 640 before);
        # router logits (308 rows): 77 tasks of 4 rows (154 x 2 before); attn_qkv on the XT (3840 rows): 160 tasks, one
        # round of its 210 WGs. Kept at 16 / 8: attn_q, MoE gate-up / down (larger rpt measured slower or no better).
        # fn_bench 96 --tp: 16.16 -> 15.81 ms/token. Re-swept with the loads-first Q8_0 rows (k3_gguf_ops.h gg_mv1): block
        # half hc up on the XT 64 -> 32 rows (320 tasks), 10.77 -> 10.70 ms/token; the others stay (XTX block 32 / 64 / 128,
        # FFN 16 / 40 / 48 / 64, hc down 4: no better)
        # T > 1 (the k3_gguf_mt module, swept on the T=4 verify pass at fixed positions, tmp/ceiling/s13/sweep_t4.py): ssm_out
        # and attn_output 16 -> 8, router logits 2 -> 4, shared-expert down 16 -> 32, hc down 8 -> 4 rows per task: 17.28 ->
        # 16.44 ms per pass (per card no further gain above noise; hc up, qkv, attn_q, MoE, head, z flat or worse); re-swept
        # with the T > 1 prefetch off: all at their optimum but the shared-expert down, 32 -> 64 (16.16 -> 16.09)
        RPT = dict(hc_attn_up=(96, 32), hc_ffn_up=(32, 32), output_hc_up=(64, 64), router=(4, 4), attn_qkv=(16, 24),
                   ssm_out=(16, 16), attn_output=(16, 16), down_shexp=(16, 16), hc_down=(8, 8)) if Tn == 1 else \
            dict(hc_attn_up=(16, 16), hc_ffn_up=(16, 16), output_hc_up=(16, 16), router=(4, 4), attn_qkv=(16, 16),
                 ssm_out=(8, 8), attn_output=(8, 8), down_shexp=(64, 64), hc_down=(4, 4))
        E, S, res, pos, ple_emb, epoch, slots = [], [], [], [], [], [], []
        # the per-pass inputs, one contiguous block per card (one H2D copy per pass from one pinned host block, the same
        # bytes for both cards): 64 words of scalars (the engine's params [16], pos at word 16, epoch at word 17), then the
        # residual [Tn * HC, D], then the PLE input [Tn, D]; every piece 256-byte aligned. The host rewrites the whole block
        # each pass (the kernels only read params / pos / epoch; res and the PLE input were rewritten per pass before)
        W0, nres, nple = 64, Tn * HC * D, Tn * D
        self.stage_words = (W0, nres, nple); self.stage = []
        self.hstage = torch.zeros(W0 + nres + nple, dtype=torch.float32, pin_memory=True)
        hs = self.hstage.numpy(); self.h_i32 = hs[:W0].view(np.int32)
        self.h_res = hs[W0:W0 + nres].reshape(Tn, HC, D); self.h_ple = hs[W0 + nres:].reshape(Tn, D)
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                B = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
                q8b = lambda n: B((n + 31) // 32 * 36, dt=torch.uint8)
                E.append(k3.Engine(module=self.module, grid=k3.resident_grid(self.module), T=Tn, limit_us=self.limit_us))
                blk = B(W0 + nres + nple); self.stage.append(blk)
                E[c].params = blk[0:16].view(torch.int32)        # before any segment takes its address
                pos.append(blk[16:17].view(torch.int32)); epoch.append(blk[17:18].view(torch.int32))
                res.append(blk[W0:W0 + nres].view(Tn * HC, D)); ple_emb.append(blk[W0 + nres:].view(Tn, D))
                assert blk.data_ptr() % 256 == 0 and res[c].data_ptr() == blk.data_ptr() + 4 * W0 and res[c].data_ptr() % 256 == 0
                assert ple_emb[c].data_ptr() == blk.data_ptr() + 4 * (W0 + nres) and ple_emb[c].data_ptr() % 256 == 0
                assert E[c].params.data_ptr() == blk.data_ptr() and pos[c].data_ptr() == blk.data_ptr() + 64 and epoch[c].data_ptr() == blk.data_ptr() + 68
                assert not E[c].segs
                av_ = max(nh * hd, 384 * kc[c])
                S.append(dict(xn=B(Tn * HC, D), xq=q8b(Tn * HC * D), lo=B(Tn * 320), loq=q8b(Tn * 320), inj=B(Tn * HC), gate=B(Tn * HC, D),
                              mixed=B(Tn * D), mq=q8b(Tn * D), blkp=B(Tn * D), blk=B(Tn * D), qkv=B(Tn * CH[c]), z=B(Tn * 384 * kc[c]),
                              al=B(Tn * 3 * kc[c]), be=B(Tn * 3 * kc[c]), gates=B(Tn * 6 * kc[c]), conv=B(Tn * CH[c]),
                              qk=B(Tn * 2 * kc[c] * 128), att=B(Tn * av_), fin=B(Tn * 384 * kc[c]), fq=q8b(Tn * av_),
                              qf=B(Tn * nh * 2 * hd), kr=B(Tn * nkv * hd), vr=B(Tn * nkv * hd), qo=B(Tn * nh * hd), ko=B(Tn * nkv * hd),
                              gat=B(Tn * nh * hd), logit=B(Tn * self.n_exp), ids=B(Tn * self.k_used, dt=torch.int32), wts=B(Tn * self.k_used),
                              h=B(Tn * self.k_used * ffc[c]), hq=q8b(Tn * self.k_used * ffc[c]), moe=B(Tn * D), shh=B(Tn * ffc[c]),
                              shq=q8b(Tn * ffc[c]), shx=B(Tn * D), sgl=B(Tn), ffp=B(Tn * D), ffo=B(Tn * D),
                              pq=q8b(Tn * D), pkey=B(Tn * HC, D), pval=B(Tn * D), pkn=B(Tn * HC, D), pqn=B(Tn * HC, D),
                              pgated=B(Tn * HC, D), pg=B(Tn * HC), pnrm=B(Tn * HC, D), pco=B(Tn * HC * D),
                              one_id=torch.zeros(Tn, dtype=torch.int32, device='cuda'), one_w=torch.ones(Tn, device='cuda')))
            slots.append(Slots(self.devs[c], self.n_ex, Tn * D))
        # the XT (card 1) direction of the one-token exchanges by pull: card 1 stores its partial into its own uncached slot
        # (local, ~0.5 us cheaper to complete than the PCIe push at x4) and sends only the flag; card 0, which waits for
        # the XT anyway, reads the partial over PCIe after the flag (~3 us). XT push 2.48 -> 1.60 us, exchange 4.52 -> 3.77;
        # fn_bench 96 --tp 9.95 -> 9.89 ms/token. T > 1: push (40 KB payloads; T=4 pass +0.07 ms by pull). K3_XPULL=0: push
        pull = Slots(self.devs[1], self.n_ex, Tn * D) if os.environ.get('K3_XPULL', '1') == '1' and not self.local_only and Tn == 1 else None
        self._pull = pull
        self.engines, self.S, self.res, self.pos, self.inputs, self.epoch, self.slots = E, S, res, pos, ple_emb, epoch, slots
        self.state = {}; self.layer_seg = [[], []]; self.seg_names = [{}, {}]
        ON = lambda c: torch.cuda.device(self.devs[c])
        if self.idx_cache: self._alloc_idx()
        # QSA scratch per card (not in self.state: per pass, never snapshotted): the pool scores [T][pool rows] (one buffer
        # for all 12 layers: a layer's SCORE starts only after the previous layer's SELECT, through the residual chain) and
        # each layer's selection rows [T][QSA_SELW]
        prow = (self.max_ctx + 3) // 4
        self.qsa_score, self.qsa_sel = [], [[], []]
        # the split attention (the Coder's modules, k3.ATTN_SPLIT_MODULES: kernels/k3_attn_split.h): the longest cell list a
        # pass row can have (the dense tape stops below QSA_EXACT unless dense_beyond_qsa; a QSA row keeps <= QSA_EXACT
        # cells) and one record buffer per card for every attention layer of both tapes (a layer's PART starts only after
        # the previous layer's COMB, through the residual chain), allocated by the first split attention segment
        self.attn_cells = self.max_ctx if (self.qsa_limit is None and not self.qsa) else min(self.max_ctx, capacity.QSA_EXACT)
        self._attn_part = [None, None]
        def akw(c):
            if not E[c].split_attn(): return {}
            if self._attn_part[c] is None:
                ns = (self.attn_cells + k3.ATTN_SS - 1) // k3.ATTN_SS
                self._attn_part[c] = torch.zeros(Tn * nh * ns * k3.ATTN_RS, device='cuda')
            return dict(max_cells=self.attn_cells, part=self._attn_part[c])
        if self.qsa:
            for c in (0, 1):
                with ON(c): self.qsa_score.append(torch.zeros(Tn, prow, device='cuda'))
        # TEST-ONLY (K3_TEST_HOOKS=qsa-scores; tools/eval/qsa_depth.py, impl:qsa-depth): each QSA layer writes its scores and its
        # normed + RoPE'd indexer query into rows of its own (qsa_score_l / qsa_q_l [card][layer], the same shapes), so all
        # 12 layers' rows can be read after a pass; only those pointers differ, every op, operand and order is the default
        # tape's. Off by default: one SCORE buffer and one query scratch per card, as above
        self.qsa_layer_bufs = bool(self.qsa) and os.environ.get('K3_TEST_HOOKS') == QSA_LAYER_HOOKS
        self.qsa_score_l, self.qsa_q_l = [[], []], [[], []]
        # the second tape (two-tape QSA) shares everything above and the states / weights below: st_ allocates a state once,
        # and the weight loaders are memoized for the build (the same tensors in both tapes, read once)
        st_ = lambda key, make: self.state[key] if key in self.state else self.state.setdefault(key, make())
        def new_engines():
            out = []
            for c in (0, 1):
                with ON(c):
                    e = k3.Engine(module=self.module, grid=k3.resident_grid(self.module), T=Tn, limit_us=self.limit_us)
                    e.params = self.stage[c][0:16].view(torch.int32); assert e.params.data_ptr() == pos[c].data_ptr() - 64; out.append(e)
            return out

        # pre-wait prefetch byte cap per matvec (k3_gguf_ops.h k3_extra_prefetch; Seg.aux = cap + 1, 0 = the whole task):
        # the first 16 KB of each task (fn_bench 96 --tp 10.74 -> 10.40 ms/token; uncapped hc down tasks are 87 KB, three
        # serial rounds before the dependency wait). Env K3_PF_CAP = 'name-regex:bytes,...' overrides (re.search on the
        # tensor name, or f32_router_logits / f32_shexp_gate; first match wins; 0 = off)
        # T > 1: off (T=4 pass 16.98 -> 16.60 ms; a 4 / 8 / 32 KB cap, or keeping it on the router logits, hc down or qkv /
        # q: no better or worse)
        pf_cap = [(re.compile(k), int(v)) for k, v in (e.rsplit(':', 1) for e in os.environ.get('K3_PF_CAP', '.:16384' if Tn == 1 else '.:0').split(',') if e)]

        # BF16 rows per task (OP_BF16_MV only, so REAP-308's Q8_0 / F32 segments keep RPT's values): BF16_RPT below, then the
        # call site's value; env K3_BF16_RPT = 'name-regex:rpt,...' goes first (re.search on the tensor name, or
        # f32_router_logits / f32_shexp_gate; first match wins; 'a/b': card 0 / 1). Swept on the Coder with the split-K
        # kernel (k3_gguf_ops.h bf16_sk / bf16_short; tmp/m1/s6/bf/sweep_coder.py, interleaved, bitwise in every config):
        # T = 1: hc down 8 -> 1 row (each row is one workgroup's 8 waves now; 40 -> 320 tasks): 11.79 -> 10.71 ms/token;
        # inject, alpha / beta, router, PLE, hc up and output_hc flat or worse. T > 1: hc down 4 -> 2, alpha / beta 16 -> 4,
        # router logits 4 -> 2: T=4 pass 20.98 -> 20.11 ms (hc down 1 / 3 / 8, alpha / beta 2 / 8, router 1 / 8: no better or
        # worse). With bf16_short's grouped token loop (one row per wave per pass at T > 1): hc up / output_hc up 16 -> 24,
        # 20.10 -> 19.78 ms (interleaved x 5; 8 / 12 / 32 / 48: 20.24 / 20.13 / 19.85 / 20.98)
        BF16_RPT = [(r'\.hc_(attn|ffn)_down\.', '1')] if Tn == 1 else \
            [(r'\.hc_(attn|ffn)_down\.', '2'), (r'\.ssm_(alpha|beta)\.', '4'), (r'^f32_router_logits$', '2'),
             (r'(\.hc_(attn|ffn)_up|^output_hc_up)\.', '24')]
        bf_rpt = [(re.compile(k), [int(u) for u in (v.split('/') * 2)[:2]])
                  for k, v in [e.rsplit(':', 1) for e in os.environ.get('K3_BF16_RPT', '').split(',') if e] + BF16_RPT]
        bfr = lambda name, rpt, c: next((v[c] for k, v in bf_rpt if k.search(name)), rpt)

        # GG (K-quant) rows per task of the GSQ-RCO Coder (k3_gguf_gsq modules only: REAP-308 keeps RPT and the call sites'
        # values): GG_RPT below, then RPT / the call site's value; env K3_GG_RPT = 'name-regex:rpt,...' goes first (any
        # module; re.search on the tensor name, first match wins; 'a/b': card 0 / 1). Every OP_GG_MV path clamps the last
        # task (r1 = min(N, r0 + rpt)), so any value is bitwise the same. Swept per card on the Coder's TP pass at grids
        # 384 / 336 (tmp/m1/s6/rs/sweep_gg.py, interleaved, bitwise in every config):
        # T = 1: ssm_out 16 -> 8 (the XTX's rows were on the path: 160 tasks for its 384 workgroups), shared-expert down
        # 16 -> 32 (the XT's): 10.24 -> 10.07 ms in-process with shared gate / up 16 -> 8, which was dropped again (+0.025 ms
        # on its own, under the 0.05 ms keep bar) (qkv / attn q / attn output / z / head / attn k, v: flat or worse; ssm_out
        # 4 / 6 / 12 / 32, shared down 8 / 48 / 64, shared gate / up 32: no better).
        # T > 1: attn_qkv on the XT 16 -> 8 (3840 rows: 480 tasks for its 336 workgroups), shared-expert down 64 -> 128:
        # T=4 pass 18.47 -> 18.28 ms in-process (qkv 4 / 6 / 12 / 24 / 32, ssm_out 4 / 6 / 12 / 16, attn output 4 / 6 / 12 /
        # 16, attn q, z, head, shared gate / up, attn k / v: flat or worse)
        GG_RPT = [] if 'gsq' not in self.module else \
            [(r'\.ssm_out\.', '8'), (r'\.ffn_down_shexp\.', '32')] if Tn == 1 else \
            [(r'\.attn_qkv\.', '16/8'), (r'\.ffn_down_shexp\.', '128')]
        gg_rpt = [(re.compile(k), [int(u) for u in (v.split('/') * 2)[:2]])
                  for k, v in [e.rsplit(':', 1) for e in os.environ.get('K3_GG_RPT', '').split(',') if e] + GG_RPT]
        assert all(r >= 1 for _, v in gg_rpt for r in v), 'K3_GG_RPT / GG_RPT: rows per task >= 1'
        ggr = lambda name, rpt, c: next((v[c] for k, v in gg_rpt if k.search(name)), rpt)

        def mv(c, W, x, out, dep, rpt=16, silu_scale=None, xf=None):
            """W . x -> out. x: the GG formats' activation (q8_1 blocks, or fp32 quantized per workgroup); xf: the fp32
            activation [T, cols] a BF16 W reads instead (OP_BF16_MV; llama.cpp's mul_mat_vec_f reads fp32)"""
            if W['fmt'] == 'BF16':
                assert xf is not None, f"{W['name']}: BF16 weights at a call site without an fp32 activation"
                rpt = bfr(W['name'], rpt, c)
                si = E[c].bf16_mv(W['w'], W['rows'], W['cols'], xf, out, rows_per_task=rpt, dep=dep, silu_scale=silu_scale)
            else:
                assert x is not None, f"{W['name']}: {W['fmt']} weights at a call site without a GG activation"
                rpt = ggr(W['name'], rpt, c)
                si =E[c].gg_matvec(W['w'], W['fmt'], W['rows'], W['cols'], W['rb'], x, out, rows_per_task=rpt, dep=dep, silu_scale=silu_scale)
            pf_set(c, si, W['name'])
            self.seg_names[c][si] = W['name']; return si

        def mv_router(c, W, x, out, rpt, dep, name):
            """router logits / shared-expert gate from fp32 x: OP_F32_MV (F32 weights) or OP_BF16_MV (BF16)"""
            if W['fmt'] == 'F32': si = E[c].f32_mv(W['w'], x, out, rows_per_task=rpt, dep=dep)
            else: si = E[c].bf16_mv(W['w'], W['rows'], W['cols'], x, out, rows_per_task=bfr(name, rpt, c), dep=dep)
            pf_set(c, si, name); return si

        def pf_set(c, si, name):
            cap = next((v for k, v in pf_cap if k.search(name)), None)
            if cap is not None: E[c].segs[si].aux = cap + 1

        def xchg(c, partial, out, x, dep):
            if pull is not None:   # card 1 publishes into its own slot; card 0 reads it from there (K3_XPULL=1)
                return E[c].xchg(partial, out, slots[1 - c].slot(x), slots[c].slot(x), epoch[c], x, c, local_only=self.local_only, dep=dep,
                                 pull=pull.slot(x) + 64, pull_mode=1 if c == 1 else 2)
            return E[c].xchg(partial, out, slots[1 - c].slot(x), slots[c].slot(x), epoch[c], x, c, local_only=self.local_only, dep=dep)

        self.features = torch.zeros(len(self.feature_layers), Tn, D, device=self._dev(self.feature_card))

        def hc_mix(prefix, dep, inject=True, post=None, feat=None):
            nw = self._rep_f32(f'{prefix}_norm.weight', (HC, D)); dn = self._rep(f'{prefix}_down.weight')
            iw = self._rep(f'{prefix}_inject.weight') if inject else None; up = self._rep(f'{prefix}_up.weight')
            M, IJ = [], []
            # the q8_1 copy of xn only when a GG matrix reads it (BF16 down / inject read the fp32 xn: no dead producer)
            xq_used = dn[0]['fmt'] != 'BF16' or (inject and iw[0]['fmt'] != 'BF16')
            for c in (0, 1):
                with ON(c):
                    s = S[c]; xq = s['xq'] if xq_used else None
                    if post is None: r = E[c].rms_mul(res[c], nw[c], s['xn'], EPS, dep=dep[c], q8=xq)
                    else: r = E[c].also(E[c].hc_post_rms(s[post[0]], s['inj'], res[c], nw[c], s['xn'], EPS, q8=xq, dep=post[1][c]), post[2][c])
                    a = mv(c, dn[c], xq, s['lo'], r, RPT['hc_down'][c], xf=s['xn'])
                    ri = r
                    if feat is not None and c == self.feature_card:   # the layer's input residual, before the FFN half overwrites it:
                        ri = E[c].hc_avg(res[c], self.features[feat], dep=r)   # the inject (and so the next residual update) waits for it
                    ij = mv(c, iw[c], xq, s['inj'], ri, 8, xf=s['xn']) if inject else r
                    if up[c]['fmt'] == 'BF16':   # fp32 lo, silu(lo / HC) by the op itself (any T)
                        a = mv(c, up[c], None, s['gate'].reshape(-1), a, RPT[prefix.split('.')[-1] + '_up'][c] if Tn == 1 else 16, silu_scale=1.0 / HC, xf=s['lo'])
                    elif Tn == 1: a = mv(c, up[c], s['lo'], s['gate'].reshape(-1), a, RPT[prefix.split('.')[-1] + '_up'][c], silu_scale=1.0 / HC)
                    else:   # (LDS quantization of all T tokens' silu(lo / HC) per workgroup measured slower: T=4 pass 18.6 -> 20.4 ms)
                        a = E[c].gg_q8_1(s['lo'], s['loq'], dep=a, silu_scale=1.0 / HC); a = mv(c, up[c], s['loq'], s['gate'].reshape(-1), a, 16)
                    M.append(E[c].hc_mean(s['xn'], s['gate'], s['mixed'], dep=a, q8=s['mq'])); IJ.append(ij)
            return M, IJ

        # the tapes (spec-qsa.md S6, the two-tape fix): one or two per card over ONE set of buffers (the stage block with
        # params / pos / epoch, the exchange slots, every state, the KV and indexer caches, the activations, the logits
        # shards): 'dense' (the trunk module: full attention; with QSA wired it also writes each position's raw indexer key
        # into IDXRAW: k_proj BF16 + KV_APPEND M=64, the same q8_0 bytes as IDX_APPEND) and 'qsa' (the QSA module). Wired
        # QSA runs the dense tape for passes ending below QSA_EXACT and the QSA tape from there (_pass_launch; the pools of
        # the dense positions are backfilled from IDXRAW at the switch, _backfill): the QSA module's code object costs any
        # tape ~0.8 ms per token (verify_q3: +0.9 ms at depth 0), so short contexts keep the trunk's speed
        kinds = (['dense', 'qsa'] if self.dual else ['qsa']) if self.qsa else ['dense']
        self._tapes, self._tape_meta, self._idx_kn = {}, {}, {}
        for kind in kinds:
            qsa_t = kind == 'qsa'; idxraw_t = kind == 'dense' and self.qsa
            if kind != kinds[0]: E[:] = new_engines()
            self.layer_seg = [[], []]; self.seg_names = [{}, {}]; ai = 0
            dep, post = [-1, -1], None
            for il in range(self.nl):
                p = f'blk.{il}'
                for c in (0, 1): self.layer_seg[c].append((il, len(E[c].segs)))
                if il in self.ple_layers:                       # replicated (one layer, small)
                    kw, vw = self._rep(f'{p}.ple_key.weight'), self._rep(f'{p}.ple_value.weight')
                    nk_, nq_, nc_ = (self._rep_f32(f'{p}.ple_norm_{n}.weight', (HC, D)) for n in ('key', 'query', 'conv'))
                    ct = self.T[f'{p}.ple_conv1d.weight']; cw = self.dr.read(int(ct.data_offset), int(ct.n_bytes), device=self._dev(0))
                    cws = [cw, cw.to(self._dev(1))]; cws = self.weights.setdefault(('ple_conv', f'{p}.ple_conv1d.weight'), cws)
                    for c in (0, 1):
                        with ON(c):
                            s = S[c]; hist = st_((c, il, 'ple'), lambda: torch.zeros(HC * D, 9, device='cuda'))
                            pxp = st_((c, il, 'plex'), lambda: torch.zeros(Tn * HC * D, device='cuda'))
                            px, pd = ple_emb[c].reshape(-1), dep[c]; pf = px
                            if Tn > 1 and kw[c]['fmt'] != 'BF16' and vw[c]['fmt'] != 'BF16': pd = E[c].gg_q8_1(px, s['pq'], dep=pd); px = s['pq']
                            elif Tn > 1: assert kw[c]['fmt'] == vw[c]['fmt'] == 'BF16', 'PLE key / value: mixed BF16 and GG formats'
                            k_ = mv(c, kw[c], px, s['pkey'].reshape(-1), pd, xf=pf); v_ = mv(c, vw[c], px, s['pval'], pd, xf=pf)
                            kn = E[c].rms_mul(s['pkey'], nk_[c], s['pkn'], EPS, dep=k_)
                            qn = E[c].rms_mul(res[c], nq_[c], s['pqn'], EPS, dep=v_)
                            g_ = E[c].also(E[c].ple_gate(s['pkn'], s['pqn'], s['pval'], s['pgated'], s['pg'], dep=kn), qn)
                            d_ = E[c].rms_mul(s['pgated'], nc_[c], s['pnrm'], EPS, dep=g_)
                            d_ = E[c].ple_conv(s['pnrm'].reshape(-1), cws[c], hist, s['pco'], pxp, dep=d_)
                            dep[c] = E[c].add_sum(s['pgated'].reshape(-1), s['pco'], res[c].reshape(-1), dep=d_)
                M, inj = hc_mix(f'{p}.hc_attn', dep, post=post, feat=self.feature_layers.index(il) if il in self.feature_layers else None)
                BLK = []
                if il % self.fai != self.fai - 1:              # GDN, split by key heads
                    qkv_r = [[(128 * ks[c], 128 * (ks[c] + kc[c])), (2048 + 128 * ks[c], 2048 + 128 * (ks[c] + kc[c]))] + [(4096 + 128 * a, 4096 + 128 * b) for a, b in vruns[c]] for c in (0, 1)]
                    qkv = self._rows(f'{p}.attn_qkv.weight', qkv_r)
                    assert self.T[f'{p}.ssm_conv1d.weight'].tensor_type.name == 'F32'
                    cw = self._full(f'{p}.ssm_conv1d.weight').view(torch.float32).view(-1, 4)
                    conv_w = [torch.cat([cw[r0:r1] for r0, r1 in qkv_r[c]]).contiguous() for c in (0, 1)]; conv_w[1] = conv_w[1].to(self._dev(1))
                    conv_w = self.weights.setdefault(('conv', f'{p}.ssm_conv1d.weight'), conv_w)
                    z = self._rows(f'{p}.attn_gate.weight', [[(128 * a, 128 * b) for a, b in vruns[c]] for c in (0, 1)])
                    al = self._rows(f'{p}.ssm_alpha.weight', vruns); be = self._rows(f'{p}.ssm_beta.weight', vruns)
                    dt = self._f32_idx(f'{p}.ssm_dt.bias', VH); sa = self._f32_idx(f'{p}.ssm_a', VH)
                    nw = self._rep_f32(f'{p}.ssm_norm.weight')
                    bs = GGML_QUANT_SIZES[self.T[f'{p}.ssm_out.weight'].tensor_type][0]
                    assert all(128 * v % bs == 0 for c in (0, 1) for r in vruns[c] for v in r), f'ssm_out {bs}-blocks vs the value-head split'
                    so = self._cols(f'{p}.ssm_out.weight', [[(128 * a // bs, 128 * b // bs) for a, b in vruns[c]] for c in (0, 1)])
                    for c in (0, 1):
                        with ON(c):
                            s = S[c]; H = 3 * kc[c]
                            conv_st = st_((c, il, 'conv'), lambda: torch.zeros(CH[c], 3, device='cuda'))
                            cxp = st_((c, il, 'convx'), lambda: torch.zeros(Tn * CH[c], device='cuda'))
                            gblk = st_((c, il, 'gdn'), lambda: torch.zeros(k3.gdn_block_size(H, 128, kc[c], Tn), device='cuda'))
                            a_ = mv(c, al[c], s['mq'], s['al'], M[c], xf=s['mixed']); b_ = mv(c, be[c], s['mq'], s['be'], M[c], xf=s['mixed'])
                            gt = E[c].also(E[c].gdn_gates(s['al'], s['be'], dt[c], sa[c], s['gates'], dep=a_), b_)
                            q_ = mv(c, qkv[c], s['mq'], s['qkv'], M[c], RPT['attn_qkv'][c])
                            c_ = E[c].gdn_conv(s['qkv'], conv_w[c], conv_st, s['conv'], cxp, dep=q_)
                            l_ = E[c].gdn_l2(s['conv'], s['qk'], 2 * kc[c], 128, EPS, x_stride=CH[c], dep=c_)
                            z_ = mv(c, z[c], s['mq'], s['z'], M[c])
                            nv = Tn * 128 * H; fqv = s['fq'][:nv // 32 * 36]
                            st = E[c].also(E[c].gdn_step(s['qk'], s['conv'][256 * kc[c]:], s['gates'], gblk, s['att'][:nv], H, 128, kc[c], CH[c], dep=l_), gt)
                            E[c].gdn_save(s['qk'], s['conv'][256 * kc[c]:], s['gates'], gblk, H, 128, kc[c], CH[c], dep=st)   # off the critical path
                            nm = E[c].also(E[c].gdn_norm(s['att'][:nv], nw[c], s['z'], s['fin'], H, 128, EPS, dep=st, q8=fqv), z_)
                            BLK.append(mv(c, so[c], fqv, s['blkp'], nm, RPT['ssm_out'][c]))
                else:                                            # attention, split by kv head
                    aq = self._rows(f'{p}.attn_q.weight', [[(nh * 2 * hd * c, nh * 2 * hd * (c + 1))] for c in (0, 1)])
                    ak = self._rows(f'{p}.attn_k.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
                    av = self._rows(f'{p}.attn_v.weight', [[(hd * c, hd * (c + 1))] for c in (0, 1)])
                    qn_, kn_ = self._rep_f32(f'{p}.attn_q_norm.weight'), self._rep_f32(f'{p}.attn_k_norm.weight')
                    bph = hd // GGML_QUANT_SIZES[self.T[f'{p}.attn_output.weight'].tensor_type][0]   # blocks per head
                    assert bph * GGML_QUANT_SIZES[self.T[f'{p}.attn_output.weight'].tensor_type][0] == hd
                    ao = self._cols(f'{p}.attn_output.weight', [[(nh * bph * c, nh * bph * (c + 1))] for c in (0, 1)])
                    pb = (2 * nkv * hd // 32 * 34) if self.q8 else 4 * nkv * hd
                    if qsa_t or idxraw_t:   # the indexer (replicated on both cards: each computes the same selection for its kv head)
                        ikw, iqw = self._rep_bf16(f'{p}.indexer.k_proj.weight'), self._rep_bf16(f'{p}.indexer.q_proj.weight')
                        ikn, iqn = self._rep_f32(f'{p}.indexer.k_norm.weight'), self._rep_f32(f'{p}.indexer.q_norm.weight')
                        idd, idh = self.geom['idx_dim'], self.geom['idx_heads']
                        assert (idd, idh) == (k3.QSA_D, k3.QSA_H) and ikw[0]['rows'] == idd and iqw[0]['rows'] == idd * idh, (idd, idh)
                        self._idx_kn[ai] = ikn
                    for c in (0, 1):
                        with ON(c):
                            s = S[c]; cache = st_((c, il, 'kv'), (lambda: self._kv_host_rows(c, ai, pb)) if self.kv_host else
                                                  (lambda: torch.zeros(self.max_ctx * pb, dtype=torch.uint8, device='cuda')))
                            # claim order: the short k / v chain (k, v, k norm + rope, hadamards, kv append) before q's 384 tasks,
                            # so its one-task glue segments are claimed while q's rows still stream
                            k_ = mv(c, ak[c], s['mq'], s['kr'], M[c]); v_ = mv(c, av[c], s['mq'], s['vr'], M[c])
                            kd = E[c].qk_nr(s['kr'], hd, nkv, hd, kn_[c], pos[c], s['ko'], self.nrot, self.theta_scale, EPS, dep=k_)
                            vd = v_
                            if self.q8: kd = E[c].hadamard(s['ko'], hd, dep=kd); vd = E[c].hadamard(s['vr'], 64, dep=vd)
                            ap = E[c].also(E[c].kv_append(s['ko'], s['vr'], cache, pos[c], q8=self.q8, dep=kd), vd)
                            if qsa_t:
                                # the indexer reads the same block input as q / k / v (llama.cpp's cur): the fp32 mixed row
                                if 'ikr' not in s: s['ikr'] = B(Tn * idd)
                                if 'iq' not in s: s['iq'], s['iqo'] = B(Tn * idd * idh), B(Tn * idd * idh)
                                raw, pool = self.state[(c, 'idxraw')][ai], self.state[(c, 'idxpool')][ai]
                                sel = torch.zeros(Tn * k3.QSA_SELW, dtype=torch.int32, device='cuda'); self.qsa_sel[c].append(sel)
                                if self.qsa_layer_bufs:   # (test-only: this layer's own score / query rows, read after the pass)
                                    score_, iqo_ = torch.zeros(Tn, prow, device='cuda'), B(Tn * idd * idh)
                                    self.qsa_score_l[c].append(score_); self.qsa_q_l[c].append(iqo_)
                                else: score_, iqo_ = self.qsa_score[c], s['iqo']
                                ik = mv(c, ikw[c], None, s['ikr'], M[c], 4, xf=s['mixed'])
                                ia = E[c].idx_append(s['ikr'], raw, pool, pos[c], ikn[c], self.theta_scale, EPS, n_rot=self.nrot, dep=ik)
                                iq = mv(c, iqw[c], None, s['iq'], M[c], 4, xf=s['mixed'])
                                iqd = E[c].qk_nr(s['iq'], idd, idh, idd, iqn[c], pos[c], iqo_, self.nrot, self.theta_scale, EPS, dep=iq)
                            q_ = mv(c, aq[c], s['mq'], s['qf'], M[c])
                            if idxraw_t:   # IDXRAW[pos] = q8_0(k_proj(mixed)): the bytes IDX_APPEND writes (split rows at T > 1)
                                if 'ikr' not in s: s['ikr'] = B(Tn * idd)
                                if Tn > 1 and 'ikb' not in s: s['ikb'] = B(Tn * idd // 2)
                                raw = self.state[(c, 'idxraw')][ai]
                                if Tn == 1:
                                    ik = mv(c, ikw[c], None, s['ikr'], M[c], 4, xf=s['mixed']); ka, kb = s['ikr'][:idd // 2], s['ikr'][idd // 2:]
                                else:
                                    hb = (idd // 2) * ikw[c]['cols'] * 2
                                    wa, wb = dict(ikw[c], w=ikw[c]['w'][:hb], rows=idd // 2), dict(ikw[c], w=ikw[c]['w'][hb:], rows=idd // 2)
                                    ka, kb = s['ikr'][:Tn * idd // 2], s['ikb']
                                    ik = mv(c, wa, None, ka, M[c], 4, xf=s['mixed']); ik2 = mv(c, wb, None, kb, M[c], 4, xf=s['mixed'])
                                ra = E[c].kv_append(ka, kb, raw, pos[c], q8=True, dep=ik)
                                if Tn > 1: E[c].also(ra, ik2)
                            qd = E[c].qk_nr(s['qf'], 2 * hd, nh, hd, qn_[c], pos[c], s['qo'], self.nrot, self.theta_scale, EPS, dep=q_)
                            if self.q8: qd = E[c].hadamard(s['qo'], hd, dep=qd)
                            na = Tn * nh * hd
                            if qsa_t:   # SCORE after this pass's pools (IDX_APPEND), SELECT (also waits for q), ATTN_SEL (+ the KV row)
                                sc_ = E[c].also(E[c].qsa_score(iqo_, pool, pos[c], score_, dep=iqd, ntask=qsa_score_ntask(prow)), ia)
                                sl_ = E[c].also((E[c].qsa_select_par if qsa_psel() else E[c].qsa_select)(score_, pos[c], sel, prow, dep=sc_), qd)
                                a_ = E[c].also(E[c].attn_sel(s['qo'], cache, pos[c], sel, s['att'][:na], nh, nkv, hd, 1.0 / np.sqrt(hd), q8=self.q8, dep=sl_, **akw(c)), ap)
                            else:
                                a_ = E[c].also(E[c].attn_dec(s['qo'], cache, pos[c], s['att'][:na], nh, nkv, hd, 1.0 / np.sqrt(hd), q8=self.q8, dep=ap, **akw(c)), qd)
                            if self.q8: a_ = E[c].hadamard(s['att'][:na], 64, rows_per_task=4, dep=a_)
                            fqa = s['fq'][:na // 32 * 36]
                            a_ = E[c].gate_mul(s['att'][:na], s['qf'], 2 * hd, hd, s['gat'], hd, dep=a_, q8=fqa)
                            BLK.append(mv(c, ao[c], fqa, s['blkp'], a_, RPT['attn_output'][c]))
                    ai += 1
                XB = []
                for c in (0, 1):
                    with ON(c): XB.append(xchg(c, S[c]['blkp'], S[c]['blk'], 2 * il, BLK[c]))
                # ffn: routed (router replicated, experts split by intermediate rows) beside the shared expert (split the same way)
                M, inj = hc_mix(f'{p}.hc_ffn', None, post=('blk', XB, inj))
                rw = self._router_w(f'{p}.ffn_gate_inp.weight', self.n_exp); sw = self._router_w(f'{p}.ffn_gate_inp_shexp.weight', 1)
                rr = [[(32 * b0, 32 * b1)] for b0, b1 in mb]
                sgu, suu = self._rows(f'{p}.ffn_gate_shexp.weight', rr), self._rows(f'{p}.ffn_up_shexp.weight', rr)
                # the down projections are split by K at the 32-unit boundaries of mb: in the tensor's own blocks (Q2_0: 64)
                mbt = {}
                for n in ('ffn_down_shexp', 'ffn_down_exps'):
                    qk = GGML_QUANT_SIZES[self.T[f'{p}.{n}.weight'].tensor_type][0]
                    assert all(32 * v % qk == 0 for r in mb for v in r), f'{p}.{n}: {qk}-blocks vs the intermediate split {mb} (in 32s)'
                    mbt[n] = [(32 * b0 // qk, 32 * b1 // qk) for b0, b1 in mb]
                sdd = self._cols(f'{p}.ffn_down_shexp.weight', [[b] for b in mbt['ffn_down_shexp']])
                g = self._exp_rows(f'{p}.ffn_gate_exps.weight', [r[0] for r in rr]); u = self._exp_rows(f'{p}.ffn_up_exps.weight', [r[0] for r in rr])
                d = self._exp_cols(f'{p}.ffn_down_exps.weight', mbt['ffn_down_exps'])
                # the shared expert: one fused gate-up op when one format decodes both and the module has it, else two matvecs
                # and the swiglu glue (llama.cpp's unfused path)
                sh_fused = sgu[0]['fmt'] == suu[0]['fmt'] and ('GU_DENSE', sgu[0]['fmt']) in (E[0].caps or ())
                FF = []
                for c in (0, 1):
                    with ON(c):
                        s = S[c]
                        # claim order: the router right after its logits (claimed before the logits finish, not ~10 us late
                        # behind the shared expert's tasks), then the shared expert (off the critical path), then the routed
                        # experts (the shared expert's q8 / down after the routed gate / up: 16.22 vs 16.16 ms/token)
                        r_ = mv_router(c, rw[c], s['mixed'], s['logit'], RPT['router'][c], M[c], 'f32_router_logits')
                        r_ = E[c].router(s['logit'], self.k_used, s['ids'], s['wts'], dep=r_)
                        sl = mv_router(c, sw[c], s['mixed'], s['sgl'], 8, M[c], 'f32_shexp_gate')
                        if sh_fused:
                            # the shared expert as dense ops (the same weights for every token: the multi-token row dot; bitwise
                            # the MoE path with one expert of weight 1)
                            s_ = E[c].gg_gu_dense(sgu[c]['w'], suu[c]['w'], sgu[c]['fmt'], sgu[c]['rows'], D, sgu[c]['rb'], s['mq'], s['shh'], dep=M[c], up_fmt=suu[c]['fmt'])
                            s_ = E[c].gg_q8_1(s['shh'], s['shq'], dep=s_)
                        else:   # gate and up each in its own format: two matvecs, then shq = q8_1(silu(gate) * up)
                            if 'shu' not in s: s['shu'] = torch.zeros(Tn * ffc[c], device='cuda')
                            sg_ = mv(c, sgu[c], s['mq'], s['shh'], M[c]); su_ = mv(c, suu[c], s['mq'], s['shu'], M[c])
                            s_ = E[c].also(E[c].gg_q8_1(s['shh'], s['shq'], dep=sg_, mul=s['shu']), su_)
                        s_ = E[c].also(mv(c, sdd[c], s['shq'], s['shx'], s_, RPT['down_shexp'][c]), sl)
                        r_ = E[c].gg_moe_gu(g[c]['w'], u[c]['w'], g[c]['fmt'], g[c]['slab'], ffc[c], D, s['ids'], s['mq'], s['h'], rows_per_task=8, dep=r_, up_fmt=u[c]['fmt'])
                        r_ = E[c].gg_q8_1(s['h'], s['hq'], dep=r_)
                        r_ = E[c].gg_moe_down(d[c]['w'], d[c]['fmt'], d[c]['slab'], ffc[c], D, s['ids'], s['wts'], s['hq'], s['moe'], rows_per_task=8, dep=r_)
                        f_ = E[c].also(E[c].shexp_add(s['moe'], s['shx'], s['sgl'], s['ffp'], dep=r_), s_)
                        FF.append(xchg(c, s['ffp'], s['ffo'], 2 * il + 1, f_))
                post = ('ffo', FF, inj)
                if il + 1 == self.nl or (il + 1) in self.ple_layers:
                    for c in (0, 1):
                        with ON(c): dep[c] = E[c].also(E[c].hc_post(S[c]['ffo'], S[c]['inj'], res[c], res[c], dep=FF[c]), inj[c])
                    post = None
            for c in (0, 1): self.layer_seg[c].append(('head', len(E[c].segs)))
            M, _ = hc_mix('output_hc', dep, inject=False)
            nv = int(self.T['output.weight'].shape[1]); vh = int(round(nv * self.vocab_share))
            out = self._rows('output.weight', [[(0, vh)], [(vh, nv)]])
            self.vh = vh
            first = kind == kinds[0]
            if first: self.logits_half = []
            self.head_w = out   # the head's row shards per card (the MTP drafter's head reads them too: caminus.decode.mtp)
            # the token path (step_token / verify_tokens): each card's max over its own rows -> (value, index) and the status
            # words, read back into small pinned buffers (one sync per card, no logits copied between the cards)
            if first: self.d_max, self.h_max, self.h_st = [], [], []
            for c in ((0, 1) if first else ()):
                with ON(c):
                    self.d_max.append((torch.zeros(Tn, device='cuda'), torch.zeros(Tn, dtype=torch.int64, device='cuda')))
                self.h_max.append((torch.zeros(Tn, pin_memory=True), torch.zeros(Tn, dtype=torch.int64, pin_memory=True)))
                self.h_st.append(torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True))
            if first: self.ev_done = []
            for c in ((0, 1) if first else ()):
                with ON(c): self.ev_done.append(torch.cuda.Event())
            for c in (0, 1):
                with ON(c):
                    if first: self.logits_half.append(torch.zeros(Tn, out[c]['rows'], device='cuda'))
                    lg = self.logits_half[c]
                    mv(c, out[c], S[c]['mq'], lg, M[c], 64)
                    E[c].build(strict=not self.tape_only)
            # the device bound guard (k3_run's prologue) must decide the same on both cards (one card skipping its tape while
            # the other waits on its exchanges would hold the other card up to limit_us): the same bounds on both
            g = [(sorted(cap for _, cap in E[j].guard_caps.values()), E[j].guard_commit, E[j].T) for j in (0, 1)]
            assert g[0] == g[1], f'k3 TP: the cards guard different bounds {g}'
            self._tapes[kind] = list(E); self._tape_meta[kind] = (self.layer_seg, self.seg_names)
        self.engines = self._tapes[kinds[0]]; self.layer_seg, self.seg_names = self._tape_meta[kinds[0]]
        # the backfill's fp32 rows per card: [attention layers][_bf_rows][128] (capacity.tp_qsa_bytes qsa_backfill)
        self._bf_rows = min(capacity.QSA_BACKFILL, self.max_ctx); self._bf_kr, self._bf_pos = [], []
        if self.dual:
            for c in (0, 1):
                with ON(c):
                    self._bf_kr.append(torch.zeros(len(self._idx_kn), self._bf_rows, k3.QSA_D, device='cuda'))
                    self._bf_pos.append(torch.zeros(1, dtype=torch.int32, device='cuda'))
        torch.cuda.set_device(self.devs[0])

    @property
    def staged(self):
        """the position the staging buffer holds for the next launch (h_i32[16], which the device reads; _pass refuses a launch
        whose i0 differs). With seq and token one of the three counters the prompt cache and the session read"""
        return int(self.h_i32[16])

    def reset(self):
        """a new sequence: zero every recurrent state (and the KV caches' use: positions restart at 0); clears a device
        bound-guard report (self.poisoned). seq (the sequence epoch) grows: caminus.decode.longctx refuses a snapshot of an earlier
        sequence"""
        for t in self.state.values(): t.zero_()
        self.seq += 1
        self.fresh = True
        self._pool_ok = 0
        self.poisoned = None

    def prefetch(self, tokens, positions):
        """start reading the PLE rows of positions whose tokens are known already (the next verify takes them; host
        only, the same bytes): a speculative loop calls it for the last accepted token before it drafts"""
        if self.ple_layers: self.ple.prefetch(tokens, positions)

    def step(self, tokens, i):
        """position i on both cards at once (ntok == 1; commits the previous token); returns the full logits on card 0"""
        assert self.ntok == 1
        return self.verify(tokens, i, 0 if self.fresh else 1)[0]

    def step_token(self, tokens, i):
        """step(), but returns only the greedy next token (argmax of the logits, the first index on ties): each card
        takes the max of its own rows and only (value, index) comes back, no logits copied between the cards"""
        assert self.ntok == 1
        return self._run(tokens, i, 0 if self.fresh else 1, True)[0]

    def verify(self, tokens, i0, commit):
        """positions i0 .. i0 + ntok - 1 (tokens[i0 + t]: the last accepted token, then the drafts) on both cards at once;
        first commits the previous pass's first `commit` tokens into the recurrent states (lazy commit: 0 after a reset).
        Returns the logits [ntok, vocab] on card 0.
        A pass whose waits gave up (a bail: the process's queues were evicted longer than limit_us, or a hang) is run
        again with commit 0, up to max_retries times: a bailed launch still runs every task exactly once (later waits
        give up at once), so its commit replays (gdn conv / step, PLE conv: from the stored state and the saved records,
        never from this pass's activations) are done, and the kernel skips the record saves (OP_GDN_SAVE) once a wait
        has given up; everything else the pass writes is rewritten by the retry from the host inputs. The retry is
        bitwise the uninterrupted pass (tools/gates/forcebail.py)."""
        self._run(tokens, i0, commit, False)
        return torch.cat([self.logits_half[0], self.logits_half[1].to(self._dev(0))], dim=1)

    def launch_tokens(self, tokens, i0, commit):
        """verify()'s pass launched without waiting and without reading anything back (the device accept path, caminus.
        dflash_k3: the drafter's launch is queued behind it on the same streams and reads its logits shards); then
        finish_pass(i0, commit). The same staging and launches as _run (the same tapes)"""
        Tn = self.ntok
        self._check(tokens, i0, commit, 'FlashNextTP pass')
        hp = self.ple.start(tokens, range(i0, i0 + Tn)) if self.ple_layers else None
        for t in range(Tn): self.h_res[t] = self._emb_row(int(tokens[i0 + t]))
        if hp is not None:
            e = self.ple.embeddings(hp); assert e.shape == self.h_ple.shape, (e.shape, self.h_ple.shape)
            self.h_ple[...] = e
        self.h_i32[16] = i0; self._tok = False; self._during = None
        self.fresh = False
        self._pass_launch(i0, commit)

    def finish_pass(self, i0, commit):
        """wait for launch_tokens' pass; a bailed pass is run again (commit 0, as _run) before returning. Returns True when
        it was rerun (anything queued behind the first launch read a bailed pass's outputs and must run again). At most
        max_retries + 1 attempts in all (the first launch + max_retries reruns), as _run"""
        bad = self._pass_wait(i0)
        if not bad: return False
        for attempt in range(1, self.max_retries + 1):
            self.bails.append((i0, attempt - 1, bad))
            print(f'k3 TP: a wait gave up at position {i0} (attempt {attempt}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
            bad = self._pass(None, None, i0, 0)
            if not bad: return True
        self.bails.append((i0, self.max_retries, bad))
        raise RuntimeError(f'k3 TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))

    def verify_tokens(self, tokens, i0, commit):
        """verify(), but returns the greedy token after each position (a list of ntok ints: argmax per row, the first
        index on ties, as verify(...).argmax(dim=1)) instead of the logits"""
        return self._run(tokens, i0, commit, True)

    def _emb_row(self, tok):
        """token_embd row tok, dequantized to float32 (cached: the same bytes through the same dequantize)"""
        r = self._emb.get(tok)
        if r is None:
            te = self.T['token_embd.weight']
            assert 0 <= tok < int(te.shape[1]), f'token {tok} out of range'
            r = quants.dequantize(np.ascontiguousarray(np.asarray(te.data)[[tok]]), te.tensor_type).astype(np.float32).reshape(-1)
            if len(self._emb) >= 16384: self._emb.clear()
            self._emb[tok] = r
        return r

    @staticmethod
    def _guesses(tokens, i, k, window=512):
        """up to k tokens likely to follow tokens[:i + 1]: the tokens that followed earlier occurrences of tokens[i] in the
        last `window` tokens, longest match of the last 3 tokens first, then the latest (a guess only: any answer is safe)"""
        seq = [int(t) for t in tokens[max(0, i + 1 - window):i + 1]]; n = len(seq)
        if n < 2: return []
        last = seq[-1]; hits = [j for j in range(n - 2, -1, -1) if seq[j] == last]
        def mlen(j):
            m = 1
            while m < 3 and j - m >= 0 and seq[j - m] == seq[n - 1 - m]: m += 1
            return m
        out = []
        for j in sorted(hits, key=mlen, reverse=True):     # stable: the latest first among equal match lengths
            if seq[j + 1] not in out:
                out.append(seq[j + 1])
                if len(out) >= k: break
        return out

    def _check(self, tokens, i0, commit, what):
        """the host bound check of a pass, before anything is staged, read or launched (k3.ContextBoundsError): positions
        i0 .. i0 + ntok - 1 inside the KV cache [0, max_ctx) and (the dense tape, unless dense_beyond_qsa) below
        capacity.QSA_EXACT, commit in
        [0, ntok], tokens[i0 + ntok - 1] present"""
        if self.poisoned: raise k3.ContextBoundsError(f'{what}: refused until reset(): {self.poisoned}')
        k3.check_span(i0, self.ntok, self.max_ctx, commit=commit, ntok=self.ntok, what=what)
        lim = getattr(self, 'qsa_limit', capacity.QSA_EXACT)
        if lim is not None and int(i0) + self.ntok > lim:
            raise k3.ContextBoundsError(f'{what}: positions {i0} .. {int(i0) + self.ntok - 1} past {lim}: full attention equals the '
                                        f"model's QSA only up to {lim} positions (the Coder with max_ctx > {lim} runs QSA; "
                                        f'FlashNextTP(dense_beyond_qsa=True): test-only override)')
        if tokens is not None and len(tokens) < i0 + self.ntok:
            raise k3.ContextBoundsError(f'{what}: positions {i0} .. {i0 + self.ntok - 1} but {len(tokens)} tokens')

    def _run(self, tokens, i0, commit, tok):
        """stage the pass's inputs in the pinned host block, then _pass (retried on a bail); tok: return the tokens"""
        Tn = self.ntok
        self._check(tokens, i0, commit, 'FlashNextTP pass')   # the KV rows the pass writes: before the PLE reads start
        # host work off the path: the PLE rows' reads (the Coder: O_DIRECT, ~0.2 ms of device latency) are submitted
        # first and stay in flight while the embedding rows are written into the pinned block; a speculative loop may
        # have started some already (prefetch)
        hp = self.ple.start(tokens, range(i0, i0 + Tn)) if self.ple_layers else None
        for t in range(Tn): self.h_res[t] = self._emb_row(int(tokens[i0 + t]))   # broadcast to the HC streams
        if hp is not None:
            e = self.ple.embeddings(hp); assert e.shape == self.h_ple.shape, (e.shape, self.h_ple.shape)
            self.h_ple[...] = e
        self.h_i32[16] = i0; self._tok = tok
        self._during = (tokens, i0) if tok and Tn == 1 and self.ple_guess > 0 and self.ple_layers else None
        self.fresh = False
        for attempt in range(self.max_retries + 1):
            bad = self._pass(None, None, i0, commit if attempt == 0 else 0)
            if not bad: break
            self.bails.append((i0, attempt, bad))
            print(f'k3 TP: a wait gave up at position {i0} (attempt {attempt + 1}), running the pass again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        else:
            raise RuntimeError(f'k3 TP: a wait gave up at position {i0} in {self.max_retries + 1} attempts: ' + '; '.join(bad))
        if not tok: return None
        (v0, x0), (v1, x1) = [(v.numpy(), x.numpy()) for v, x in self.h_max]
        # the first maximum over [card 0's rows | card 1's rows]: card 0's on a tie; a NaN counts as the maximum (argmax)
        return [int(x0[t]) if (v0[t] >= v1[t] or v0[t] != v0[t]) else self.vh + int(x1[t]) for t in range(Tn)]

    def _pass(self, xt, ple, i0, commit):
        """one launch per card from the staged host block (xt / ple: unused, the signature kept for test hooks); returns the
        bail reports (empty: both cards finished every wait). A retry restages the whole block (the bailed pass rewrote
        res). self._tok (set by _run): also read back each card's (max, argmax) of its logits rows"""
        self._pass_launch(i0, commit)
        return self._pass_wait(i0)

    def _pass_launch(self, i0, commit):
        """_pass's first half: the launches on both cards (and the logits' max / argmax read back when self._tok)"""
        tok = self._tok
        self._check(None, i0, commit, 'FlashNextTP._pass')   # (a test hook may call it directly)
        if self.h_i32[16] != i0: raise k3.ContextBoundsError(f'FlashNextTP._pass: staged position {self.h_i32[16]} != {i0}')
        if getattr(self, 'dual', False): self._pick_tape(i0)
        base = self.token * self.n_ex; self.token += 1   # a fresh epoch per launch (a retry too): exchange flags only grow
        self.h_i32[1] = commit; self.h_i32[17] = base
        ct = getattr(self, '_card_ev', None)   # (test doubles build the engine without __init__)
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if ct is not None: ct[c].record()              # (card_timing: the pass's start on this card's stream)
                self.stage[c].copy_(self.hstage, non_blocking=True)   # the pinned block is rewritten only after the syncs below
                self.engines[c].run(check=False)
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]):
                if tok:
                    torch.max(self.logits_half[c], dim=1, out=self.d_max[c])
                    self.h_max[c][0].copy_(self.d_max[c][0], non_blocking=True); self.h_max[c][1].copy_(self.d_max[c][1], non_blocking=True)
                self.h_st[c].copy_(self.engines[c].sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
                self.ev_done[c].record()      # after everything this pass put on the card's stream

    def enable_card_timing(self, on=True):
        """per-card pass spans for the server's metrics (M7 ops; off by default, so every gate and bench runs the same calls as
        before): a timing event recorded on each card's stream before the pass's stage copy, and ev_done made a timing event,
        so after _pass_wait's synchronize (the wait every pass has anyway) card_ms = each card's span from the pass's start on
        its stream to its end (the stage copy, k3_run, the read-backs: the target only; the drafter's step queued behind it
        on the device-accept path is not in it), with no added sync. A bailed attempt's span is replaced by the retry's. The
        exchange wait inside the span is not separable here (it is a spin inside the persistent kernel's OP_XCHG: the
        *_trace builds have device timestamps, the serving code objects do not)"""
        if on and self._card_ev is None:
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): self.ev_done[c] = torch.cuda.Event(enable_timing=True)
            self._card_ev = []
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): self._card_ev.append(torch.cuda.Event(enable_timing=True))
        elif not on and self._card_ev is not None:
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): self.ev_done[c] = torch.cuda.Event()
            self._card_ev = None
        self.card_ms = None

    def _pick_tape(self, i0):
        """two-tape QSA: the tape of the pass at i0 (self.engines; _pass_wait and the bail retries read the same one) and
        the pool bookkeeping. Any pass rewrites the IDXRAW rows i0 .. i0 + T - 1, so only the pools j < i0 / 4 stay valid;
        the QSA tape needs exactly those (it pools the ones its own rows complete, and a row never sees a pool past it), so
        the ones the dense tape left unpooled are backfilled first; after it the pools its rows completed are valid too
        (a later pass at a smaller i0, after rejected drafts, lowers the mark again). Snapshots need nothing: a restore
        to position n keeps the rows < n as the fill wrote them, and the next pass at n lowers the mark to n / 4"""
        q = int(i0) + self.ntok > capacity.QSA_EXACT
        self.engines = self._tapes['qsa' if q else 'dense']
        self._pool_ok = min(self._pool_ok, int(i0) // 4)
        if q:
            if self._pool_ok < int(i0) // 4: self._backfill(self._pool_ok, int(i0) // 4)
            self._pool_ok = (int(i0) + self.ntok) // 4

    def _backfill(self, j0, j1):
        """IDXPOOL pools j0 .. j1 - 1 of every attention layer on both cards from their IDXRAW rows (positions 4 j0 .. 4 j1 - 1,
        written by the dense tape): the rows dequantized as the device does ((float) q * (float) d) into fp32, then
        IDX_APPEND over them (Engine.idx_append(rows=)), up to _bf_rows positions per launch (one segment per layer, no
        dependencies), on each card's current stream behind the passes queued there; synchronous (once per crossing)"""
        p, end = 4 * int(j0), 4 * int(j1)
        k3.check_span(p, end - p, self.max_ctx, what='FlashNextTP._backfill')
        raw, pool = [self.state[(c, 'idxraw')] for c in (0, 1)], [self.state[(c, 'idxpool')] for c in (0, 1)]
        na = raw[0].shape[0]; assert na == len(self._idx_kn) and raw[0].shape[2] == k3.QSA_ROWB
        while p < end:
            n = min(self._bf_rows, end - p)
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]):
                    b = raw[c][:, p:p + n].reshape(na, n, 4, 34)
                    # (float) q * (float) d straight into the counted rows (_bf_kr), no temporaries: the old q.float(), d.float()
                    # and q * d copies left 2 + 12 + 12 MiB in torch's free cache per card at the crossing (M6 unit 1, measured:
                    # past the XT's idle + projection); the same fp32 products (exact conversions, one multiply), so the same bits
                    kv = self._bf_kr[c][:, :n].view(na, n, 4, 32)
                    kv.copy_(b[..., 2:].view(torch.int8)); kv.mul_(b[..., :2].view(torch.float16))
                    kr = self._bf_kr[c]
                    self._bf_pos[c].fill_(p)
                    e = k3.Engine(module=k3.QSA_OF[self.module], grid=k3.resident_grid(self.module), T=1, limit_us=self.limit_us)
                    for ai in range(na):
                        e.idx_append(kr[ai, :n], raw[c][ai], pool[c][ai], self._bf_pos[c], self._idx_kn[ai][c], self.theta_scale, EPS,
                                     n_rot=self.nrot, rows=n)
                    e.build().run(check=True)
            p += n

    def _pass_wait(self, i0):
        """_pass's second half: host work while the cards run, the wait, the bound and bail reports"""
        if self._during is not None:   # host work while the cards run (once per token, not on a retry)
            tk, i = self._during; self._during = None
            try: self.ple.guess(tk, i, self._guesses(tk, i, self.ple_guess))
            except (OSError, IOError) as e:   # an optional read failed: drop the guess, never leave the pass unwaited
                self.ple._ready = None; self.ple._pre = None
                print(f'k3 TP: PLE guess prefetch dropped at position {i}: {e!r}', file=sys.stderr, flush=True)
        bad, oob = [], []
        for c in (0, 1): self.ev_done[c].synchronize()
        if getattr(self, '_card_ev', None) is not None:   # card_timing: each card's span of this attempt, from events already complete (no extra sync)
            self.card_ms = tuple(self._card_ev[c].elapsed_time(self.ev_done[c]) for c in (0, 1))
        for c in (0, 1):   # a device bound-guard report first, on either card: not a bail, never retried
            b = k3.bounds_status(self.h_st[c].tolist())
            if b: oob.append(f'card {c} (device {self.devs[c]}): {b}')
        if oob:
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]): torch.cuda.synchronize(); self.engines[c].sync.zero_()
            self.poisoned = f'the device bound guard caught an out-of-range value at pass position {i0}: ' + '; '.join(oob)
            raise k3.ContextBoundsError(f'k3 TP: {self.poisoned}')
        for c in (0, 1):
            st = self.h_st[c].tolist()
            if st[1]:
                with torch.cuda.device(self.devs[c]):
                    torch.cuda.synchronize()
                    names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                    bad.append(f'card {c} (device {self.devs[c]}): segment {st[2]} = {names.get(self.engines[c].segs[st[2]].op)} (seen {st[3]}); {self.engines[c].stall_report()}')
                    self.engines[c].sync.zero_()
        return bad

    def check_replicas(self):
        """the replicated state must be the same bits on both cards: (the residual, the last router choice and, with QSA,
        every layer's last selection rows)"""
        r0, r1 = self.res[0].cpu(), self.res[1].cpu()
        i0, i1 = self.S[0]['ids'].cpu(), self.S[1]['ids'].cpu()
        sel = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(self.qsa_sel[0], self.qsa_sel[1]))
        return bool(torch.equal(r0, r1)), bool(torch.equal(i0, i1)) and sel
