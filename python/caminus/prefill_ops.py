"""The batched prefill's BF16 GEMMs and glue (docs/agentic/spec-prefill-cache.md M5 S3): the per-token K3 decode ops of a
layer, run over a whole chunk of T tokens with ordinary launches. Device code: kernels/prefill_glue.hip
(build/prefill_glue.hsaco) and kernels/gemm.hip's bf16 WMMA (build/gemm.hsaco, caminus.gemm.gemm_tn).

Which reference each op equals (gate G1, tests/test_prefill_glue.py):
- BITWISE K3's decode op on the same inputs (its expressions in its order, so a prefilled token's op output is the decode
  op's): rms_mul (OP_RMS_MUL), hc_post_rms (OP_HC_POST_RMS), hc_post (OP_HC_POST), hc_mean (OP_HC_MEAN), router
  (OP_ROUTER: top-k by rank on the probabilities' fp32 bit patterns, equal probabilities -> the lower expert index first),
  gdn_conv + its state (OP_GDN_CONV over T tokens), ple_gate (OP_PLE_GATE), ple_conv + its history (OP_PLE_CONV, optionally
  fused with OP_ADD_SUM), add_sum (OP_ADD_SUM), and bf16_mv (OP_BF16_MV's split-K order: the thin BF16 matrices, router
  logits, shared-expert gate, ssm alpha / beta, hc inject). The norms need not store fp32 xn (on the prefill's path they
  do not: one 84 MB write stream per hc mix at T = 2048, 0.39 ms of hc_post_rms's 0.94 on the XT, and 80 MiB per card):
  they write the row scale (and the bf16 copy a WMMA reads), and hc_mean / bf16_mv / ple_gate / ple_conv given (scale, w)
  recompute xn = (x * scale) * w, the norm's own expression on the same stored bits: still bitwise the decode op chain.
- MEASURED TOLERANCE: bf16_gemm, the wide BF16 matrices (hc down / up, output_hc down / up, ple key / value, the indexer
  projections) on gemm.hip's WMMA: the activation rounded to bf16 (round to nearest even), the bf16 weights exact, fp32
  accumulation in a fixed K order (32-wide K stages in order, two 16-deep WMMAs each), fp32 output. The decode's
  OP_BF16_MV keeps the activation in fp32 (llama.cpp's mul_mat_vec_f), so the order cannot match; llama.cpp's own batched op
  for these (ggml_cuda_mul_mat_cublas, hipBLAS with BF16 compute on RDNA3 above 8 columns) rounds the activation AND the
  output to bf16. Gate: the WMMA result's error against fp64 is at most llama.cpp's batched op's on the same inputs.
  An output element depends only on its token's row and its weight row (K order fixed, no split K), so the result is
  the same whatever T, the chunk's split points or the row's place in a tile (G2b).
Prefill state therefore cannot equal decode state end to end: S2's quantized GEMMs are llama.cpp's batched MMQ, whose
float order differs from the decode's MMVQ-order matvecs. The bitwise ops make each op's own contribution zero; G4
(D-281 end to end) judges the whole.

Router logits (lead decision 11-03, docs/AGENTIC-PLAN.md wf_af0b9af2-d20): router_logits() runs on the WMMA by default
(ROUTER_LOGITS = 'wmma': the mixed row's bf16 copy times the BF16 router matrix, the tolerance tier above, 17-24 ms per
2048-token chunk faster), which is what llama.cpp's own prefill does (its batched mul_mat of the BF16 router weights is
hipBLAS BF16); 'exact' is OP_BF16_MV's order through bf16_mv (bitwise the decode's logits). One switch: Glue(dev,
router_logits='exact'), or mode= per call, so the end-to-end gate (S6, G4) can A/B them.

Routing safety: the router's ids feed caminus.mmq.group, whose verbatim ggml mm_ids_helper corrupts its tables on ids out
of range or repeated within a token (and MMQ then writes out of bounds). pf_router flags NaN / inf logits and checks its
own ids (in range, distinct; never seen wrong), group() clamps every id on the device before the helper, and both set
bits of a sticky int32 flag word (ROUTE_*). The host must call check_routing on that word before it uses anything the
chunk produced (S6: before the hand-off commits state); it raises RoutingError and the flag stays set until
clear_routing. Between the launch and the check nothing can go out of bounds (the clamped ids are valid by construction).

Memory: every buffer is the caller's (GlueWorkspace preallocates a card's for the largest chunk, counted by
capacity.prefill_need_mib); nothing here allocates. Each launch is refused (ValueError) unless its buffers are on the
device, contiguous, of the right dtype and cover what the kernel reads and writes WITHIN THEIR OWN VIEW (a short view of
a workspace block is refused, not written past), and unless no buffer it writes overlaps another of its buffers (exact
in-place uses only where the kernel allows them: rms_mul y = x, ple_gate gated = kn, add_sum y = a or b; caminus.bufcheck)."""
import ctypes
import torch
from .hip import Module
from . import gemm as _gemm
from .bufcheck import disjoint, view_bytes

# the routing flag word's bits (pf_router, mmq's k3_mmq_ids_clamp): sticky until clear_routing
ROUTE_NONFINITE = 1    # a router logit was NaN or +-inf
ROUTE_RANGE = 2        # an expert id out of [0, n_exp) (replaced by the clamp)
ROUTE_DUP = 4          # an expert id repeated within a token (replaced by the clamp)
ROUTE_TILES = 8        # caminus.moe: a grouping / tile table entry out of its range (skipped on the device; never expected)
ROUTER_LOGITS = 'wmma'   # the prefill's router logits: 'wmma' (default, lead 11-03) or 'exact' (bf16_mv, bitwise the decode)
WMMA_OUT = 'bf16'        # the WMMA GEMMs' output: 'bf16' (default, user 11-04: each result rounded to bf16, RNE, as llama.cpp's
                         # batched BF16 mul_mat output above 8 columns on RDNA3, so the prefill reproduces llama.cpp's own batched
                         # numerics; router top-10 == llama.cpp's on 100 % of tokens) or 'fp32' (the accumulation as is; fix:prefill-s6)


class RoutingError(RuntimeError):
    """the device flagged a chunk's routing (NaN / inf logits, ids clamped): the chunk's results must not be used"""


def _route_words(errs):
    out = []
    for e in errs:
        if not isinstance(e, torch.Tensor) or e.dtype != torch.int32 or e.numel() < 1 or not e.is_cuda:
            raise ValueError('prefill_ops: a routing flag word is an int32 device tensor (GlueWorkspace.err, mmq.Workspace.err)')
        out.append(e.view(-1)[:1])
    return out


def routing_flags(*errs):
    """the OR of the routing flag words (waits for each word's device to finish the work queued before it)"""
    f = 0
    for e in _route_words(errs): f |= int(e.item())
    return f


def check_routing(*errs, what='prefill routing'):
    """raise RoutingError if any flag word is set (the words stay set: every later check refuses too until clear_routing)"""
    f = routing_flags(*errs)
    if f:
        why = [s for b, s in ((ROUTE_NONFINITE, 'a NaN / inf router logit'), (ROUTE_RANGE, 'an expert id out of range (clamped)'),
                              (ROUTE_DUP, 'an expert id repeated within a token (replaced)'),
                              (ROUTE_TILES, 'a MoE grouping / tile table entry out of range (skipped)')) if f & b]
        raise RoutingError(f'{what}: the device flagged {", ".join(why) or "an unknown bit"} (flags {f:#x}); refusing the chunk')
    return 0


def clear_routing(*errs):
    for e in _route_words(errs): e.zero_()

D_GLUE = 2560          # the glue kernels' row width (GLUE_NI x 256: n_embd)
TPAD = 128             # the WMMA GEMMs' token padding (gemm.hip's BM = 128 tiles)
NI_OF_K = {2560: 5, 10240: 20}
# the WMMA tile per (N, K) (measured on the XT, tests/test_prefill_glue.py speed; gemm.hip's three instances), and the
# fallback rule: the largest of 128 x 128 / 64 x 128 / 128 x 64 that tiles N
WMMA_TILE = {(320, 10240): (128, 64), (10240, 320): (64, 128), (10240, 2560): (128, 128), (2560, 2560): (128, 128),
             (512, 2560): (64, 128), (128, 2560): (64, 128), (256, 2560): (64, 128)}


def tpad(T):
    return (int(T) + TPAD - 1) // TPAD * TPAD


def _b(t, n):
    """(t, the bytes of n of its elements) for bufcheck.disjoint"""
    return (t, int(n) * t.element_size()) if t is not None else (None, 0)


class Glue:
    """the prefill glue kernels on one device (every launch with that device current, on its current stream).
    router_logits: the default mode of router_logits() ('wmma' / 'exact'; None = ROUTER_LOGITS)
    wmma_out: the default output of bf16_gemm ('fp32' / 'bf16'; None = WMMA_OUT), so also of the 'wmma' router logits"""

    def __init__(self, device, router_logits=None, wmma_out=None):
        self.dev = torch.device(device)
        self.router_mode = ROUTER_LOGITS if router_logits is None else router_logits
        if self.router_mode not in ('wmma', 'exact'): raise ValueError(f'Glue: router_logits {router_logits!r} (wmma or exact)')
        self.wmma_out = WMMA_OUT if wmma_out is None else wmma_out
        if self.wmma_out not in ('fp32', 'bf16'): raise ValueError(f'Glue: wmma_out {wmma_out!r} (fp32 or bf16)')
        with torch.cuda.device(self.dev): self.m = Module('prefill_glue')

    # -- checks ------------------------------------------------------------------------------------------------------------
    def _t(self, t, what, dtype, n):
        """t: a contiguous tensor on this device of dtype with >= n elements of its own (its view: a short view of a larger
        block is refused, never written past)"""
        if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
            raise ValueError(f'prefill_ops: {what} must be a contiguous tensor on {self.dev}')
        if t.dtype != dtype: raise ValueError(f'prefill_ops: {what} must be {dtype} (is {t.dtype})')
        if t.numel() < int(n): raise ValueError(f'prefill_ops: {what} holds {t.numel()} < {n} elements (its view; {view_bytes(t)} B)')
        return t

    def _go(self, fn, n, *args, threads=256):
        with torch.cuda.device(self.dev):
            getattr(self.m, fn)(((int(n) + threads - 1) // threads,), (threads,), *args)

    @staticmethod
    def _nul(t):
        return t if t is not None else ctypes.c_void_p(0)

    # -- the hyper-connection glue ------------------------------------------------------------------------------------------
    def rms_mul(self, x, w, y, R, eps, ybf=None, scale=None):
        """y[r] = rms_norm(x[r]) * w[r % wr] for R rows of 2560 (w [wr, 2560], wr divides R). Outputs, each optional (at
        least one): y fp32 [R, 2560] (in place, y is x, allowed), ybf its bf16 copy, scale [R] the row's rsqrt(mean(x^2) +
        eps) (a consumer recomputes y = (x * scale) * w: bitwise y)"""
        K = D_GLUE; R = int(R); wr = w.numel() // K
        if R < 1 or wr < 1 or w.numel() != wr * K or R % wr: raise ValueError(f'rms_mul: R {R}, w {w.numel()} (rows of {K})')
        if y is None and ybf is None and scale is None: raise ValueError('rms_mul: no output')
        self._t(x, 'x', torch.float32, R * K); self._t(w, 'w', torch.float32, wr * K)
        if y is not None: self._t(y, 'y', torch.float32, R * K)
        if ybf is not None: self._t(ybf, 'ybf', torch.bfloat16, R * K)
        if scale is not None: self._t(scale, 'scale', torch.float32, R)
        disjoint('rms_mul', dict(x=_b(x, R * K), w=_b(w, wr * K), y=_b(y, R * K), ybf=_b(ybf, R * K), scale=_b(scale, R)),
                 ('y', 'ybf', 'scale'), same_ok=[('y', 'x')])
        with torch.cuda.device(self.dev):
            self.m.pf_rms_mul((R,), (256,), x, w, self._nul(y), self._nul(ybf), self._nul(scale), ctypes.c_int(R), ctypes.c_int(wr), ctypes.c_float(eps))

    def hc_post_rms(self, block, inj, res, w, T, HC, eps, rs=0.25, xn=None, xnbf=None, scale=None):
        """res[t, h] += block[t] * 2 sigmoid(inj[t, h] * rs) (in place), then the next mix's norm of each (token, stream)
        row: xn fp32 [T, HC, 2560], xnbf its bf16 copy, scale [T * HC] (each optional)"""
        T, HC, D = int(T), int(HC), D_GLUE
        self._t(block, 'block', torch.float32, T * D); self._t(inj, 'inj', torch.float32, T * HC)
        self._t(res, 'res', torch.float32, T * HC * D); self._t(w, 'w', torch.float32, HC * D)
        if xn is not None: self._t(xn, 'xn', torch.float32, T * HC * D)
        if xnbf is not None: self._t(xnbf, 'xnbf', torch.bfloat16, T * HC * D)
        if scale is not None: self._t(scale, 'scale', torch.float32, T * HC)
        disjoint('hc_post_rms', dict(block=_b(block, T * D), inj=_b(inj, T * HC), res=_b(res, T * HC * D), w=_b(w, HC * D),
                                     xn=_b(xn, T * HC * D), xnbf=_b(xnbf, T * HC * D), scale=_b(scale, T * HC)), ('res', 'xn', 'xnbf', 'scale'))
        with torch.cuda.device(self.dev):
            self.m.pf_hc_post_rms((T * HC,), (256,), block, inj, res, w, self._nul(xn), self._nul(xnbf), self._nul(scale), ctypes.c_int(T),
                                  ctypes.c_int(HC), ctypes.c_float(rs), ctypes.c_float(eps))

    def hc_post(self, block, inj, res, T, HC, D=D_GLUE, rs=0.25):
        """res[t, h, i] += block[t, i] * 2 sigmoid(inj[t, h] * rs)"""
        T, HC, D = int(T), int(HC), int(D)
        self._t(block, 'block', torch.float32, T * D); self._t(inj, 'inj', torch.float32, T * HC); self._t(res, 'res', torch.float32, T * HC * D)
        disjoint('hc_post', dict(block=_b(block, T * D), inj=_b(inj, T * HC), res=_b(res, T * HC * D)), ('res',))
        self._go('pf_hc_post', T * D, block, inj, res, ctypes.c_int(T), ctypes.c_int(HC), ctypes.c_int(D), ctypes.c_float(rs))

    def hc_mean(self, xn, gate, out, T, HC, D=D_GLUE, rs=0.25, scale=None, w=None):
        """out[t, i] = rs * sum_h xn[t, h, i] * sigmoid(gate[t, h, i]); xn given [T, HC, D], or (scale, w given) the
        residual [T, HC, D] whose norm is recomputed: xn = (res * scale[t, h]) * w[h]"""
        T, HC, D = int(T), int(HC), int(D)
        self._t(xn, 'xn', torch.float32, T * HC * D); self._t(gate, 'gate', torch.float32, T * HC * D); self._t(out, 'out', torch.float32, T * D)
        if (scale is None) != (w is None): raise ValueError('hc_mean: scale and w go together')
        if scale is not None: self._t(scale, 'scale', torch.float32, T * HC); self._t(w, 'w', torch.float32, HC * D)
        disjoint('hc_mean', dict(xn=_b(xn, T * HC * D), gate=_b(gate, T * HC * D), out=_b(out, T * D), scale=_b(scale, T * HC),
                                 w=_b(w, HC * D)), ('out',))
        self._go('pf_hc_mean', T * D, xn, self._nul(scale), self._nul(w), gate, out, ctypes.c_int(T), ctypes.c_int(HC), ctypes.c_int(D), ctypes.c_float(rs))

    # -- the router -----------------------------------------------------------------------------------------------------------
    def router(self, logits, T, E, k, ids, wts, err, clamp=6.103515625e-5):
        """softmax over E logits per token, the top k (rank on the probabilities' bit patterns; ties: lower index first),
        weights normalised by their sum (slot order) clamped at `clamp`: ids [T, k] int32, wts [T, k]. err: the routing flag
        word (int32 [>= 1] on this device, sticky: ROUTE_NONFINITE for a NaN / inf logit, ROUTE_RANGE / ROUTE_DUP if the
        ids ever came out invalid, then replaced by 0 .. k - 1); the caller checks it (check_routing) before using the chunk"""
        T, E, k = int(T), int(E), int(k)
        if T < 1 or not (1 <= E <= 1024 and 1 <= k <= min(64, E)): raise ValueError(f'router: T {T}, E {E}, k {k}')
        self._t(logits, 'logits', torch.float32, T * E); self._t(ids, 'ids', torch.int32, T * k); self._t(wts, 'wts', torch.float32, T * k)
        self._t(err, 'err', torch.int32, 1)
        disjoint('router', dict(logits=_b(logits, T * E), ids=_b(ids, T * k), wts=_b(wts, T * k), err=_b(err, 1)), ('ids', 'wts', 'err'))
        with torch.cuda.device(self.dev):
            self.m.pf_router((T,), (256,), logits, ctypes.c_int(E), ctypes.c_int(k), ctypes.c_float(clamp), ids, wts, err)

    def router_logits(self, w, E, x, logits, T, xbf=None, mode=None, K=D_GLUE):
        """the router's logits for T tokens: logits[t, e] = W[e] . x[t] (W bf16 [E, K] raw bytes or bfloat16; x fp32 [T, K]:
        the hc_mean output). mode (default: this Glue's router_mode, ROUTER_LOGITS = 'wmma'):
          'wmma'   x rounded to bf16 into xbf [tpad(T), K] (to_bf16), then the WMMA GEMM: logits [tpad(T), E] (rows past T
                   are the zero padding's; a MEASURED TOLERANCE, the tier llama.cpp's own batched router op is in)
          'exact'  bf16_mv in OP_BF16_MV's order: logits [T, E], bitwise K3's decode logits (xbf unused)
        returns the mode used"""
        mode = self.router_mode if mode is None else mode
        if mode == 'exact':
            self.bf16_mv(w, E, K, x, logits, T)
        elif mode == 'wmma':
            if xbf is None: raise ValueError('router_logits: the wmma mode needs xbf (bf16 [tpad(T), K]: GlueWorkspace.mixed_bf)')
            self.to_bf16(x, xbf, T, K); self.bf16_gemm(w, E, K, xbf, logits, T)
        else:
            raise ValueError(f'router_logits: mode {mode!r} (wmma or exact)')
        return mode

    # -- GDN conv -------------------------------------------------------------------------------------------------------------
    def gdn_conv(self, x, w, state, y, C, T, ldx=None, ldy=None):
        """y[t, c] = silu(conv4([state | x][t .. t + 3], w[c])) for the chunk's T tokens, then state = the last 3 inputs.
        x / y [T, C] (row strides ldx / ldy); y must not overlap x"""
        C, T = int(C), int(T); ldx = C if ldx is None else int(ldx); ldy = C if ldy is None else int(ldy)
        if T < 1 or C < 1 or ldx < C or ldy < C: raise ValueError(f'gdn_conv: C {C} T {T} ldx {ldx} ldy {ldy}')
        self._t(x, 'x', torch.float32, (T - 1) * ldx + C); self._t(w, 'w', torch.float32, C * 4)
        self._t(state, 'state', torch.float32, C * 3); self._t(y, 'y', torch.float32, (T - 1) * ldy + C)
        # (y on x: a token reads the inputs of the 3 before it; the state on x: pf_conv_state reads x while it writes)
        disjoint('gdn_conv', dict(x=_b(x, (T - 1) * ldx + C), w=_b(w, C * 4), state=_b(state, C * 3), y=_b(y, (T - 1) * ldy + C)), ('y', 'state'))
        self._go('pf_gdn_conv', T * C, x, w, state, y, ctypes.c_int(C), ctypes.c_int(T), ctypes.c_int(ldx), ctypes.c_int(ldy))
        self._go('pf_conv_state', C, x, state, ctypes.c_int(C), ctypes.c_int(T), ctypes.c_int(ldx))

    # -- PLE ------------------------------------------------------------------------------------------------------------------
    def ple_gate(self, kn, qn, val, gated, gate, T, HC, D=D_GLUE, ldv=None, qscale=None, qw=None):
        """per token and stream: gate = sigmoid(sgn(s) sqrt(max(|s|, 1e-6))), s = kn . qn / sqrt(D); gated = val * gate.
        kn / gated [T, HC, D] (gated may be kn), val [T, D] (row stride ldv), gate [T, HC]; qn given [T, HC, D], or (qscale,
        qw given) the residual whose norm is recomputed"""
        T, HC, D = int(T), int(HC), int(D); ldv = D if ldv is None else int(ldv)
        if HC > 8: raise ValueError(f'ple_gate: HC {HC} > 8')
        if (qscale is None) != (qw is None): raise ValueError('ple_gate: qscale and qw go together')
        for t, n in (('kn', kn), ('qn', qn), ('gated', gated)): self._t(n, t, torch.float32, T * HC * D)
        if qscale is not None: self._t(qscale, 'qscale', torch.float32, T * HC); self._t(qw, 'qw', torch.float32, HC * D)
        self._t(val, 'val', torch.float32, (T - 1) * ldv + D); self._t(gate, 'gate', torch.float32, T * HC)
        # gated = kn exactly is the kernel's in-place form (a workgroup reads its token's key rows before it writes them);
        # any other overlap (gated on qn / val, a shifted kn) mixes tokens
        disjoint('ple_gate', dict(kn=_b(kn, T * HC * D), qn=_b(qn, T * HC * D), gated=_b(gated, T * HC * D), qscale=_b(qscale, T * HC),
                                  qw=_b(qw, HC * D), val=_b(val, (T - 1) * ldv + D), gate=_b(gate, T * HC)), ('gated', 'gate'),
                 same_ok=[('gated', 'kn')])
        with torch.cuda.device(self.dev):
            self.m.pf_ple_gate((T,), (256,), kn, qn, self._nul(qscale), self._nul(qw), val, ctypes.c_int(ldv), gated, gate, ctypes.c_int(HC), ctypes.c_int(D))

    def ple_conv(self, x, w_f16, hist, C, T, dil=3, y=None, a=None, res=None, xscale=None, xw=None, D=D_GLUE):
        """conv[t, c] = silu of the dilated 4-tap causal conv over [hist | x] (w f16 [C, 4] as raw uint8 or float16); x given
        [T, C], or (xscale [T * C / D], xw [C] given) the pre-norm rows whose norm is recomputed. y: conv out [T, C]; res / a:
        res += a + conv (OP_ADD_SUM fused); then hist = the last 3 dil (normed) inputs. Neither y nor res may overlap x"""
        C, T, dil, D = int(C), int(T), int(dil), int(D)
        if 3 * dil > 32 or T < 1 or C % D: raise ValueError(f'ple_conv: dil {dil} (history <= 32), T {T}, C {C} (rows of {D})')
        if y is None and res is None: raise ValueError('ple_conv: neither y nor res')
        if (res is None) != (a is None): raise ValueError('ple_conv: res and a go together')
        if (xscale is None) != (xw is None): raise ValueError('ple_conv: xscale and xw go together')
        self._t(x, 'x', torch.float32, T * C); self._t(hist, 'hist', torch.float32, C * 3 * dil)
        if xscale is not None: self._t(xscale, 'xscale', torch.float32, T * (C // D)); self._t(xw, 'xw', torch.float32, C)
        if w_f16.dtype == torch.uint8: self._t(w_f16, 'w', torch.uint8, C * 8)
        else: self._t(w_f16, 'w', torch.float16, C * 4)
        for nm, t in (('y', y), ('res', res)):
            if t is not None: self._t(t, nm, torch.float32, T * C)
        if a is not None: self._t(a, 'a', torch.float32, T * C)
        # (y / res on x: a token reads the inputs 3, 6, 9 before it; hist on anything: pf_ple_hist reads x while it writes)
        disjoint('ple_conv', dict(x=_b(x, T * C), xscale=_b(xscale, T * (C // D)), xw=_b(xw, C), w=(w_f16, C * 8), hist=_b(hist, C * 3 * dil),
                                  y=_b(y, T * C), a=_b(a, T * C), res=_b(res, T * C)), ('hist', 'y', 'res'))
        n = self._nul
        self._go('pf_ple_conv', T * C, x, n(xscale), n(xw), w_f16, hist, n(y), n(a), n(res), ctypes.c_int(C), ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(dil))
        self._go('pf_ple_hist', C, x, n(xscale), n(xw), hist, ctypes.c_int(C), ctypes.c_int(T), ctypes.c_int(D), ctypes.c_int(dil))

    def add_sum(self, a, b, y, n):
        """y += a + b"""
        n = int(n)
        for nm, t in (('a', a), ('b', b), ('y', y)): self._t(t, nm, torch.float32, n)
        # element-wise: y = a or y = b exactly is safe (each thread reads its element before it writes it); a shifted
        # overlap reads elements other threads have already written
        disjoint('add_sum', dict(a=_b(a, n), b=_b(b, n), y=_b(y, n)), ('y',), same_ok=[('y', 'a'), ('y', 'b')])
        self._go('pf_add_sum', n, a, b, y, ctypes.c_long(n))

    # -- BF16 matrices ----------------------------------------------------------------------------------------------------------
    def bf16_mv(self, w, N, K, x, out, T, ldo=None, rpb=None, xscale=None, xw=None, xd=D_GLUE):
        """out[t, r] = W[r] . x[t] in OP_BF16_MV's order (bitwise the decode op per token): W bf16 [N, K] (raw bytes or
        bfloat16), x fp32 [T, K], out [T, N] (row stride ldo). K 2560 or 10240. x given, or (xscale [T * K / xd], xw [K]
        given) the pre-norm rows whose norm is recomputed: x = (x * xscale[t, k / xd]) * xw[k] (the hc inject on the
        residual). rpb: rows per workgroup (default: halved from N while the grid has < 2048 workgroups, down to 8)"""
        N, K, T = int(N), int(K), int(T); ldo = N if ldo is None else int(ldo)
        if K not in NI_OF_K: raise ValueError(f'bf16_mv: K {K} (2560 or 10240: OP_BF16_MV\'s split-K shapes)')
        if T < 1 or N < 1 or ldo < N: raise ValueError(f'bf16_mv: N {N}, T {T}, ldo {ldo}')
        if w.dtype == torch.uint8: self._t(w, 'w', torch.uint8, 2 * N * K)
        else: self._t(w, 'w', torch.bfloat16, N * K)
        self._t(x, 'x', torch.float32, T * K); self._t(out, 'out', torch.float32, (T - 1) * ldo + N)
        if (xscale is None) != (xw is None): raise ValueError('bf16_mv: xscale and xw go together')
        if xscale is not None:
            if K % int(xd) or int(xd) % 2: raise ValueError(f'bf16_mv: xd {xd} must divide K {K} (and be even)')
            self._t(xscale, 'xscale', torch.float32, T * (K // int(xd))); self._t(xw, 'xw', torch.float32, K)
        disjoint('bf16_mv', dict(w=(w, 2 * N * K), x=_b(x, T * K), xscale=_b(xscale, T * (K // int(xd))), xw=_b(xw, K),
                                 out=_b(out, (T - 1) * ldo + N)), ('out',))
        if w.data_ptr() % 4 or x.data_ptr() % 8: raise ValueError('bf16_mv: w 4-byte, x 8-byte aligned')
        ni = NI_OF_K[K]; tt = 8 if ni == 5 else 2
        ntile = (T + tt - 1) // tt
        if rpb is None:   # (measured on the XT, router logits N 256 at T 2048: rpb 256 0.768 ms, 64 0.627, 32 0.613, 16 0.613, 4 0.673)
            rpb = N
            while ntile * ((N + rpb - 1) // rpb) < 2048 and rpb > 8: rpb = (rpb + 1) // 2
        rpb = max(1, int(rpb))
        with torch.cuda.device(self.dev):
            getattr(self.m, f'pf_bf16_mv_n{ni}')((ntile, (N + rpb - 1) // rpb), (256,), w, x, self._nul(xscale), self._nul(xw), ctypes.c_int(int(xd)),
                                                 out, ctypes.c_int(N), ctypes.c_int(T), ctypes.c_int(ldo), ctypes.c_int(rpb))

    def silu_bf16(self, x, y, T, K, rs, ldx=None):
        """y [tpad(T), K] bf16 = bf16(silu(fma(rs, x, 0))) for the T rows of x (row stride ldx), zero rows to tpad(T)"""
        T, K = int(T), int(K); ldx = K if ldx is None else int(ldx); Tp = tpad(T)
        self._t(x, 'x', torch.float32, (T - 1) * ldx + K); self._t(y, 'y', torch.bfloat16, Tp * K)
        disjoint('silu_bf16', dict(x=_b(x, (T - 1) * ldx + K), y=_b(y, Tp * K)), ('y',))
        self._go('pf_silu_bf16', Tp * K, x, ctypes.c_int(ldx), y, ctypes.c_int(T), ctypes.c_int(Tp), ctypes.c_int(K), ctypes.c_float(rs))

    def to_bf16(self, x, y, T, K, ldx=None):
        """y [tpad(T), K] bf16 = bf16(x) for the T rows of x, zero rows to tpad(T)"""
        T, K = int(T), int(K); ldx = K if ldx is None else int(ldx); Tp = tpad(T)
        self._t(x, 'x', torch.float32, (T - 1) * ldx + K); self._t(y, 'y', torch.bfloat16, Tp * K)
        disjoint('to_bf16', dict(x=_b(x, (T - 1) * ldx + K), y=_b(y, Tp * K)), ('y',))
        self._go('pf_to_bf16', Tp * K, x, ctypes.c_int(ldx), y, ctypes.c_int(T), ctypes.c_int(Tp), ctypes.c_int(K))

    def bf16_gemm(self, w, N, K, xbf, out, T, tile=None, wmma_out=None):
        """out [tpad(T), N] fp32 = xbf [tpad(T), K] (bf16) . W[N, K]^T on gemm.hip's WMMA (fixed K order, fp32 out); the
        rows past T are the zero padding's (discarded). W: raw bytes or bfloat16. wmma_out (default: this Glue's): 'fp32'
        the accumulation as is, 'bf16' each result rounded to bf16 (RNE) and stored as fp32 (gemm.hip's *_rbf kernels)"""
        wmma_out = self.wmma_out if wmma_out is None else wmma_out
        if wmma_out not in ('fp32', 'bf16'): raise ValueError(f'bf16_gemm: wmma_out {wmma_out!r} (fp32 or bf16)')
        N, K, T = int(N), int(K), int(T); Tp = tpad(T)
        tile = tile or WMMA_TILE.get((N, K)) or next(t for t in ((128, 128), (64, 128), (128, 64)) if _gemm.supported(Tp, N, K, t))
        if not _gemm.supported(Tp, N, K, tile): raise ValueError(f'bf16_gemm: {Tp} x {N} x {K} does not tile by {tile}')
        wb = (w.view(torch.bfloat16) if w.dtype == torch.uint8 else w)
        self._t(wb, 'w', torch.bfloat16, N * K); self._t(xbf, 'xbf', torch.bfloat16, Tp * K); self._t(out, 'out', torch.float32, Tp * N)
        disjoint('bf16_gemm', dict(w=_b(wb, N * K), xbf=_b(xbf, Tp * K), out=_b(out, Tp * N)), ('out',))
        with torch.cuda.device(self.dev):
            _gemm.gemm_tn(xbf.view(-1)[:Tp * K].view(Tp, K), wb.view(-1)[:N * K].view(N, K), out=out.view(-1)[:Tp * N].view(Tp, N), tile=tile,
                          round_bf16=(wmma_out == 'bf16'))


# -- the glue workspace -------------------------------------------------------------------------------------------------
ALIGN = 256


def glue_dims(geom, card, gdn_keys=10):
    """the per-card shapes the glue workspace needs (geom: capacity.geometry)"""
    kc = gdn_keys if card == 0 else geom['ssm_groups'] - gdn_keys
    per_v = geom['ssm_heads'] // geom['ssm_groups']
    return dict(D=geom['D'], HC=geom['hc'], lr=geom['hc_lr'], n_exp=geom['n_exp'], k=geom['k_used'], ab=per_v * kc,
                conv=(2 + per_v) * geom['ssm_state'] * kc, gk=kc, gs=geom['ssm_state'])


def glue_layout(T, d):
    """name -> bytes of one card's glue workspace for chunks of up to T tokens (d: glue_dims). The hc norms are not stored
    in fp32 (their consumers recompute them from res and scale: Glue.hc_mean / bf16_mv / ple_gate / ple_conv with scale).
    Aliases (the chunk loop's order, S6): gate [tpad, HC*D] fp32 = the hc up output, the PLE key / normed key / gated
    value, the GDN conv output (C <= HC*D); xn_bf [tpad, HC*D] bf16 = the PLE embedding's bf16 copy ([tpad, D]); mixed =
    the PLE value ([tpad, D]); scale2 = the PLE's second norm scale. res is the chunk's residual stream [T, HC, D].
    The router: mixed_bf [tpad, D] bf16 (the hc_mean output's bf16 copy, the WMMA router logits' input), logits
    [tpad, n_exp] (the WMMA writes the padding rows too), ids / wts [T, k], err the routing flag word (ROUTE_*, sticky;
    zeroed here, check_routing before the chunk is used). The GDN half (caminus.prefill_gdn, M5 S4): gdn_qk [T, 2 gk, gs]
    the l2-normed q / k heads, gdn_gates [T, 2 ab] (gate, beta_sig), gdn_att [T, ab, gs] the recurrence's output (its
    norm writes mmq.Workspace.x_ssm_out; the conv output is `gate` above, its input mmq.Workspace.o_qkv)"""
    T = int(T); Tp = tpad(T); HD = d['HC'] * d['D']
    assert d['conv'] <= HD, d
    L = dict(res=4 * T * HD, xn_bf=2 * Tp * HD, gate=4 * Tp * HD, mixed=4 * Tp * d['D'], scale=4 * T * d['HC'], scale2=4 * T * d['HC'],
             lo=4 * Tp * d['lr'], lo_bf=2 * Tp * d['lr'], inj=4 * T * d['HC'], pg=4 * T * d['HC'],
             mixed_bf=2 * Tp * d['D'], logits=4 * Tp * d['n_exp'], ids=4 * T * d['k'], wts=4 * T * d['k'], err=4, sgl=4 * T,
             alpha=4 * T * d['ab'], beta=4 * T * d['ab'],
             gdn_qk=4 * T * 2 * d['gk'] * d['gs'], gdn_gates=4 * T * 2 * d['ab'], gdn_att=4 * T * d['ab'] * d['gs'])
    return {n: (b + ALIGN - 1) // ALIGN * ALIGN for n, b in L.items()}


def glue_bytes(T, d):
    return sum(glue_layout(T, d).values())


class GlueWorkspace:
    """one card's preallocated glue buffers for chunks of up to T tokens (one allocation, carved by glue_layout; checked
    against the card's cap first). Attributes: fp32 / bf16 / int32 views by name"""
    def __init__(self, device, T, dims, check=True):
        from . import capacity
        self.dev, self.T, self.dims = torch.device(device), int(T), dict(dims)
        self.layout = glue_layout(self.T, self.dims); self.nbytes = sum(self.layout.values())
        if check: capacity.check([self.dev.index], [self.nbytes / capacity.MIB], f'prefill glue workspace (T={self.T})')
        self.block = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.dev)   # (zeroed: the bf16 padding rows start at 0)
        off = 0
        for n, b in self.layout.items():
            v = self.block[off:off + b]
            v = v.view(torch.bfloat16) if n.endswith('_bf') else v.view(torch.int32) if n in ('ids', 'err') else v.view(torch.float32)
            setattr(self, n, v); off += b
        self.err = self.err[:1]                                                         # (its 256-byte slot: one flag word)

    def check_routing(self, what='prefill routing'):
        """raise RoutingError if this card's routing flag word is set (prefill_ops.check_routing)"""
        return check_routing(self.err, what=what)
