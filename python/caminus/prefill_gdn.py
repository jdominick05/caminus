"""The batched prefill's GDN (gated delta net) ops (docs/agentic/spec-prefill-cache.md M5 S4). Device code:
kernels/prefill_gdn.hip (build/prefill_gdn.hsaco).

Every op is its K3 decode op's expressions in its order over a chunk of T tokens, so it is BITWISE that op on the same
inputs (gate G1, tests/test_prefill_gdn.py, against the K3 op itself run per token and in T = 4 passes):
- l2 (OP_GDN_L2): the q / k heads of the conv output, normed per head;
- gates (OP_GDN_GATES): softplus(alpha + dt) * a and sigmoid(beta);
- step (OP_GDN_STEP): the gated delta rule as a PER-TOKEN fp32 recurrence (not a chunked scan). One workgroup per
  slice of a value head's state columns keeps them in registers and walks the chunk's tokens in order with the decode
  op's lane map and update code, so the outputs and the state after the chunk are the decode's after the same T steps.
  The state [H, 128, 128] is K3's own (the first H * 128 * 128 floats of a layer's gdn block, k3.gdn_block_size): read
  once, written once after the chunk, so the first K3 pass after a prefill runs with commit 0 (lazy commit, S6). With
  rec / mprev the decode's pending commit is replayed first (OP_GDN_SAVE's records, as OP_GDN_STEP replays them), so a
  prefill may follow decode passes whose accepted tokens are not committed yet. Snapshots: the state after each listed
  chunk position is written on the fly into the caller's buffers (the prompt cache's checkpoints, S8: no chunk has to
  be split at a checkpoint; the conv state / PLE history at a position are the last inputs before it, which
  prefill_ops' pf_conv_state / pf_ple_hist can write from the chunk's rows);
- norm (OP_GDN_NORM): the ssm_norm RMS norm times sigmoid(z), fp32 out (the ssm_out MMQ's input, mmq.Workspace.x_ssm_out;
  no q8 copy: MMQ quantizes its own input).
Chunk geometry does not change a bit (G2b): a token's l2 / gates / norm depend on its own rows only, and the recurrence
runs token by token whatever the chunk boundaries, so a chunk of T1 then one of T2 (state carried) is one of T1 + T2.

Memory: every buffer is the caller's (prefill_ops.GlueWorkspace holds the chunk's gdn_qk / gdn_gates / gdn_att, counted
in capacity.prefill_need_mib); nothing here allocates. Snapshot buffers are the prompt cache's (S8's tier), not counted
here. Each launch is refused (ValueError) unless its buffers are on the device, contiguous fp32, aligned as the kernel's
float4 accesses need, and cover what it touches WITHIN THEIR OWN VIEW, and unless no buffer it writes overlaps another
of its buffers (caminus.bufcheck). The step's snapshot positions are checked on the host (increasing, in [1, T], at most
MAX_SNAP) and its mprev against the records given."""
import ctypes
import torch
from .hip import Module
from .bufcheck import disjoint, view_bytes

S = 128                  # the state's head size (op_gdn_step's specialisation)
CPW = 4                  # state columns per wave
MAX_SNAP = 8             # snapshot positions per launch (kernels/prefill_gdn.hip GDN_MAX_SNAP)
# the step's kernels: name -> (function, waves of 4 state columns per workgroup). Every one gives the same bits (G2a).
# g8 / g8b8 / g8w4: 8 lanes per state column, the inputs staged through LDS in blocks of 16 / 8 / 16 tokens, 8 / 8 / 4
# waves; w8b16: OP_GDN_STEP's own 32-lane map, LDS blocks of 16; direct: the first version (32 lanes, no LDS, one token of
# register prefetch). kernels/prefill_gdn.hip says why g8 is the fastest
STEP_KERNELS = {'g8': ('pf_gdn_step_g8', 8), 'g8b8': ('pf_gdn_step_g8b8', 8), 'g8w4': ('pf_gdn_step_g8w4', 4),
                'w8b16': ('pf_gdn_step_w8b16', 8), 'direct': ('pf_gdn_step_direct', 2)}
STEP_KERNEL = 'g8'       # the default: fastest on both cards' shapes (tests/test_prefill_gdn.py speed)


class GdnSnaps(ctypes.Structure):
    _fields_ = [('n', ctypes.c_int), ('pos', ctypes.c_int * MAX_SNAP)]


def rec_floats(H, nk):
    """floats in one of OP_GDN_SAVE's saved-input records (k [nk, S], v [H, S], gates [2H])"""
    return nk * S + H * S + 2 * H


def _b(t, n):
    return (t, int(n) * t.element_size()) if t is not None else (None, 0)


class Gdn:
    """the prefill GDN kernels on one device (every launch with that device current, on its current stream)"""

    def __init__(self, device):
        self.dev = torch.device(device)
        with torch.cuda.device(self.dev): self.m = Module('prefill_gdn')

    def _t(self, t, what, n, align=4):
        """t: a contiguous fp32 tensor on this device with >= n elements of its own (its view), its pointer align-aligned"""
        if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
            raise ValueError(f'prefill_gdn: {what} must be a contiguous tensor on {self.dev}')
        if t.dtype != torch.float32: raise ValueError(f'prefill_gdn: {what} must be float32 (is {t.dtype})')
        if t.numel() < int(n): raise ValueError(f'prefill_gdn: {what} holds {t.numel()} < {n} elements (its view; {view_bytes(t)} B)')
        if t.data_ptr() % align: raise ValueError(f'prefill_gdn: {what} must be {align}-byte aligned')
        return t

    def l2(self, x, y, T, heads, eps, ldx=None, K=S):
        """y[t, h] = OP_GDN_L2 of head h of token t (x + t * ldx + h * K, K values): the q / k heads of the conv output
        (heads = 2 nk, ldx = the conv row). y [T * heads, K]"""
        T, heads, K = int(T), int(heads), int(K); ldx = heads * K if ldx is None else int(ldx)
        if T < 1 or heads < 1 or not 1 <= K <= 256 or ldx < heads * K: raise ValueError(f'gdn l2: T {T} heads {heads} K {K} ldx {ldx}')
        nx = (T - 1) * ldx + heads * K
        self._t(x, 'x', nx); self._t(y, 'y', T * heads * K)
        disjoint('gdn l2', dict(x=_b(x, nx), y=_b(y, T * heads * K)), ('y',))
        with torch.cuda.device(self.dev):
            self.m.pf_gdn_l2((T * heads,), (256,), x, y, ctypes.c_int(heads), ctypes.c_int(K), ctypes.c_int(ldx), ctypes.c_float(eps / K))

    def gates(self, alpha, beta, dt, a, out, T, H, lda=None, ldb=None):
        """out [T, 2H] = (softplus(alpha + dt) * a, sigmoid(beta)) per token: alpha / beta [T, H] (row strides lda / ldb),
        dt / a [H] (the layer's ssm_dt bias and ssm_a of this card's value heads)"""
        T, H = int(T), int(H); lda = H if lda is None else int(lda); ldb = H if ldb is None else int(ldb)
        if T < 1 or H < 1 or lda < H or ldb < H: raise ValueError(f'gdn gates: T {T} H {H} lda {lda} ldb {ldb}')
        na, nb = (T - 1) * lda + H, (T - 1) * ldb + H
        self._t(alpha, 'alpha', na); self._t(beta, 'beta', nb); self._t(dt, 'dt', H); self._t(a, 'a', H); self._t(out, 'out', 2 * T * H)
        disjoint('gdn gates', dict(alpha=_b(alpha, na), beta=_b(beta, nb), dt=_b(dt, H), a=_b(a, H), out=_b(out, 2 * T * H)), ('out',))
        with torch.cuda.device(self.dev):
            self.m.pf_gdn_gates(((T * H + 255) // 256,), (256,), alpha, ctypes.c_int(lda), beta, ctypes.c_int(ldb), dt, a, out,
                                ctypes.c_int(H), ctypes.c_int(T))

    def step(self, qk, v, gates, state, out, T, H, nk, ldv, rec=None, mprev=0, snap=None, snap_pos=(), snap_stride=None, kernel=None):
        """the gated delta rule over the chunk's T tokens (OP_GDN_STEP, bitwise): qk [T, 2, nk, 128] (l2), v: value head h
        of token t at v + t * ldv + h * 128, gates [T, 2H], state [H, 128, 128] (updated in place: the state after the
        chunk), out [T, H, 128]. rec / mprev: replay mprev of OP_GDN_SAVE's records first (rec [>= mprev, rec_floats]).
        snap_pos: chunk positions p (1 <= p <= T, increasing, <= MAX_SNAP of them): the state after p tokens goes to
        snap + i * snap_stride (default stride H * 128 * 128 floats). kernel: one of STEP_KERNELS (default STEP_KERNEL)"""
        T, H, nk, ldv, mprev = int(T), int(H), int(nk), int(ldv), int(mprev)
        kernel = STEP_KERNEL if kernel is None else kernel
        if kernel not in STEP_KERNELS: raise ValueError(f'gdn step: kernel {kernel!r} (one of {sorted(STEP_KERNELS)})')
        fn, waves = STEP_KERNELS[kernel]
        if T < 1 or H < 1 or nk < 1 or H % nk: raise ValueError(f'gdn step: T {T}, H {H}, nk {nk} (H a multiple of nk)')
        if ldv < H * S or ldv % 4: raise ValueError(f'gdn step: ldv {ldv} (>= H * {S}, a multiple of 4)')
        nst = H * S * S; nq, nv, ng, no = T * 2 * nk * S, (T - 1) * ldv + H * S, T * 2 * H, T * H * S
        self._t(qk, 'qk', nq, 16); self._t(v, 'v', nv, 16); self._t(gates, 'gates', ng); self._t(state, 'state', nst, 16)
        self._t(out, 'out', no, 16)
        RS = rec_floats(H, nk); nr = 0
        if mprev < 0: raise ValueError(f'gdn step: mprev {mprev}')
        if mprev:
            if rec is None: raise ValueError('gdn step: mprev > 0 needs the saved-input records (rec)')
            if RS % 4: raise ValueError(f'gdn step: a record of {RS} floats is not float4-aligned (H {H} must be even)')
            nr = mprev * RS; self._t(rec, 'rec', nr, 16)
        pos = [int(p) for p in snap_pos]
        if len(pos) > MAX_SNAP: raise ValueError(f'gdn step: {len(pos)} snapshot positions (at most {MAX_SNAP})')
        if any(not 1 <= p <= T for p in pos) or any(b <= a for a, b in zip(pos, pos[1:])):
            raise ValueError(f'gdn step: snapshot positions {pos} must increase within [1, T = {T}]')
        ss = nst if snap_stride is None else int(snap_stride); nsn = 0
        if pos:
            if snap is None: raise ValueError('gdn step: snapshot positions without a snapshot buffer')
            if ss < nst or ss % 4: raise ValueError(f'gdn step: snap_stride {ss} (>= {nst}, a multiple of 4)')
            nsn = (len(pos) - 1) * ss + nst; self._t(snap, 'snap', nsn, 16)
        disjoint('gdn step', dict(qk=_b(qk, nq), v=_b(v, nv), gates=_b(gates, ng), state=_b(state, nst), out=_b(out, no),
                                  rec=_b(rec, nr) if mprev else (None, 0), snap=_b(snap, nsn) if pos else (None, 0)),
                 ('state', 'out', 'snap'))
        sp = GdnSnaps(); sp.n = len(pos)
        for i, p in enumerate(pos): sp.pos[i] = p
        null = ctypes.c_void_p(0)
        with torch.cuda.device(self.dev):
            getattr(self.m, fn)((H * (S // (CPW * waves)),), (32 * waves,), qk, v, ctypes.c_int(ldv), gates, state,
                                                     rec if mprev else null, ctypes.c_int(mprev), out, snap if pos else null,
                                                     ctypes.c_long(ss), sp, ctypes.c_int(H), ctypes.c_int(nk), ctypes.c_int(T))

    def norm(self, o, w, z, y, T, H, eps, K=S):
        """y[t, h] = ((o * rsqrt(mean(o^2) + eps)) * w) * sigmoid(z) per (token, value head) (OP_GDN_NORM, no q8 copy):
        o / z / y [T, H, K], w [K]"""
        T, H, K = int(T), int(H), int(K); n = T * H * K
        if T < 1 or H < 1 or not 1 <= K <= 256: raise ValueError(f'gdn norm: T {T} H {H} K {K}')
        self._t(o, 'o', n); self._t(w, 'w', K); self._t(z, 'z', n); self._t(y, 'y', n)
        disjoint('gdn norm', dict(o=_b(o, n), w=_b(w, K), z=_b(z, n), y=_b(y, n)), ('y',))
        with torch.cuda.device(self.dev):
            self.m.pf_gdn_norm((T * H,), (256,), o, w, z, y, ctypes.c_int(K), ctypes.c_float(eps))
