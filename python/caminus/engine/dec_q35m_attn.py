"""The qwen35moe decode attention as its own kernel (kernels/dec_q35m_attn.hip): one launch per attention layer, outside
the K3 tape, with attn_dec's inputs and output (k3.Engine.attn_dec on the split pair: q fp32 [16][256], the F16 or
Q8_0 cache of 2 kv heads x 256, the int32 position word on the device, out fp32 [16][256]; the Q8 tape's rotations stay
outside). Not served yet: nothing builds it into a tape.

  A = DecQ35mAttn(q8=False)            # its code object, the workspace (counters + split records) and the error word
  A(q, cache, pos, out, scale)         # queued on the current torch stream (or stream=); no wait, no read-back
  A.error()                            # (a sync) True when a launch saw pos + 1 outside [0, the cache's rows]

The splits per launch (ns) default to splits(rows): rocKE's rule on this kernel's grid (enough workgroups for the card,
at least MIN_KEYS_PER_SPLIT keys each); the device reads the position itself, so the same ns serves every depth (a split
past the last cell writes no record and only counts in)."""
import ctypes
import torch
from ..device import capacity
from ..device.bufcheck import LaunchCheck
from ..device.hip import Module
from . import k3

D, G, NKV = 256, 8, 2           # head dim, query heads per kv head, kv heads (kernels/dec_q35m_attn.hip DQ_*)
H = G * NKV
NW = 4                          # waves per workgroup (DQ_NW)
RUN = 16 * NW                   # keys per run: a split takes every ns-th run
GROUPS = 4                      # workgroups per split: (kv head, dv half)
RS, CNT = 132, 64               # floats per split record, counter words before the records (DQ_RS, DQ_CNT)
NS_MAX = 64                     # splits the default workspace holds
WG_PER_WGP = 3                  # workgroups a WGP holds at once (34 KB of static LDS each)
MIN_KEYS_PER_SPLIT = 512        # (rocKE: 4 runs of 32 keys per wave of 4)

_mods = {}
_check = LaunchCheck('dec_q35m_attn', align=16)


def workspace_floats(ns_max=NS_MAX):
    """fp32 words of a launch's workspace: CNT counters, then the records [GROUPS][ns][G][RS]"""
    return CNT + GROUPS * int(ns_max) * G * RS


def splits(cells, wgps=None):
    """splits for lists of up to `cells` keys: enough workgroups to fill the card (WG_PER_WGP per WGP over the GROUPS
    workgroups of a split), at least MIN_KEYS_PER_SPLIT keys each, at most NS_MAX"""
    if wgps is None: wgps = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return max(1, min(NS_MAX, wgps * WG_PER_WGP // GROUPS, int(cells) // MIN_KEYS_PER_SPLIT))


class DecQ35mAttn:
    """one layer's launcher on the current device: q8 picks the kernel (dec_q35m_attn_q8 / _f16); the workspace is this
    object's (layers that run one after another on one stream may share it: each launch re-arms its counters)"""

    def __init__(self, q8, ns_max=NS_MAX):
        dev = torch.cuda.current_device()
        if dev not in _mods: _mods[dev] = Module('dec_q35m_attn')
        self.q8, self.ns_max = bool(q8), int(ns_max)
        self.fn = getattr(_mods[dev], 'dec_q35m_attn_q8' if self.q8 else 'dec_q35m_attn_f16')
        n = workspace_floats(self.ns_max)
        capacity.check([dev], [(n + 1) * 4 / 2 ** 20], 'dec_q35m_attn workspace')
        self.ws = torch.zeros(n, dtype=torch.float32, device='cuda')
        self.err = torch.zeros(1, dtype=torch.int32, device='cuda')

    def rows(self, cache):
        """positions the cache holds (k3.kv_rows: whole positions of NKV x D values each of K and V)"""
        return k3.kv_rows(cache, NKV * D, self.q8)

    def __call__(self, q, cache, pos, out, scale, ns=None, stream=None):
        rows = self.rows(cache)
        ns = splits(rows) if ns is None else int(ns)
        if not 1 <= ns <= self.ns_max: raise ValueError(f'dec_q35m_attn: {ns} splits outside [1, {self.ns_max}] (the workspace)')
        _check(self.ws.device, 'attention', dict(q=(q, torch.float32, H * D), out=(out, torch.float32, H * D),
                                                 pos=(pos, torch.int32, 1, 4), ws=(self.ws, torch.float32, workspace_floats(ns)),
                                                 err=(self.err, torch.int32, 1, 4)), ('out', 'ws', 'err'))
        if not isinstance(cache, torch.Tensor) or not cache.is_cuda or not cache.is_contiguous() or cache.data_ptr() % 16:
            raise ValueError('dec_q35m_attn: the cache must be a contiguous, 16-byte-aligned device tensor')
        self.fn((ns * GROUPS,), (NW * 32,), q, cache, pos, ctypes.c_int(rows), ctypes.c_float(float(scale)), ctypes.c_int(ns),
                self.ws, out, self.err, stream=stream)

    def error(self):
        """True when a launch since the last reset() saw pos + 1 outside [0, rows] (a sync)"""
        return bool(self.err.item())

    def reset(self):
        self.err.zero_()
