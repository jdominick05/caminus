# This module and kernels/gdn.hip implement flash-linear-attention's (fla) chunk_gated_delta_rule_fwd_h and _bwd_dhu
# (fla/ops/common/chunk_delta_h.py, https://github.com/fla-org/flash-linear-attention) for HIP, following its algorithm and
# operand roundings; fla is distributed under the following license (THIRD_PARTY.md):
#
# MIT License
#
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Gated DeltaNet chunk kernels: drop-ins for fla's state recurrences (kernels/gdn_*.hip).
Supported: scalar gate g (fp32, chunk-local cumulative log2), K == 128, V % 16 == 0, chunk 64, H == HV, bf16 inputs,
equal-length sequences, state layout [K, V] or [V, K] (state_v_first, which fla's GatedDeltaNet layer uses). Anything else goes to fla's own implementation."""
import ctypes
import torch
from .hip import Module

_mod = {}


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('gdn')
    return _mod[dev]


def _null(): return ctypes.c_void_p(None)


def bwd_dhu_supported(q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None, state_v_first=False,
                      cu_seqlens=None, chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False):
    B, T, H, K = q.shape
    return (g is not None and gk is None and cu_seqlens is None and chunk_size == 64
            and K == 128 and do.shape[2] == H and do.shape[-1] % 16 == 0 and g.dtype == torch.float32
            and all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in (q, k, w, do, dv)) and g.is_contiguous()
            and not use_graph)


def bwd_dhu(q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None, state_v_first=False, cu_seqlens=None,
            chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False):
    """same signature and outputs as fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_bwd_dhu: (dh, dh0, dv2)"""
    B, T, H, K = q.shape; V = do.shape[-1]; NT = (T + chunk_size - 1) // chunk_size
    dh = q.new_empty(B, NT, H, V, K) if state_v_first else q.new_empty(B, NT, H, K, V); dv2 = torch.empty_like(dv)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dht_c = dht.float().contiguous() if dht is not None else None
    _m().gdn_bwd_dhu((B * H * (V // 16),), (128,), q, k, w, g, do, dv, dht_c if dht_c is not None else _null(),
                     dh, dv2, dh0 if dh0 is not None else _null(), ctypes.c_int(T), ctypes.c_int(H), ctypes.c_int(V),
                     ctypes.c_float(scale), ctypes.c_int(int(state_v_first)))
    return dh, dh0, dv2


def fwd_h_supported(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64,
                    save_new_value=True, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None,
                    chunk_offsets=None, use_graph=False, **_):
    B, T, H, K = k.shape
    return (g is not None and gk is None and cu_seqlens is None and chunk_size == 64
            and K == 128 and u.shape[2] == H and u.shape[-1] % 16 == 0 and g.dtype == torch.float32
            and all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in (k, w, u)) and g.is_contiguous()
            and not use_graph and not _)


def fwd_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True,
          state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_offsets=None,
          use_graph=False):
    """same signature and outputs as fla.ops.common.chunk_delta_h.chunk_gated_delta_rule_fwd_h: (h, v_new, final_state)"""
    B, T, H, K = k.shape; V = u.shape[-1]; NT = (T + chunk_size - 1) // chunk_size
    h = k.new_empty(B, NT, H, V, K) if state_v_first else k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u) if save_new_value else None
    final = (k.new_zeros(B, H, V, K, dtype=torch.float32) if state_v_first else k.new_zeros(B, H, K, V, dtype=torch.float32)) if output_final_state else None
    h0 = initial_state.float().contiguous() if initial_state is not None else None
    _m().gdn_fwd_h((B * H * (V // 16),), (128,), k, w, u, g, h0 if h0 is not None else _null(), h,
                   v_new if v_new is not None else _null(), final if final is not None else _null(),
                   ctypes.c_int(T), ctypes.c_int(H), ctypes.c_int(V), ctypes.c_int(int(state_v_first)))
    return h, v_new, final


def install(forward=False):
    """route fla's gated-delta-rule backward state recurrence (bwd_dhu) through ours where supported. forward=True
    also routes fwd_h, which is correct but slower than fla's on gfx1100 today (0.2-0.7x; fla's forward kernel is
    efficient here, only its backward was not), so it stays off until a WMMA version beats it."""
    import fla.ops.gated_delta_rule.chunk as chunk
    pairs = [('chunk_gated_delta_rule_bwd_dhu', bwd_dhu, bwd_dhu_supported)]
    if forward: pairs.append(('chunk_gated_delta_rule_fwd_h', fwd_h, fwd_h_supported))
    for name, ours, ok in pairs:
        orig = getattr(chunk, name)
        if getattr(orig, '_rstack', False): continue
        def dispatch(*a, _ours=ours, _ok=ok, _orig=orig, **kw):
            return _ours(*a, **kw) if _ok(*a, **kw) else _orig(*a, **kw)
        dispatch._rstack = True; dispatch._orig = orig
        setattr(chunk, name, dispatch)
