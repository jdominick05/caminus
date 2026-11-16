"""The qwen3moe one-card engine's Q8 KV option without a card (caminus.engine.q3moe_one.Q3MoEOne(q8=True), its batched
prefill caminus.prefill.q3m.Q3MPrefill and the registry entry), on stand-ins for the torch.cuda and HIP calls (every tensor
in host memory; nothing is launched), with the tiny synthetic file (tests/engine/tiny_q3m.py).
  - the module tables: k3_q3m_lean_kvq8 is k3_q3m_lean's Q8 build (same ops, pairs and lean dispatch check), its two-card
    form k3_q3m_lean_tp_kvq8 (tests/engine/test_q3moe_tp_kvq8_cpu.py), no verify form, outside the dense line's Q8 table; in the installed runtime set with q3m_prefill_kvq8 (served: caminus serve --kv q8); its Engine has the error word after Sync
  - the wire layout the kernels index ([K blocks | V blocks], block (part, kv head g, block b) at (part M / 32 + 4 g + b)
    x 34 bytes): caminus.ref_kv.encode_q8_kv puts each block there, and a Q8 row is encode_q8_kv(hadamard128(K), V) of the
    F16 row's values (K decoded and rotated back within the blocks' rounding)
  - the rotation (kernels/own_kv_rot128.h, its reference caminus.ref_kv.hadamard128): the float64 Sylvester Hadamard128
    over sqrt(128) is orthonormal, and on float64 vectors (H q) . (H k) = q . k to rounding; hadamard128 is within fp32
    rounding of it, its scale is the kernel header's literal; a lane-by-lane emulation of the kernel's statements (each
    wave's lanes holding dims l, l + 32, l + 64, l + 96, the shuffle stages h = 1 .. 16, then h = 32 and 64 in registers,
    the quarter select and the scale) gives hadamard128's bits; the scores of rotated decoded rows against rotated q
    match the unrotated float64 scores within the codec's error
  - the engine: k3_q3m_lean_kvq8, the cache a uint8 tensor of (max_ctx + ATS_PAD) Q8 rows, the prompt cache's state views
    max_ctx rows of bytes, every OP_OWN_ATTS segment on its layer's cache; the projection counts Q8 rows (equal to the
    engine's own check, and the F16 projection less exactly the row bytes saved); kv_format_id names the format
  - refusals: q8 not a bool; a Q8 cache on any other object or tape form; the Q8 object without q8; a float16 cache on the
    Q8 module (and a uint8 one on the F16 module); the verify engine on a Q8 engine; an IQ prefill object with a Q8 engine
  - a refused encoding: the error word set with a stalled wait as well, read back by _pass_launch's one status copy (no
    second read): _pass_wait raises InvalidKVEncoding before the timeout path (no bail recorded, the word not cleared), the
    engine stays broken across reset() and refuses every pass; step() marks it too; Engine.status_words is the F16 module's
    8 words and the Q8 module's 8 + the error word, contiguous
  - the prefill on a Q8 engine: the Q8 kernels' module beside the F16 object, qk_rope / attn with the Q8 kernels, its KV
    error word passed, its rows checked on the byte cache; the error word set: the settle raises InvalidKVEncoding and the
    engine is broken; numerics() names the cache
  - the prompt cache: the checkpoint layout's row is the Q8 row; the engine key carries the KV format (F16 keys unchanged:
    no 'kv' entry), so a Q8 park never matches an F16 engine; the format id names the rotation, and a key with the
    earlier draft's unrotated id hashes differently (its rows never restore into this engine)
  - the park round trip (caminus serve --kv q8; tests/engine/kv_park_rt.py): each layer's Q8 byte rows back from a
    checkpoint slot, from a park in the same process and after a restart (a new engine and cache on the directory), byte for
    byte, rows past the restored positions still zero; negative controls (two layers' reads swapped, one layer read a row
    late) differ in exactly those layers (the late one also past the positions). An F16
    server's park is not offered to a Q8 server and a Q8 park not to an F16 one (cold, rows untouched); the park's own
    format restores it (the positive control)
  - the registry: arch.kv_q8 ('f16' default, 'q8', anything else refused) and Q3MoE.auto_need counting Q8 rows; every
    other entry but the two-card qwen3moe one refuses opt['kv'] = 'q8' in build and auto_need before reading anything (never
    an F16 engine instead)
  - tools/gates/kv_against.py (the G-model gates' --against): llama.cpp's floor and explicit bars pass / fail as stated on
    synthetic logits; a saved file that is not whole rows is refused
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3moe_kvq8_cpu.py --scratch DIR"""
import argparse, contextlib, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def raises(fn, exc=ValueError):
    try: fn()
    except exc: return True
    except Exception as e: print('   (raised', type(e).__name__, e, ')'); return False
    return False


# ---- stand-ins: torch.cuda / HIP on host memory (no device is opened) ------------------------------------------------------
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=48, total_memory=24560 << 20)
torch.cuda.get_device_name = lambda d=None: 'stand-in'


class FakeEv:
    def record(self, *a): pass

    def synchronize(self): pass


torch.cuda.Event = lambda *a, **k: FakeEv()
torch.cuda.synchronize = lambda d=None: None

import tiny_q3m
from caminus import arch, errors, ref_kv
from caminus.cache import prompt
from caminus.device import capacity, hip, kernels
from caminus.engine import k3, q3moe_one
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_verify import Q3MoEVerify
from caminus.prefill import q3m, q3m_ops

capacity.check = lambda *a, **k: None
LOG = []


class Rec:
    def __init__(self, name): self.name = name

    def __getattr__(self, kern): return lambda grid, block, *args: LOG.append((self.name, kern, tuple(grid), args))


hip.Module = lambda name: Rec(name)

# ---- the module tables -------------------------------------------------------------------------------------------------------
Q8M = 'k3_q3m_lean_kvq8'
check(k3.Q3M_KV_Q8_BASE[Q8M] == 'k3_q3m_lean' and k3.Q3M_KV_Q8_OF['k3_q3m_lean'] == Q8M and Q8M in k3.Q3M_LEAN_MODULES
      and Q8M in k3.Q3M_MODULES and k3.MODULE_CAPS[Q8M] == k3.MODULE_CAPS['k3_q3m_lean'],
      'k3_q3m_lean_kvq8: k3_q3m_lean\'s Q8 build, in the lean modules (the same ops whitelist and (op, format) pairs)')
check(k3.Q3M_TP_OF.get(Q8M) == 'k3_q3m_lean_tp_kvq8' and Q8M not in k3.Q3M_TP_OF.values() and Q8M not in k3.Q3M_VER_MODULES
      and Q8M not in k3.Q35_KV_Q8_BASE and Q8M in kernels.RUNTIME_SET and 'q3m_prefill_kvq8' in kernels.RUNTIME_SET,
      'its two-card form is k3_q3m_lean_tp_kvq8 (tests/engine/test_q3moe_tp_kvq8_cpu.py); no verify form, not in the dense Q8 '
      'table; it and q3m_prefill_kvq8 in the installed runtime set (served: caminus serve --kv q8)')
E = k3.Engine(module=Q8M, grid=8, T=1)
check(E.sync.numel() == k3.SYNC_WORDS + 1 and E.kv_error is not None and E.kv_error.data_ptr() == E.sync.data_ptr() + 4 * k3.SYNC_WORDS
      and k3.Engine(module='k3_q3m_lean', grid=8).kv_error is None,
      'its Engine: one int32 error word after Sync (the F16 module\'s Engine has none)')
check(raises(lambda: k3.Engine(module=Q8M, grid=8, T=2)), 'its Engine refuses T = 2 (decode only)')

# ---- the wire layout the kernels index ----------------------------------------------------------------------------------------
nkv, hd = 4, 128
lay = ref_kv.Q8KVLayout(nkv, hd)
check(lay.row_bytes == 1088 == k3.kv_pos_bytes(nkv * hd, True), 'Qwen3-30B-A3B\'s Q8 row: 1,088 bytes (2,048 as F16)')
rng = np.random.default_rng(3)
kf = (rng.standard_normal((1, nkv, hd)) * 0.7).astype(np.float16).astype(np.float32)
vf = (rng.standard_normal((1, nkv, hd)) * 0.7).astype(np.float16).astype(np.float32)
row = ref_kv.encode_q8_kv(kf, vf, lay)[0]
good = True
for part, x in ((0, kf[0]), (1, vf[0])):
    for g in range(nkv):
        for b in range(hd // 32):
            o = (part * (nkv * hd // 32) + g * 4 + b) * 34   # (kernels/k3_own_atts.h, q3m_prefill.hip: the block's offset)
            one = ref_kv.encode_q8_kv(x[g, 32 * b:32 * b + 32].reshape(1, 1, 32), np.zeros((1, 1, 32), np.float32), ref_kv.Q8KVLayout(1, 32))[0][:34]
            good &= bytes(row[o:o + 34]) == bytes(one)
            good &= o % 4 == 2 * (b & 1)                     # (the dword parity the kernels' loads assume)
check(good, 'every block of a row sits at (part M / 32 + 4 g + b) x 34 bytes, 2 (b & 1) past a dword (the kernels\' indexing)')
kr = ref_kv.hadamard128(kf)
rrow = ref_kv.encode_q8_kv(kr, vf, lay)[0]
dk, dv = ref_kv.decode_q8_kv(rrow.reshape(1, -1), lay)
amax = np.abs(np.concatenate([kr, vf]).reshape(-1, 32)).max(1)
err = np.abs(np.concatenate([dk, dv]).reshape(-1, 32) - np.concatenate([kr, vf]).reshape(-1, 32)).max(1)
back = ref_kv.hadamard128(dk)   # (H is its own inverse: Sylvester, symmetric, orthonormal)
check(bool((err <= amax / 127 * (0.5 + 127 * 2 ** -11) + 1e-12).all()) and float(np.abs(back - kf).max()) < 0.02 * float(np.abs(kf).max())
      and rrow[(M_ := nkv * hd) // 32 * 34:].tobytes() == row[M_ // 32 * 34:].tobytes(),
      f'a Q8 row is encode_q8_kv(hadamard128(K), V) of the F16 row\'s values: decoded within half a step (+ the half scale\'s rounding) of '
      f'each block (max {float((err / (amax / 127)).max()):.3f} steps); K rotated back within {float(np.abs(back - kf).max()):.4f}; V blocks unrotated')

# ---- the rotation: kernels/own_kv_rot128.h against caminus.ref_kv.hadamard128 and the float64 transform ----------------------
H = np.array([[1.0]])
for _ in range(7): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(128)
rq = np.random.default_rng(8)
qa, ka = rq.standard_normal((64, 128)) * 3, rq.standard_normal((64, 128)) * 2
s0, s1 = (qa * ka).sum(1), ((qa @ H.T) * (ka @ H.T)).sum(1)
check(float(np.abs(H @ H.T - np.eye(128)).max()) < 1e-14 and np.allclose(H, H.T) and float(np.abs(s1 - s0).max()) < 1e-12 * float(np.abs(s0).max() + 1),
      f'H128 / sqrt(128) (float64 Sylvester) is orthonormal (|H H^T - I| {float(np.abs(H @ H.T - np.eye(128)).max()):.1e}) and symmetric; on float64 '
      f'vectors (H q) . (H k) = q . k (max diff {float(np.abs(s1 - s0).max()):.1e} for scores up to {float(np.abs(s0).max()):.0f})')
hdr = open(os.path.join(HERE, '..', '..', 'kernels', 'own_kv_rot128.h')).read()
lit = [ln.split()[2] for ln in hdr.splitlines() if ln.startswith('#define OWN_KV_ROT128_SCALE')]
xs = np.concatenate([rq.standard_normal((40, 128)), rq.standard_normal((8, 128)) * 1e3, np.eye(128)[:8] * 65504,
                     rq.standard_normal((8, 128)) * np.where(np.arange(128) % 17 == 0, 60.0, 0.1)]).astype(np.float32)
hx = ref_kv.hadamard128(xs)
d64 = np.abs(hx - xs.astype(np.float64) @ H.T).max(1) / np.abs(xs).max(1)
check(len(lit) == 1 and np.float32(float.fromhex(lit[0].rstrip('f'))) == ref_kv.HADAMARD128_SCALE == np.float32(1 / np.sqrt(128))
      and hx.dtype == np.float32 and float(d64.max()) < 2e-6 and raises(lambda: ref_kv.hadamard128(xs.astype(np.float64)))
      and raises(lambda: ref_kv.hadamard128(np.zeros((2, 64), np.float32))),
      f'ref_kv.hadamard128: within fp32 rounding of the float64 transform (worst {float(d64.max()):.1e} x max |x|, outliers, 65504 spikes); '
      f'its scale is the kernel header\'s literal {lit[0] if lit else None} = float32(1 / sqrt(128)); float64 or 64-value input refused')


def emu_kernel(x):
    """kernels/own_kv_rot128.h lane by lane in float32: wave qw (0 .. 3) of a head, lane l holds u_m = x[l + 32 m]; the
    shuffle stages (partner = lane ^ h: own + partner, or partner - own when lane has bit h), then h = 32 and 64 in
    registers, the select of quarter qw, the scale; dim 32 qw + l gets that wave's lane l result"""
    f = np.float32
    out = np.empty(128, f)
    for qw in range(4):
        u = [np.array([f(x[l + 32 * m]) for l in range(32)], f) for m in range(4)]
        h = 1
        while h < 32:
            lanes = np.arange(32); hi = (lanes & h) != 0
            u = [np.where(hi, um[lanes ^ h] - um, um + um[lanes ^ h]).astype(f) for um in u]
            h *= 2
        a0, a1, a2, a3 = u[0] + u[1], u[0] - u[1], u[2] + u[3], u[2] - u[3]
        sel = (a0 + a2, a1 + a3, a0 - a2, a1 - a3)[qw]
        out[32 * qw:32 * qw + 32] = (sel * ref_kv.HADAMARD128_SCALE).astype(f)
    return out


emu = np.stack([emu_kernel(x) for x in xs])
check(np.array_equal(emu.view(np.uint32), hx.view(np.uint32)) and float((np.abs(emu - xs.astype(np.float64) @ H.T).max(1) / np.abs(xs).max(1)).max()) < 2e-6,
      f'the kernel\'s butterflies emulated lane by lane ({len(xs)} heads) give ref_kv.hadamard128 bit for bit, and match numpy\'s float64 H128 '
      f'within fp32 rounding')
kh16 = (rq.standard_normal((32, 4, 128)) * 0.8 * np.where(np.isin(np.arange(128), (3, 77)), 20.0, 1.0)).astype(np.float16).astype(np.float32)
qh = rq.standard_normal((32, 128))   # (K with two outlier dims, as real K heads have)
dkr, _ = ref_kv.decode_q8_kv(ref_kv.encode_q8_kv(ref_kv.hadamard128(kh16), kh16, lay), lay)
dku, _ = ref_kv.decode_q8_kv(ref_kv.encode_q8_kv(kh16, kh16, lay), lay)
sr, su, sx = (dkr[:, 0].astype(np.float64) @ (H @ qh.T)), (dku[:, 0].astype(np.float64) @ qh.T), (kh16[:, 0].astype(np.float64) @ qh.T)
er, eu = float(np.abs(sr - sx).mean()), float(np.abs(su - sx).mean())
check(er < 0.6 * eu and float(np.abs(sr - sx).max()) < 0.01 * float(np.abs(sx).max()),
      f'scores of rotated decoded K against rotated q: the exact scores within the codec\'s error, and on K with two outlier dims (x 20) '
      f'a mean error of {er:.3f} against {eu:.3f} unrotated (scores up to {float(np.abs(sx).max()):.0f})')
check(raises(lambda: ref_kv.encode_q8_kv(np.full((1, nkv, hd), np.float32(np.float16(np.inf))), vf, lay)),
      'an fp16-overflowed (inf) K value is refused by the reference (the kernels refuse the block and set the error word)')

# ---- the engine on the tiny file ------------------------------------------------------------------------------------------------
P = os.path.join(A.scratch, 'tiny-q3m.gguf'); info = tiny_q3m.write(P)
MC = 64
m = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True)
f = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False)
M = m.nkv * m.hd; rb = k3.kv_pos_bytes(M, True)
check(m.module == Q8M and m.E.module == Q8M and f.module == 'k3_q3m_lean' and m.kv_format_id == q3moe_one.KV_FORMAT_Q8
      and not hasattr(f, 'kv_format_id'),
      f'Q3MoEOne(q8=True) decodes on {Q8M} with kv_format_id {m.kv_format_id!r}; the F16 engine keeps k3_q3m_lean and has none')
ats = [sg for sg in m.E.segs if sg.op == k3.OP_OWN_ATTS]
check(all(d['cache'].dtype == torch.uint8 and d['cache'].numel() == (MC + k3.Engine.ATS_PAD) * rb for d in m.L)
      and all(m.state[(0, il, 'kv')].numel() == MC * rb and m.state[(0, il, 'kv')].dtype == torch.uint8 for il in range(m.nl))
      and len(ats) == m.nl and [sg.W for sg in ats] == [d['cache'].data_ptr() for d in m.L]
      and [sg.op for sg in m.E.segs] == [sg.op for sg in f.E.segs] and m.E.refuse is None,
      f'its caches: uint8, (max_ctx + {k3.Engine.ATS_PAD}) rows of {rb} B; the state views max_ctx rows of bytes; each OP_OWN_ATTS '
      f'on its layer\'s cache; the tape the F16 one\'s ops in order')
pq, pf_ = Q3MoEOne.projection_mib(P, MC, q8=True), Q3MoEOne.projection_mib(P, MC)
saved = m.nl * (MC + k3.Engine.ATS_PAD) * (4 * M - rb) / 2 ** 20
check(abs(m.vram_mib - Q3MoEOne.projection_mib(P, MC, splits=m.splits, q8=True)) < 1e-9 and abs(pf_ - pq - saved) < 1e-9
      and abs(Q3MoEOne.kv_pos_mib(P, True) - m.nl * rb / 2 ** 20) < 1e-12 and abs(Q3MoEOne.kv_pos_mib(P) - m.nl * 4 * M / 2 ** 20) < 1e-12,
      f'projection: the engine\'s own check = projection_mib(q8=True); F16 - Q8 = the row bytes saved ({saved:.4f} MiB here); '
      f'kv_pos_mib counts Q8 rows')

# ---- refusals ---------------------------------------------------------------------------------------------------------------------
check(raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=1)), 'q8=1 (not a bool) is refused')
check(raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True, module='k3_q3m_lean')),
      'a Q8 cache on the F16 module is refused')
check(raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True, attn_split=False)),
      'a Q8 cache on another tape form is refused (the lean form only)')
check(raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, module=Q8M)), 'the Q8 object without q8 is refused')
E2 = k3.Engine(module=Q8M, grid=8)
B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt)
args = lambda cache: (B(m.nh * 128 + 2 * M), B(k3.Engine.atts_scratch_floats(m.nkv, 8)), B(258), cache, B(1, torch.int32),
                      B(m.nh * 128, torch.float16), m.nh, m.nkv, 128, 0.1)
check(raises(lambda: E2.own_atts(*args(B((MC + 64) * 2 * M, torch.float16)), splits=8))
      and raises(lambda: k3.Engine(module='k3_q3m_lean', grid=8).own_atts(*args(B((MC + 64) * rb, torch.uint8)), splits=8))
      and not raises(lambda: E2.own_atts(*args(B((MC + 64) * rb, torch.uint8)), splits=8)),
      'own_atts: a float16 cache on the Q8 module and a uint8 cache on the F16 module are refused; a Q8 cache on the Q8 module builds')
check(raises(lambda: Q3MoEVerify(m, 2, vram_check=False)), 'the verify engine refuses a Q8 engine (its module reads F16 rows)')
check(raises(lambda: q3m.Q3MPrefill(m, chunk=64, check=False, module=q3m_ops.IQ_MODULE)), 'an IQ prefill object with a Q8 engine is refused')

# ---- a refused encoding --------------------------------------------------------------------------------------------------------
check(E.status_words().numel() == k3.STATUS_WORDS + 1 and E.status_words()[k3.STATUS_WORDS:].data_ptr() == E.kv_error.data_ptr()
      and k3.Engine(module='k3_q3m_lean', grid=8).status_words().numel() == k3.STATUS_WORDS,
      'Engine.status_words: the 8 status words, + the error word right after them on the Q8 module (one copy brings both)')
mm = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True)
check(mm.h_st.numel() == k3.STATUS_WORDS + 1 and f.h_st.numel() == k3.STATUS_WORDS,
      'the pinned status block: 9 words on the Q8 engine (the error word), 8 on the F16 one (unchanged)')
mm.E.run = lambda check=True, trace=None: None            # (no launch: the copies below run on the host tensors)
mm._stage([1] * 8, 5); mm._tok = False
mm.E.kv_error.fill_(2); mm.E.sync[2 * k3.MAXSEG + 1] = 1; mm.E.sync[2 * k3.MAXSEG + 2] = 0   # (and a wait that gave up)
mm._pass_launch(5, 0)
ITEM = torch.Tensor.item; reads = []
torch.Tensor.item = lambda self: reads.append(1) or ITEM(self)
r = raises(lambda: mm._pass_wait(5), errors.InvalidKVEncoding)
torch.Tensor.item = ITEM
check(r and mm.broken and not mm.bails and int(mm.E.kv_error[0]) == 2 and int(mm.h_st[k3.STATUS_WORDS]) == 2 and not reads
      and issubclass(errors.InvalidKVEncoding, errors.NumericalFailure),
      'a refused block with a stalled wait: InvalidKVEncoding (a NumericalFailure) from the launch\'s one status copy, before the '
      'timeout path; no .item() read back, no bail, the word kept')
mm.reset()
check(mm.broken and raises(lambda: mm.step_token([1] * 8, 0), errors.InvalidKVEncoding)
      and raises(lambda: mm.step(1), errors.InvalidKVEncoding) and raises(lambda: mm.verify([1] * 8, 0, 0), errors.InvalidKVEncoding),
      'reset() keeps the engine broken; step, step_token and verify are refused')
ms = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True)
def run_fail(check=True, trace=None): ms.E.kv_error.fill_(2); ms.E.check_kv_error()
ms.E.run = run_fail
check(raises(lambda: ms.step(3), errors.InvalidKVEncoding) and ms.broken and ms.pos_host == 0,
      'step(): the run\'s refused encoding marks the engine broken (the position does not advance)')

# ---- the prefill on a Q8 engine -------------------------------------------------------------------------------------------------
mp = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True)
pf = q3m.Q3MPrefill(mp, chunk=64, check=False)
check(pf.q8 and pf.ops.q8 and pf.ops.module == 'q3m_prefill' and pf.ops.kvmod.name == 'q3m_prefill_kvq8'
      and pf.numerics().get('kv') == 'q8_0' and 'kv' not in q3m.Q3MPrefill(f, chunk=64, check=False).numerics(),
      'Q3MPrefill on a Q8 engine: q3m_prefill for the GEMMs and glue, q3m_prefill_kvq8 for the rows and attention; numerics name the cache')
LOG.clear()
toks = list(range(1, 41))
pf._chunk(toks, 0, 40)
rope = [e for e in LOG if e[1].startswith('q3p_qk_rope')]; att = [e for e in LOG if e[1].startswith('q3p_attn')]
check(rope and all(e[0] == 'q3m_prefill_kvq8' and e[1] == 'q3p_qk_rope_q8' and e[3][-1] is pf.kverr for e in rope)
      and att and all(e[0] == 'q3m_prefill_kvq8' and e[1] == 'q3p_attn_q8' for e in att)
      and len(rope) == mp.nl and not any(e[1] in ('q3p_qk_rope', 'q3p_attn') for e in LOG),
      f'a chunk: {len(rope)} q3p_qk_rope_q8 (with the KV error word) and {len(att)} q3p_attn_q8 launches; no F16 row kernel')
check(raises(lambda: pf.ops.qk_rope(pf.qkv, 512, mp.nh, mp.nkv, mp.L[0]['qk_par'], 0, pf.q16, mp.L[0]['cache'], 4))
      and raises(lambda: pf.ops.qk_rope(pf.qkv, 512, mp.nh, mp.nkv, mp.L[0]['qk_par'], MC - 2, pf.q16, mp.L[0]['cache'], 4, rows=MC, err=pf.kverr), k3.ContextBoundsError)
      and raises(lambda: pf.ops.attn(pf.q16, f.L[0]['cache'], 0, 4, mp.nkv, 0.1, pf.att16)),
      'the Q8 ops refuse a missing error word, positions past the byte cache\'s rows, and an F16 cache')
pf.kverr.fill_(2)
check(raises(pf._run_settle, errors.InvalidKVEncoding) and mp.broken and raises(lambda: pf._chunk(toks, 0, 40), errors.InvalidKVEncoding),
      'the prefill\'s error word set: its settle raises InvalidKVEncoding, the engine is broken, the next chunk is refused')

# ---- the prompt cache --------------------------------------------------------------------------------------------------------------
from types import SimpleNamespace as NS
Lq, Lf = prompt.Layout(m), prompt.Layout(f)
check(Lq.pb == [rb] and Lf.pb == [4 * M] and Lq.kv_rows == MC, f'checkpoint layout: a Q8 row of {rb} B per layer (F16 {4 * M} B)')
pfk = NS(numerics=lambda: dict(q='x'), crossover=8)
kq, kf_ = prompt.engine_key(m, pfk, build='fixture'), prompt.engine_key(f, pfk, build='fixture')
check(kq['tp'].get('kv') == q3moe_one.KV_FORMAT_Q8 and 'kv' not in kf_['tp'] and kq['module'] == Q8M and kf_['module'] == 'k3_q3m_lean'
      and prompt.key_hash(kq) != prompt.key_hash(kf_),
      'engine key: the Q8 engine\'s carries its KV format and module (an F16 key has no kv entry: unchanged), so neither restores into the other')
OLD_ID = 'q8_0-v1:unrotated:f16-rounded-v1'   # (the earlier draft's rows: K unrotated)
m.kv_format_id = OLD_ID; ko = prompt.engine_key(m, pfk, build='fixture'); m.kv_format_id = q3moe_one.KV_FORMAT_Q8
check(q3moe_one.KV_FORMAT_Q8 != OLD_ID and 'hadamard128' in q3moe_one.KV_FORMAT_Q8 and ko['tp'].get('kv') == OLD_ID
      and prompt.key_hash(ko) != prompt.key_hash(kq),
      f'the format id names the rotation ({q3moe_one.KV_FORMAT_Q8!r}); a key with the earlier unrotated id hashes differently: its rows never restore here')
from caminus.cmd import replay
check(replay.kv_of_key(kq) == 'q8' and replay.kv_of_key(kf_) == 'f16' and replay.kv_of_key(None) == 'f16'
      and arch.ARCHS['qwen3moe'].kv_check(dict(kv=replay.kv_of_key(kq))) is True
      and arch.ARCHS['qwen3moe'].kv_check(dict(kv=replay.kv_of_key(kf_))) is False,
      'caminus replay: a bundle whose engine key is the Q8 engine\'s builds the entry with kv q8, an F16 key (no kv entry) with f16')

# ---- the park round trip on the Q8 engine (caminus serve --kv q8: its byte rows through a checkpoint slot, a park and an unpark
# after a restart), negative controls, and the format mismatch both ways (tests/engine/kv_park_rt.py) ---------------------------
import kv_park_rt as RT
RT_CTX = 256
mk1 = lambda q8: (lambda: Q3MoEOne(P, max_ctx=RT_CTX, vram_check=False, link_keepalive=False, q8=q8))


def rt_swap(w, eng):            # (layer 0's and layer 1's segments read into each other's rows)
    if 'kv0.c0' in w and 'kv1.c0' in w: w['kv0.c0'], w['kv1.c0'] = w['kv1.c0'], w['kv0.c0']
    return w


def rt_late(w, eng):            # (layer 1 read one row late)
    if 'kv1.c0' in w:
        pb = prompt.Layout(eng).pb[0]; n = w['kv1.c0'][0].numel()
        w['kv1.c0'] = [prompt._u8(eng.state[(0, 1, 'kv')])[pb:pb + n]]
    return w


check(prompt.Layout(mk1(True)()).pb == [rb], f'round trip engine: Q3MoEOne(q8=True) at max_ctx {RT_CTX}, a {rb}-byte Q8 row per layer')
RT.round_trip(mk1(True), check, os.path.join(A.scratch, 'cache-q8'), info['V'], 'Q8 one card',
              [('layers 0 and 1 swapped', rt_swap, [(0, 0), (0, 1)], []), ('layer 1 one row late', rt_late, [(0, 1)], [(0, 1)])])
RT.mismatch(mk1(False), mk1(True), check, os.path.join(A.scratch, 'cache-f16-then-q8'), info['V'], 'an F16 park, a Q8 server (one card)')
RT.mismatch(mk1(True), mk1(False), check, os.path.join(A.scratch, 'cache-q8-then-f16'), info['V'], 'a Q8 park, an F16 server (one card)')

# ---- the registry ------------------------------------------------------------------------------------------------------------------
check(arch.kv_q8({}) is False and arch.kv_q8({'kv': 'f16'}) is False and arch.kv_q8({'kv': 'q8'}) is True
      and raises(lambda: arch.kv_q8({'kv': 'q4'})), "arch.kv_q8: 'f16' by default, 'q8', anything else refused")
q = arch.ARCHS['qwen3moe']
nq, _, sq, needq = q.auto_need({'model': P, 'kv': 'q8'}, 4)
nf, _, sf, needf = q.auto_need({'model': P}, 4)
check(abs(sq - (Q3MoEOne.kv_pos_mib(P, True) * 1.05 + 1.0)) < 1e-12 and needq(MC)[0] < needf(MC)[0]
      and abs((needf(MC)[0] - needq(MC)[0]) - (saved + 6 * (sf - sq))) < 1e-9 and nq == nf,
      'Q3MoE.auto_need with kv q8: Q8 checkpoint slots and the Q8 projection (less than F16 by the rows and slots saved)')

bad = []
for name, a in arch.ARCHS.items():
    if name in ('qwen3moe', 'qwen3moe-tp'): continue   # (the two-card entry builds Q8 too: tests/engine/test_q3moe_tp_kvq8_cpu.py)
    for what, fn in (('build', lambda: a.build({'kv': 'q8'})), ('auto_need', lambda: a.auto_need({'kv': 'q8'}, 4))):
        try: fn(); bad.append((name, what, 'no error'))
        except ValueError as e:
            if "kv 'q8'" not in str(e): bad.append((name, what, str(e)[:80]))
        except Exception as e: bad.append((name, what, type(e).__name__ + ' ' + str(e)[:80]))
check(not bad and all(a.kv_check({}) is False and a.kv_check({'kv': 'f16'}) is False for a in arch.ARCHS.values())
      and arch.ARCHS['qwen3moe'].kv_check({'kv': 'q8'}) is True and arch.ARCHS['qwen3moe-tp'].kv_check({'kv': 'q8'}) is True,
      f"every other registry entry ({len(arch.ARCHS) - 2}) refuses opt['kv'] = 'q8' in build and auto_need before reading anything; "
      f"'f16' and no kv pass everywhere" + (f' (wrong: {bad})' if bad else ''))

# ---- the gates' --against (tools/gates/kv_against.py) ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'gates'))
import kv_against, q3moe_gmodel
rs = np.random.default_rng(5); V = 64; n = 50
base = rs.standard_normal((n, V)).astype(np.float32) * 3
near = base + rs.standard_normal((n, V)).astype(np.float32) * 0.01      # (a small change: tiny KLD, argmax kept)
far = base + rs.standard_normal((n, V)).astype(np.float32) * 1.0        # (a large one)
okf, lf = kv_against.gate(q3moe_gmodel.score, near, base, llama_f16=base, llama_q8=base + rs.standard_normal((n, V)).astype(np.float32) * 0.05)
okb, lb = kv_against.gate(q3moe_gmodel.score, far, base, llama_f16=base, llama_q8=near)
oke, _ = kv_against.gate(q3moe_gmodel.score, near, base, max_kld=1e-2, min_top1=98.0)
okx, _ = kv_against.gate(q3moe_gmodel.score, far, base, max_kld=1e-2, min_top1=98.0)
bf, gf = os.path.join(A.scratch, 'short.bin'), os.path.join(A.scratch, 'rows.bin')
base.reshape(-1)[:-1].tofile(bf); base.tofile(gf)                       # (one float short of n rows; n whole rows)
check(okf and not okb and oke and not okx and any(l.startswith('FAIL') for l in lb) and raises(lambda: kv_against.load_rows(bf, n, 'x'), SystemExit)
      and np.array_equal(kv_against.load_rows(gf, n, 'x'), base),
      '--against: a small change passes llama.cpp\'s floor and the explicit bars, a large one fails both; rows that are not whole rows are refused')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
