"""The prompt cache's checkpoint layout counts KV rows the way the dense qwen35 engines allocate them (CPU only, nothing
launched). Dense35Own and Dense35TP with spec_T > 1 keep spec_T - 1 tail rows past max_ctx for verify passes; they declare
kv_rows, and caminus.cache.prompt.Layout divides each KV view by it.
  - max_ctx 8192, spec_T 4, on the tiny synthetic models (tests/engine/tiny_q35.py): both engines declare 8195 rows, the
    layout accepts them, and its row stride is each card's own KV row (2 * kv heads * head dim f16 values)
  - the strict whole-rows check refuses the same views without kv_rows (a tail that does not fill whole rows of max_ctx)
    and a view with a wrong stride (a tensor that is not a whole number of kv_rows rows); a bad kv_rows is refused
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/cache/test_layout_kv_rows.py --scratch DIR"""
import argparse, contextlib, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- stand-ins: the engines' torch.cuda / HIP calls on host memory (no device is opened) -------------------------------
_zeros, _to = torch.zeros, torch.Tensor.to
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()

import tiny_q35
from caminus.cache.prompt import Layout
from caminus.engine import q35_dense_tp
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_tp import Dense35TP


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    """the receive slots' addresses only (distinct per card and slot)"""
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n = n

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q35_dense_tp._hip = lambda: FakeHip()
q35_dense_tp.Slots = FakeSlots


def refused(f):
    try: f()
    except ValueError: return True
    return False


MAX_CTX, SPEC_T = 8192, 4
ROWS = MAX_CTX + SPEC_T - 1
P1 = os.path.join(A.scratch, 'tiny-q35-8k.gguf'); i1 = tiny_q35.write(P1, n_ctx=MAX_CTX)
P2 = os.path.join(A.scratch, 'tiny-q35-tp-8k.gguf'); i2 = tiny_q35.write(P2, tp=True, n_ctx=MAX_CTX)
engines = (('Dense35Own', Dense35Own(P1, max_ctx=MAX_CTX, vram_check=False, spec_T=SPEC_T), 1),
           ('Dense35TP', Dense35TP(P2, devices=(0, 1), max_ctx=MAX_CTX, vram_check=False, ffn_share=0.5, spec_T=SPEC_T), 2))

for name, m, ncard in engines:
    kv = [k for k in m.state if k[-1] == 'kv']
    hkv = tiny_q35.HKV if ncard == 1 else 2
    row = 2 * (hkv // ncard) * tiny_q35.HD * 2        # (k and v, this card's kv heads, f16)
    check(m.kv_rows == ROWS and kv and all(m.state[k].numel() * m.state[k].element_size() == ROWS * row for k in kv),
          f'{name}: kv_rows {m.kv_rows} = max_ctx {MAX_CTX} + spec_T {SPEC_T} - 1, and each of its {len(kv)} KV views holds '
          f'exactly that many {row}-byte rows')
    L = Layout(m)
    check(L.kv_rows == ROWS and L.pb == [row] * ncard,
          f'{name}: the layout accepts the verify tail; row stride per card {L.pb}')
    bare = types.SimpleNamespace(devs=m.devs, ntok=m.ntok, max_ctx=m.max_ctx, state=m.state)
    check(refused(lambda: Layout(bare)),
          f'{name}: the same views without kv_rows are refused ({ROWS} rows are not whole rows of max_ctx {MAX_CTX})')
    k0 = kv[0]
    wrong = dict(m.state); wrong[k0] = torch.zeros(ROWS * row - row // 2, dtype=torch.uint8)
    check(refused(lambda: Layout(types.SimpleNamespace(devs=m.devs, ntok=m.ntok, max_ctx=m.max_ctx, kv_rows=ROWS, state=wrong))),
          f'{name}: a KV view with a wrong stride (not a whole number of {ROWS} rows) is refused')
    for bad in (MAX_CTX - 1, True, 8195.0):
        check(refused(lambda: Layout(types.SimpleNamespace(devs=m.devs, ntok=m.ntok, max_ctx=m.max_ctx, kv_rows=bad, state=m.state))),
              f'{name}: kv_rows {bad!r} is refused')

# a small context, where the old stride (the view's bytes // max_ctx) is wrong: 3 tail rows of 1024 bytes over 2048
# positions add 1 byte per row; kv_rows gives the exact stride
SMALL = 2048
P3 = os.path.join(A.scratch, 'tiny-q35-2k.gguf'); tiny_q35.write(P3, n_ctx=SMALL)
m = Dense35Own(P3, max_ctx=SMALL, vram_check=False, spec_T=SPEC_T)
row = 2 * tiny_q35.HKV * tiny_q35.HD * 2
nbytes = (SMALL + SPEC_T - 1) * row
L = Layout(m)
check(L.pb == [row] and nbytes // SMALL != row,
      f'Dense35Own at max_ctx {SMALL}: the layout stride is {L.pb[0]} bytes (exact); the old bytes // max_ctx gives {nbytes // SMALL}')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
