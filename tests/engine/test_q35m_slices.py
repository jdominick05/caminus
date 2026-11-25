"""Moe35Own's sliced pass (engine/q35_moe.py _build_slices on k3.Engine.split) on the card: GPU, one KV format per run.
  1. the slicing is exact: the sliced pass with each layer's split pair as its own launch between the slices
     (attn_launch 'pair') gives the single launch's bits: every token's logits, the Gated DeltaNet and conv states and
     the KV rows the tokens wrote, from position 0 and from --depth (the rows before it left as allocated), at 3, 4 and 5
     workgroups per WGP. The attention kernel in the pair's place on the same states: its top-1 agreement and largest
     logit difference, printed.
  2. the attention kernel at real depth: a --prompt token prompt (this repository's docs/dev/*.md, the model's own
     tokenizer; F16: the batched prefill, Q8: one-token passes), its state saved, then --score teacher-forced passes from
     that state on the single launch and again on the sliced pass with the kernel. The kernel sums in another order
     than the split pair (and decodes Q8_0 rows to fp16), so the bits differ by design. Gate: mean KLD(single || sliced)
     within the G-model gate's bound for this file, llama.cpp's own mean KLD against the float64 reference with the same
     KV type (tools/gates/q35_gmodel.py --ref-only, 128 positions: 1.62e-3 F16, 1.76e-3 q8_0), and top-1 agreement on all
     but at most one pass. Printed beside it: the kernel against itself at half its splits (only the summation order
     changes; XT, 16,384: 1.5e-4 F16, 1.0e-4 Q8), the scale of difference any reordering of these sums makes here
     (through the routers' near ties).
  tools/guard.sh LOG 900 python -I tests/engine/test_q35m_slices.py GGUF [--kv f16|q8] [--depth 20000] [--prompt 16384]
      [--score 32] [--grids 3,4,5]
Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--kv', default='f16', choices=('f16', 'q8'))
ap.add_argument('--depth', type=int, default=20000); ap.add_argument('--prompt', type=int, default=16384)
ap.add_argument('--score', type=int, default=32); ap.add_argument('--grids', default='3,4,5')
A = ap.parse_args()
from caminus import arch, ggufio
from caminus.bpe import BPE
from caminus.engine import k3
from caminus.engine.q35_dense import RECURRENT
from caminus.engine.q35_moe import Moe35Own
from caminus.engine.q3moe_one import card_per_wgp
from caminus.prefill import q35moe as PF

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


NEVER = 1 << 30
TOKS = [248044, 9707, 11, 1879, 374, 13, 1234, 524]
q8 = A.kv == 'q8'
max_ctx = max(A.depth + len(TOKS), A.prompt + A.score) + 8
batched = not q8   # (the batched prefill: F16 KV only)
extra = (PF.buffer_need_mib(PF.shapes_of(A.gguf), 2048) + arch.PREFILL_MODULES_MIB if batched else 0) + 256   # (+ the saved states, the rows read back)
m = Moe35Own(A.gguf, max_ctx=max_ctx, q8=q8, attn_from=NEVER, vram_extra_mib=extra)
print(f'{torch.cuda.get_device_name(0)}; module {m.module}; max_ctx {max_ctx}; {len(m.slices)} slices + {len(m.pair_slices)} '
      f'attention launches a sliced pass', flush=True)


def kv_rows(lo, hi):
    """the rows [lo, hi) of every attention block's cache, on the host"""
    out = []
    for c in m.attn_caches:
        r = c.view(m.kv_rows, -1)
        out.append(r[lo:hi].cpu().clone())
    return out


def states():
    return {k: t.cpu().clone() for k, t in m.state.items() if k[2] in RECURRENT}


def run(p0, attn_from, launch='kernel'):
    """the tokens from position p0 on fresh states: (logits rows, recurrent states, KV rows written)"""
    m.reset(); m.attn_from, m.attn_launch = attn_from, launch
    if p0: m.pos_host = p0
    lg = [m.step(t).float().cpu().clone() for t in TOKS]
    return torch.stack(lg), states(), kv_rows(p0, p0 + len(TOKS))


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.uint8 else a,
                                                                       b.view(torch.uint8) if b.dtype != torch.uint8 else b)


# ---- 1. the slicing is exact --------------------------------------------------------------------------------------------
for g in [int(x) for x in A.grids.split(',')]:
    m.per_wgp = g
    with torch.cuda.device(m.dev): m._build_tape()
    for p0 in (0, A.depth):
        L1, S1, K1 = run(p0, NEVER)
        L2, S2, K2 = run(p0, 0, 'pair')
        st = all(same(S1[k], S2[k]) for k in S1); kv = all(same(a, b) for a, b in zip(K1, K2))
        check(same(L1, L2) and st and kv and not m.bails,
              f'grid {g} per WGP ({m.E.grid} workgroups), from position {p0}: the sliced pass with the pair launched alone is the '
              f'single launch bit for bit ({len(TOKS)} tokens: logits {same(L1, L2)}, states {st}, KV rows {kv}; bails {len(m.bails)})')
        L3, _, _ = run(p0, 0)
        top = int((L1.argmax(1) == L3.argmax(1)).sum())
        print(f'  the attention kernel in its place: top-1 {top}/{len(TOKS)}, max |d logit| {float((L1 - L3).abs().max()):.3e}, '
              f'finite {bool(torch.isfinite(L3).all())}', flush=True)
        check(bool(torch.isfinite(L3).all()) and not m.bails, f'grid {g}, from position {p0}: the kernel\'s sliced pass runs (finite logits, no bail)')

# ---- 2. the attention kernel at real depth --------------------------------------------------------------------------------
m.per_wgp = card_per_wgp()
with torch.cuda.device(m.dev): m._build_tape()
F = ggufio.GGUFFile(A.gguf); tok = BPE(F.meta); F.close()
text = ''.join(open(p, encoding='utf-8').read() for p in sorted(glob.glob(os.path.join(ROOT, 'docs', 'dev', '*.md'))))
seq = tok.encode(text[:8 * (A.prompt + A.score)])   # (about 4 characters a token: enough, and the encoder is Python)
while len(seq) < A.prompt + A.score: seq = seq + seq
seq = seq[:A.prompt + A.score]
P = A.prompt
m.reset(); m.attn_from = NEVER
if batched:
    PF.Q35MoEPrefill(m, chunk=2048).run(seq, 0, 0, P)
else:
    for i in range(P): m._run_tok(seq[i], i, 0 if i == 0 else 1, False)
saved = (states(), m.pos_host, m.fresh, m.last_T)
print(f'prompt: {P} tokens ({"batched prefill" if batched else "one-token passes"}); scoring positions {P} .. {P + A.score - 1}', flush=True)


def score(attn_from, splits=None):
    st, m.pos_host, m.fresh, m.last_T = saved[0], saved[1], saved[2], saved[3]
    with torch.cuda.device(m.dev):
        for k, t in m.state.items():
            if k in st: t.copy_(st[k])
    m.attn_from, m.attn_splits = attn_from, splits
    rows = []
    for i in range(P, P + A.score):
        commit = 0 if (batched and i == P) else 1
        m._run_tok(seq[i], i, commit, False)
        rows.append(m.buf['logits'].double().cpu().clone())
    return torch.stack(rows)


def kld(a, b):
    lp, lq = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return (lp.exp() * (lp - lq)).sum(-1), int((a.argmax(1) == b.argmax(1)).sum())


from caminus.engine import dec_q35m_attn
ns = dec_q35m_attn.splits(P + 1)
R1 = score(NEVER); R2 = score(0); R3 = score(0, max(1, ns // 2))
kl, top = kld(R1, R2); kf, tf = kld(R2, R3)
print(f'  KLD(single || sliced) mean {float(kl.mean()):.3e}, max {float(kl.max()):.3e}; top-1 {top}/{A.score}; '
      f'max |d logit| {float((R1 - R2).abs().max()):.3e}; the kernel at {ns} splits against {max(1, ns // 2)}: mean '
      f'{float(kf.mean()):.3e}, max {float(kf.max()):.3e}, top-1 {tf}/{A.score}', flush=True)
BOUND = 1.76e-3 if q8 else 1.62e-3
check(float(kl.mean()) <= BOUND and top >= A.score - 1 and not m.bails,
      f'at depth {P} ({A.kv} KV): the attention kernel against the split pair: mean KLD {float(kl.mean()):.3e} <= {BOUND} '
      f'(llama.cpp\'s own against float64), top-1 {top} >= {A.score - 1}')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
