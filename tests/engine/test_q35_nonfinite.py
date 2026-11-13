"""The dense qwen35 engine fails loudly on corrupt weights (GPU, one card, under the guard): the startup self-check passes
on a healthy load; one fp16 block scale of a block-53 matrix overwritten with NaN on the card (the shape of a served
load seen once in ~15: every row non-finite from block ~52 on) makes the next step raise NonFiniteLogits, the report
names that tensor as changed on the card since load, and the engine stays refused after reset() (sticky). A second
engine reloaded from the file passes again. Prints PASS / FAIL lines and RESULT.

  tools/guard.sh LOG 900 python -I tests/engine/test_q35_nonfinite.py GGUF
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus.engine import k3                      # noqa: E402
from caminus.engine.q35_dense import Dense35Own     # noqa: E402
from caminus.errors import NonFiniteLogits          # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--block', type=int, default=53)
A = ap.parse_args()
fails = []


def check(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
    if not ok: fails.append(msg)


eng = Dense35Own(A.gguf, max_ctx=256)
try:
    eng.selfcheck(); check(eng.broken is None, 'startup self-check passes on a healthy load')
except NonFiniteLogits as e:
    check(False, f'startup self-check on a healthy load: {e}')
# (a matrix whose blocks start with their fp16 scale d: Q4_K / Q5_K / Q8_0; Q6_K planes start with quants)
d = next(v for v in eng.L[A.block].values() if isinstance(v, dict) and v.get('fmt') in ('Q4_K', 'Q5_K', 'Q8_0'))
name = d['name']
check(k3.NonFiniteLogits is NonFiniteLogits, 'k3.NonFiniteLogits is caminus.errors.NonFiniteLogits')
w = d['w'].reshape(-1).view(torch.uint8)
with torch.cuda.device(eng.dev):
    w[0:2] = torch.tensor([0x00, 0x7E], dtype=torch.uint8, device=w.device)   # (fp16 NaN = 0x7E00 in row 0's first block d)
    torch.cuda.synchronize()
eng.reset()
raised = None
toks = [248044, 9707, 11, 1879, 374]
try:
    for i in range(len(toks)):
        eng.step_token(toks, i)            # (the served greedy pass: argmax read back, its max checked)
except NonFiniteLogits as e:
    raised = str(e)
check(raised is not None, f'a NaN scale in {name} raises NonFiniteLogits on the next steps')
check(raised is not None and name in raised and 'changed on the card' in raised,
      f'the report names {name} as changed on the card since load: {(raised or "")[:220]}')
eng.reset()
try:
    eng.step_token(toks, 0); check(False, 'reset() clears broken (it must not)')
except NonFiniteLogits:
    check(True, 'the engine stays refused after reset() (broken is sticky)')
del eng, d, w
torch.cuda.empty_cache()
eng2 = Dense35Own(A.gguf, max_ctx=256)
try:
    eng2.selfcheck(); check(True, 'a reload from the file passes the self-check again')
except NonFiniteLogits as e:
    check(False, f'the reload: {e}')
print('RESULT ' + ('PASS' if not fails else f'FAIL ({len(fails)})'))
sys.exit(1 if fails else 0)
