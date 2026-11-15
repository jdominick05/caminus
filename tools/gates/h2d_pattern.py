"""The two-card load's upload pattern on ONE card: a fresh 2-80 MiB host buffer (as ggufio.Tensor.pread makes one), a cut
copy of half its rows or half its columns (as q35_dense_tp.cut makes one), the copy of that cut to the card through the
CAMINUS_H2D arm (engine/upload.py), then both host buffers dropped (the read buffer stays alive through the upload, as in
the load), again. The card copies are kept until a budget is full, then read back into pinned memory, compared bit for bit
with the bytes rebuilt from the same seed, and freed: one round. No kernel launch.
  CAMINUS_UPLOAD_LOG=ULOG [CAMINUS_H2D=pageable|pinned] HIP_VISIBLE_DEVICES=<the non-display card> \\
      tools/guard.sh LOG 300 python -I tools/gates/h2d_pattern.py [--seconds 180] [--budget-mib 1024] [--seed 1]
Exactly one card visible (run it on the card that drives no display). Half the uploads are 1-2 MiB (the size of the
attention k / v slices, 1.12 and 1.64 MiB on a two-card 9B load), half 2-40 MiB; each source is twice its upload.
Torch's cached blocks are released after every round, so the card holds at most the budget plus one upload. Rounds run
until --seconds have passed. peer needs two cards and is refused.
Prints one line per round, then RESULT PASS | FAIL rounds R uploads U GiB G mismatched M (exit 0 / 1). Refuses to run
outside the guard."""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'python'))
import numpy as np
import torch

from caminus.device import capacity
from caminus.engine import upload

MIB = 1 << 20


def gen(i, n):
    """upload i's source bytes: n (a multiple of 8) in a fresh bytearray"""
    b = np.frombuffer(bytearray(n), np.uint8)
    b.view(np.uint64)[:] = np.arange(n // 8, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i)
    return b


def cut_of(b, R, rb, cols):
    """the cut copy: rows R/4 .. 3R/4 (cols False) or the first half of every row (cols True)"""
    m = b.reshape(R, rb)
    if cols: return np.ascontiguousarray(np.concatenate([m[:, :rb // 2]], axis=1))
    return np.ascontiguousarray(np.concatenate([m[R // 4:3 * R // 4]], axis=0))


def shape_of(rng):
    """(rows, row bytes) of the source: a Q4_K-like row of 2304 bytes, rows a multiple of 8, sized so the cut (half the
    source) is at least the drawn upload size: 1-2 MiB or 2-40 MiB (at most 9 KiB over)"""
    rb = 2304
    mib = rng.uniform(1.0, 2.0) if rng.random() < 0.5 else rng.uniform(2.0, 40.0)
    return -(-int(2 * mib * MIB) // (8 * rb)) * 8, rb


def run(seconds=180.0, budget_mib=1024, seed=1, mode=None, log=None, max_uploads=None, chunk=None):
    """the bench on torch device 0 -> dict(rounds, uploads, bytes, bad=[(i, why)], mode)"""
    up = upload.Uploader([0], mode, log if log is not None else upload.UploadLog.from_env(), what='h2d_pattern')
    capacity.check([0], [budget_mib + 40 + upload.extra_mib(up.mode, 1)[0] + 64], f'h2d_pattern (budget {budget_mib} MiB)')
    rng = np.random.default_rng(seed)
    up.begin('h2d_pattern', f'seed{seed}')
    t_end = time.monotonic() + seconds
    rounds = i = 0; bad = []; buf = None
    while time.monotonic() < t_end and (max_uploads is None or i < max_uploads):
        held, used = [], 0
        while used < budget_mib * MIB and (max_uploads is None or i < max_uploads):
            R, rb = shape_of(rng); cols = bool(i % 2)
            b = gen(i, R * rb); x = cut_of(b, R, rb, cols)
            with torch.cuda.device(0): w = up.up(x, f'bench.{i}', 0, 'bench')
            held.append((i, R, rb, cols, w)); used += x.nbytes; del x, b; i += 1
        if buf is None:
            capacity.check_host_pinned(chunk or upload.CHUNK, 'h2d_pattern read-back buffer')
            buf = torch.empty(chunk or upload.CHUNK, dtype=torch.uint8, pin_memory=True)
        torch.cuda.synchronize(0)
        nb = 0
        for j, R, rb, cols, w in held:
            r = upload.read_back(w, cut_of(gen(j, R * rb), R, rb, cols), buf)
            if r is not None: bad.append((j, f'first differing byte {r[0]} of {r[1]}'))
        rounds += 1
        print(f'round {rounds}: {len(held)} uploads, {used / 2 ** 30:.2f} GiB; {len(bad)} mismatched so far', flush=True)
        del held, w
        torch.cuda.empty_cache()                     # (the cached blocks of this round's varied sizes: never past the budget)
        if bad: break
    up.finish()
    return dict(rounds=rounds, uploads=up.n_up, bytes=up.n_bytes, bad=bad, mode=up.mode)


def result_line(r):
    ok = r['uploads'] > 0 and not r['bad']
    return (f'RESULT {"PASS" if ok else "FAIL"} rounds {r["rounds"]} uploads {r["uploads"]} GiB {r["bytes"] / 2 ** 30:.2f} '
            f'mismatched {len(r["bad"])} mode {r["mode"]}')


def main(argv):
    a = list(argv)

    def opt(name, f, d):
        if name in a:
            i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
        return d
    secs, budget, seed = opt('--seconds', float, 180.0), opt('--budget-mib', int, 1024), opt('--seed', int, 1)
    if a: print(__doc__); return 2
    if not os.environ.get('CAMINUS_GUARD'): print('h2d_pattern: run it under tools/guard.sh'); return 2
    if torch.cuda.device_count() != 1:
        print(f'h2d_pattern: {torch.cuda.device_count()} cards visible; make one visible (HIP_VISIBLE_DEVICES=<the non-display card>)')
        return 2
    r = run(secs, budget, seed)
    for j, why in r['bad'][:20]: print('MISMATCH', j, why)
    print(result_line(r), flush=True)
    return 0 if r['uploads'] and not r['bad'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
