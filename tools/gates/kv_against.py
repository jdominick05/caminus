"""The KV-format comparison the Qwen3-30B-A3B G-model gates run with --against (tools/gates/q3moe_gmodel.py,
q3moe_depth_gmodel.py): this run's logits (the Q8 cache) against another run's on the same rows (the F16 cache, saved with
--save-logits), KLD(that || this) and top-1 agreement, gated.

The bound, in the floor rule's shape (mean KLD within a floor llama.cpp sets for itself, top-1 within one point):
  - llama.cpp's own cost of the same change: KLD(its F16-KV rows || its q8_0-KV rows) for the same tokens and its top-1
    agreement between the two. Our Q8-vs-F16 mean KLD must be at most llama.cpp's, our top-1 agreement at least
    llama.cpp's less one point. Needs --llama with q8_0-KV rows (--llama-kv q8_0) and --against-llama with its F16-KV rows.
  - or explicit bars (--against-kld MEAN --against-top1 PERCENT), for rows without llama.cpp references: say so when used.
No bound given: the gate refuses to run (a comparison printed and never gated cannot fail)."""
import os
import numpy as np


def load_rows(path, n, what):
    """f32 rows [n, vocab] from a raw file, refused when its size is not n whole rows"""
    size = os.path.getsize(path)
    if size == 0 or size % (4 * n):
        raise SystemExit(f'{what} {path}: {size} bytes is not {n} rows of f32 logits')
    return np.fromfile(path, dtype=np.float32).reshape(n, -1)


def add_args(ap):
    ap.add_argument('--save-logits', default=None, help="write this run's scored logits (f32 [rows, vocab])")
    ap.add_argument('--against', default=None, help="another run's saved logits of the same rows (the F16 cache's): gated")
    ap.add_argument('--against-llama', default=None, help="llama.cpp's F16-KV rows of the same tokens (with --llama q8_0-KV rows): "
                                                          "llama.cpp's own F16 -> q8_0 cost is the bound")
    ap.add_argument('--against-kld', type=float, default=None, help='explicit bar: mean KLD(saved || this) at most this')
    ap.add_argument('--against-top1', type=float, default=None, help='explicit bar: top-1 agreement at least this percent')


def check_args(ap, A):
    """argparse errors for an --against without a bound, or a bound without --against"""
    floor = A.against_llama is not None
    bars = A.against_kld is not None or A.against_top1 is not None
    if A.against is None:
        if floor or bars: ap.error('--against-llama / --against-kld / --against-top1 need --against')
        return
    if floor and bars: ap.error('--against: either --against-llama (llama.cpp\'s floor) or --against-kld / --against-top1, not both')
    if floor and (A.llama is None or A.llama_kv != 'q8_0'):
        ap.error('--against-llama needs --llama with llama.cpp\'s q8_0-KV rows and --llama-kv q8_0')
    if bars and (A.against_kld is None or A.against_top1 is None):
        ap.error('--against: give both --against-kld and --against-top1')
    if not floor and not bars:
        ap.error('--against needs a bound: --against-llama F16KV.bin (with --llama q8_0-KV rows) or --against-kld / --against-top1')


def gate(score, ours, saved, llama_f16=None, llama_q8=None, max_kld=None, min_top1=None):
    """(ok, lines): KLD(saved || ours) and top-1 agreement against llama.cpp's floor (llama_f16 / llama_q8: its rows of the
    same tokens) or the explicit bars. score(ref, cand) -> dict(mean, top1, n, ...) is the calling gate's own"""
    s = score(saved, ours)
    pct = 100.0 * s['top1'] / s['n']
    lines = [f'  Q8 vs F16  mean KLD {s["mean"]:.3e} (median {s["median"]:.3e}, p90 {s["p90"]:.3e}, max {s["max"]:.3e}); '
             f'top-1 {s["top1"]}/{s["n"]} ({pct:.1f} %)  (the saved run as the reference)']
    if llama_f16 is not None:
        f = score(llama_f16, llama_q8)
        bar_kld, bar_top1 = f['mean'], 100.0 * f['top1'] / f['n'] - 1.0
        lines.append(f'  llama.cpp F16-KV vs q8_0-KV mean KLD {f["mean"]:.3e}; top-1 {f["top1"]}/{f["n"]} (the floor: its own cost of 8-bit KV)')
        how = 'llama.cpp\'s'
    else:
        bar_kld, bar_top1, how = float(max_kld), float(min_top1), 'the explicit'
    c1, c2 = s['mean'] <= bar_kld, pct >= bar_top1
    lines.append(('PASS' if c1 else 'FAIL') + f' Q8 vs F16 cache: mean KLD {s["mean"]:.3e} <= {how} {bar_kld:.3e}')
    lines.append(('PASS' if c2 else 'FAIL') + f' Q8 vs F16 cache: top-1 {pct:.1f} % >= {how} {bar_top1:.1f} %')
    return c1 and c2, lines
