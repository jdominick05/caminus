"""The two-card weight load alone, with a read-back: Dense35TP's _load on both cards (no tapes, no K3 launch), then every card
tensor the engine keeps read back ONCE after the whole load, into pinned host memory, and compared bit for bit with the same
bytes rebuilt from the file (Dense35TP.host_bytes). The read-back never runs between uploads: a sync per upload would change
the timing of the path under test.
  CAMINUS_UPLOAD_LOG=ULOG [CAMINUS_H2D=pageable|pinned|peer] \\
      tools/guard.sh LOG 900 python -I tools/gates/tp_load_only.py GGUF [--repeat K] [--settle S] [--max-ctx N] [--ffn-share F]
GGUF: a dense qwen35 file (Swift 1.5, ThinkingCap). Both cards visible; card 0 is torch device 0. --ffn-share F: card 0's
part of the feed-forward rows; default: the split caminus serve runs these two cards at (caminus.arch's Dense35TP.split of
the first two visible cards, named through caminus.device.host from sysfs; 0.532 for XTX + XT); a pair it cannot name is
refused (give --ffn-share). --repeat K loads K times in this process (the engine dropped and the allocator emptied
between), --settle S seconds between loads (default 0). For one process per load, loop in the shell and stop at the first
nonzero exit (0 only with RESULT PASS in LOG; guard 99 = a fault: stop all GPU work):
  for i in 1 2 3; do CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tools/gates/tp_load_only.py GGUF || break; done
The grid is fixed (per_wgp 3) so the per-card grid probe, a kernel launch, is skipped; torch's own zero fills and the conv
channels' gather run as in every load. --max-ctx (default 64) keeps the KV caches small.
Prints one line per load (arm, uploads, GiB, seconds, tensors read back, mismatches; the first mismatches by name, card and
byte offset), then RESULT PASS | FAIL loads K uploads U mismatched M (exit 0 / 1). Refuses to run outside the guard."""
import gc
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'python'))
import numpy as np
import torch

from caminus.device import capacity
from caminus.engine import upload
from caminus.engine.q35_dense_tp import Dense35TP


class LoadOnly(Dense35TP):
    """Dense35TP without its tapes: the weights, the states and the upload log only"""

    def _build_tapes(self):
        pass


def load_only(path, devices=(0, 1), max_ctx=64, ffn_share=0.5, **kw):
    return LoadOnly(path, devices=devices, max_ctx=max_ctx, per_wgp=3, ffn_share=ffn_share, **kw)


def served_share(env=None, gs=None):
    """card 0's feed-forward share as caminus serve runs the two visible cards -> (share, 'XTX + XT'); ValueError when they
    cannot be named (torch device i = the i-th HIP_VISIBLE_DEVICES index, or HIP index i when it is unset)"""
    from caminus import arch
    from caminus.device import host
    env = os.environ if env is None else env
    for k in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        if (env.get(k) or '').strip(): raise ValueError(f'{k} is set: the visible cards cannot be named; give --ffn-share')
    gs = host.gpus() if gs is None else gs
    by_hip = {g['hip']: g for g in gs if g.get('hip') is not None}
    vis = (env.get('HIP_VISIBLE_DEVICES') or '').strip()
    try: order = [int(x) for x in vis.split(',') if x.strip()] if vis else sorted(by_hip)
    except ValueError: raise ValueError(f'HIP_VISIBLE_DEVICES={vis!r} is not a list of HIP indices; give --ffn-share') from None
    if len(order) < 2 or any(i not in by_hip for i in order[:2]):
        raise ValueError(f'the two visible cards (HIP {order[:2]}) are not both known to caminus.device.host; give --ffn-share')
    cards = [by_hip[i] for i in order[:2]]
    return float(arch.ARCHS['qwen35-own-tp'].split(cards)['ffn_share']), ' + '.join(c['label'] for c in cards)


def readback(m, chunk=None):
    """every kept card tensor of m against its bytes rebuilt from the file -> (tensors, bytes, [(name, card, kind, why)])"""
    chunk = chunk or upload.CHUNK
    for d in m.devs: torch.cuda.synchronize(d)
    capacity.check_host_pinned(chunk, 'tp_load_only read-back buffer')
    buf = torch.empty(chunk, dtype=torch.uint8, pin_memory=True)
    n = nb = 0; bad = []
    for name, c, kind, w, args in m.uploads:
        want = m.host_bytes(name, c, kind, args)
        n += 1; nb += want.nbytes
        if tuple(w.shape) != tuple(want.shape) or w.element_size() != want.itemsize:
            bad.append((name, c, kind, f'shape {tuple(w.shape)} x {w.element_size()} B, want {want.shape} x {want.itemsize} B')); continue
        r = upload.read_back(w, want, buf)
        if r is not None: bad.append((name, c, kind, f'first differing byte {r[0]} of {r[1]}'))
    return n, nb, bad


def result_line(loads, uploads, bad, mode):
    return f'RESULT {"PASS" if loads and not bad else "FAIL"} loads {loads} uploads {uploads} mismatched {len(bad)} mode {mode}'


def main(argv):
    a = list(argv)

    def opt(name, f, d):
        if name in a:
            i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
        return d
    K, settle = opt('--repeat', int, 1), opt('--settle', float, 0.0)
    max_ctx, share = opt('--max-ctx', int, 64), opt('--ffn-share', float, None)
    if len(a) != 1: print(__doc__); return 2
    if not os.environ.get('CAMINUS_GUARD'): print('tp_load_only: run it under tools/guard.sh'); return 2
    path = a[0]; mode = upload.mode_of()
    if share is None:
        try: share, pair = served_share()
        except ValueError as e: print(f'tp_load_only: {e}'); return 2
        how = f'(the served split of {pair})'
    else:
        how = '(given)'
    print(f'tp_load_only: {os.path.basename(path)} mode {mode} repeat {K} settle {settle} s max_ctx {max_ctx} ffn_share {share} '
          f'{how} log {"on" if os.environ.get("CAMINUS_UPLOAD_LOG") else "off"}', flush=True)
    loads = uploads = 0; bad = []
    for k in range(K):
        t0 = time.perf_counter(); m = load_only(path, max_ctx=max_ctx, ffn_share=share); t1 = time.perf_counter()
        n, nb, b = readback(m); t2 = time.perf_counter()
        loads += 1; uploads += m.h2d.n_up; bad += b
        print(f'load {k + 1}/{K}: {m.h2d.n_up} uploads, {m.h2d.n_bytes / 2 ** 30:.2f} GiB in {t1 - t0:.1f} s; read back {n} tensors, '
              f'{nb / 2 ** 30:.2f} GiB in {t2 - t1:.1f} s: {len(b)} mismatched', flush=True)
        for x in b[:20]: print('MISMATCH', *x, flush=True)
        m.F.close(); del m; gc.collect(); torch.cuda.empty_cache()
        if b: break                                  # (a mismatch: stop, report)
        if k + 1 < K and settle > 0: time.sleep(settle)
    print(result_line(loads, uploads, bad, mode), flush=True)
    return 0 if loads and not bad else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
