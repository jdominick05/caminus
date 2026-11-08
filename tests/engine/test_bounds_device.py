"""Device bound guard (defense in depth behind the host checks): the pass's positions pos .. pos + T - 1 against the KV
cache's rows (OP_KV_APPEND / OP_ATTN_DEC) and the lazy commit against [0, T] (the OP_GDN_CONV / OP_GDN_STEP / OP_PLE_CONV
replays), in both forms:
- kernel: k3_run's prologue (kernels/k3.hip, the production path): out of range, no task runs: nothing is written (cache
  rows, outputs, states and the position / commit words keep their bytes);
- launch: the separate guard launch (kernels/k3_guard.hip: k3_run2 tapes, tapes with several position tensors; forced
  here with Engine.guard_launch): out of range it clamps the words to 0 in place (rows 0 .. T - 1 written, a commit-0
  replay).
Either way the sync block reports it (k3.bounds_status), Engine.run raises k3.ContextBoundsError (never retried) and the
next launch runs normally. The cache sits inside a larger buffer with canary bytes on both sides, so a write past it
lands on the canaries (caught) instead of faulting. In range, the outputs are saved for a bitwise comparison with the
base build (--base runs only the in-range cases, through a tree without the guard), and both forms must give the same
bits.
Runs on the XT ALONE (HIP_VISIBLE_DEVICES=1; refuses an XTX) under guard.sh:
  HIP_VISIBLE_DEVICES=1 guard.sh LOG 120 python -I tests/engine/test_bounds_device.py [--save OUT.npz] [--base]
  python -I tests/engine/test_bounds_device.py --cmp A.npz B.npz"""
import os, sys
a = sys.argv[1:]
if '--cmp' in a:
    import numpy as np
    x, y = (np.load(p) for p in a[a.index('--cmp') + 1:a.index('--cmp') + 3])
    bad = [k for k in sorted(set(x.files) | set(y.files)) if k not in x.files or k not in y.files or x[k].tobytes() != y[k].tobytes()]
    print(f'{len(x.files)} vs {len(y.files)} arrays; ' + (f'DIFFER: {bad}' if bad else 'all bitwise equal')); sys.exit(1 if bad else 0)
SRC = os.environ.get('SRC', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(SRC, 'python'))
import numpy as np
import torch
from caminus.engine import k3

BASE = '--base' in a
SAVE = a[a.index('--save') + 1] if '--save' in a else None
assert os.environ.get('HIP_VISIBLE_DEVICES') == '1' and torch.cuda.device_count() == 1, 'run on the XT alone: HIP_VISIBLE_DEVICES=1'
name = torch.cuda.get_device_name(0); assert 'XTX' not in name, f'refusing to run on {name} (the monitors card)'
print(f'device: {name}', flush=True)
torch.cuda.set_per_process_memory_fraction(0.5, 0)
fails, saved = [], {}
MODE = 'kernel'     # the guard form of the current cases (see the docstring)
n_refused = {'kernel': 0, 'launch': 0}
CAN = 0xA5          # canary byte (before / after the cache, and the cache's initial fill)
R, PAD = 16, 4096   # cache rows; canary bytes on each side


def check(c, what):
    if not c: fails.append(what); print(f'  FAIL {what}', flush=True)


def run(e, expect_refusal):
    """launch; returns the bounds message (None: clean). Base trees have no bounds API: in-range cases only"""
    if not BASE: e.guard_launch = MODE == 'launch'
    e.build()
    if BASE: e.run(); return None
    try:
        e.run(); msg = None
    except k3.ContextBoundsError as ex:
        msg = str(ex)
    except RuntimeError as ex:
        check(False, f'a wait-bail RuntimeError instead of a bounds error: {ex}'); msg = None
    check((msg is not None) == expect_refusal, f'refusal {"missing" if expect_refusal else "unexpected"}: {msg}')
    if msg is not None:
        n_refused[MODE] += 1
        how = 'skipped its whole tape' if MODE == 'kernel' else 'clamped to 0'
        check(how in msg, f'{MODE}: report {msg!r} lacks {how!r}')
    st = e.sync[2 * k3.MAXSEG:].tolist()
    check(not any(st), f'sync status not cleared after the launch: {st}')
    return msg


def keep(key, arr):
    """an in-range result: saved by the kernel-form pass, and the launch-form pass must give the same bits"""
    if MODE == 'kernel': saved[key] = arr
    else: check(arr.tobytes() == saved[key].tobytes(), f'{key}: the launch-form guard changed an in-range result')


def kv_case(module, T, pos0, q8=True, tag=''):
    """kv_append then attn_dec of T tokens at positions pos0 .. pos0 + T - 1 into an R-row cache"""
    D, H, NKV = 256, 12, 1   # (12 query heads: the Coder modules' split attention pair supports only that)
    M = NKV * D; pb = 2 * (M // 32) * 34 if q8 else 4 * M   # (= k3.kv_pos_bytes; the base tree lacks it)
    g = torch.Generator().manual_seed(1000 * T + (pos0 % 997))
    buf = torch.full((PAD + R * pb + PAD,), CAN, dtype=torch.uint8, device='cuda'); cache = buf[PAD:PAD + R * pb]
    e = k3.Engine(module=module, grid=8, limit_us=20_000, T=T)
    pos = torch.tensor([pos0], dtype=torch.int32, device='cuda')
    k_ = torch.randn(T * M, generator=g).cuda(); v_ = torch.randn(T * M, generator=g).cuda(); q = torch.randn(T * H * D, generator=g).cuda()
    out = torch.full((T * H * D,), -7.0, device='cuda')
    s0 = e.kv_append(k_, v_, cache, pos, q8=q8)
    e.attn_dec(q, cache, pos, out, H, NKV, D, 1.0 / 16, q8=q8, dep=s0)
    ok_range = 0 <= pos0 and pos0 + T <= R
    msg = run(e, not ok_range); torch.cuda.synchronize()
    # in range: rows pos0 ..; out of range: none (kernel: the tape is skipped) or rows 0 .. T - 1 (launch: clamped to 0)
    eff = pos0 if (ok_range or BASE) else (None if MODE == 'kernel' else 0)
    b = buf.cpu().numpy(); c = b[PAD:PAD + R * pb].reshape(R, pb)
    check((b[:PAD] == CAN).all() and (b[PAD + R * pb:] == CAN).all(), f'{tag}: canary bytes overwritten')
    for r in range(R):
        written = not (c[r] == CAN).all(); want = eff is not None and any(eff + t == r for t in range(T))
        check(written == want, f'{tag}: cache row {r} written={written}, expected {want}')
    if ok_range:
        keep(f'{tag}.cache', c.copy()); keep(f'{tag}.out', out.cpu().numpy())
    elif not BASE:
        want = f'pass position {pos0} (+ T tokens) outside the KV cache rows [0, {R})'
        check(msg is not None and want in msg, f'{tag}: report {msg!r} (want {want!r})')
        if MODE == 'kernel':
            check(int(pos.item()) == pos0, f'{tag}: position word changed ({int(pos.item())})')
            check((out.cpu() == -7.0).all().item(), f'{tag}: attention output written by a skipped tape')
        else:
            check(int(pos.item()) == 0, f'{tag}: position word not clamped ({int(pos.item())})')
    print(f'  [{MODE}] {tag}: {"refused: " + msg if msg else "in range, ran"}', flush=True)
    return e


def commit_case(module, T, commit, tag):
    """gdn_conv / ple_conv / gdn_step with params[1] = commit (the lazy commit: rows of the previous pass's inputs)"""
    g = torch.Generator().manual_seed(77 + T)
    rnd = lambda *s: torch.randn(*s, generator=g).cuda()
    e = k3.Engine(module=module, grid=8, limit_us=20_000, T=T)
    C = 512; x, w, st, xp = rnd(T * C), rnd(C, 4), rnd(C, 3), rnd(T * C); out = torch.full((T * C,), -7.0, device='cuda')
    pc_x, pc_w, hist, pc_xp = rnd(T * C), (rnd(C, 4) * 0.1).half(), rnd(C, 9), rnd(T * C); pc_out = torch.full((T * C,), -7.0, device='cuda')
    H, nk, S = 2, 1, 128
    qk, v, gates = rnd(T * 2 * nk * S), rnd(T * H * S), rnd(T * 2 * H)
    blk = rnd(k3.gdn_block_size(H, S, nk, T)) * 0.01; gout = torch.full((T * H * S,), -7.0, device='cuda')
    a_ = e.gdn_conv(x, w, st, out, xp, rows_per_task=128)
    b_ = e.ple_conv(pc_x, pc_w, hist, pc_out, pc_xp, rows_per_task=128, dep=a_)
    e.gdn_step(qk, v, gates, blk, gout, H, S, nk, H * S, dep=b_)
    e.params[1] = commit
    outs = (('conv_out', out), ('conv_state', st), ('ple_out', pc_out), ('ple_hist', hist), ('gdn_out', gout), ('gdn_block', blk))
    before = {n: t.cpu().numpy().copy() for n, t in outs}
    msg = run(e, not 0 <= commit <= T); torch.cuda.synchronize()
    if 0 <= commit <= T:
        for n, t in outs: keep(f'{tag}.{n}', t.cpu().numpy())
    elif not BASE:
        if MODE == 'kernel':   # the tape was skipped: every output and state keeps its bytes
            for n, t in outs: check(t.cpu().numpy().tobytes() == before[n].tobytes(), f'{tag}: {n} written by a skipped tape')
            check(int(e.params[1].item()) == commit, f'{tag}: commit word changed')
        else:   # clamped to a commit-0 replay: the same results as the in-range commit-0 run of the same inputs
            ref = tag.rsplit('.commit', 1)[0] + '.commit0'
            for n, t in outs: check(t.cpu().numpy().tobytes() == saved[f'{ref}.{n}'].tobytes(), f'{tag}: {n} differs from the commit-0 run')
            check(int(e.params[1].item()) == 0, f'{tag}: commit word not clamped')
        check(msg is not None and f'lazy commit {commit} outside [0, {T}]' in msg, f'{tag}: report {msg!r}')
    print(f'  [{MODE}] {tag}: {"refused: " + msg if msg else "in range, ran"}', flush=True)


for MODE, module in [(m, mod) for m in (('kernel',) if BASE else ('kernel', 'launch')) for mod in ('k3_gguf_gsq', 'k3_gguf')]:
    for T in (1, 2, 4):
        print(f'{module} T={T} ({k3.module_for(module, T)})', flush=True)
        for q8 in (True, False):
            kv_case(module, T, R - T, q8, f'{module}.T{T}.q8{int(q8)}.last')           # the last rows: in range
            kv_case(module, T, 0, q8, f'{module}.T{T}.q8{int(q8)}.first')
            if BASE: continue
            kv_case(module, T, R, q8, f'{module}.T{T}.q8{int(q8)}.at_cap')            # every token past the end
            kv_case(module, T, -1, q8, f'{module}.T{T}.q8{int(q8)}.neg')
            kv_case(module, T, 2 ** 31 - 1, q8, f'{module}.T{T}.q8{int(q8)}.huge')
            if T > 1: kv_case(module, T, R - T + 1, q8, f'{module}.T{T}.q8{int(q8)}.straddle')   # the last token one past
        for cm in range(T + 1): commit_case(module, T, cm, f'{module}.T{T}.commit{cm}')
        if not BASE:
            commit_case(module, T, T + 1, f'{module}.T{T}.commit{T + 1}')
            commit_case(module, T, -1, f'{module}.T{T}.commit-1')
        # after the refusals the engine runs normally: a fresh in-range launch on the same module
        kv_case(module, T, 3, True, f'{module}.T{T}.after')

if SAVE: np.savez(SAVE, **saved); print(f'saved {len(saved)} arrays to {SAVE}')
print(('FAIL: ' + '; '.join(fails)) if fails else f'PASS ({"base, in range only" if BASE else "every out-of-range pass refused and reported, canaries intact"}); '
      f'refusals: kernel form {n_refused["kernel"]} (tape skipped), launch form {n_refused["launch"]} (clamped)')
sys.exit(1 if fails else 0)
