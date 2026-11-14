"""The dense qwen35 line's own batched prefill (Q35OwnPrefill on Dense35Own: Swift/ThinkingCap/GSQ) end
to end against the float64 reference (tools/reference/q35_ref.py: N teacher-forced positions of a public text):

  G-model    positions [0, P) prefilled, then P .. N - 1 one K3 pass each: KLD(ref || ours) and top-1 at those positions,
             against the decode-only engine at the same positions and against llama.cpp's own prefill-then-decode (its
             logits rows from build/gg_golden --lc 'tfx NAME TOKENS P N-P P 1 OUT', one card, matching KV): the gate is mean
             KLD <= llama.cpp's and top-1 >= llama.cpp's (without --llama only the numbers are printed)
  geometry   the prefill in chunks of 16 and at the configured chunk size, and with the prompt cache's stops: the same cache rows and
             recurrent states, bit for bit, so the same logits after
  bounds     a span past max_ctx is refused before any write; the engine's poisoned flag refuses a prefill
  speed      (--speed L) prompt tok/s of a prefill of L positions from empty at chunk 2048 and 1024 (medians of 3; reported,
             not gated), and the slowest attention launch of the deepest chunk (every launch well under 200 ms)
  tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill.py GGUF --ref REF.npz [--kv f16|q8] [--llama LL.bin --llama-kv f16|q8] [--from P] [--speed L]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>).
--llama-kv declares the reference's actual format; verify its stored run metadata
before using it. No reference or tolerance is regenerated here. Q8 model quality
versus F16 and fallback timing at depth remain separate acceptance requirements."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q35_dense import Dense35Own
from caminus.prefill.driver import PrefillRefused
from caminus.prefill.q35own import Q35OwnPrefill, buffer_need_mib, shapes_of

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None)
    ap.add_argument('--kv', choices=('f16', 'q8'), default='f16')
    ap.add_argument('--llama-kv', choices=('f16', 'q8'), default='f16',
                    help='actual cache format of the recorded llama reference (legacy default: f16)')
    ap.add_argument('--from', dest='P', type=int, default=64)
    ap.add_argument('--speed', type=int, default=0)
    a = ap.parse_args(argv)
    if a.llama and a.llama_kv != a.kv:
        ap.error('--llama-kv must match --kv; verify the reference run metadata, do not reuse another format\'s floor')
    if a.P < 1 or a.speed < 0:
        ap.error('--from must be positive and --speed nonnegative')
    return a


def state_views(eng, prefix):
    """Exact byte views, including each Q8 row's scales and final V block.

    Decode after the prefill changes later rows; only the committed prefix
    belongs in this geometry comparison. Speculative tail rows stay excluded.
    """
    if isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 <= prefix <= eng.max_ctx:
        raise ValueError('state prefix must be an integer within max_ctx')
    row_bytes = k3.kv_pos_bytes(eng.nkv * eng.hd, eng.q8)
    out = []
    for layer in eng.L:
        if layer['kind'] == 'attn':
            cache = layer['cache']
            expected = torch.uint8 if eng.q8 else torch.float16
            if cache.dtype != expected or not cache.is_contiguous():
                raise ValueError('cache dtype or layout does not match the engine KV format')
            raw = cache.view(torch.uint8).reshape(-1)
            if raw.numel() % row_bytes or raw.numel() < prefix * row_bytes:
                raise ValueError('cache does not contain whole physical rows covering the prefix')
            out.append(raw[:prefix * row_bytes])
    for name in ('conv_st', 'gdn'):
        for layer in eng.L:
            if layer['kind'] == 'gdn':
                tensor = layer[name] if name == 'conv_st' else layer[name][:eng.NV * eng.S * eng.S]
                out.append(tensor.view(torch.uint8).reshape(-1))
    return out


def state_bytes(eng, prefix):
    return [view.clone() for view in state_views(eng, prefix)]


def same_state(left, right):
    return len(left) == len(right) and all(torch.equal(x, y) for x, y in zip(left, right))


def main():
    A = parse_args()
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']; N, P = len(seq), A.P
    if not 0 < P < N or ref.ndim != 2 or ref.shape[0] != N or not np.isfinite(ref).all():
        raise ValueError('reference must contain finite logits for every token and 0 < --from < token count')
    print(torch.cuda.get_device_name(0), flush=True)
    ok = True


    def check(c, msg):
        nonlocal ok
        ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


    def lsm(v):
        v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


    def score(rows, cand):
        if rows.shape != cand.shape or not np.isfinite(rows).all() or not np.isfinite(cand).all():
            raise ValueError('quality comparison requires matching shapes and finite logits at every position')
        kl = [float(np.sum(np.exp(lsm(r)) * (lsm(r) - lsm(c)))) for r, c in zip(rows, cand)]
        return float(np.mean(kl)), float(np.percentile(kl, 90)), float(np.max(kl)), int(sum(np.argmax(r) == np.argmax(c) for r, c in zip(rows, cand)))


    max_ctx = max(N + 64, A.speed + 64)
    CHUNK = 2048 if A.speed else 128
    eng = Dense35Own(A.gguf, max_ctx=max_ctx, q8=A.kv == 'q8', vram_extra_mib=buffer_need_mib(shapes_of(A.gguf), CHUNK))
    pf = Q35OwnPrefill(eng, chunk=CHUNK)
    attn_L = [L for L in eng.L if L['kind'] == 'attn']
    snap_mib = 3 * sum(view.numel() for view in state_views(eng, P)) / 2 ** 20 + 2
    capacity.check([eng.dev], [snap_mib], f'test_q35_prefill: its state snapshots ({snap_mib:.0f} MiB)')
    print(f'NOTE target KV {A.kv}; geometry compares {k3.kv_pos_bytes(eng.nkv * eng.hd, eng.q8)} bytes per cache row', flush=True)


    def decode_only():
        eng.reset(); return np.stack([eng.step(t).float().cpu().numpy() for t in seq])


    def prefill_then_decode(chunk=None, stops=()):
        eng.reset()
        st = []
        pf.run(seq, 0, 0, P, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
        rows = state_bytes(eng, P)
        eng.pos_host = P
        out = np.stack([eng.step(t).float().cpu().numpy() for t in seq[P:]])
        return out, rows, st


    dec = decode_only()
    ours, rows1, _ = prefill_then_decode()
    so, sd = score(ref[P:], ours), score(ref[P:], dec[P:])
    fmt = lambda s: f'mean KLD {s[0]:.3e} (p90 {s[1]:.3e}, max {s[2]:.3e}), top-1 {s[3]}/{N - P}'
    print(f'  prefill [0, {P}) then decode: {fmt(so)}', flush=True)
    print(f'  decode only, the same positions: {fmt(sd)}', flush=True)
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(N - P, -1)
        sl = score(ref[P:], ll)
        print(f'  llama.cpp prefill then decode: {fmt(sl)}', flush=True)
        check(so[0] <= sl[0], f'G-model: mean KLD vs float64 {so[0]:.3e} <= llama.cpp\'s prefill-then-decode {sl[0]:.3e}')
        check(so[3] >= sl[3], f'G-model: top-1 {so[3]} >= llama.cpp\'s {sl[3]}')
    else:
        print('NOTE G-model versus llama.cpp was not gated: no matching recorded --llama reference supplied', flush=True)
    check(so[0] <= 2.0 * sd[0] + 1e-4, f'prefill-then-decode KLD within 2x the decode-only engine\'s at the same positions + 1e-4 ({so[0]:.3e} vs {sd[0]:.3e})')
    o16, rows16, _ = prefill_then_decode(chunk=16)
    stops = tuple(s for s in (16, 40, 41) if s < P)
    ost, rowsst, st = prefill_then_decode(stops=stops)
    check(same_state(rows1, rows16), 'chunks of 16: the same complete cache rows and recurrent states, bit for bit')
    check(np.array_equal(o16, ours), 'chunks of 16: the same logits after, bit for bit')
    check(same_state(rows1, rowsst) and np.array_equal(ost, ours) and st == list(stops), f'stops at {stops} (at_stop {st}): the same complete states and logits')
    eng.reset()
    try:
        pf.run(seq * (max_ctx // N + 2), 0, 0, eng.max_ctx + 1); check(False, 'a span past max_ctx was not refused')
    except PrefillRefused as e:
        check(True, f'a span past max_ctx is refused before any write ({str(e)[:70]})')
    eng.poisoned = 'test'
    try:
        pf.run(seq, 0, 0, P); check(False, 'a poisoned engine was not refused')
    except PrefillRefused:
        check(True, 'a poisoned engine refuses the prefill')
    eng.poisoned = None
    if A.speed:
        L_ = A.speed; toks = (seq * (L_ // N + 1))[:L_]
        # first the slowest attention launch: the deepest rows of the last chunk (one launch walks every position below its
        # rows; the cache's contents do not change its time). The full prefills only if it is well under the 200 ms bound
        O, L0 = pf.O, attn_L[0]
        n = min(pf.chunk, L_); p = L_ - n; r = O.attn_blocks(p, n)[-1]
        capacity.check([eng.dev], [2 * n * eng.nh * eng.hd * 4 / 2 ** 20 + 2], 'test_q35_prefill: the attention timing buffers')
        q = torch.randn(n * eng.nh * eng.hd, device=f'cuda:{eng.dev}'); out = torch.zeros_like(q)
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            (pf.KV if eng.q8 else O).attn_wm(q, L0['cache'], p, n, eng.nh, eng.nkv, eng.hd, 1.0 / np.sqrt(eng.hd), out, eng.max_ctx, *r)
            torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
        dt = sorted(ts)[1]; del q, out
        check(dt < 100.0, f'the slowest attention launch (blocks {r[0]} .. {r[1] - 1} of a chunk at {p}, up to position {L_ - 1}): {dt:.1f} ms (< 100)')
        if dt < 100.0:
            for chunk in (2048, 1024):
                if chunk > pf.chunk: continue
                ts = []
                for _ in range(3):
                    eng.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
                    pf.run(toks, 0, 0, L_, chunk=chunk); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                med = sorted(ts)[1]
                print(f'NOTE speed (prefill): {L_} positions from empty at chunk {chunk}: {1e3 * med:.0f} ms, {L_ / med:.0f} tok/s '
                      f'(runs {", ".join(f"{1e3 * t:.0f}" for t in ts)} ms)', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
