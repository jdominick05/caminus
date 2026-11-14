"""The dense qwen35 line's own batched prefill on TWO cards (caminus.prefill.q35own_tp.Q35OwnPrefillTP on Dense35TP) end to
end against the float64 reference (tools/reference/q35_ref.py: N teacher-forced positions of a public text), as
tests/prefill/test_q35_prefill.py for one card:

  G-model    positions [0, P) prefilled on both cards, then P .. N - 1 one two-card K3 pass each: KLD(ref || ours) and top-1,
             against the two-card decode-only engine at the same positions and against llama.cpp's own prefill-then-decode
             (LL.bin, the one-card gate's file): the gate is mean KLD <= llama.cpp's and top-1 >= llama.cpp's
  replicas   the replicated residual the same bits on both cards after every decode pass that follows the prefill
  geometry   the prefill in chunks of 16 and in one chunk, and with the prompt cache's stops: the same cache rows and
             recurrent states on both cards, bit for bit, so the same logits after
  bounds     a span past max_ctx is refused before any write; the engine's poisoned flag refuses a prefill
  speed      (--speed L) prompt tok/s of a prefill of L positions from empty at chunk 2048 and 1024 (medians of 3; reported)
  tools/guard.sh LOG 1800 python -I tests/prefill/test_q35_prefill_tp.py GGUF --ref REF.npz [--kv f16|q8] [--llama LL.bin --llama-kv f16|q8] [--from P]
      [--speed L] [--ffn-share F]                          (HIP_VISIBLE_DEVICES=0,1: card 0 the XTX)"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q35_dense_tp import Dense35TP
from caminus.prefill.driver import PrefillRefused
from caminus.prefill.q35own_tp import Q35OwnPrefillTP, buffer_need_tp_mib, tp_shapes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--from', dest='P', type=int, default=64)
    ap.add_argument('--speed', type=int, default=0); ap.add_argument('--ffn-share', type=float, default=0.5)
    ap.add_argument('--kv', choices=('f16', 'q8'), default='f16')
    ap.add_argument('--llama-kv', choices=('f16', 'q8'), default='f16',
                    help='declared reference format; verify its stored run metadata too (legacy default:f16)')
    a = ap.parse_args(argv)
    if a.llama and a.llama_kv != a.kv:
        ap.error('--llama-kv must match --kv; reference provenance must also be verified')
    if a.P <= 0 or a.speed < 0 or not 0 < a.ffn_share < 1:
        ap.error('--from must be positive, --speed nonnegative, and --ffn-share between zero and one')
    return a


def state_views(eng, prefix):
    """Read-only complete state bytes, excluding unwritten KV and verify-tail rows."""
    if isinstance(prefix, (bool, np.bool_)) or not isinstance(prefix, (int, np.integer)) or not 0 <= prefix <= eng.max_ctx:
        raise ValueError('snapshot prefix must be within max_ctx')
    q8 = bool(getattr(eng, 'q8', False))
    pb = k3.kv_pos_bytes(eng.nkv // 2 * eng.hd, q8)
    out = []
    for key, value in sorted(eng.state.items()):
        if not value.is_contiguous(): raise ValueError('snapshot state must be contiguous')
        raw = value.view(torch.uint8).view(-1)
        if key[-1] == 'kv':
            if value.dtype != (torch.uint8 if q8 else torch.float16): raise ValueError('snapshot KV dtype mismatch')
            if raw.numel() % pb or raw.numel() < prefix * pb: raise ValueError('snapshot requires whole physical rows')
            raw = raw[:prefix * pb]
        out.append((key, raw))
    return out


def snapshot_state(eng, prefix):
    return [value.clone() for _, value in state_views(eng, prefix)]


def snapshot_nbytes(eng, prefix, card):
    return sum(value.numel() for key, value in state_views(eng, prefix) if key[0] == card)


def same_state(before, after):
    return len(before) == len(after) and all(torch.equal(a, b) for a, b in zip(before, after))


def main(argv=None):
    A = parse_args(argv)
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']; N, P = len(seq), A.P
    if ref.ndim != 2 or ref.shape[0] != N or not np.isfinite(ref).all() or not 0 < P < N:
        raise ValueError('reference logits must be finite and cover every token; --from must be within the sequence')
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
    ok = True


    def check(c, msg):
        nonlocal ok
        ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


    def lsm(v):
        v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


    def score(rows, cand):
        if rows.ndim != 2 or rows.shape != cand.shape or not len(rows) or not np.isfinite(cand).all():
            raise ValueError('quality rows must have matching finite shapes')
        kl = [float(np.sum(np.exp(lsm(r)) * (lsm(r) - lsm(c)))) for r, c in zip(rows, cand)]
        return float(np.mean(kl)), float(np.percentile(kl, 90)), float(np.max(kl)), int(sum(np.argmax(r) == np.argmax(c) for r, c in zip(rows, cand)))


    max_ctx = max(N + 64, A.speed + 64)
    CHUNK = 2048 if A.speed else 128
    eng = Dense35TP(A.gguf, max_ctx=max_ctx, ffn_share=A.ffn_share,
                    vram_extra_mib=buffer_need_tp_mib(tp_shapes(A.gguf, A.ffn_share), CHUNK), q8=A.kv == 'q8')
    pf = Q35OwnPrefillTP(eng, chunk=CHUNK)
    print(f'module {eng.module}; KV {A.kv}; ffn rows {eng.f0} / {eng.ffn - eng.f0}', flush=True)
    attn_L = [dd for dd in eng.L if dd[0]['kind'] == 'attn']; gdn_L = [dd for dd in eng.L if dd[0]['kind'] == 'gdn']
    nkvc = eng.nkv // 2


    def state_bytes():
        return snapshot_state(eng, P)


    def snap_bytes(c):
        return snapshot_nbytes(eng, P, c)


    snap_mib = [3 * snap_bytes(c) / 2 ** 20 + 2 for c in (0, 1)]   # (three snapshots are held at once: rows1, rows16, rowsst)
    capacity.check(list(eng.devs), snap_mib, f'test_q35_prefill_tp: its state snapshots ({snap_mib[0]:.0f} / {snap_mib[1]:.0f} MiB)')
    rep_bad = []


    def run_rows(toks):
        rows = []
        for t in toks:
            rows.append(eng.step(t).float().cpu().numpy())
            if not eng.check_replicas()[0]: rep_bad.append(eng.pos_host - 1)
        return np.stack(rows)


    def decode_only():
        eng.reset(); return run_rows(seq)


    def prefill_then_decode(chunk=None, stops=()):
        eng.reset()
        st = []
        pf.run(seq, 0, 0, P, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
        rows = state_bytes()
        eng.pos_host = P
        return run_rows(seq[P:]), rows, st


    dec = decode_only()
    ours, rows1, _ = prefill_then_decode()
    so, sd = score(ref[P:], ours), score(ref[P:], dec[P:])
    fmt = lambda s: f'mean KLD {s[0]:.3e} (p90 {s[1]:.3e}, max {s[2]:.3e}), top-1 {s[3]}/{N - P}'
    print(f'  two cards: prefill [0, {P}) then decode: {fmt(so)}', flush=True)
    print(f'  two cards: decode only, the same positions: {fmt(sd)}', flush=True)
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(N - P, -1)
        sl = score(ref[P:], ll)
        print(f'  llama.cpp prefill then decode: {fmt(sl)}', flush=True)
        check(so[0] <= sl[0], f'G-model: mean KLD vs float64 {so[0]:.3e} <= llama.cpp\'s prefill-then-decode {sl[0]:.3e}')
        check(so[3] >= sl[3], f'G-model: top-1 {so[3]} >= llama.cpp\'s {sl[3]}')
    else:
        print('NOT EVALUATED G-model against llama.cpp: no matching reference supplied', flush=True)
    check(so[0] <= 2.0 * sd[0] + 1e-4, f'prefill-then-decode KLD within 2x the decode-only engine\'s at the same positions + 1e-4 ({so[0]:.3e} vs {sd[0]:.3e})')
    o16, rows16, _ = prefill_then_decode(chunk=16)
    ost, rowsst, st = prefill_then_decode(stops=(16, 40, 41))
    check(same_state(rows1, rows16), 'chunks of 16: the same cache rows and recurrent states on both cards, bit for bit')
    check(np.array_equal(o16, ours), 'chunks of 16: the same logits after, bit for bit')
    check(same_state(rows1, rowsst) and np.array_equal(ost, ours) and st == [16, 40, 41], f'stops at 16, 40, 41 (at_stop {st}): the same states and logits')
    check(not rep_bad, 'the replicated residual is the same bits on both cards after every decode pass (after a prefill too)'
          + ('' if not rep_bad else f' (differs at {rep_bad[:8]})'))
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
        for chunk in (2048, 1024):
            if chunk > pf.chunk: continue
            ts = []
            for _ in range(3):
                eng.reset(); pf._sync_devices(); t0 = time.perf_counter()
                pf.run(toks, 0, 0, L_, chunk=chunk); pf._sync_devices(); ts.append(time.perf_counter() - t0)
            med = sorted(ts)[1]
            print(f'NOTE speed (prefill, two cards): {L_} positions from empty at chunk {chunk}: {1e3 * med:.0f} ms, {L_ / med:.0f} tok/s '
                  f'(runs {", ".join(f"{1e3 * t:.0f}" for t in ts)} ms)', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
