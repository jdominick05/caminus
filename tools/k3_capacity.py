"""K3 TP context capacity: the per-card VRAM budget table, the refusal check, and the measurement behind the fixed part.
  python -I tools/k3_capacity.py table [--model coder|reap] [--ntok 4] [--prefill 2048] [--split 10,12]   host only (sysfs +
        GGUF metadata): per-card budget at 32K / 128K / 262K, without and with the drafters (MTP, DFlash, both), each checked
        against the cap (the guard's line - margin, plus the card's current use); --split: the engine's gdn_keys,moe_blocks
        (FlashNextTP's defaults 10,12), which the prefill rows depend on
  guard.sh LOG SECS python -I tools/k3_capacity.py measure MAX_CTX NTOK [--model coder|reap] [--drafter mtp|dflash] [--ring N] [--no-idx] [--split GK,MB]
        loads FlashNextTP (vram_check off; with a drafter: its smoke factory, checks on), runs 3 passes and prints device-level
        used MiB per card (sysfs) idle / after the load / after the passes (guard.sh's log line has the peaks) and the
        context-sized buffers' MiB. --ring N: the DFlash ring's rows (default capacity.dflash_ring; the old sizing was the
        power of two >= max(4096, MAX_CTX)). --split GK,MB: the engine's TP split (no drafter; default 10,12), with the
        projected fixed part (capacity.fixed_mib) printed beside the measured one. --gen N (<= 64): then N tokens generated with the drafter at small positions
        (DFlash: its device-accept loop), so its own launches are in the peaks
See caminus/capacity.py for the per-position bytes and the measured fixed parts."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from caminus.device import capacity  # noqa: E402

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
MODELS = dict(coder=devpaths.coder(), reap=devpaths.reap())
SYSFS = ('/sys/class/drm/card1/device', '/sys/class/drm/card2/device')   # device 0 = the XTX (card1), 1 = the XT (card2), as guard.sh


def sysfs_mib(which):
    return [int(open(f'{d}/mem_info_vram_{which}').read()) / capacity.MIB for d in SYSFS]


def opt(a, name, default, f=str):
    if name in a:
        i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
    return default


def dflash_geom():
    """(layers, kv heads, head dim, window) of the DFlash drafter (GGUF metadata only; no GPU context)"""
    from caminus.engine import flashnext  # noqa: F401  (gguf-py on the path)
    from gguf import GGUFReader
    from caminus.decode.dflash import DRAFTER
    f = {x.name: x for x in GGUFReader(DRAFTER, 'r').fields.values()}
    hp = lambda k: int(f[f'dflash.{k}'].contents())
    return hp('block_count'), hp('attention.head_count_kv'), hp('attention.key_length'), hp('attention.sliding_window')


def table(a):
    model = opt(a, '--model', 'coder'); ntok = opt(a, '--ntok', 4, int); prefill = opt(a, '--prefill', 2048, int)
    split = tuple(int(v) for v in opt(a, '--split', '10,12').split(','))
    from caminus.prefill.feed import feed_shape      # (first: puts gguf-py on the path)
    from gguf import GGUFReader
    rd = GGUFReader(MODELS[model], 'r'); g = capacity.geometry(rd); dg = dflash_geom()
    try: fs = feed_shape(rd, MODELS[model])                # (the batched prefill's feed shapes from this file)
    except (AssertionError, KeyError) as e:
        fs = None; print(f'feed shapes: not a Coder feed ({type(e).__name__} {e}): the Coder\'s are counted (caminus.prefill.plan.CODER_FEED)')
    del rd
    total, used = sysfs_mib('total'), sysfs_mib('used')
    # the guard's own kill lines (capacity.GUARD_LINE_MIB by DRM card, as the runtime checks: capacity.caps_mib)
    cap = [min(capacity.GUARD_LINE_MIB[os.path.basename(os.path.dirname(d))], capacity.CAP_FRAC * t) - capacity.MARGIN_MIB for d, t in zip(SYSFS, total)]
    print(f'model {model}: {g}; T={ntok}; DFlash geometry (layers, kv heads, head dim, window) {dg}, ring '
          f'{capacity.dflash_ring(dg[3], ntok)} rows ({capacity.dflash_ring_bytes(*dg[:3], capacity.dflash_ring(dg[3], ntok)) / capacity.MIB:.0f} MiB, every max_ctx)')
    print(f'caps MiB (the guard\'s lines {capacity.GUARD_LINE_MIB} - {capacity.MARGIN_MIB}): XTX {cap[0]:.0f}, XT {cap[1]:.0f}; in use now (sysfs): XTX {used[0]:.0f}, XT {used[1]:.0f}')
    pp = {k: v / 4 for k, v in capacity.tp_ctx_bytes(g, 4).items()}
    print(f'per position per card: {sum(pp.values()):.0f} B (KV {pp["kv"]:.0f}, indexer raw {pp["idxraw"]:.0f}, pooled {pp["idxpool"]:.0f})')
    hdr = f'{"max_ctx":>8} {"drafters":<12} {"fixed XTX":>9} {"fixed XT":>8} {"ctx XTX":>8} {"ctx XT":>8} {"total XTX":>9} {"total XT":>8} {"free XTX":>8} {"free XT":>8}  verdict'
    print(hdr); print('-' * len(hdr))
    for n in (2048, 32768, 131072, 262144):
        for dr, pf in (((), None), (('mtp',), None), (('dflash',), None), (('mtp', 'dflash'), None), (('dflash',), prefill)):
            if pf == 0: continue
            b = capacity.budget(g, n, ntok, model, dr, dflash_geom=dg, prefill=pf, split=split, feed=fs)
            fixed = [sum(r[1 + c] for r in b['rows'] if 'fixed' in r[0]) for c in (0, 1)]
            ctx = [b['total'][c] - fixed[c] for c in (0, 1)]
            free = [cap[c] - used[c] - b['total'][c] for c in (0, 1)]
            over = [c for c in (0, 1) if free[c] < 0]
            v = 'OK' if not over else 'REFUSED (' + ', '.join(('XTX', 'XT')[c] for c in over) + ' over the cap)'
            if not b['fixed_known']: v += ' [fixed part unmeasured]'
            if 'mtp' in dr and n > capacity.QSA_EXACT: v += '; MTP past 2051: its own layer dense (drafts inexact, slower)'
            if pf: ctx = [ctx[c] - sum(r[1 + c] for r in b['rows'] if r[0].startswith('prefill')) for c in (0, 1)]
            print(f'{n:>8} {("+".join(dr) or "none") + (f"+pf{pf}" if pf else ""):<12} {fixed[0]:>9.0f} {fixed[1]:>8.0f} {ctx[0]:>8.0f} {ctx[1]:>8.0f} {b["total"][0]:>9.0f} '
                  f'{b["total"][1]:>8.0f} {free[0]:>8.0f} {free[1]:>8.0f}  {v}')
    print('MiB per card. "ctx" = the context-sized buffers: KV + indexer caches, + the drafters\' as coded (MTP: H rows of '
          'hc*D fp32 + f16 KV per position on both cards; DFlash: a fixed f16 K/V ring of capacity.dflash_ring rows on the XT, '
          'whatever max_ctx) and, past 2051 on the Coder, the QSA tapes\' buffers (capacity.tp_qsa_bytes). "+pfN": with the batched '
          'prefill\'s buffers for N-token chunks (capacity.prefill_need_mib at --split: GEMM workspace + two-card feed + glue workspace + '
          'the exchange\'s receive buffer + the prefill modules and streams; in the totals, not in ctx; --prefill N, 0 = no such rows). "free" subtracts the card\'s use now (the '
          'desktop, the driver, other processes), as the runtime checks do.')


def measure(a):
    model = opt(a, '--model', 'coder'); drafter = opt(a, '--drafter', None); ring = opt(a, '--ring', None, int); idx = '--no-idx' not in a
    ngen = opt(a, '--gen', 0, int)
    sp = opt(a, '--split', None, lambda v: tuple(int(x) for x in v.split(',')))
    if sp is not None and drafter is not None: print('--split: without --drafter'); sys.exit(2)
    if not 0 <= ngen <= 64 or (ngen and drafter is None): print('--gen N: 0 .. 64 tokens, with --drafter'); sys.exit(2)
    if not idx: a.remove('--no-idx')
    max_ctx, ntok = int(a[0]), int(a[1])
    idle = sysfs_mib('used')
    import torch
    from caminus.engine.flashnext_tp import FlashNextTP
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    t0 = time.time()
    if drafter == 'dflash':
        from caminus.decode import dflash as dk
        cfg = dict(gguf=MODELS[model])
        m, dr, _info = dk.smoke_factory(dict(cfg, drafter=dk.DRAFTER), ntok, max_ctx=max_ctx, ring=ring)
    elif drafter == 'mtp':
        from caminus.decode import mtp
        m, dr, _info = mtp.smoke_factory(dict(gguf=MODELS[model]), ntok, max_ctx=max_ctx)
    else:
        m = FlashNextTP(MODELS[model], ntok=ntok, max_ctx=max_ctx, idx_cache=idx, vram_check=False,
                        **(dict(gdn_keys=sp[0], moe_blocks=sp[1]) if sp else {}))
    for d in (0, 1): torch.cuda.synchronize(d)
    t_load = time.time() - t0
    loaded = sysfs_mib('used')
    toks = [248044, 760, 22414, 5285, 315, 13, 11, 13, 760, 22414, 5285, 315]
    for p in range(3):
        if ntok == 1: m.step(toks, p)
        else: m.verify(toks, p * ntok, 0 if p == 0 else ntok)
    for d in (0, 1): torch.cuda.synchronize(d)
    after = sysfs_mib('used'); gen = None
    if ngen:   # the drafter's own launches (device accept for DFlash): a short generation at small positions
        from caminus.decode import spec
        st = {}; t1 = time.time()
        out = dr.generate(m, toks, ngen, stats=st) if drafter == 'dflash' else spec.generate(m, toks, ngen, dr, st)
        for d in (0, 1): torch.cuda.synchronize(d)
        gen = sysfs_mib('used')
        print(f'  generated {len(out) - len(toks)} tokens with {drafter} in {time.time() - t1:.1f} s ({st.get("passes")} passes, accepted '
              f'{st.get("accepted")}); target bails {len(m.bails)}, drafter bails {len(getattr(getattr(dr, "df", dr), "bails", []))}; '
              f'tokens {out[len(toks):]}')
    for d in (0, 1): torch.cuda.empty_cache()
    steady = sysfs_mib('used')
    cb = capacity.tp_ctx_bytes(m.geom, max_ctx, idx); ctx = sum(cb.values()) / capacity.MIB
    print(f'model {model} max_ctx {max_ctx} T={ntok} idx {idx} drafter {drafter}' + (f' (ring {dr.df.ring} rows)' if drafter == 'dflash' else '')
          + f': load {t_load:.1f} s')
    for name, v in (('idle', idle), ('after load', loaded), ('after 3 passes', after)) + ((('after the generation', gen),) if gen else ()) + (('after empty_cache', steady),):
        print(f'  {name:<18} XTX {v[0]:8.0f} MiB  XT {v[1]:8.0f} MiB')
    print(f'  torch allocated / reserved: ' + ', '.join(f'dev {d}: {torch.cuda.memory_allocated(d) / capacity.MIB:.0f} / '
                                                       f'{torch.cuda.memory_reserved(d) / capacity.MIB:.0f}' for d in (0, 1)))
    print(f'  context-sized buffers per card: {ctx:.1f} MiB ({ {k: round(v / capacity.MIB, 1) for k, v in cb.items()} })')
    print(f'  fixed (after passes - idle - ctx): XTX {after[0] - idle[0] - ctx:.0f}  XT {after[1] - idle[1] - ctx:.0f} MiB '
          f'(the peak: guard.sh\'s line minus idle minus ctx)')
    if sp:
        pj = capacity.fixed_mib(model, ntok, capacity.model_bytes(m.rd), split=sp, tensors=capacity.split_tensors(m.rd))
        print(f'  split {sp[0]} / {sp[1]}: projected fixed (capacity.fixed_mib) XTX {pj[0]:.0f}  XT {pj[1]:.0f} MiB vs measured '
              f'{after[0] - idle[0] - ctx:.0f} / {after[1] - idle[1] - ctx:.0f}: ' + ('PROJECTION HOLDS' if all(pj[c] >= after[c] - idle[c] - ctx for c in (0, 1)) else 'PROJECTION LOW'))
    print(f'MEASURE idle={idle[0]:.0f},{idle[1]:.0f} ctx={ctx:.1f} after={after[0]:.0f},{after[1]:.0f}' + (f' gen={gen[0]:.0f},{gen[1]:.0f}' if gen else ''))


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    {'table': table, 'measure': measure}[a[0]](a[1:])
