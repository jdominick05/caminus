"""caminus.decode.longctx gates at <= 2051 positions (where K3's dense attention is the model's): GPU, under guard.sh, one stage per process.
  guard.sh LOG SECS python -I tests/decode/test_longctx.py plain|mtp|dflash [PROBES_DIR]
  [K3_TEST_HOOKS=ring-controls] guard.sh LOG SECS python -I tests/decode/test_longctx.py lc-ring PROBES_DIR D
plain:  fill(n) + teacher_forced(64) == rows n .. n + 63 of one teacher-forced run from position 0, bitwise, for n % 4 == 0
        and n % 4 != 0; A (logits + greedy), restore, B, restore, A again == the first A bitwise; the resumed greedy ==
        spec.generate from scratch on the same tokens.
mtp / dflash (host accept): a fill with the drafter attached, snapshot; spec A, restore, spec B, restore, spec A again: the
        same tokens, the same drafts pass for pass and the same acceptance; == plain greedy resumed, == spec.generate from
        scratch.
lc-mtp / lc-dflash D (D = 8192 / 32768: past 2051, the target runs the model's QSA): one fill of the depth-D context with the
        drafter attached, snapshot; per probe question (needle 50 %, rare 90 %): plain greedy resumed vs spec (host accept)
        resumed, the same tokens (N = 48); lc-dflash also the device-accept loop (DFlashK3Drafter.generate: its own prompt
        passes from scratch) == plain on the first probe; bails 0. Acceptance is reported (draft quality: past 2051 the MTP
        layer's own attention is dense, caminus.decode.mtp). The fill stops at a pass over 150 ms.
lc-ring D (D a multiple of 4096: 8192 / 32768): the DFlash ring of 4096 rows (A, the default) against the old sizing (B: the
        power of two >= max_ctx, a full ring) as two drafters on one target (Tee: both get every inject; each draft call asks
        both and compares their draft ids bitwise; the loop verifies A's). One fill to D - 8 (past the ring, 8 positions
        before a wrap), snapshot, then from it: (1) the two probe questions, spec (host accept) == plain; (2) the wrap: 160
        tokens from D - 4 with the drafts corrupted on every pass whose anchor is below D + 64 (0 / 1 / 2 drafts kept: j <=
        0 / 1 / 2, rejected rows across the wrap), == plain; (3) snapshot / restore: probe P (512 tokens), restore, a leg of
        3072 generated tokens (past D - 8 + 2049: A's window slots overwritten), restore, P again: the same tokens, the
        same drafts call for call, the same acceptance, == plain; (4) with K3_TEST_HOOKS=ring-controls (else skipped and
        said so): A's restore without its ring after a leg is refused (StaleRingError), and with the slot check off too
        A's drafts differ from (3)'s (the controls are not vacuous) while the tokens stay == plain; (5) the device-accept
        loop (DFlashK3Drafter.generate from scratch, its own prompt passes) for A and for B with the same corruption at the
        wrap: their drafts equal pass for pass, tokens == plain. Gate: every Tee call outside the controls A == B.
Every stage is policed by caminus.engine.passpolice: the per-pass limit applies to the time net of the KFD queue eviction provably
inside the pass's launches (host page compaction, db8bdfe), and the final "bails" check counts only bails that no eviction
explains; a POLICE summary (evicted passes, evicted ms, longest pass raw and net, compaction_proactiveness, every forgiven
pass and bail) is printed at exit, on a failure too. Positive controls (test-only, with K3_TEST_HOOKS=police-controls):
K3_TEST_STALL=<pass>:<ms> (a host stall inside a fill pass) and K3_TEST_BAIL=<pass> (a forced bail) must fail the stage.
Text: tools/eval/lc_probes.py's contexts (CPython stdlib source + synthetic needles; public)."""
import atexit, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'python'))
import torch
from caminus.decode import longctx, spec
from caminus.engine import passpolice

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
MODEL = devpaths.coder()
stage = sys.argv[1]
PD = sys.argv[2] if len(sys.argv) > 2 else devpaths.artifacts('qsa', 'lc')
ids = lambda p: [int(v) for v in open(os.path.join(PD, 'd1792', p)).read().split()]
ctx = ids('ctx.ids'); qA = ids('q_needle_10.ids')[1792:]; qB = ids('q_rare_50.ids')[1792:]
for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
fails = []
POL = []          # the engine's police (caminus.engine.passpolice): its summary is printed at exit, after a failure too
atexit.register(lambda: [print(p.summary(), flush=True) for p in POL])


def police(m):
    """attach the police before the engine's first pass (every launch attempt is seen)"""
    POL.append(passpolice.of(m)); return m


def check_bails(m, dr=None):
    """the bails check: every target bail explained by a KFD eviction inside its attempt (strict without the counters);
    the drafter's own bails are reported, not policed (as before)"""
    un = passpolice.of(m).unexplained_bails()
    check(not un, f'bails {len(m.bails)}: {len(m.bails) - len(un)} explained by a KFD eviction inside the bailed attempt, '
                  f'{len(un)} not' + (f' ({"; ".join(un)})' if un else ''))
    if dr is not None: print(f'drafter bails {len(getattr(dr, "bails", ()))} (reported, not policed)', flush=True)


def check(c, what):
    print(('ok   ' if c else 'FAIL ') + what, flush=True)
    if not c: fails.append(what)


def beq(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


class Tee:
    """two DFlashK3Drafter on one target (a: the 4096-row ring, b: the full-ring reference): inject / prefill / snapshot /
    restore / reset go to both; each draft call asks both (a first) and compares their draft ids; the loop verifies a's
    drafts, or corrupt(call number, anchor position, drafts)'s. mark() / since(): one segment's (calls, equal, a's drafts)"""
    def __init__(self, a, b):
        self.a, self.b, self.corrupt, self.n = a, b, None, 0
        self.max_pos = min(a.max_pos, b.max_pos); self.calls = []   # (anchor position, a's drafts, b's drafts)
        self._m = 0
    def reset(self): self.a.reset(); self.b.reset()
    def inject(self, i0, feats): self.a.inject(i0, feats); self.b.inject(i0, feats)
    def prefill(self, toks, n): self.a.prefill(toks, n); self.b.prefill(toks, n)
    def snapshot(self): return dict(a=self.a.snapshot(), b=self.b.snapshot())
    def restore(self, s): self.a.restore(s['a']); self.b.restore(s['b'])
    def __call__(self, toks, k):
        p = len(toks) - 1; da = list(self.a(toks, k)); db = list(self.b(toks, k))
        self.calls.append((p, da, db)); self.n += 1
        return da if self.corrupt is None else list(self.corrupt(self.n - 1, p, da))
    def mark(self): self._m = len(self.calls)
    def since(self):
        c = self.calls[self._m:]
        return dict(calls=len(c), equal=sum(a == b for _, a, b in c), drafts=[a for _, a, _ in c], pos=[p for p, _, _ in c],
                    first_diff=next(((p, a, b) for p, a, b in c if a != b), None))


class Log:
    """a drafter wrapper recording each call's drafts (the rest of the interface forwarded)"""
    def __init__(self, d): self.d, self.log = d, []
    def __call__(self, toks, k):
        r = list(self.d(toks, k)); self.log.append(r); return r
    def __getattr__(self, n): return getattr(self.d, n)


if stage == 'plain':
    from caminus.engine.flashnext_tp import FlashNextTP
    m = police(FlashNextTP(MODEL, ntok=4, max_ctx=2048))
    for n in (1500, 1501, 1503):
        ref = {}
        def keep(p, lo):
            for j in range(lo.shape[0]):
                if p + j >= n: ref[p + j] = lo[j].float().cpu()
        m.reset(); longctx.teacher_forced(m, dict(i0=0, commit=0), ctx, n + 64, rows=keep)
        st = longctx.fill(m, ctx, n, progress=False)
        got, _ = longctx.teacher_forced(m, st, ctx, 64)
        same = sum(beq(got[j], ref[n + j]) for j in range(64))
        check(same == 64, f'n={n} (n % 4 = {n % 4}): fill + teacher_forced(64) == one run from 0: {same}/64 rows bitwise '
                          f'(fill {st["tok_s"]:.0f} tok/s, longest pass {st["max_pass_ms"]:.1f} ms)')
    n = 1501; A, B = ctx[:n] + qA, ctx[:n] + qB
    st = longctx.fill(m, ctx, n, progress=False); snap = longctx.snapshot(m, st)
    print(f'snapshot: {snap["bytes"] / 2**20:.1f} MiB on the cards', flush=True)
    A1, _ = longctx.teacher_forced(m, st, ctx, 64)                       # straight after the fill
    s = longctx.restore(m, snap); gA1, _ = longctx.generate(m, s, A, 16)
    s = longctx.restore(m, snap); gB, _ = longctx.generate(m, s, B, 16)
    s = longctx.restore(m, snap); A2, _ = longctx.teacher_forced(m, s, ctx, 64)
    s = longctx.restore(m, snap); gA2, _ = longctx.generate(m, s, A, 16)
    check(beq(A1, A2), 'A (teacher forced 64) after fill == A after restore (B in between), bitwise')
    check(gA1 == gA2, f'greedy A after restore == greedy A after restore with B in between ({gA1[-16:]})')
    gS = spec.generate(m, A, 16, longctx.Repeat())
    check(gS == gA1, 'resumed greedy A == spec.generate(A) from scratch')
    check(gB != gA1, 'probe B differs from A (the probes are not vacuous)')
    check_bails(m)
elif stage in ('mtp', 'dflash'):
    if stage == 'mtp':
        from caminus.decode import mtp
        m, dr, _ = mtp.smoke_factory(dict(gguf=MODEL), 4)
    else:
        from caminus.decode import dflash as dk
        m, dr, _ = dk.smoke_factory(dict(gguf=MODEL, drafter=dk.DRAFTER), 4); dr.device_accept = False   # the generic loop
    police(m)
    # the suffix after the fill is a whole number of T-token passes: each probe takes (-len(q)) % 4 more context tokens before
    # its question (kept from when DFlashK3Drafter drafted only k = T - 1, so this gate's numbers stay comparable; any suffix
    # length is covered by tests/decode/test_spec_policy.py since impl:dflash-loop)
    n = 1501; N = 40
    A, B = (ctx[:n + (-len(q)) % 4] + q for q in (qA, qB))
    st = longctx.fill(m, ctx, n, draft=dr, progress=False); snap = longctx.snapshot(m, st, dr)
    L1, s1 = Log(dr), {}; gA1, _ = longctx.generate(m, st, A, N, draft=L1, stats=s1)
    s = longctx.restore(m, snap, dr); gB, _ = longctx.generate(m, s, B, N, draft=dr)
    s = longctx.restore(m, snap, dr); L2, s2 = Log(dr), {}; gA2, _ = longctx.generate(m, s, A, N, draft=L2, stats=s2)
    s = longctx.restore(m, snap, dr); gP, _ = longctx.generate(m, s, A, N)
    check(gA1 == gA2, f'{stage}: spec A after restore (B in between) == spec A after the fill: tokens')
    check(L1.log == L2.log, f'{stage}: the same drafts pass for pass ({len(L1.log)} draft calls)')
    check(s1['accepted'] == s2['accepted'] and s1['passes'] == s2['passes'],
          f'{stage}: the same acceptance ({s1["accepted"]} / {s2["accepted"]} accepted drafts, {s1["passes"]} / {s2["passes"]} passes)')
    check(gP == gA1, f'{stage}: resumed spec A == resumed plain greedy A')
    gS = spec.generate(m, A, N, dr)
    check(gS == gA1, f'{stage}: resumed spec A == spec.generate(A) from scratch')
    check(sum(map(len, L1.log)) > 0 and s1['accepted'] > 0, f'{stage}: drafts were accepted (not vacuous)')
    check_bails(m, dr)
elif stage in ('lc-mtp', 'lc-dflash'):
    D = int(sys.argv[3]); N = 48
    idd = lambda p: [int(v) for v in open(os.path.join(PD, f'd{D}', p)).read().split()]
    cD = idd('ctx.ids'); qs = [idd(f'{p}.ids')[D:] for p in ('q_needle_50', 'q_rare_90')]
    mc = (D + 256 + N + 16 + 255) // 256 * 256
    if stage == 'lc-mtp':
        from caminus.decode import mtp
        m, dr, _ = mtp.smoke_factory(dict(gguf=MODEL), 4, max_ctx=mc)
    else:
        from caminus.decode import dflash as dk
        m, dr, _ = dk.smoke_factory(dict(gguf=MODEL, drafter=dk.DRAFTER), 4, max_ctx=mc); dr.device_accept = False
    police(m)
    print(f'{stage} D={D}: target qsa {m.qsa} max_ctx {m.max_ctx}, modules {[e.module for e in m.engines]}; drafter max_pos {dr.max_pos}', flush=True)
    st = longctx.fill(m, cD, D, draft=dr, every=30.0, out=sys.stdout, max_pass_ms=150.0); snap = longctx.snapshot(m, st, dr)
    print(f'fill {D} with the drafter: {st["seconds"]:.1f} s ({st["tok_s"]:.0f} tok/s), longest pass {st["max_pass_ms"]:.1f} ms '
          f'({st["max_pass_net_ms"]:.1f} ms net of evictions; {st["evicted_passes"]} evicted passes, {st["evicted_ms"]} ms; police {st["police"]})', flush=True)
    plains = []
    for qi, q in enumerate(qs):
        A = cD[:D + (-len(q)) % 4] + q
        s = longctx.restore(m, snap, dr); gP, _ = longctx.generate(m, s, A, N); plains.append((A, gP))
        s = longctx.restore(m, snap, dr); s1 = {}; gS, _ = longctx.generate(m, s, A, N, draft=dr, stats=s1)
        check(gS == gP, f'{stage} D={D} probe {qi}: spec (host accept) == plain greedy, {N} tokens '
                        f'(accepted {s1["accepted"]} drafts in {s1["passes"]} passes)')
    if stage == 'lc-dflash':
        dr.device_accept = True; A, gP = plains[0]; s2 = {}
        gD = dr.generate(m, A, N, stats=s2)
        check(gD == gP, f'{stage} D={D}: device-accept loop (own prompt passes) == plain greedy ({s2["accepted"]} accepted in {s2["passes"]} passes)')
    check_bails(m, dr)
elif stage == 'lc-ring':
    import time
    from caminus.decode import dflash as dk
    from caminus.device import capacity
    from caminus.engine.flashnext_tp import FlashNextTP
    from gguf import GGUFReader
    D = int(sys.argv[3]); N, NP, L, NW = 48, 512, 3072, 160
    assert D % 4096 == 0, 'lc-ring: D a multiple of 4096 (a ring wrap at D)'
    idd = lambda p: [int(v) for v in open(os.path.join(PD, f'd{D}', p)).read().split()]
    cD = idd('ctx.ids'); qs = [idd(f'{p}.ids')[D:] for p in ('q_needle_50', 'q_rare_90')]
    Dw = D - 8; Pw = cD[:Dw + 4]
    mc = (Dw + 4 + L + NP + 64 + 255) // 256 * 256
    full = 4096
    while full < mc: full *= 2   # the old sizing: the power of two >= max(4096, max_pos)
    dg = dk.geom(dk.DRAFTER)
    layers = [int(v) for v in {x.name: x for x in GGUFReader(dk.DRAFTER, 'r').fields.values()}['dflash.target_layers'].contents()]
    m = police(FlashNextTP(MODEL, ntok=4, max_ctx=mc, feature_layers=layers, vram_drafters=(('dflash', dg, None), ('dflash', dg, full))))
    A = dk.DFlashK3Drafter(dk.DFlashK3(m, dk.DRAFTER)); B = dk.DFlashK3Drafter(dk.DFlashK3(m, dk.DRAFTER, ring=full))
    A.device_accept = B.device_accept = False; V = A.df.vocab
    check(A.df.ring == capacity.dflash_ring(dg[3], 4) == 4096 and B.df.ring == full, f'rings: A {A.df.ring}, B {B.df.ring}')
    print(f'lc-ring D={D}: target qsa {m.qsa} max_ctx {mc}; A ring {A.df.ring} rows ({A.df.ring_bytes() / 2**20:.0f} MiB), B ring {B.df.ring} '
          f'({B.df.ring_bytes() / 2**20:.0f} MiB); compaction_proactiveness {passpolice.proactiveness()}', flush=True)
    tee = Tee(A, B); gate = []   # the segments whose Tee calls must be A == B
    st = longctx.fill(m, cD, Dw, draft=tee, every=30.0, out=sys.stdout, max_pass_ms=150.0)
    print(f'fill {Dw} with both drafters: {st["seconds"]:.1f} s ({st["tok_s"]:.0f} tok/s), longest pass {st["max_pass_ms"]:.1f} ms '
          f'({st["max_pass_net_ms"]:.1f} ms net of evictions; {st["evicted_passes"]} evicted passes, {st["evicted_ms"]} ms; police {st["police"]})', flush=True)
    # cost of the two ways to restore a window: A's ring copy (snapshot + restore) vs re-injecting 2048 positions (512 inject
    # launches; timed as the last fill pass injected again: the same features, so the same rows)
    df = A.df; ti = []
    for _ in range(20):
        torch.cuda.synchronize(df.dx); t0 = time.perf_counter(); df.inject(Dw - 4); df.flush(); ti.append(time.perf_counter() - t0)
    tc = []
    for _ in range(5):
        t0 = time.perf_counter(); s_ = df.snapshot(); t1 = time.perf_counter(); df.restore(s_); tc.append((t1 - t0, time.perf_counter() - t1)); del s_
    ti.sort()
    print(f'restore cost: A ring copy snapshot {1e3 * sorted(x[0] for x in tc)[2]:.2f} ms + restore {1e3 * sorted(x[1] for x in tc)[2]:.2f} ms '
          f'(medians of 5, {df.ring_bytes() / 2**20:.0f} MiB on the XT) vs re-injecting the window: one inject launch {1e3 * ti[10]:.2f} ms '
          f'(median of 20) x 512 = {512 * ti[10]:.2f} s, plus {2048 * 5 * 2560 * 4 / 2**20:.0f} MiB of saved features', flush=True)
    t0 = time.perf_counter(); snap = longctx.snapshot(m, st, tee); print(f'snapshot (target + both drafters) {1e3 * (time.perf_counter() - t0):.1f} ms', flush=True)
    # (1) the probes
    for qi, q in enumerate(qs):
        P = cD[:Dw + (-len(q)) % 4] + q
        s = longctx.restore(m, snap, tee); gP, _ = longctx.generate(m, s, P, N)
        s = longctx.restore(m, snap, tee); s1 = {}; tee.mark(); gS, _ = longctx.generate(m, s, P, N, draft=tee, stats=s1); r = tee.since(); gate.append(r)
        check(gS == gP, f'(1) probe {qi}: spec (host accept) == plain greedy, {N} tokens (accepted {s1["accepted"]} drafts in {s1["passes"]} passes)')
        print(f'    probe {qi}: A == B on {r["equal"]} / {r["calls"]} draft calls', flush=True)
    # (2) the wrap: rejected rows across D
    def wrap_corrupt(n, p, d):
        if p >= D + 64: return d
        kp = n % 3; return d[:kp] + [(x + 1) % V for x in d[kp:]]
    s = longctx.restore(m, snap, tee); gPw, _ = longctx.generate(m, s, Pw, NW)
    s = longctx.restore(m, snap, tee); sw = {}; tee.mark(); tee.corrupt = wrap_corrupt
    gSw, _ = longctx.generate(m, s, Pw, NW, draft=tee, stats=sw); tee.corrupt = None; r = tee.since(); gate.append(r)
    nb = sum(1 for p in r['pos'] if p < D + 64); past = [(a == b) for p, a, b in tee.calls[len(tee.calls) - r['calls']:] if p >= D]
    check(gSw == gPw, f'(2) wrap: {NW} tokens from {Dw + 4} with corrupted drafts below {D + 64} ({nb} calls, anchors {min(r["pos"])} .. '
                      f'{max(p for p in r["pos"] if p < D + 64)}; accepted {sw["accepted"]} of {3 * sw["gen_passes"]} drafts) == plain')
    print(f'    wrap: A == B on {r["equal"]} / {r["calls"]} calls ({sum(past)} of {len(past)} at or past the wrap at {D})', flush=True)
    # (3) snapshot / restore across a leg that overwrites A's window slots
    s = longctx.restore(m, snap, tee); s1 = {}; tee.mark(); g1, _ = longctx.generate(m, s, Pw, NP, draft=tee, stats=s1); r1 = tee.since(); gate.append(r1)
    s = longctx.restore(m, snap, tee); tee.mark(); gL, stL = longctx.generate(m, s, Pw, L, draft=tee); rL = tee.since(); gate.append(rL)
    print(f'    leg: to position {stL["i0"]} (> {Dw} + 2049: A\'s window slots of {Dw} overwritten); A == B on {rL["equal"]} / {rL["calls"]} calls', flush=True)
    s = longctx.restore(m, snap, tee); s2 = {}; tee.mark(); g2, _ = longctx.generate(m, s, Pw, NP, draft=tee, stats=s2); r2 = tee.since(); gate.append(r2)
    s = longctx.restore(m, snap, tee); gPp, _ = longctx.generate(m, s, Pw, NP)
    check(g1 == g2 and r1['drafts'] == r2['drafts'] and s1['accepted'] == s2['accepted'] and s1['passes'] == s2['passes'],
          f'(3) restore after the leg: P again == P ({NP} tokens; {r1["calls"]} / {r2["calls"]} draft calls, the same drafts; accepted '
          f'{s1["accepted"]} / {s2["accepted"]} in {s1["passes"]} / {s2["passes"]} passes)')
    check(g1 == gPp, f'(3) P (spec) == plain greedy, {NP} tokens')
    # (4) the negative controls
    if os.environ.get('K3_TEST_HOOKS') == dk.RING_HOOKS:
        s = longctx.restore(m, snap, tee); longctx.generate(m, s, Pw, L, draft=tee)    # A's window slots overwritten again
        A.df.test_controls(no_ring_restore=True)
        s = longctx.restore(m, snap, tee)
        try: longctx.generate(m, s, Pw, NP, draft=tee); err = None
        except dk.StaleRingError as e: err = str(e)
        check(err is not None, f'(4) control: A restored without its ring after the leg is refused: {err}')
        A.df.test_controls(no_ring_restore=True, no_tag_check=True)
        s = longctx.restore(m, snap, tee); tee.mark(); g3, _ = longctx.generate(m, s, Pw, NP, draft=tee); r3 = tee.since()
        A.df.test_controls()
        nd = sum(a != b for a, b in zip(r1['drafts'], r3['drafts'])) + abs(len(r1['drafts']) - len(r3['drafts']))
        check(nd > 0, f'(4) control: with the slot check off too, A reads the stale ring: {nd} of {r1["calls"]} draft calls differ from (3) '
                      f'(A != B on {r3["calls"] - r3["equal"]} calls; the first at anchor {r3["first_diff"] and r3["first_diff"][0]})')
        check(g3 == gPp, '(4) control: the tokens stay == plain (drafts only change acceptance)')
    else:
        print(f'(4) controls SKIPPED (set K3_TEST_HOOKS={dk.RING_HOOKS} to run them)', flush=True)
    # (5) the device-accept loop from scratch, A and B, the same corruption at the wrap (by pass number: from position Dw + 3,
    # every pass advances 1 .. 3 positions, so the first 72 passes cross D)
    dev_corrupt = lambda n, d: d if n >= 72 else d[:n % 3] + [(x + 1) % V for x in d[n % 3:]]
    outs = {}
    for nm, dr in (('A', A), ('B', B)):
        dr.device_accept = True; dr.corrupt = dev_corrupt; dr.log = []; s3 = {}
        g = dr.generate(m, Pw, NW, stats=s3); outs[nm] = (g, dr.log, s3); dr.corrupt = None; dr.log = None; dr.device_accept = False
        print(f'    device accept {nm}: {s3["passes"]} passes, accepted {s3["accepted"]} of {3 * s3["gen_passes"]} drafts', flush=True)
    (gA, lA, sA), (gB, lB, sB) = outs['A'], outs['B']
    check(lA == lB and sA['accepted'] == sB['accepted'], f'(5) device accept: A\'s drafts == B\'s pass for pass ({sum(a == b for a, b in zip(lA, lB))} / '
                                                         f'{len(lA)} equal, {len(lB)} in B; accepted {sA["accepted"]} / {sB["accepted"]})')
    check(gA == gPw and gB == gPw, f'(5) device accept from scratch == plain greedy (A {gA == gPw}, B {gB == gPw})')
    calls = sum(r['calls'] for r in gate); eq = sum(r['equal'] for r in gate)
    check(calls > 0 and eq == calls, f'Tee: A == B (draft ids bitwise) on {eq} / {calls} host-loop draft calls (first difference '
                                     f'{next((r["first_diff"] for r in gate if r["first_diff"]), None)})')
    print(f'lc-ring D={D}: per-step draft agreement A vs B {eq} / {calls} (host loop) + {sum(a == b for a, b in zip(lA, lB))} / {len(lA)} '
          f'(device accept); bails target {len(m.bails)}, A {len(A.df.bails)}, B {len(B.df.bails)}', flush=True)
    check_bails(m, A)
else:
    print(__doc__); sys.exit(2)
print('PASS' if not fails else f'FAIL: {fails}')
sys.exit(1 if fails else 0)
