"""Gates of the batched prefill driver (docs/specs/spec-prefill-cache.md M5 S6, milestone A: correct and fast up to 2,051
tokens; caminus.prefill.coder, the ChunkStack with the attention core, the entry points longctx.fill / spec.generate /
DFlashK3Drafter.generate).

  tools/guard.sh LOG 1800 python -I tests/prefill/test_prefill_s6.py CODER STAGE... [--kv q8|f16] [--json F] [--small] [--reps 3]

--kv is TEST-ONLY, default q8. F16 supports the smoke / chunk / resume / DFlash / exact-hit / allocation stages below.
Stages using recorded Q8 llama.cpp numerical floors or performance baselines refuse F16; matching F16 baselines are
still required for the model and speed gates. State hashes cover the full selected target KV row in either mode.

One process, one engine for the GPU stages: FlashNextTP T = 4 at max_ctx 4096 (the Coder's QSA engine, dual tape: the dense
trunk below 2051, the QSA tape past it), the DFlash drafter (DFlashK3, features on the XTX), the prefill at chunk 2048.
compaction_proactiveness is printed with every timing.

Stages:
  cpu     CPU only (a stand-in engine, nothing runs): Prefill.validate's refusals (a token out of range anywhere in the
          prompt, the drafter's vocabulary, positions past max_ctx / QSA_EXACT / the drafter's rows, a poisoned engine, an
          MTP drafter) and its acceptance of the same prompts without the fault
  smoke   both cards: refusals on the loaded engine (a bad token, positions past Prefill.limit) leave every state and cache byte untouched; a 255-token prefill then one
          K3 pass: replicas equal, flags clear, the first logits close to K3's own decode
  g4a     G4a, the prefill's own error, like for like: K3 prefill-then-decode vs K3's own one-token decode (teacher-forced T = 4
          passes from position 0: bitwise one-token decode, tools/gates/tp_multi.py) on the same positions.
          short: the D-281 texts, for every p = 1 .. 255: reset, the batched prefill of [0, p) FORCED, token p through K3;
          765 positions, bar KLD <= 0.0068 and top-1 >= 96.6 %. routed: the same through longctx.fill(prefill=) with the
          measured crossover (p < T*: K3 passes, bitwise the reference; gated separately with the same bar). 2K: prefill
          N = 1792 / 2048 in one chunk, then 256 positions teacher-forced through K3 (2048: past 2051, the QSA tape on the
          prefill's IDXRAW rows); bars KLD <= the floor's bootstrap upper bound (0.00288 / 0.00137), top-1 >= 99.0 %;
          the max position and the mean without it reported.
          G4b (the same runs): vs llama.cpp's own prefill-then-decode at the same ub (short: ub512/sl_*, == ub 2048 by the
          stored control; 2K: ub2048/pd*_r1), bar KLD <= 0.0076, top-1 >= 95.9 %; vs llama.cpp's one-token decode reported,
          not gated. KLD: floor.py's math (tools/gates/g4_floor.py: float64, full vocabulary, KL(reference || ours)).
          --router exact / --wmma-out bf16 (the WMMA GEMMs rounded to bf16 as llama.cpp's batched op): the A/B switches
  g2b     the prefill of [0, 2048) in chunks of 2048 / 1024 / 512 / 777 + 777 + 494 and a 1000 + 1048 split: every state
          (GDN, conv, PLE: all of FlashNextTP.state but the position-indexed caches), the KV and IDXRAW rows < 2048, the
          DFlash features and the next 8 positions' K3 logits (the QSA tape) bitwise equal
  g3      G3-lite: a cold prefill of [0, 2048) vs a prefill to p (512 and 1001), a snapshot, a probe that rewrites rows >= p
          (another text's prefill), restore, the prefill of [p, 2048): states, KV / IDXRAW rows, the next 8 logits bitwise
  dflash  DFlash after a prefill: DFlashK3Drafter.generate(prefill=) == plain greedy after the same prefill (spec.generate
          with filler drafts, prefill=), bitwise, on 8 prompts x 96 new tokens; its acceptance against the same generation
          on a K3-filled context (no prefill): accepted drafts per pass, paired, within noise (|mean diff| <= 2 SE + 0.05)
  speed   end to end from empty, 512 / 1024 / 2048-token prompts: TTFT (reset, prefill of n - 1, the hand-off, the last
          token's K3 pass, the first token on the host) with and without the DFlash inject, tok/s = n / TTFT, median of
          --reps interleaved; the K3-only TTFT at 512 beside it; llama.cpp's llama-bench (spec S0) for reference.
          --moe ab: every prompt also through the other routed-expert path (ChunkStack.moe 'mmq' vs 'own', interleaved
          in the same rounds): the before / after of caminus.prefill.moe; --q6k ab: likewise the other dense Q6_K paths
          (ChunkStack.q6k own / wmma / mmq; --q6k X runs every stage on path X)
  long    a 3000-token prompt through the entry points: longctx.fill(prefill=) (batched to the engine's limit: the whole
          prompt on the QSA engine (S7), else 2051 and K3 passes from there) then 64 positions teacher-forced vs K3's own
          decode (KLD <= 0.0068, top-1 >= 99 %, replicas, not poisoned); DFlash after its prefill spec == plain bitwise,
          acceptance vs the K3-filled context; snapshot / restore with the drafter after a prefill of 2000: a probe with
          other given tokens, restore, the continuation == cold
  exact   M6 unit 2, the exact cache hit (a request with zero new tokens past a cached state) vs the cold run: prompts of
          29 .. 32 tokens (K3 passes, every residue mod 4), 255 .. 258 (batched, every residue) and 2100 (past 2051), plain and
          DFlash (generic loop): (a) the prompt-end checkpoint (spec.prompt_end), snapshot, a probe, restore, the request;
          (b) zero new tokens from a state at n with its last pass uncommitted (the loop reruns position n - 1); first logits
          bitwise and tokens equal; zero new tokens from a commit-0 state refused before anything launches (every byte
          unchanged); TTFT of the cold run, (a), (b) and (b) with stored logits (a lower bound)
The QSA prompt path (M5 S7; positions past 2051 batched on the QSA engine; --max-ctx 33024 for the depth stages):
  g2b7    G2b across the 2,051 boundary and at depth: [0, N) (N = 8192 when max_ctx allows, else 4000) of qsa/lc/d8192 in
          chunks of 2048 / 1024 / 777 and as runs [0, 2051) + rest, [0, 3000) + rest == one run in chunks of 2048, bitwise:
          every state, the KV / IDXRAW rows < N, the IDXPOOL rows < N / 4, the DFlash features of the export's window from
          position 3000 (the rows the last run writes in every variant), the next 8 K3 logits (QSA tape)
  g3q     G3-lite past 2051: a cold prefill of [0, N) vs a prefill to p (2500, 3001; and N - 1000 at 8K), a snapshot, a probe
          rewriting rows >= p, restore, the prefill of [p, N): states, KV / IDXRAW / IDXPOOL rows, the features the resume
          wrote, the next 8 logits bitwise
  g4d     G4 at depth N = 8192 and 32768 (when max_ctx >= N + 256): the batched prefill of [0, N), then positions N .. N + 255
          teacher-forced through K3. G4b vs llama.cpp's own prefill-then-decode at ub 2048 ($ARTIFACTS/impl-prefill-s7/g4,
          tools/gates/g4_floor.py depth-jobs): KLD <= 0.0076, top-1 >= 95.9 %. G4a vs K3's own one-token decode (teacher-
          forced T = 4 passes from position 0): KLD <= the floor's 95 % bootstrap upper bound (llama.cpp's prefill-then-
          decode vs its one-token decode at that depth: g4_depth.json), top-1 >= the floor's minus 1 point; vs llama.cpp's
          one-token decode reported. The max position and the mean without it reported
  probes  the copy probes of qsa/lc at 8K / 32K (needle and rare-token, 10 / 50 / 90 % depth): the prompt through the batched
          prefill (spec.generate(prefill=)), 24 greedy tokens; each answer found (or not) as llama.cpp's reference finds it
          (lc_baseline's rule); token-identical continuations and first divergences reported
  speed7  TTFT from empty at 4096 / 8192 / 32768 tokens (the prompt's n - 1 positions batched, the hand-off, the last token's
          K3 pass, the first token), with and without the DFlash inject, median of --reps interleaved; an 8K append at depth
          24576 (positions 24576 .. 32767, the deepest 8K window inside 32K); G5 and llama-bench (spec section 0) beside them;
          the VRAM check idle + projection >= measured after the 32K prefill with DFlash
  vram7   in a fresh process (run it alone or first): a 32768-token prompt (max_ctx - 256 at most) through
          DFlashK3Drafter.generate(prefill=) + 4 tokens, then idle + projection >= the device-level use per card (torch's
          allocated / reserved MiB beside it)
  alloc   allocation only (no prefill, no pass): the engine at --max-ctx (262144: the model's cap) + DFlash + the prefill's
          buffers; the device-level VRAM per card vs the guard lines (XTX 23,800 / XT 19,850 MiB) and idle + projection
  tstar   the crossover T*: the time to the first token for an append of L tokens, batched vs K3 passes, with the DFlash
          drafter, from empty and at depth 1024, interleaved medians; T* = the smallest L from which the batched path wins
          at both depths
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, contextlib, hashlib, json, os, sys, time
if '--graphs' in sys.argv: os.environ.setdefault('DEBUG_CLR_GRAPH_PACKET_CAPTURE', '0')   # (before HIP initializes: caminus.prefill.graph.ENV)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates'))
from caminus.engine import flashnext   # noqa: E402,F401  (puts gguf-py on the path)
import g4_floor as G           # noqa: E402  (the floor's KLD / stats: the same math)
from common import ARTIFACTS, D281_SHORT, LC, tokens   # noqa: E402

FAILS, NUMBERS = [], {}
G4 = os.path.join(ARTIFACTS, 'impl-prefill-s4', 'g4')
TEXTS = ('prose', 'code', 'shell')
CTX = {1792: os.path.join(LC, 'd1792', 'ctx.ids'), 2048: os.path.join(G4, 'ctx2304.ids')}
BAR_2K = {1792: 0.00288, 2048: 0.00137}     # the floor's 95 % bootstrap upper bounds at ub 2048 (spec G4)
S0 = {512: (1538.8, 'ub 256'), 2048: (2200.4, 'ub 512')}   # llama-bench best cells, spec S0 (no pp1024 cell was measured)


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(-1).view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def sha_rows(t, lo, hi):
    """sha() of t[:, lo:hi] without a device copy: the leading slices t[l, lo:hi] are contiguous and hashed in order, the same
    bytes as the contiguous copy (whose device allocation grew torch's free cache on the XT: M6 unit 1)"""
    torch = __import__('torch'); h = hashlib.sha256()
    for l in range(t.shape[0]): h.update(t[l, lo:hi].detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---- stage: cpu ---------------------------------------------------------------------------------------------------------
SPLIT = {}                       # FlashNextTP's TP split (--tp-split); empty: the engine's default 10 / 12
F16_STAGES = frozenset(('cpu', 'smoke', 'g2b', 'g3', 'dflash', 'exact', 'g2b7', 'g3q', 'alloc'))


def check_kv_stages(kv, stages):
    if kv == 'f16' and set(stages) - F16_STAGES:
        raise ValueError('F16 stages with recorded Q8 baselines are unavailable; matching F16 baselines are required')

class FakeEngine:
    def __init__(self): self.poisoned, self.max_ctx, self.ntok, self.seq, self.token = None, 4096, 4, 0, 0


class FakeDF:
    def __init__(self, vocab): self.emb_vocab = vocab


class FakeDFDrafter:
    def __init__(self, vocab, max_pos=4096): self.df, self.max_pos = FakeDF(vocab), max_pos
    def inject(self, i0, f): pass
    def prefill(self, t, n): pass
    def flush(self): pass


class FakeMTP:
    max_pos = 4096
    def inject(self, i0, f): pass
    def prefill(self, t, n): pass


def stage_cpu():
    from caminus.prefill import coder as P
    from caminus.engine import k3
    from caminus.decode.mtp import PolicyDrafter
    pf = P.Prefill.__new__(P.Prefill); pf.m = FakeEngine(); pf.n_vocab = 248320; pf.crossover = 16
    good = list(range(1000, 1300))
    def refused(what, fn):
        try: fn()
        except P.PrefillRefused as e: return check(True, f'cpu: refused before anything runs: {what} ({str(e)[:100]})')
        except Exception as e: return check(False, f'cpu: {what}: raised {type(e).__name__}: {e}')
        return check(False, f'cpu: {what}: NOT refused')
    def ok(what, fn):
        try: fn(); return check(True, f'cpu: accepted: {what}')
        except Exception as e: return check(False, f'cpu: {what}: refused ({type(e).__name__}: {e})')
    ok('a 300-token prompt, positions 0 .. 298', lambda: pf.validate(good, 0, 299))
    bad = list(good); bad[290] = 248320
    refused('a token past the vocabulary at position 290, outside the batched span [0, 100) (the whole prompt is checked)', lambda: pf.validate(bad, 0, 100))
    bad2 = list(good); bad2[3] = -1
    refused('a negative token id', lambda: pf.validate(bad2, 0, 299))
    refused('a token inside the target vocabulary but past the drafter\'s', lambda: pf.validate([5, 6, 7, 9000], 0, 3, draft=FakeDFDrafter(8000)))
    ok('the same prompt with the drafter\'s vocabulary covering it', lambda: pf.validate([5, 6, 7, 900], 0, 3, draft=FakeDFDrafter(8000)))
    refused('positions reaching QSA_EXACT (2051)', lambda: pf.validate(list(range(3000)), 0, 2052))
    ok('positions up to 2050 (the last dense position)', lambda: pf.validate(list(range(3000)), 0, 2051))
    pf.m.max_ctx = 1024
    refused('positions past max_ctx', lambda: pf.validate(list(range(3000)), 0, 1025)); pf.m.max_ctx = 4096
    refused('positions past the drafter\'s rows', lambda: pf.validate(list(range(3000)), 0, 600, draft=FakeDFDrafter(248320, max_pos=512)))
    refused('an empty span', lambda: pf.validate(good, 5, 5))
    refused('a span past the prompt', lambda: pf.validate(good, 0, 301))
    refused('an MTP drafter (no per-pass target rows from a prefill)', lambda: pf.validate(good, 0, 299, draft=FakeMTP()))
    refused('a PolicyDrafter holding an MTP drafter', lambda: pf.validate(good, 0, 299, draft=PolicyDrafter({'dflash': FakeDFDrafter(248320), 'mtp': FakeMTP()})))
    ok('a PolicyDrafter holding DFlash only', lambda: pf.validate(good, 0, 299, draft=PolicyDrafter({'dflash': FakeDFDrafter(248320)})))
    pf.m.poisoned = 'test'
    refused('a poisoned engine', lambda: pf.validate(good, 0, 299)); pf.m.poisoned = None
    # S7: a QSA engine's limit is its max_ctx (the QSA prompt path); a prompt past the feature export needs a drafter that can skip
    pf.m.qsa = True; pf.m.max_ctx = 33024
    ok('a QSA engine: positions 0 .. 33023 (its max_ctx)', lambda: pf.validate(list(range(1000, 1000 + 33024)), 0, 33024))
    refused('a QSA engine: positions past its max_ctx', lambda: pf.validate(list(range(1000, 1000 + 33100)), 0, 33025))
    refused('a prompt past the feature export with a drafter that cannot skip_inject', lambda: pf.validate(list(range(1000, 9000)), 0, 7999, draft=FakeDFDrafter(248320, max_pos=33024)))
    sk = FakeDFDrafter(248320, max_pos=33024); sk.skip_inject = lambda p: None
    ok('the same prompt with a drafter that skips (DFlashK3Drafter.skip_inject)', lambda: pf.validate(list(range(1000, 9000)), 0, 7999, draft=sk))
    ok('a prompt within the feature export with a drafter that cannot skip', lambda: pf.validate(list(range(1000, 3000)), 0, 1999, draft=FakeDFDrafter(248320, max_pos=33024)))
    check(pf.limit == 33024, f'cpu: Prefill.limit on a QSA engine == its max_ctx ({pf.limit})')
    pf.m.qsa = False; pf.m.max_ctx = 4096
    check(pf.limit == 2051, f'cpu: Prefill.limit without QSA == QSA_EXACT ({pf.limit})')
    check(pf.route(16) and not pf.route(15), 'cpu: route(): the batched path from the crossover on, K3 passes below it')
    from caminus.decode import spec
    refused('a short prompt routed to K3 passes (below T*) with a bad token: spec.prefill_span checks the whole prompt',
            lambda: spec.prefill_span(pf, [5, 6, 248320, 7, 8], None))
    ok('the same short prompt without the bad token: routed to K3 (span 0)', lambda: spec.prefill_span(pf, [5, 6, 9, 7, 8], None) == 0 or (_ for _ in ()).throw(AssertionError('span')))
    stage_cpu_exact()


class PassLog:
    """a stand-in engine for the exact-hit refusals: records every pass, launches nothing"""
    def __init__(self, T=4, max_ctx=4096):
        self.ntok, self.max_ctx, self.qsa_limit, self.poisoned, self.passes, self.features = T, max_ctx, None, None, [], None
    def verify_tokens(self, seq, i0, commit):
        self.passes.append((i0, commit, len(seq))); return [7] * self.ntok
    def verify(self, seq, i0, commit):
        import torch
        self.passes.append((i0, commit, len(seq))); return torch.zeros(self.ntok, 16)
    def reset(self): pass


def stage_cpu_exact():
    """M6 unit 2: requests with zero new tokens (an exact cache hit) and invalid resume states, on the host: refused before
    any pass (spec.resume_point, ExactHitRefused), or the last pass's uncommitted row of position n - 1 rerun"""
    from caminus.decode import spec, longctx
    toks = list(range(100, 140))
    def refused(what, fn):
        m = PassLog()
        try: fn(m)
        except spec.ExactHitRefused as e: return check(not m.passes, f'cpu exact: refused, nothing launched: {what} ({str(e)[:90]})')
        except Exception as e: return check(False, f'cpu exact: {what}: raised {type(e).__name__}: {e}')
        return check(False, f'cpu exact: {what}: NOT refused (passes {m.passes})')
    refused('zero new tokens from a committed state (i0 = 40, commit 0: a fill / prefill of every token)',
            lambda m: longctx.generate(m, dict(i0=40, commit=0), toks, 8))
    refused('zero new tokens, commit 0, through spec.loop', lambda m: spec.loop(m, toks, 40, 0, 48, longctx.Repeat()))
    refused('zero new tokens, commit 0, through longctx.first_logits', lambda m: longctx.first_logits(m, dict(i0=40, commit=0), toks))
    refused('a state past the request (i0 = 44 > 40 tokens)', lambda m: longctx.generate(m, dict(i0=44, commit=4), toks, 8))
    refused('a lazy commit past T (5 > 4)', lambda m: longctx.generate(m, dict(i0=36, commit=5), toks, 8))
    refused('a lazy commit past i0 (3 > i0 2)', lambda m: spec.loop(m, toks, 2, 3, 48, longctx.Repeat()))
    refused('an empty request', lambda m: spec.loop(m, [], 0, 0, 4, longctx.Repeat()))
    for c in (1, 2, 4):
        m = PassLog(); out, st = longctx.generate(m, dict(i0=40, commit=c), toks, 3)
        check(m.passes[0][:2] == (39, c - 1) and len(out) == 43,
              f'cpu exact: zero new tokens with commit {c}: the first pass reruns position 39 (i0 39, commit {c - 1}; got {m.passes[0][:2]}), '
              f'then {len(out) - 40} tokens')
    m = PassLog(); lg, st = longctx.first_logits(m, dict(i0=40, commit=2), toks)
    check(m.passes == [(39, 1, 43)] and st['i0'] == 40 and st['commit'] == 1, f'cpu exact: first_logits of an exact hit: one pass at 39 (commit 1), state (40, 1) ({m.passes}, {st})')
    m = PassLog(); st = spec.prompt_end(m, toks[:31], None, None)
    check(st == dict(i0=28, commit=4) and [p[:2] for p in m.passes] == [(i, 0 if i == 0 else 4) for i in range(0, 28, 4)],
          f'cpu exact: prompt_end of a 31-token prompt on K3 passes stops before the last prompt pass: state {st}, passes at {[p[0] for p in m.passes]}')


# ---- the engine ---------------------------------------------------------------------------------------------------------
class Rig:
    def __init__(self, torch, coder, max_ctx=4096, chunk=2048, router=None, wmma_out=None, moe=None, q6k=None, q6k_buffers=None,
                 plan_bf16=None, q8=True):
        from gguf import GGUFReader
        from caminus.decode import dflash as dk
        from caminus.prefill import coder as P
        from caminus.engine.flashnext_tp import FlashNextTP
        self.torch = torch
        rd = GGUFReader(dk.DRAFTER, 'r'); f = {x.name: x for x in rd.fields.values()}
        layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
        dg = dk.geom(dk.DRAFTER)
        self.idle = vram_used()
        t0 = time.time()
        # (the prefill as it is built below, counted before the target loads; plan_bf16: the 'wmma' buffers counted now and
        # allocated later, by stage_speed's dense-path A/B: caminus.prefill.plan.check_counted)
        self.m = FlashNextTP(coder, ntok=4, max_ctx=max_ctx, q8=q8, feature_layers=layers, feature_card=0, vram_drafters=(('dflash', dg, None),), **SPLIT,
                             vram_prefill=dict(chunk=chunk, q6k=q6k, q6k_buffers=q6k_buffers if plan_bf16 is None else plan_bf16))
        self.df = dk.DFlashK3(self.m, dk.DRAFTER); self.dr = dk.DFlashK3Drafter(self.df)
        self.pf = P.Prefill(self.m, chunk=chunk, router_logits=router, wmma_out=wmma_out, moe=moe, q6k=q6k, q6k_buffers=q6k_buffers)
        for d in (0, 1): torch.cuda.synchronize(d)
        print(f'rig: loaded in {time.time() - t0:.1f} s; target KV {"q8" if q8 else "f16"}; qsa {self.m.qsa} dual {self.m.dual}; feature layers {layers}; vram plan prefill '
              f'{[round(x) for x in self.m.vram_plan["prefill_mib"]]} MiB; crossover T* {self.pf.crossover}; torch free cache released at construction {[round(x, 1) for x in self.pf.released_mib]} MiB (tape addresses checked first: {__import__('caminus.device.capacity', fromlist=['x']).LAST_RELEASE}); router logits {self.pf.S.Gl[0].router_mode}; WMMA out {self.pf.S.Gl[0].wmma_out}; routed experts {self.pf.S.moe}; dense Q6_K {self.pf.S.q6k}', flush=True)

    def sync(self):
        for d in (0, 1): self.torch.cuda.synchronize(d)

    def ref(self, toks, lo, hi):
        """K3's own decode logits of positions [lo, hi) (teacher-forced T = 4 passes from position 0, after a reset): float32 [n, V]"""
        from caminus.decode import longctx
        m = self.m; m.reset(); keep = []
        def rows(i0, lg):
            a, b = max(i0, lo), min(i0 + lg.shape[0], hi)
            if a < b: keep.append(lg[a - i0:b - i0].float().cpu().numpy())
        longctx.teacher_forced(m, dict(i0=0, commit=0), toks, hi, rows=rows)
        return np.concatenate(keep)

    def after_prefill(self, toks, p, n_pos, chunk=None):
        """reset, the batched prefill of [0, p) (forced), then n_pos positions teacher-forced through K3: float32 [n_pos, V]"""
        from caminus.decode import longctx
        m = self.m; m.reset()
        st = self.pf.run(toks, 0, 0, p, chunk=chunk)
        lg, _ = longctx.teacher_forced(m, dict(i0=st['i0'], commit=0), toks, n_pos)
        return lg.numpy()

    def states(self, n, feat_from=0):
        """hashes of every state: the recurrent ones whole, the KV / IDXRAW rows < n, the IDXPOOL rows < n / 4 (the prefill
        pools from token 0, S7), the features of [max(feat_from, n - feat_rows, 0), n) within the export's window (the last run's)"""
        m = self.m; out = {}
        from caminus.engine import k3
        pb = k3.kv_pos_bytes(m.hd, m.q8)   # one KV head per TP card; state storage is raw bytes in either mode
        for k, t in m.state.items():
            if k[-1] == 'kv': out[k] = sha(t[:n * pb])
            elif k[-1] == 'idxraw': out[k] = sha_rows(t, 0, n)
            elif k[-1] == 'idxpool': out[k] = sha_rows(t, 0, n // 4)
            else: out[k] = sha(t)
        S = self.pf.S; lo = max(int(feat_from), n - S.feat_rows, 0, S.feat_base); hi = min(n, S.feat_base + S.feat_rows)
        out['features'] = sha_rows(S.AW[S.fcard].feats, lo - S.feat_base, hi - S.feat_base) if hi > lo else ''
        return out


def vram_used():
    """device-level MiB used [XTX, XT] (sysfs: every process)"""
    from caminus.device import capacity
    return [int(open(f'/sys/class/drm/{c}/device/mem_info_vram_used').read()) / capacity.MIB for c in ('card1', 'card2')]


def stats_of(ref, oth, pos):
    s = G.stats(ref, oth, pos)
    ks = np.array(s['kld']); i = int(ks.argmax())
    s['max_pos'] = int(pos[i]); s['mean_wo_max'] = float(np.delete(ks, i).mean()) if len(ks) > 1 else 0.0
    return s


def short_line(name, s):
    return G.line(name, s) + f'; max at {s["max_pos"]}, mean without it {s["mean_wo_max"]:.5f}'


def slim(s):
    return {k: v for k, v in s.items() if k not in ('kld', 'agree')}


# ---- stage: smoke -------------------------------------------------------------------------------------------------------
def stage_smoke(R):
    torch = R.torch; m, pf = R.m, R.pf
    from caminus.prefill import coder as P
    toks = tokens(os.path.join(D281_SHORT, 'e2e_code_decode', 'tokens.txt'))
    m.reset(); h0 = R.states(4096)
    bad = list(toks); bad[200] = 10 ** 6
    for what, fn in (('a token out of range at position 200', lambda: pf.fill(bad, 255)),
                     (f'positions past the limit {pf.limit}' + (' (the QSA engine\'s max_ctx, S7)' if m.qsa else ' (QSA_EXACT)'),
                      lambda: pf.fill(list(range(100, 100 + pf.limit + 8)), pf.limit + 1)),
                     ('an out-of-range token through spec.generate', lambda: __import__('caminus.decode.spec', fromlist=['x']).generate(m, bad[:255], 4, __import__('caminus.decode.longctx', fromlist=['x']).Repeat(), prefill=pf))):
        try: fn(); okr = False
        except P.PrefillRefused: okr = True
        R.sync()
        check(okr and R.states(4096) == h0, f'smoke: {what} refused before anything ran: every state, KV / IDXRAW row and feature byte unchanged')
    ref = R.ref(toks, 0, 256)
    t0 = time.perf_counter(); lg = R.after_prefill(toks, 255, 1); dt = time.perf_counter() - t0
    r0, r1 = m.check_replicas()
    s = stats_of(ref[255:256], lg, [255])
    R.sync(); used = vram_used(); vp = m.vram_plan
    proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(R.idle[c] + proj[c] >= used[c], f'smoke VRAM, {nm}: idle {R.idle[c]:.0f} + projected {proj[c]:.0f} (prefill {vp["prefill_mib"][c]:.0f}, '
              f'drafters {vp["drafters_mib"][c]:.0f}) = {R.idle[c] + proj[c]:.0f} MiB >= the measured device-level use {used[c]:.0f} MiB after a prefill '
              f'(margin {R.idle[c] + proj[c] - used[c]:+.0f})')
    NUMBERS['vram'] = dict(idle=R.idle, projected=proj, measured=used)
    check(r0 and r1 and m.poisoned is None and bool(np.isfinite(lg).all()),
          f'smoke: prefill [0, 255) + K3 pass at 255 ({dt * 1e3:.0f} ms): replicas {r0}/{r1}, not poisoned, logits finite; KLD vs K3 decode '
          f'{s["kld_mean"]:.5f}, top-1 {"==" if s["top1"] == 100 else "!="}')


# ---- stage: g4a / g4b ---------------------------------------------------------------------------------------------------
def stage_g4(R, small):
    from caminus.decode import longctx
    m, pf = R.m, R.pf
    out = {}
    # short window
    parts, parts_rt, parts_b, parts_l1 = [], [], [], []
    routed_bitwise = []
    for t in TEXTS:
        toks = tokens(os.path.join(D281_SHORT, f'e2e_{t}_decode', 'tokens.txt'))
        ps = list(range(1, 256 if not small else 9))
        ref = R.ref(toks, 0, 256)
        lc_pd = np.memmap(os.path.join(G4, 'ub512', f'sl_{t}.bin'), dtype=np.float32, mode='r').reshape(255, G.NV)
        lc_1t = G.dump_rows(t, ps)
        got, got_rt = [], []
        t0 = time.perf_counter()
        for p in ps:
            got.append(R.after_prefill(toks, p, 1)[0])
            m.reset(); st = longctx.fill(m, toks, n=p, prefill=pf, progress=False)
            lg, _ = longctx.teacher_forced(m, st, toks, 1); got_rt.append(lg.numpy()[0])
            if p < pf.crossover: routed_bitwise.append(np.array_equal(got_rt[-1].view(np.int32), ref[p].view(np.int32)))
        got, got_rt = np.stack(got), np.stack(got_rt)
        dt = time.perf_counter() - t0
        s = stats_of(ref[ps], got, ps); srt = stats_of(ref[ps], got_rt, ps)
        sb = stats_of(lc_pd[[p - 1 for p in ps]], got, ps); s1 = stats_of(lc_1t, got, ps)
        parts.append(s); parts_rt.append(srt); parts_b.append(sb); parts_l1.append(s1)
        print('  ' + short_line(f'G4a short {t} (forced)', s) + f'  [{dt:.0f} s for {len(ps)} x (prefill + K3 pass) twice]', flush=True)
        print('  ' + short_line(f'G4a short {t} (routed)', srt), flush=True)
        print('  ' + short_line(f'G4b short {t} vs llama.cpp prefill-then-decode', sb), flush=True)
        print('  ' + short_line(f'(reported) short {t} vs llama.cpp one-token decode', s1), flush=True)
    for name, pp in (('forced', parts), ('routed', parts_rt), ('g4b', parts_b), ('vs_llama_1tok', parts_l1)):
        a = G.merge(pp); ks = np.array(a['kld']); i = int(ks.argmax()); a['mean_wo_max'] = float(np.delete(ks, i).mean())
        out[f'short_{name}'] = slim(a)
    sf, sr, sb = out['short_forced'], out['short_routed'], out['short_g4b']
    # reported, not gated (user 11-04): with the bf16 WMMA outputs (llama.cpp's batched numerics, the default) the prefill moves toward
    # llama.cpp's own prefill and slightly away from K3's own decode; llama.cpp's own prefill-vs-decode seam is 0.0068 / 97.6 %.
    print(f'  (reported, not gated) G4a short, the batched prefill forced for every p: {sf["n"]} positions, mean KLD {sf["kld_mean"]:.5f} '
          f'(llama.cpp\'s own seam 0.0068), top-1 {sf["top1"]:.2f} % (96.6), p99 {sf["kld_p99"]:.4f}, max {sf["kld_max"]:.4f}', flush=True)
    check(sr['kld_mean'] <= 0.0068 and sr['top1'] >= 96.6 and all(routed_bitwise),
          f'G4a short, routed (longctx.fill(prefill=), T* {pf.crossover}): mean KLD {sr["kld_mean"]:.5f}, top-1 {sr["top1"]:.2f} %; the '
          f'{len(routed_bitwise)} positions below T* (K3 passes) bitwise the reference: {sum(routed_bitwise)} / {len(routed_bitwise)}')
    check(sb['kld_mean'] <= 0.0076 and sb['top1'] >= 95.9, f'G4b short vs llama.cpp\'s own prefill-then-decode (ub 512 == 2048): mean KLD {sb["kld_mean"]:.5f} '
          f'(bar 0.0076), top-1 {sb["top1"]:.2f} % (bar 95.9)')
    print(f'  (reported, not gated) short vs llama.cpp one-token decode: mean KLD {out["short_vs_llama_1tok"]["kld_mean"]:.5f}, '
          f'top-1 {out["short_vs_llama_1tok"]["top1"]:.2f} %', flush=True)
    # 2K windows
    for N in ((1792,) if small else (1792, 2048)):
        toks = tokens(CTX[N]); npos = 16 if small else 256; pos = list(range(N, N + npos))
        t0 = time.perf_counter(); ref = R.ref(toks, N, N + npos); t1 = time.perf_counter()
        got = R.after_prefill(toks, N, npos); t2 = time.perf_counter()
        lc_pd = np.memmap(os.path.join(G4, 'ub2048', f'pd{N}_r1.bin'), dtype=np.float32, mode='r').reshape(256, G.NV)[:npos]
        lc_1t = np.memmap(os.path.join(G4, 'ub512', f'ref{N}.bin'), dtype=np.float32, mode='r').reshape(256, G.NV)[:npos]
        s, sb, s1 = stats_of(ref, got, pos), stats_of(lc_pd, got, pos), stats_of(lc_1t, got, pos)
        print('  ' + short_line(f'G4a 2K N={N}', s) + f'  [ref {t1 - t0:.0f} s, prefill + decode {t2 - t1:.0f} s]', flush=True)
        print('  ' + short_line(f'G4b 2K N={N} vs llama.cpp prefill-then-decode ub 2048', sb), flush=True)
        print('  ' + short_line(f'(reported) 2K N={N} vs llama.cpp one-token decode', s1), flush=True)
        out[f'2k_{N}'] = slim(s); out[f'2k_{N}_g4b'] = slim(sb); out[f'2k_{N}_vs_llama_1tok'] = slim(s1)
        check(s['kld_mean'] <= BAR_2K[N] and s['top1'] >= 99.0, f'G4a 2K N={N} (chunk 2048): mean KLD {s["kld_mean"]:.5f} (bar {BAR_2K[N]}), top-1 {s["top1"]:.2f} % '
              f'(bar 99.0); max {s["kld_max"]:.4f} at {s["max_pos"]}, mean without it {s["mean_wo_max"]:.5f}')
        check(sb['kld_mean'] <= 0.0076 and sb['top1'] >= 95.9, f'G4b 2K N={N} vs llama.cpp\'s prefill-then-decode at ub 2048: mean KLD {sb["kld_mean"]:.5f} '
              f'(bar 0.0076), top-1 {sb["top1"]:.2f} %')
    NUMBERS['g4'] = out


# ---- stage: g2b / g3 ----------------------------------------------------------------------------------------------------
def tail(R, toks, n, k=8):
    from caminus.decode import longctx
    lg, _ = longctx.teacher_forced(R.m, dict(i0=n, commit=0), toks, k)
    return lg.numpy()


def stage_g2b(R):
    m, pf = R.m, R.pf
    toks = tokens(CTX[2048]); N = 2048
    def one(splits):
        m.reset(); a = 0
        for b, c in splits: pf.run(toks, a, 0, b, chunk=c); a = b
        R.sync(); st = R.states(N); lg = tail(R, toks, N); return st, lg
    base_st, base_lg = one([(N, 2048)])
    for name, splits in (('chunk 1024', [(N, 1024)]), ('chunk 512', [(N, 512)]), ('chunk 777 (777 + 777 + 494)', [(N, 777)]),
                         ('runs [0, 1000) + [1000, 2048), chunk 2048', [(1000, 2048), (N, 2048)])):
        st, lg = one(splits)
        diff = sorted(str(k) for k in base_st if st[k] != base_st[k])
        check(not diff and np.array_equal(lg.view(np.int32), base_lg.view(np.int32)),
              f'G2b: [0, 2048) {name} == one chunk of 2048, bitwise: every state, KV / IDXRAW row and feature ({len(base_st)} items'
              + (f'; differ: {diff[:4]}' if diff else '') + '), the next 8 K3 logits (QSA tape)')


def stage_g3(R):
    from caminus.decode import longctx
    m, pf = R.m, R.pf
    toks = tokens(CTX[2048]); N = 2048
    other = tokens(os.path.join(D281_SHORT, 'e2e_prose_decode', 'tokens.txt'))
    m.reset(); pf.run(toks, 0, 0, N); R.sync(); cold_st = R.states(N); cold_lg = tail(R, toks, N)
    for p in (512, 1001):
        m.reset(); st = pf.run(toks, 0, 0, p)
        snap = longctx.snapshot(m, st)
        probe = list(toks[:p]) + list(other) * 4
        pf.run(probe, p, 0, p + 900)                                   # rewrites the KV / IDXRAW / feature rows p .. p + 899
        st2 = longctx.restore(m, snap); longctx.release(snap)          # (done with it: its copies back to the driver)
        pf.run(toks, st2['i0'], st2['commit'], N); R.sync()
        rs = R.states(N); lg = tail(R, toks, N)
        diff = sorted(str(k) for k in cold_st if rs[k] != cold_st[k])
        check(not diff and np.array_equal(lg.view(np.int32), cold_lg.view(np.int32)),
              f'G3-lite: prefill to {p}, snapshot, a probe rewriting rows {p} .. {p + 899}, restore, resume to {N} == the cold prefill of {N}, bitwise '
              f'(every state, KV / IDXRAW row and feature; the next 8 logits)' + (f'; differ: {diff[:4]}' if diff else ''))


# ---- stage: dflash ------------------------------------------------------------------------------------------------------
def prompts(small):
    P = []
    for t in TEXTS:
        toks = tokens(os.path.join(D281_SHORT, f'e2e_{t}_decode', 'tokens.txt'))
        P += [(f'{t}[:128]', toks[:128]), (f'{t}[:255]', toks[:255])]
    P += [('d1792[:1792]', tokens(CTX[1792])[:1792]), ('ctx2304[:2000]', tokens(CTX[2048])[:2000])]
    return P[:2] if small else P


def stage_dflash(R, small):
    from caminus.decode import spec, longctx
    m, pf, dr = R.m, R.pf, R.dr
    n_new = 32 if small else 96
    rows = []; same_all = True
    for name, pr in prompts(small):
        sa = {}; a = dr.generate(m, pr, n_new, sa, prefill=pf)
        b = spec.generate(m, pr, n_new, longctx.Repeat(), prefill=pf)
        sc = {}; c = dr.generate(m, pr, n_new, sc)
        same = a == b; same_all &= same
        acc = lambda s: s['accepted'] / max(1, s['gen_passes'])
        rows.append(dict(prompt=name, n=len(pr), spec_eq_plain=same, acc_prefill=acc(sa), acc_k3=acc(sc), passes_prefill=sa['gen_passes'],
                         passes_k3=sc['gen_passes'], text_eq_k3fill=a == c, prefill=sa.get('prefill')))
        print(f'  dflash {name:16s} n {len(pr):5d}: spec {"==" if same else "!="} plain after the prefill; accepted per pass '
              f'{acc(sa):.3f} (prefill, {sa["gen_passes"]} passes) vs {acc(sc):.3f} (K3-filled, {sc["gen_passes"]}); the same text as the '
              f'K3-filled run: {a == c}; inject {sa["prefill"]["inject_launches"] if sa.get("prefill") else 0} launches '
              f'{sa["prefill"]["inject_ms"] if sa.get("prefill") else 0:.0f} ms', flush=True)
    d = np.array([r['acc_prefill'] - r['acc_k3'] for r in rows]); se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    check(same_all, f'DFlash after a prefill: spec == plain greedy after the same prefill, bitwise, on {len(rows)} prompts x {n_new} new tokens')
    check(abs(float(d.mean())) <= 2 * se + 0.05, f'DFlash acceptance after a prefill vs a K3-filled context: mean accepted per pass '
          f'{np.mean([r["acc_prefill"] for r in rows]):.3f} vs {np.mean([r["acc_k3"] for r in rows]):.3f}, paired difference {d.mean():+.3f} '
          f'(2 SE {2 * se:.3f} + 0.05)')
    NUMBERS['dflash'] = dict(rows=rows, diff_mean=float(d.mean()), diff_se=se)


# ---- stage: long (prompts past the batched limit, through the entry points) ---------------------------------------------
def stage_long(R, small):
    """a 3000-token prompt (d8192's first tokens): on the QSA engine the batched prefill covers it whole (S7: rows past 2051 on
    the QSA prompt path, its pools from token 0, so the QSA tape backfills nothing), else it stops at 2051 (not a multiple of
    T) and the rest runs as K3 passes from commit 0 on the QSA tape; the DFlash inject's last group zero-padded over a row
    the next pass rewrites"""
    from caminus.decode import spec, longctx
    m, pf, dr = R.m, R.pf, R.dr
    LIMIT = pf.limit
    toks = tokens(os.path.join(LC, 'd8192', 'ctx.ids')); N = 3000; npos = 16 if small else 256   # 256 like every G4 window (user 11-04: since S7 batches the whole prompt this is a real prefill-vs-decode seam test; 64 positions made one near-tie flip a fail)
    # longctx.fill(prefill=) to N, then teacher-forced positions N .. N + npos - 1, against K3's own decode there
    ref = R.ref(toks, N, N + npos)
    m.reset(); st = longctx.fill(m, toks, n=N, prefill=pf, progress=False)
    lg, _ = longctx.teacher_forced(m, st, toks, npos); got = lg.numpy()
    r0, r1 = m.check_replicas(); s = stats_of(ref, got, list(range(N, N + npos)))
    check(st['prefilled'] == min(N, LIMIT) and r0 and r1 and m.poisoned is None and s['top1'] >= 99.0 and s['kld_mean'] <= 0.0068,
          f'long: longctx.fill(n={N}, prefill=): {st["prefilled"]} positions batched (limit {LIMIT}), {st["passes"]} K3 passes from there; then '
          f'{npos} positions teacher-forced vs K3\'s own decode: mean KLD {s["kld_mean"]:.5f}, top-1 {s["top1"]:.2f} %, max {s["kld_max"]:.4f} at '
          f'{s["max_pos"]}; replicas {r0}/{r1}, not poisoned')
    NUMBERS['long_tf'] = slim(s)
    # DFlash after a prefill of a prompt past the limit: spec == plain, acceptance vs the K3-filled context
    n_new = 32 if small else 64; pr = toks[:N]
    sa = {}; a = dr.generate(m, pr, n_new, sa, prefill=pf)
    b = spec.generate(m, pr, n_new, longctx.Repeat(), prefill=pf)
    sc = {}; c = dr.generate(m, pr, n_new, sc)
    acc = lambda x: x['accepted'] / max(1, x['gen_passes'])
    check(a == b and sa['prefill']['p1'] == min(N - 1, LIMIT), f'long: DFlash after the prefill of a {N}-token prompt (batched to {sa["prefill"]["p1"]}, K3 passes to {N - 1}): spec == plain '
          f'after the same prefill, bitwise ({n_new} tokens); accepted per pass {acc(sa):.3f} ({sa["gen_passes"]} passes) vs {acc(sc):.3f} on a K3-filled '
          f'context ({sc["gen_passes"]}); the same text as the K3-filled run: {a == c}')
    NUMBERS['long_dflash'] = dict(spec_eq_plain=a == b, acc_prefill=acc(sa), acc_k3=acc(sc), text_eq_k3fill=a == c)
    # snapshot / restore with the drafter after a prefill: a probe with other given tokens, restore, the same continuation
    other = tokens(os.path.join(D281_SHORT, 'e2e_shell_decode', 'tokens.txt'))
    n0, given = 2000, toks[2000:2010]
    def cont(st_):
        out, _ = longctx.generate(m, st_, list(toks[:n0]) + list(given), 40, draft=dr); return out
    st0 = longctx.fill(m, toks, n=n0, draft=dr, prefill=pf, progress=False); cold = cont(st0)
    st1 = longctx.fill(m, toks, n=n0, draft=dr, prefill=pf, progress=False); snap = longctx.snapshot(m, st1, dr)
    longctx.generate(m, st1, list(toks[:n0]) + list(other[:30]), 40, draft=dr)          # the probe: other given tokens, rows >= n0 rewritten
    st2 = longctx.restore(m, snap, dr); longctx.release(snap); again = cont(st2)
    check(again == cold and st1['prefilled'] == n0, f'long: snapshot / restore with the DFlash drafter after a prefill of {n0}: a probe with other given '
          f'tokens, restore, the continuation (10 given + 40 new tokens) == the cold run\'s, token for token')


# ---- stage: speed / tstar -----------------------------------------------------------------------------------------------
def stage_speed(R, reps, moe_ab=False, q6k_ab=False):
    from caminus.decode import spec, longctx
    from caminus.prefill import tp as pt
    m, pf, dr = R.m, R.pf, R.dr
    toks = tokens(CTX[2048])
    res = {}
    base = (pf.S.moe, pf.S.q6k); paths = [base]
    if moe_ab: paths.append(('mmq' if base[0] == 'own' else 'own', base[1]))
    if q6k_ab: paths += [(base[0], q) for q in pt.Q6K_PATHS if q != base[1]]
    if any(q == 'wmma' for _, q in paths): pf.S.alloc_q6k_buffers()
    tagged = len(paths) > 1
    print(f'  speed: compaction_proactiveness {comp()}; (routed-expert, dense Q6_K) paths {paths}', flush=True)
    def ttft(n, how, path=base):
        pr = toks[:n]; pf.S.moe, pf.S.q6k = path; R.sync(); t0 = time.perf_counter()
        try:
            if how == 'prefill': out = spec.generate(m, pr, 1, longctx.Repeat(), prefill=pf)
            elif how == 'prefill+dflash': st = {}; out = dr.generate(m, pr, 1, st, prefill=pf)
            else: out = spec.generate(m, pr, 1, longctx.Repeat())
            R.sync(); return (time.perf_counter() - t0) * 1e3
        finally: pf.S.moe, pf.S.q6k = base
    for p in paths: ttft(512, 'prefill', p); ttft(512, 'prefill+dflash', p)                 # (warm-up)
    cfg = [(n, h, p) for n in (512, 1024, 2048) for h in ('prefill', 'prefill+dflash') for p in paths] + [(512, 'k3', base)]
    T = {c: [] for c in cfg}
    for r in range(reps):
        for c in cfg: T[c].append(ttft(*c))
    for (n, h, p), v in T.items():
        lab = f' moe:{p[0]} q6k:{p[1]}' if tagged and h != 'k3' else ''
        med = float(np.median(v)); res[f'{n} {h}' + lab] = dict(ms=v, median_ms=med, tok_s=n / med * 1e3)
        ll = S0.get(n)
        print(f'  TIME {n:5d}-token prompt from empty, {h:15s}' + (f' moe {p[0]:4s} q6k {p[1]:4s}' if lab else '') + f': TTFT median {med:7.1f} ms = {n / med * 1e3:6.0f} tok/s (runs '
              f'{", ".join(f"{x:.0f}" for x in v)})' + (f'; llama-bench pp{n} best {ll[0]:.0f} tok/s ({ll[1]}, S0)' if ll and h != 'k3' else ''), flush=True)
    NUMBERS['speed'] = dict(res, compaction=comp())


def stage_tstar(R, reps, Ls=(4, 8, 12, 16, 24, 32, 48, 64, 96, 128)):
    from caminus.decode import spec, longctx
    m, pf, dr = R.m, R.pf, R.dr
    toks = tokens(CTX[2048]); D0 = 1024
    print(f'  tstar: compaction_proactiveness {comp()}', flush=True)
    def at_depth(L, batched):
        # the depth's context first, untimed (a from-empty run between two trials resets the sequence: no snapshot reuse)
        m.reset(); dr.reset(); st2 = pf.run(toks, 0, 0, D0, dr); R.sync(); t0 = time.perf_counter()
        n = D0 + L
        if batched: pf.run(toks, st2['i0'], st2['commit'], n - 1, dr); i0 = n - 1; commit = 0
        else: i0, commit = st2['i0'], st2['commit']
        spec.loop(m, list(toks[:n]), i0, commit, n + 1, dr)
        R.sync(); return (time.perf_counter() - t0) * 1e3
    def from_empty(L, batched):
        R.sync(); t0 = time.perf_counter(); s = {}
        if batched: old = pf.crossover; pf.crossover = 1; dr.generate(m, toks[:L], 1, s, prefill=pf); pf.crossover = old
        else: dr.generate(m, toks[:L], 1, s)
        R.sync(); return (time.perf_counter() - t0) * 1e3
    from_empty(32, True); at_depth(32, True)
    res = {}
    for L in Ls:
        t = {k: [] for k in ('e_b', 'e_k', 'd_b', 'd_k')}
        for r in range(reps):
            t['e_b'].append(from_empty(L, True)); t['e_k'].append(from_empty(L, False))
            t['d_b'].append(at_depth(L, True)); t['d_k'].append(at_depth(L, False))
        med = {k: float(np.median(v)) for k, v in t.items()}; res[L] = dict(med, runs=t)
        print(f'  T* L={L:4d}: from empty batched {med["e_b"]:7.1f} / K3 {med["e_k"]:7.1f} ms; at depth {D0} batched {med["d_b"]:7.1f} / K3 '
              f'{med["d_k"]:7.1f} ms (DFlash inject included, medians of {reps})', flush=True)
    win = [L for L in Ls if res[L]['e_b'] < res[L]['e_k'] and res[L]['d_b'] < res[L]['d_k']]
    ts = next((L for L in Ls if all(x in win for x in Ls if x >= L)), None)
    print(f'  T* (the smallest L from which the batched path wins at both depths, in every longer L measured): {ts}', flush=True)
    NUMBERS['tstar'] = dict(rows=res, tstar=ts, compaction=comp())


# ---- stage: exact (M6 unit 2: the exact cache hit) ----------------------------------------------------------------------
def stage_exact(R, reps, small):
    """an exact cache hit (a request with zero new tokens past a cached state) against the cold run of the same prompt, for
    prompts routed to K3 passes (n - 1 < T*: every residue mod 4) and batched ones (every residue mod 4; one past 2051), plain
    (Repeat) and DFlash on the generic loop (DFlashK3Drafter(device_accept=False): the exact hit resumes a state, which
    the device-accept loop never does, so generic is compared with generic):
      (a) the prompt-end checkpoint: spec.prompt_end's state (before the last prompt pass), snapshot, a probe rewriting the
          rows past it, restore, then the request: first logits (longctx.first_logits) and tokens (longctx.generate) ==
          the cold run's, bitwise / token for token;
      (b) a state at n with the last pass's rows uncommitted (commit F >= 1; plain: the cold path's last prompt pass by
          first_logits; DFlash: a K3 fill of every token with the drafter, whose inject of that pass is done, against the
          cold run of the same kind, K3 passes from 0), snapshot, probe, restore, zero new tokens: the loop reruns position
          n - 1 (spec.resume_point): first logits and tokens == cold;
      refusal: zero new tokens from a state committed through n - 1 (a batched prefill of every token: commit 0) raises
          ExactHitRefused before anything launches: every state, KV / IDXRAW row and feature byte unchanged, not poisoned.
    The cold first logits: prompt_end + first_logits (the cold run's last prompt pass with filler drafts: its forced rows
    are the same, a pass's rows depend only on positions <= theirs); the cold tokens: spec.generate(prefill=).
    Timing (interleaved medians of --reps): TTFT of the cold run, (a), (b) by rerun, and (b) with stored logits (restore +
    the argmax of the kept logits on the device: the lower bound of storing logits with the checkpoint)"""
    import torch
    from caminus.decode import spec, longctx, dflash as dk
    m, pf = R.m, R.pf
    drg = dk.DFlashK3Drafter(R.df, device_accept=False)
    code = tokens(os.path.join(D281_SHORT, 'e2e_code_decode', 'tokens.txt')); long_ = tokens(CTX[2048])
    other = tokens(os.path.join(D281_SHORT, 'e2e_shell_decode', 'tokens.txt'))
    lens = [29, 30, 31, 32, 255, 256, 257, 258, 2100] if not small else [30, 257]
    n_new = 8 if small else 16
    print(f'  exact: T* {pf.crossover}; prompts {lens}; {n_new} new tokens', flush=True)
    for n in lens:
        prompt = list((code if n <= len(code) else long_)[:n])
        # the cold first logits (the target alone: the drafter changes no target logit)
        st0 = spec.prompt_end(m, prompt, None, pf); lg_cold, _ = longctx.first_logits(m, st0, prompt)
        cold_lg = lg_cold.numpy().view(np.int32)
        for dname, draft in (('plain', None), ('dflash', drg)):
            dr_ = draft if draft is not None else longctx.Repeat()
            cold = spec.generate(m, prompt, n_new, dr_, prefill=pf)
            # (a) the prompt-end checkpoint
            st0 = spec.prompt_end(m, prompt, draft, pf); snap = longctx.snapshot(m, st0, draft); sa_st = dict(st0)
            longctx.generate(m, st0, prompt[:st0['i0']] + list(other[:12]), 8, draft)      # (the probe: rows >= i0 rewritten)
            st = longctx.restore(m, snap, draft); lg_a, _ = longctx.first_logits(m, st, prompt)
            st = longctx.restore(m, snap, draft); out_a, _ = longctx.generate(m, st, prompt, n_new, draft); longctx.release(snap)
            # (b) a state at n with the last pass's rows uncommitted: zero new tokens. plain: the cold path's last prompt pass by
            # first_logits (target only); DFlash: a K3 fill of every token with the drafter (its inject of the last pass done: a
            # state the drafter can continue), against the cold run of the same kind (K3 passes from 0, spec.generate without
            # the prefill; first logits K3's own teacher-forced decode: the same passes)
            if draft is None:
                st0 = spec.prompt_end(m, prompt, None, pf); _, st1 = longctx.first_logits(m, st0, prompt); ref_b, cold_b, how = cold_lg, cold, 'the cold path'
            else:
                ref_b = R.ref(prompt, n - 1, n)[0].view(np.int32); cold_b = spec.generate(m, prompt, n_new, draft)
                f_ = longctx.fill(m, prompt, draft=draft, progress=False); st1 = dict(i0=f_['i0'], commit=f_['commit']); how = 'a K3 fill with the drafter'
            snap = longctx.snapshot(m, st1, draft)
            longctx.generate(m, st1, prompt + list(other[:12]), 8, draft)
            st = longctx.restore(m, snap, draft); lg_b, stb = longctx.first_logits(m, st, prompt)
            st = longctx.restore(m, snap, draft); out_b, _ = longctx.generate(m, st, prompt, n_new, draft); longctx.release(snap)
            route = 'batched' if spec.prefill_span(pf, prompt, None) else 'K3 passes'
            check(np.array_equal(lg_a.numpy().view(np.int32), cold_lg) and out_a == cold,
                  f'exact (a) n {n} ({route}, n - 1 = {n - 1} mod 4 = {(n - 1) % 4}), {dname}: the prompt-end checkpoint (state ({sa_st["i0"]}, commit {sa_st["commit"]})) '
                  f'restored after a probe: first logits == cold bitwise, {n_new} tokens == cold ({out_a[n:n + 4]}...)')
            check(np.array_equal(lg_b.numpy().view(np.int32), ref_b) and out_b == cold_b and st1['commit'] >= 1,
                  f'exact (b) n {n} ({route}), {dname}: zero new tokens from the state ({st1["i0"]}, commit {st1["commit"]}) of {how}, restored after a '
                  f'probe: position {n - 1} rerun, first logits == the cold run\'s bitwise, {n_new} tokens == cold')
        # refusal: a batched prefill of every token (commit 0), zero new tokens
        if n - 1 >= pf.crossover:
            m.reset(); stp = pf.run(prompt, 0, 0, n); R.sync(); h0 = R.states(n)
            try: longctx.generate(m, dict(i0=stp['i0'], commit=stp['commit']), prompt, 4); okr = False; why = 'not refused'
            except spec.ExactHitRefused as e: okr = True; why = str(e)[:80]
            R.sync()
            check(okr and R.states(n) == h0 and m.poisoned is None, f'exact refusal n {n}: zero new tokens after a batched prefill of all {n} '
                  f'tokens (commit 0) refused before anything launched ({why}); every state, KV / IDXRAW row and feature byte unchanged, not poisoned')
    # timing: TTFT (to the first token on the host) of the cold run, (a), (b) by rerun, (b) with stored logits
    print(f'  exact timing: compaction_proactiveness {comp()}', flush=True)
    res = {}
    for n in ((257,) if small else (257, 2100)):
        prompt = list((code if n <= len(code) else long_)[:n])
        for dname, draft in (('plain', None), ('dflash', drg)):
            dr_ = draft if draft is not None else longctx.Repeat()
            # each trial sets its state up untimed (a restore is refused across a reset: its rows may be gone), then times
            # the restore and the request to its first token on the host
            def setup_a():
                st0 = spec.prompt_end(m, prompt, draft, pf); return longctx.snapshot(m, st0, draft), None
            def setup_b():
                st0 = spec.prompt_end(m, prompt, None, pf); lg, st1 = longctx.first_logits(m, st0, prompt)   # (plain: the cold path)
                if draft is not None:                                                                      # (DFlash: a K3 fill)
                    f_ = longctx.fill(m, prompt, draft=draft, progress=False); st1 = dict(i0=f_['i0'], commit=f_['commit'])
                return longctx.snapshot(m, st1, draft), lg.to(m._dev(0))
            def t_cold():
                R.sync(); t0 = time.perf_counter(); spec.generate(m, prompt, 1, dr_, prefill=pf); R.sync(); return (time.perf_counter() - t0) * 1e3
            def t_rest(setup, stored=False):
                snap, keep = setup(); R.sync(); t0 = time.perf_counter()
                st = longctx.restore(m, snap, draft)
                if stored: int(keep.argmax().item())
                else: longctx.generate(m, st, prompt, 1, draft)
                R.sync(); dt = (time.perf_counter() - t0) * 1e3; longctx.release(snap); return dt
            fs = dict(cold=t_cold, a=lambda: t_rest(setup_a), b_rerun=lambda: t_rest(setup_b), b_stored=lambda: t_rest(setup_b, True))
            for f in fs.values(): f()
            T_ = {k: [] for k in fs}
            for r in range(reps):
                for k, f in fs.items(): T_[k].append(f())
            med = {k: float(np.median(v)) for k, v in T_.items()}
            res[f'{n} {dname}'] = dict(median_ms=med, runs=T_)
            print(f'  TIME exact hit n {n} {dname}: TTFT median cold {med["cold"]:.1f} ms; (a) prompt-end checkpoint {med["a"]:.1f}; (b) rerun of n - 1 '
                  f'{med["b_rerun"]:.1f}; (b) stored logits (restore + device argmax: a lower bound) {med["b_stored"]:.1f} ms; (a) - stored = '
                  f'{med["a"] - med["b_stored"]:.1f} ms; stored logits would hold {248320 * 4 / 2 ** 20:.2f} MiB per checkpoint', flush=True)
    NUMBERS['exact'] = res


# ---- the QSA prompt path (M5 S7) ----------------------------------------------------------------------------------------
S7 = os.path.join(ARTIFACTS, 'impl-prefill-s7')
D8 = os.path.join(LC, 'd8192', 'ctx.ids')
S0_7 = {8192: (2884.1, 'ub 1024, from empty'), 'd32k': (1994.1, '8K appended at 32K depth, ub 1024')}   # spec S0 llama-bench best cells
G5 = {8192: 3000.0, 'depth': 2000.0}


def depth_n(R):
    return 8192 if R.m.max_ctx >= 8192 + 16 else min(4000, R.m.max_ctx - 96)


def stage_g2b7(R):
    m, pf = R.m, R.pf
    toks = tokens(D8); N = depth_n(R)
    def one(splits):
        m.reset(); a = 0
        for b, c in splits: pf.run(toks, a, 0, b, chunk=c); a = b
        # (the features: positions >= 3000, which the last run writes in every variant: the export holds the last run's rows)
        R.sync(); st = R.states(N, feat_from=3000); lg = tail(R, toks, N); return st, lg
    t0 = time.perf_counter(); base_st, base_lg = one([(N, 2048)]); t1 = time.perf_counter()
    print(f'  g2b7: N = {N}, one run in chunks of 2048 + 8 K3 logits {t1 - t0:.1f} s; poisoned {m.poisoned}', flush=True)
    for name, splits in (('chunk 1024', [(N, 1024)]), ('chunk 777', [(N, 777)]), ('runs [0, 2051) + rest', [(2051, 2048), (N, 2048)]),
                         ('runs [0, 3000) + rest', [(3000, 2048), (N, 2048)])):
        st, lg = one(splits)
        diff = sorted(str(k) for k in base_st if st[k] != base_st[k])
        check(not diff and np.array_equal(lg.view(np.int32), base_lg.view(np.int32)) and m.poisoned is None,
              f'G2b (S7): [0, {N}) {name} == one run in chunks of 2048, bitwise: every state, KV / IDXRAW / IDXPOOL row and the '
              f'export window\'s features ({len(base_st)} items' + (f'; differ: {diff[:4]}' if diff else '') + '), the next 8 K3 logits (QSA tape)')


def stage_g3q(R):
    from caminus.decode import longctx
    m, pf = R.m, R.pf
    toks = tokens(D8); N = depth_n(R)
    other = tokens(os.path.join(D281_SHORT, 'e2e_prose_decode', 'tokens.txt'))
    m.reset(); pf.run(toks, 0, 0, N); R.sync(); cold_lg = tail(R, toks, N)
    ps = (2500, 3001) + ((N - 1000,) if N >= 8192 else ())
    for p in ps:
        m.reset(); pf.run(toks, 0, 0, N); R.sync(); cold_st = R.states(N, feat_from=p)
        m.reset(); st = pf.run(toks, 0, 0, p)
        snap = longctx.snapshot(m, st)
        probe = list(toks[:p]) + list(other) * 4
        pf.run(probe, p, 0, p + 900)                                   # rewrites the KV / IDXRAW / IDXPOOL / feature rows from p
        st2 = longctx.restore(m, snap); longctx.release(snap)          # (done with it: its copies back to the driver)
        pf.run(toks, st2['i0'], st2['commit'], N); R.sync()
        rs = R.states(N, feat_from=p); lg = tail(R, toks, N)
        diff = sorted(str(k) for k in cold_st if rs[k] != cold_st[k])
        check(not diff and np.array_equal(lg.view(np.int32), cold_lg.view(np.int32)),
              f'G3-lite (S7): prefill to {p}, snapshot, a probe rewriting rows {p} .. {p + 899}, restore, resume to {N} == the cold prefill '
              f'of {N}, bitwise (every state, KV / IDXRAW / IDXPOOL row, the features from {max(p, N - R.pf.S.feat_rows)}; the next 8 logits)'
              + (f'; differ: {diff[:4]}' if diff else ''))


def floor_depth():
    f = os.path.join(S7, 'g4', 'g4_depth.json')
    return json.load(open(f))['depth'] if os.path.exists(f) else {}


def stage_g4d(R, small):
    m, pf = R.m, R.pf
    fl = floor_depth(); out = {}
    Ns = [N for N in (8192, 32768) if m.max_ctx >= N + 256]
    if not Ns: return check(False, f'g4d: max_ctx {m.max_ctx} < 8192 + 256 (run with --max-ctx 33024)')
    for N in Ns:
        toks = tokens(os.path.join(LC, f'd{N}', 'ctx.ids')); npos = 32 if small else 256; pos = list(range(N, N + npos))
        F = fl.get(f'N{N}_ub2048')
        if F is None: check(False, f'g4d N={N}: no floor (tools/gates/g4_floor.py depth-cmp {S7}/g4 --json {S7}/g4/g4_depth.json first)'); continue
        t0 = time.perf_counter(); ref = R.ref(toks, N, N + npos); t1 = time.perf_counter()
        got = R.after_prefill(toks, N, npos); t2 = time.perf_counter()
        r0, r1 = m.check_replicas()
        D_ = os.path.join(S7, 'g4', 'depth_ub2048')
        lc_pd = np.memmap(os.path.join(D_, f'pd{N}_r1.bin'), dtype=np.float32, mode='r').reshape(256, G.NV)[:npos]
        lc_1t = np.memmap(os.path.join(D_, f'ref{N}.bin'), dtype=np.float32, mode='r').reshape(256, G.NV)[:npos]
        s, sb, s1 = stats_of(ref, got, pos), stats_of(lc_pd, got, pos), stats_of(lc_1t, got, pos)
        k3d = stats_of(lc_pd, ref, pos)    # (reported: K3's own decode, no prefill, vs llama.cpp's prefill-then-decode)
        bar_k, bar_t = F['boot95'][1], F['top1'] - 1.0
        print(f'  floor N={N} (llama.cpp prefill-then-decode at ub 2048 vs its one-token decode): mean KLD {F["kld_mean"]:.5f}, 95 % bootstrap '
              f'{F["boot95"][0]:.5f} .. {F["boot95"][1]:.5f}, top-1 {F["top1"]:.2f} %, max {F["kld_max"]:.4f} at {F["max_pos"]}', flush=True)
        print('  ' + short_line(f'G4a depth N={N}', s) + f'  [K3 one-token reference {t1 - t0:.0f} s, prefill + decode {t2 - t1:.0f} s]', flush=True)
        print('  ' + short_line(f'G4b depth N={N} vs llama.cpp prefill-then-decode ub 2048', sb), flush=True)
        print('  ' + short_line(f'(reported) depth N={N} vs llama.cpp one-token decode', s1), flush=True)
        print('  ' + short_line(f'(reported) K3 one-token decode, no prefill, N={N} vs llama.cpp prefill-then-decode', k3d), flush=True)
        out[f'depth_{N}'] = dict(g4a=slim(s), g4b=slim(sb), vs_llama_1tok=slim(s1), k3_decode_vs_llama_pd=slim(k3d), floor=F, replicas=[r0, r1])
        check(s['kld_mean'] <= bar_k and s['top1'] >= bar_t and r0 and r1 and m.poisoned is None,
              f'G4a depth N={N} (chunk 2048): mean KLD {s["kld_mean"]:.5f} (bar: the floor\'s 95 % bootstrap upper bound {bar_k:.5f}; floor mean '
              f'{F["kld_mean"]:.5f}), top-1 {s["top1"]:.2f} % (bar {bar_t:.2f}); max {s["kld_max"]:.4f} at {s["max_pos"]}, mean without it '
              f'{s["mean_wo_max"]:.5f}; replicas {r0}/{r1}')
        check(sb['kld_mean'] <= 0.0076 and sb['top1'] >= 95.9, f'G4b depth N={N} vs llama.cpp\'s prefill-then-decode at ub 2048: mean KLD '
              f'{sb["kld_mean"]:.5f} (bar 0.0076), top-1 {sb["top1"]:.2f} % (bar 95.9)')
    NUMBERS['g4_depth'] = out


def stage_probes(R):
    from caminus.decode import spec, longctx
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'eval'))
    import lc_probes as L
    m, pf = R.m, R.pf
    meta = json.load(open(os.path.join(LC, 'probes.json'))); res = {}
    for d in meta['depths']:
        D = d['D']
        if D < 8192 or max(p['P'] for p in d['probes']) + meta['ngen'] + 8 > m.max_ctx: continue
        for p in d['probes']:
            toks = [int(v) for v in open(os.path.join(LC, f'd{D}', f'{p["name"]}.ids')).read().split()][:p['P']]
            t0 = time.perf_counter(); st = {}
            g = spec.generate(m, toks, meta['ngen'], longctx.Repeat(), st, prefill=pf); dt = time.perf_counter() - t0
            gk = list(g[len(toks):])
            gl = [int(v) for v in open(os.path.join(LC, 'ref', f'{p["name"]}_{D}.tokens')).read().split()]
            tk, tl = L.decode(gk), L.decode(gl)
            fk, fl_ = tk.startswith(p['answer']), tl.startswith(p['answer'])
            div = next((i for i, (u, v) in enumerate(zip(gk, gl)) if u != v), None)
            res[f'{p["name"]}_{D}'] = dict(found=fk, llama_found=fl_, same_as_llama=gk == gl, first_divergence=div, text=tk[:40], llama=tl[:40],
                                          prefilled=(st.get('prefill') or {}).get('p1'), seconds=dt)
            check(fk == fl_ and m.poisoned is None, f'probe {p["name"]} at {D}: the answer {"found" if fk else "NOT found"} after the batched prefill of '
                  f'{(st.get("prefill") or {}).get("p1")} positions, llama.cpp {"found" if fl_ else "not found"}: as llama.cpp answers it; '
                  f'token-identical {gk == gl} (first divergence {div}); {tk[:28]!r} [{dt:.1f} s]')
    NUMBERS['probes'] = res


def stage_speed7(R, reps):
    from caminus.decode import spec, longctx
    m, pf, dr = R.m, R.pf, R.dr
    t32 = tokens(os.path.join(LC, 'd32768', 'ctx.ids'))
    lens = [n for n in (4096, 8192, 32768) if n + 8 <= m.max_ctx]
    print(f'  speed7: compaction_proactiveness {comp()}; MemAvailable {open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]} kB; '
          f'prompts {lens}', flush=True)
    def ttft(n, how):
        pr = t32[:n]; R.sync(); t0 = time.perf_counter()
        if how == 'prefill': spec.generate(m, pr, 1, longctx.Repeat(), prefill=pf)
        else: st = {}; dr.generate(m, pr, 1, st, prefill=pf)
        R.sync(); return (time.perf_counter() - t0) * 1e3
    D0, L8 = 24576, 8192
    def at_depth():
        if D0 + L8 + 8 > m.max_ctx: return None
        m.reset(); dr.reset(); st = pf.run(t32, 0, 0, D0, dr); R.sync(); t0 = time.perf_counter()
        n = D0 + L8; pf.run(t32, st['i0'], st['commit'], n - 1, dr)
        spec.loop(m, list(t32[:n]), n - 1, 0, n + 1, dr)
        R.sync(); return (time.perf_counter() - t0) * 1e3
    ttft(4096, 'prefill'); ttft(4096, 'prefill+dflash')        # (warm-up)
    cfg = [(n, h) for n in lens for h in ('prefill', 'prefill+dflash')]
    T = {c: [] for c in cfg}; Td = []
    for r in range(reps):
        for c in cfg: T[c].append(ttft(*c))
        v = at_depth()
        if v is not None: Td.append(v)
    res = {}
    for (n, h), v in T.items():
        med = float(np.median(v)); res[f'{n} {h}'] = dict(ms=v, median_ms=med, tok_s=n / med * 1e3)
        ll = S0_7.get(n); g5 = G5.get(n)
        print(f'  TIME {n:6d}-token prompt from empty, {h:15s}: TTFT median {med:8.1f} ms = {n / med * 1e3:6.0f} tok/s (runs {", ".join(f"{x:.0f}" for x in v)})'
              + (f'; G5 >= {g5:.0f}' if g5 else '') + (f'; llama-bench pp{n} best {ll[0]:.0f} ({ll[1]}, S0)' if ll else '; no llama-bench cell in S0'), flush=True)
    if Td:
        med = float(np.median(Td)); res['8192 at depth 24576 +dflash'] = dict(ms=Td, median_ms=med, tok_s=L8 / med * 1e3)
        print(f'  TIME 8192-token append at depth {D0} (positions {D0} .. {D0 + L8 - 1}), with the DFlash inject: {med:.1f} ms = {L8 / med * 1e3:.0f} tok/s '
              f'(runs {", ".join(f"{x:.0f}" for x in Td)}); G5 (8K at 32K depth) >= {G5["depth"]:.0f}; llama-bench 8K at 32K {S0_7["d32k"][0]:.0f} (S0); '
              f'8K at 32K depth itself reaches 40,960 positions: not run (no fills beyond 32K)', flush=True)
    # VRAM after the largest prefill with DFlash
    R.sync(); used = vram_used(); vp = m.vram_plan; det = vram_detail(R.torch)
    proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(R.idle[c] + proj[c] >= used[c], f'speed7 VRAM, {nm}, after the {max(lens)}-token prefill with DFlash (max_ctx {m.max_ctx}): idle {R.idle[c]:.0f} + '
              f'projected {proj[c]:.0f} (prefill {vp["prefill_mib"][c]:.0f}, drafters {vp["drafters_mib"][c]:.0f}) = {R.idle[c] + proj[c]:.0f} MiB >= '
              f'the measured device-level use {used[c]:.0f} MiB (margin {R.idle[c] + proj[c] - used[c]:+.0f}); torch allocated {det[c][0]:.0f} / reserved {det[c][1]:.0f} MiB')
    NUMBERS['speed7'] = dict(res, compaction=comp(), vram=dict(idle=R.idle, projected=proj, measured=used, torch_alloc_reserved=det))


def vram_detail(torch):
    """torch's allocated / reserved MiB per card (the reserved minus allocated is the caching allocator's free blocks)"""
    return [(torch.cuda.memory_allocated(d) / 2 ** 20, torch.cuda.memory_reserved(d) / 2 ** 20) for d in (0, 1)]


def stage_vram7(R):
    """the VRAM check in a fresh process: the largest prompt the engine takes (max_ctx - 256, at most 32768) through
    DFlashK3Drafter.generate(prefill=) and 4 new tokens, then idle + projection >= the device-level use"""
    from caminus.decode import longctx
    m, pf, dr = R.m, R.pf, R.dr
    n = min(32768, m.max_ctx - 256)
    t32 = tokens(os.path.join(LC, 'd32768', 'ctx.ids'))[:n]
    st = {}; t0 = time.perf_counter(); dr.generate(m, t32, 4, st, prefill=pf); R.sync(); dt = time.perf_counter() - t0
    used = vram_used(); vp = m.vram_plan; det = vram_detail(R.torch)
    proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(R.idle[c] + proj[c] >= used[c], f'vram7, {nm}, {"a fresh process" if not FAILS and not NUMBERS else "after the stages before it in this process"}: a {n}-token prompt through the batched prefill + DFlash ({dt:.1f} s), max_ctx {m.max_ctx}: '
              f'idle {R.idle[c]:.0f} + projected {proj[c]:.0f} (fixed {vp["fixed_mib"][c]:.0f}, ctx {vp["ctx_mib"]:.0f}, prefill {vp["prefill_mib"][c]:.0f}, drafters '
              f'{vp["drafters_mib"][c]:.0f}) = {R.idle[c] + proj[c]:.0f} MiB >= the measured device-level use {used[c]:.0f} MiB (margin {R.idle[c] + proj[c] - used[c]:+.0f}); '
              f'torch allocated {det[c][0]:.0f} / reserved {det[c][1]:.0f} MiB')
    NUMBERS['vram7'] = dict(n=n, idle=R.idle, projected=proj, measured=used, torch_alloc_reserved=det)


def stage_alloc(R):
    R.sync(); used = vram_used(); vp = R.m.vram_plan
    proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    lines = (23800, 19850)
    det = vram_detail(R.torch)
    print(f'  alloc: torch allocated / reserved MiB per card {[(round(a), round(b)) for a, b in det]}', flush=True)
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(used[c] < lines[c] and R.idle[c] + proj[c] < lines[c],
              f'alloc (max_ctx {R.m.max_ctx} + DFlash + prefill chunk {R.pf.chunk}, nothing run), {nm}: measured {used[c]:.0f} MiB, idle {R.idle[c]:.0f} + '
              f'projected {proj[c]:.0f} (prefill {vp["prefill_mib"][c]:.0f}) = {R.idle[c] + proj[c]:.0f} MiB; the guard line {lines[c]} '
              f'(headroom {lines[c] - max(used[c], R.idle[c] + proj[c]):.0f})')
    NUMBERS['alloc'] = dict(max_ctx=R.m.max_ctx, idle=R.idle, projected=proj, measured=used, lines=lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--json', default=None); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--kv', choices=('q8', 'f16'), default='q8', help='test-only target KV format; F16 numerical/speed baselines are pending')
    ap.add_argument('--tp-split', default=None, help='GK,MB: the TP split (FlashNextTP gdn_keys / moe_blocks; default the engine\'s 10,12; 8,10: the even split)')
    ap.add_argument('--max-ctx', type=int, default=4096); ap.add_argument('--router', default=None, help='wmma | exact (the prefill router logits; default: prefill_ops)')
    ap.add_argument('--wmma-out', default=None, help='fp32 | bf16 (the WMMA GEMMs\' output, router logits included; default: prefill_ops.WMMA_OUT)')
    ap.add_argument('--moe', default=None, help='own | mmq | ab (the routed experts\' path, ChunkStack.moe; default prefill_tp.MOE_PATH; ab: '
                                                'the default, and speed times the other one interleaved)')
    ap.add_argument('--graphs', action='store_true', help='every stage with HIP-graph chunks (caminus.prefill.graph; Prefill(graphs=True))')
    ap.add_argument('--q6k', default=None, help='own | wmma | mmq | ab (the dense Q6_K GEMMs\' path, ChunkStack.q6k; default prefill_tp.Q6K_PATH; '
                                                'ab: the default, and speed times the other two interleaved)')
    a = ap.parse_args()
    try: check_kv_stages(a.kv, a.stages)
    except ValueError as e: ap.error(str(e))
    NUMBERS['kv'] = a.kv
    if a.tp_split:
        SPLIT.update(zip(('gdn_keys', 'moe_blocks'), (int(v) for v in a.tp_split.split(','))))
        print(f'split: {SPLIT}', flush=True)
    if 'cpu' in a.stages: stage_cpu()
    gpu = [s for s in a.stages if s != 'cpu']
    if gpu:
        import torch
        for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
        assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
        print('compaction_proactiveness', comp(), flush=True)
        R = Rig(torch, a.coder, max_ctx=a.max_ctx, router=a.router, wmma_out=a.wmma_out, moe=None if a.moe in (None, 'ab') else a.moe,
                q6k=None if a.q6k in (None, 'ab') else a.q6k, plan_bf16=True if a.q6k == 'ab' else None, q8=a.kv == 'q8')
        if a.graphs:
            from caminus.prefill.graph import Graphs
            R.pf.graphs = Graphs(R.pf.S, max_graphs=16); print('graphs: HIP-graph chunks on (Prefill.graphs, max 16 structures)', flush=True)
        for s in gpu:
            t0 = time.time(); print(f'== stage {s}', flush=True)
            {'smoke': lambda: stage_smoke(R), 'g4a': lambda: stage_g4(R, a.small), 'g2b': lambda: stage_g2b(R), 'g3': lambda: stage_g3(R),
             'dflash': lambda: stage_dflash(R, a.small), 'speed': lambda: stage_speed(R, a.reps, a.moe == 'ab', a.q6k == 'ab'), 'tstar': lambda: stage_tstar(R, a.reps),
             'long': lambda: stage_long(R, a.small), 'g2b7': lambda: stage_g2b7(R), 'g3q': lambda: stage_g3q(R),
             'g4d': lambda: stage_g4d(R, a.small), 'probes': lambda: stage_probes(R), 'speed7': lambda: stage_speed7(R, a.reps),
             'alloc': lambda: stage_alloc(R), 'vram7': lambda: stage_vram7(R), 'exact': lambda: stage_exact(R, a.reps, a.small)}[s]()
            R.sync(); used = vram_used(); det = vram_detail(torch)
            print(f'== stage {s} done in {time.time() - t0:.0f} s; VRAM ' + '; '.join(f'{nm} device {used[c]:.0f} MiB, torch allocated {det[c][0]:.0f} / reserved '
                  f'{det[c][1]:.0f} (free cache {det[c][1] - det[c][0]:.1f})' for c, nm in ((0, 'XTX'), (1, 'XT'))), flush=True)
            if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    if gpu and a.graphs: print(f'graphs: {R.pf.graphs.stats}; structures held {len(R.pf.graphs.g)}, {[round(x, 1) for x in R.pf.graphs.mib()]} MiB per card; '
                               f'every instantiation measured {[round(x, 1) for x in R.pf.graphs.total]} MiB in all', flush=True)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
