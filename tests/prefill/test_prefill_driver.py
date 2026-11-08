"""CPU tests of the prefill drivers' shared parts (caminus.prefill.driver): the one routing rule against the four rules it
replaced, a driver without a batched prefill (KPassOnly) in the prompt cache's plan, the refusals of the shared shell.
Nothing touches a GPU (the run() shell itself runs in tests/cache/test_prompt_cache.py cpu: its stand-in FakePF is a
PrefillDriver).

  python -I tests/prefill/test_prefill_driver.py      RESULT PASS"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def main():
    from caminus.prefill.driver import PrefillDriver, PrefillRefused, batched_span
    from caminus.prefill.kpass import KPassOnly
    from caminus.cache import prompt as pc

    # -- the one rule == the four it replaced (before 11-07, as written in each place) --------------------------------------
    old_plan = lambda P, p0, X: P > p0 and (P - p0) >= X                     # plan_request
    old_spec = lambda n, lim, X: (lambda P: P if (P >= 1 and P >= X) else 0)(min(n - 1, lim))   # spec.prefill_span
    old_fill = lambda n, lim, X: min(n, lim) if min(n, lim) >= X else 0      # longctx.fill (route(min(n, limit)))
    bad = []
    for p0, P, X in itertools.product(range(0, 70, 7), range(0, 140, 3), (1, 16, 48, 64)):
        if (batched_span(p0, P, X) is not None) != old_plan(P, p0, X): bad.append(('plan', p0, P, X))
    for n, lim, X in itertools.product(range(1, 200, 5), (40, 64, 262144), (1, 16, 48)):
        d = PrefillDriver(); d.crossover = X; d.m = type('M', (), dict(max_ctx=lim))()
        sp = d.span(0, n - 1); got = sp[1] if sp else 0
        if got != old_spec(n, lim, X): bad.append(('spec', n, lim, X, got))
        sp = d.span(0, n); got = sp[1] if sp else 0
        if got != old_fill(n, lim, X): bad.append(('fill', n, lim, X, got))
        if d.route(n) != (n >= X): bad.append(('route', n, X))
    check(not bad, f'batched_span == the four rules it replaced (plan_request, spec.prefill_span, longctx.fill, route) on a grid '
                   f'({bad[:3]})')

    # -- no batched prefill: crossover None == the old 1 << 62 in the cache's plan (KPassOnly's checkpoints unchanged) -------
    diff = []
    for n, i0, c, mk in itertools.product((5, 40, 300, 9000, 40000), (0, 3, 100, 8200), (0, 1, 4),
                                          ({}, {'root': 50}, {'user': 120}, {'root': 30, 'user': 8300}, {'user': 2})):
        if i0 + c > n or c > i0: continue
        a = pc.plan_request(n, i0, c, mk, crossover=None, limit=262144, T=1)
        b = pc.plan_request(n, i0, c, mk, crossover=1 << 62, limit=262144, T=1)
        if a != b: diff.append((n, i0, c, mk, a, b))
    check(not diff, f'plan_request(crossover=None) == the old crossover 1 << 62 (KPassOnly\'s checkpoint placement) ({diff[:2]})')
    k = KPassOnly(type('M', (), dict(max_ctx=4096, devs=[0, 1]))())
    check(k.crossover is None and not k.route(10 ** 6) and k.span(0, 4000) is None and k.numerics()['moe'] == 'none',
          'KPassOnly: no crossover, route / span never batch, numerics none')
    for fn, what in ((lambda: k.validate([1, 2, 3], 0, 2), 'validate'), (lambda: k.run([1, 2, 3], 0, 0, 2), 'run')):
        try: fn(); ok = False
        except PrefillRefused: ok = True
        check(ok, f'KPassOnly.{what} refuses (no batched prefill)')

    # -- the shared refusals ----------------------------------------------------------------------------------------------
    d = PrefillDriver(); d.crossover, d.chunk, d.n_vocab = 4, 16, 100
    d.m = type('M', (), dict(max_ctx=64, poisoned=None))()
    cases = (([1] * 10, 0, 11, 'positions past the prompt'), ([1] * 10 + [100], 0, 5, 'a token past the vocabulary (the whole prompt)'),
             ([1] * 70, 0, 70, 'positions past max_ctx'))
    for toks, a, b, what in cases:
        try: d.validate(toks, a, b); ok = False
        except PrefillRefused: ok = True
        check(ok, f'PrefillDriver.validate refuses {what}')
    dr = type('D', (), dict(emb_vocab=50))()
    try: d.check_tokens([1, 60], dr); ok = False
    except PrefillRefused: ok = True
    check(ok, 'PrefillDriver.check_tokens: a drafter\'s smaller vocabulary bounds the tokens too')
    d.m.poisoned = 'x'
    try: d.validate([1] * 10, 0, 5); ok = False
    except PrefillRefused: ok = True
    check(ok, 'PrefillDriver.validate refuses on a poisoned engine')
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
