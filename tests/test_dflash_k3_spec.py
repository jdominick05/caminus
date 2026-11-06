"""K3 DFlash drafter gates on the Coder, both cards (run under guard.sh), for both loops of caminus.dflash_k3.DFlashK3Drafter:
'dev' (device_accept: the S3 loop, the accept on the device inside the drafter's 'step' launch) and 'old' (caminus.spec.
generate's generic loop: host accept, inject fused into the next draft):
  spec == plain greedy (a k3_smoke run's tp1 tokens) on prose / code / shell, 128 tokens, rep 1 drafts == rep 2, and the two
  loops' drafts identical pass for pass;
  the device accept's tokens == the host argmax rule on the same logits shards, every pass;
  corrupted drafts at position 0 / 1 / 2 and a pseudo-random position per pass: output == plain (both loops);
  forced bails (a 1 us wait bound on the first attempt): dev: a step launch, the target's verify (then the step reruns), an
  inject launch; old: a fused draft launch, an inject launch. Output == plain, drafts identical to the unbailed run's.
usage: guard.sh LOG SECS python -I tests/test_dflash_k3_spec.py [--n 128] [--plain SMOKE.json]"""
import sys, os, json, time, argparse
R_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, os.path.join(R_, 'python')); sys.path.insert(0, os.path.join(R_, 'tools'))
import numpy as np, torch
from caminus import spec, dflash_k3 as dk
import k3_smoke as ks

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=128)
ap.add_argument('--save-logs', default=None); ap.add_argument('--cmp-logs', default=None)
# (the plain tokens of the split-attention build 27d4479: its summation order changed the greedy text at near-ties)
from caminus import devpaths  # noqa: E402
ap.add_argument('--plain', default=devpaths.artifacts('smoke', '20261103-084114.json'))
a = ap.parse_args()
ref = ks.load_ref(); cfg = dict(ref['models']['coder']); cfg['drafter'] = ref['drafter']; cfg['name'] = 'coder'
plain = json.load(open(a.plain))['stages']['tp1']['plain']
torch.cuda.set_device(0)
m, dr, info = dk.smoke_factory(cfg, 4)
df = dr.df
print('built', info, 'inject on the XTX:', df.inj_remote, flush=True)
FAIL = []


def gen(nm, path, corrupt=None, n=None):
    dr.device_accept = path == 'dev'; dr.corrupt = corrupt; dr.log = []
    st = {}; p = ks.prompt_tokens(cfg, nm, 200)
    out = spec.generate(m, p, a.n if n is None else n, dr, st)[len(p):]
    log = dr.log; dr.corrupt = None; dr.log = None
    return out, st, log


def check(tag, nm, toks):
    ok = toks == plain[nm]['tokens'][:len(toks)] and len(toks) == a.n
    if not ok:
        fd = next((i for i in range(min(len(toks), a.n)) if toks[i] != plain[nm]['tokens'][i]), None)
        FAIL.append(f'{tag} {nm}: != plain (first difference {fd})')
    return ok


def diff(x, y): return [(i, u, v) for i, (u, v) in enumerate(zip(x, y)) if u != v] + ([('len', len(x), len(y))] if len(x) != len(y) else [])


for path in ('old', 'dev'): gen('prose', path, n=32)   # warm-up
base = {}
for path in ('old', 'dev'):
    for nm in ks.TEXTS:
        for rep in (1, 2):
            dr.check_argmax = path == 'dev' and rep == 2; c0, m0 = dr.argmax_checks, dr.argmax_mismatch
            toks, st, log = gen(nm, path)
            dr.check_argmax = False
            ok = check(f'{path} rep{rep}', nm, toks)
            gp = st['gen_passes']; acc = 100 * st['accepted'] / (gp * 3)
            extra = (f'; host argmax checks {dr.argmax_checks - c0}, mismatches {dr.argmax_mismatch - m0}' if path == 'dev' and rep == 2 else '')
            if path == 'dev' and rep == 2 and (dr.argmax_mismatch - m0 or not dr.argmax_checks - c0): FAIL.append(f'dev {nm}: device accept tokens != host argmax')
            print(f'{path} {nm} rep{rep}: == plain {ok}; {(st["gen_tokens"] - 1) / st["gen_seconds"]:.1f} tok/s, {gp} passes, acceptance {acc:.2f}%, '
                  f'draft {1e3 * st["draft_seconds"] / gp:.3f} ms/pass ({"step launch device time" if path == "dev" else "host wall"}), '
                  f'verify {1e3 * st["verify_seconds"] / gp:.2f}{extra}', flush=True)
            if rep == 1: base[(path, nm)] = log
            elif log != base[(path, nm)]:
                dd = diff(base[(path, nm)], log); FAIL.append(f'{path} {nm}: rep 2 drafts != rep 1 ({len(dd)})'); print('   ', dd[:6], flush=True)
if a.save_logs: json.dump({f'{p_}:{n_}': v for (p_, n_), v in base.items()}, open(a.save_logs, 'w'))
if a.cmp_logs:   # drafts of another build / configuration (e.g. DF_HEAD_SHIFT=0), pass for pass
    other = json.load(open(a.cmp_logs))
    for (p_, n_), v in base.items():
        dd = diff(other[f'{p_}:{n_}'], v)
        if dd: FAIL.append(f'{p_} {n_}: drafts != {a.cmp_logs} ({len(dd)})'); print('   vs saved:', dd[:6], flush=True)
        print(f'drafts {p_} {n_} == saved logs: {not dd}', flush=True)
for nm in ks.TEXTS:
    dd = diff(base[('old', nm)], base[('dev', nm)])
    if dd: FAIL.append(f'{nm}: dev-loop drafts != old-loop drafts ({len(dd)})'); print('   old vs dev:', dd[:6], flush=True)
    print(f'drafts old loop == dev loop on {nm}: {not dd} ({len(base[("dev", nm)])} passes)', flush=True)
# corrupted drafts
rng = np.random.default_rng(0)
for path in ('dev', 'old'):
    for mode in ((0, 1, 2, 'rand') if path == 'dev' else (0, 'rand')):
        for nm in ks.TEXTS:
            def corrupt(i, x, mode=mode):
                j = mode if mode != 'rand' else int(rng.integers(0, len(x)))
                x = list(x); x[j] = (x[j] + 7919) % df.vocab; return x
            toks, st, log = gen(nm, path, corrupt)
            ok = check(f'{path} corrupt@{mode}', nm, toks)
            print(f'{path} corrupted drafts at {mode}: {nm} == plain {ok} ({st["gen_passes"]} passes, accepted {st["accepted"]})', flush=True)
# forced bails
orig_launch = m.launch_tokens
def target_bail(nth):
    cnt = [0]
    def launch(tokens, i0, commit):
        cnt[0] += 1
        if cnt[0] == nth:
            for E in m.engines: E.limit_us = 1
            try: orig_launch(tokens, i0, commit)
            finally:
                for E in m.engines: E.limit_us = df.limit_us
            print(f'   (target launch #{nth} at {i0} armed with a 1 us wait bound)', flush=True)
        else: orig_launch(tokens, i0, commit)
    m.launch_tokens = launch
for path, kind, nth in (('dev', 'step', 3), ('dev', 'step', 11), ('dev', 'target', 7), ('dev', 'inject', 5),
                        ('old', 'fused', 3), ('old', 'inject', 5)):
    nm = 'code'; nb0, tb0 = len(df.bails), len(m.bails)
    if kind == 'target': target_bail(nth)
    else: df.force_bail(kind, nth)
    toks, st, log = gen(nm, path)
    m.launch_tokens = orig_launch
    ok = check(f'{path} bail {kind}#{nth}', nm, toks); same = log == base[(path, nm)]
    nb = (len(m.bails) - tb0) if kind == 'target' else (len(df.bails) - nb0)
    if not same: dd = diff(base[(path, nm)], log); FAIL.append(f'{path} bail {kind}#{nth}: drafts differ ({len(dd)})'); print('   ', dd[:4], flush=True)
    if nb < 1: FAIL.append(f'{path} bail {kind}#{nth}: no bail happened')
    print(f'{path} forced bail {kind} #{nth}: bails {nb}, == plain {ok}, drafts == unbailed run {same}', flush=True)
print(f'replicas {m.check_replicas()}; target bails {len(m.bails)}; drafter bails {len(df.bails)}; stats {df.stats}')
print('RESULT ' + ('PASS' if not FAIL else f'FAIL {FAIL}'), flush=True)
sys.exit(0 if not FAIL else 1)
