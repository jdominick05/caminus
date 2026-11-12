"""A served A/B of a one-card drafter on the dense qwen35 line: `caminus serve` (tools/caminus_serve.py) once per arm (--draft
off, then the drafter), the same greedy completions in each, interleaved by round: decode tok/s (the server's own timings,
predicted_per_second over the generated tokens) per prompt class, the median of --rounds rounds, and the replies' token ids
equal between the arms (greedy: a drafter must not change one token).

The prompts: a JSON file {"prompts": {name: [token ids]}} (--ids; the drafter's file: the catalog's, under the model directory; e.g. the inputs llama.cpp's own drafter was measured on,
so the two engines read the same tokens), sent as /v1/completions token arrays (the server puts 248044 first when it is
missing: one more prompt token than llama.cpp reads), temperature 0, --n tokens, no stop strings. Each arm's server runs
alone on --card; the cards' other work stops first (a GPU window). Prints the table, PASS / FAIL for the token identity and
RESULT PASS | FAIL.
  tools/guard.sh LOG 3600 python -I benchmarks/q35_served_ab.py --model GGUF --ids IDS.json --scratch DIR --card 0 \
      [--draft dflash] [--n 128] [--rounds 3] [--max-ctx 4096]"""
import argparse, json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import serve_smoke as S          # noqa: E402  (request helpers, LIVE cleanup)
from serve_smoke1 import Server1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--ids', required=True); ap.add_argument('--scratch', required=True)
ap.add_argument('--card', default='0'); ap.add_argument('--draft', default='dflash')
ap.add_argument('--n', type=int, default=128); ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--max-ctx', default='4096')
A = ap.parse_args()
prompts = json.load(open(A.ids))['prompts']
os.makedirs(A.scratch, exist_ok=True)


class Arm(argparse.Namespace):
    pass


def server(draft):
    a = Arm(model=A.model, max_ctx=A.max_ctx, scratch=os.path.join(A.scratch, draft), alias=None, cards=1, draft=draft,
            block_imports='gguf,tokenizers', card=A.card)
    os.makedirs(a.scratch, exist_ok=True)
    s = Server1(f'serve[{draft}]', a)
    if not s.wait_ready(): raise SystemExit(f'q35_served_ab: the {draft} server did not come up')
    return s


def complete(port, ids):
    r = S.request(port, 'POST', '/v1/completions', dict(model='m', prompt=list(ids), max_tokens=A.n, temperature=0,
                                                        stream=False))
    j = r['json']; tm = j.get('timings') or {}
    toks = j['choices'][0]['text']   # (greedy text equal <=> the same tokens, for these replies)
    return float(tm.get('predicted_per_second') or 0.0), toks, int(tm.get('predicted_n') or 0)


res = {arm: {k: [] for k in prompts} for arm in ('off', A.draft)}
out = {arm: {} for arm in res}
srv = {}
try:
    for rnd in range(A.rounds):
        for arm in ('off', A.draft) if rnd % 2 == 0 else (A.draft, 'off'):   # (the order alternates per round)
            s = server(arm)
            try:
                for k, ids in prompts.items():
                    complete(s.port, ids[:8])                     # (a short warm-up request: the first launch's costs)
                    tps, toks, n = complete(s.port, ids)
                    res[arm][k].append(tps); out[arm].setdefault(k, toks)
                    print(f'round {rnd} {arm:7s} {k:6s}: {tps:6.1f} tok/s ({n} tokens)', flush=True)
            finally:
                s.stop()
finally:
    for s in list(S.LIVE): s.stop()
ok = True
print(f'\nserved decode tok/s, median of {A.rounds} (card {A.card}, {A.n} tokens, greedy):')
print(f'{"":8s}{"off":>8s}{A.draft:>9s}{"x":>7s}')
for k in prompts:
    a, b = statistics.median(res['off'][k]), statistics.median(res[A.draft][k])
    print(f'{k:8s}{a:8.1f}{b:9.1f}{b / a if a else 0:7.2f}')
for k in prompts:
    same = out['off'][k] == out[A.draft][k]
    ok &= same; print(('PASS' if same else 'FAIL'), f'{k}: the {A.draft} reply == the plain reply (token for token)')
print('RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
sys.exit(0 if ok else 1)
