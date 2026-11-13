"""llama.cpp's own speculative decoding through llama-server, on the prompts benchmarks/q35_served_ab.py serves: the
yardstick beside Caminus's served `--draft` numbers (docs/dev/benchmarks.md). Starts llama-server plain and with the
drafting flags, interleaved over --rounds (plain, spec, spec, plain, ...), asks each prompt for --n greedy tokens and
reads the server's own decode timing (timings.predicted_per_second) and draft acceptance. Prints the median per arm and
prompt, the speed-up, and whether the drafted tokens equal the plain ones.

  tools/guard.sh LOG 3600 python -I benchmarks/llamacpp_spec_ab.py --bin ~/src/llama.cpp/build-hip/bin --model GGUF \
      --ids IDS.json --dev ROCm1 --out DIR [--spec '--spec-type draft-mtp --spec-draft-n-max 3'] [--kv f16] [--ctx 12288]

IDS.json: {"prompts": {"code": [ids], "prose": [...], "shell": [...]}} (q35_served_ab.py's --ids). The ids go to
/completion as they are; caminus serve puts 248044 first (one prompt token more). --kv f16 and --ctx 12288 match
q35_served_ab.py's caminus serve (f16 KV, --max-ctx 12288); --kv q8_0 --ctx 1024 is the setup of the earlier engine-level
and DFlash yardsticks. GPU: run under tools/guard.sh, on a card nothing else uses. Prints RESULT PASS when every run
answered (token equality is reported, not gated: llama.cpp's verify need not be bitwise)."""
import argparse, json, os, shlex, statistics, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--bin', required=True, help='the llama.cpp build bin/ directory (llama-server)')
ap.add_argument('--model', required=True)
ap.add_argument('--ids', required=True)
ap.add_argument('--dev', required=True, help="llama-server -dev (e.g. ROCm0 / ROCm1: llama-server --list-devices)")
ap.add_argument('--out', required=True)
ap.add_argument('--spec', default='--spec-type draft-mtp --spec-draft-n-max 3', help='the drafting flags')
ap.add_argument('--kv', default='f16', choices=('f16', 'q8_0'))
ap.add_argument('--ctx', type=int, default=12288)
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--port', type=int, default=8097)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
PROMPTS = json.load(open(A.ids))['prompts']
SPEC = shlex.split(A.spec)


def serve(label, extra):
    try:   # (a leftover server on the port would be timed instead of this one)
        urllib.request.urlopen(f'http://127.0.0.1:{A.port}/health', timeout=2).read()
        raise SystemExit(f'{label}: something already answers on port {A.port}; stop it or pass --port')
    except OSError:
        pass
    log = open(os.path.join(A.out, f'server-{label}.log'), 'w')
    cmd = [os.path.join(A.bin, 'llama-server'), '-m', A.model, '-dev', A.dev, '-ngl', '99', '-c', str(A.ctx), '-np', '1',
           '-fa', 'on', '-ctk', A.kv, '-ctv', A.kv, '-b', '512', '-ub', '256', '-t', '8', '--cache-ram', '0',
           '--port', str(A.port)] + extra
    print('$ ' + ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{A.port}/health', timeout=2).read(); return p
        except Exception:
            if p.poll() is not None: raise SystemExit(f'{label}: llama-server exited {p.returncode} (see server-{label}.log)')
            time.sleep(2)
    p.kill(); raise SystemExit(f'{label}: llama-server not up in 600 s')


def ask(ids):
    body = json.dumps({'prompt': ids, 'n_predict': A.n, 'temperature': 0, 'top_k': 1, 'cache_prompt': False,
                       'return_tokens': True, 'ignore_eos': True, 'stream': False}).encode()
    r = urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{A.port}/completion', data=body,
                                                      headers={'Content-Type': 'application/json'}), timeout=900)
    return json.loads(r.read())


runs = {}   # (arm, prompt) -> [dict(tps, tokens, draft_n, draft_n_accepted)]
order = []
for r in range(A.rounds):
    order += [('plain', 'spec'), ('spec', 'plain')][r % 2]
bad = 0
for i, arm in enumerate(order):
    p = serve(f'{arm}-{i}', SPEC if arm == 'spec' else [])
    try:
        ask(PROMPTS['code'][:8])   # (warm-up)
        for name, ids in PROMPTS.items():
            d = ask(ids); t = d.get('timings', {}); toks = d.get('tokens', [])
            if len(toks) != A.n: bad += 1
            rec = dict(tps=t.get('predicted_per_second', 0.0), tokens=toks, draft_n=t.get('draft_n', 0),
                       draft_n_accepted=t.get('draft_n_accepted', 0))
            runs.setdefault((arm, name), []).append(rec)
            acc = f", accepted {rec['draft_n_accepted']} / {rec['draft_n']}" if rec['draft_n'] else ''
            print(f'run {i} {arm:5} {name:5}: {rec["tps"]:6.1f} tok/s ({len(toks)} tokens){acc}', flush=True)
    finally:
        p.terminate()
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: p.kill()

print(f'\nllama-server {A.kv} KV, -c {A.ctx}, spec: {A.spec}; median of {A.rounds} runs per arm')
print(f'{"prompt":<8}{"plain":>8}{"spec":>8}{"x":>7}  accepted   tokens == plain (every spec run vs the first plain run)')
out = {}
for name in PROMPTS:
    pl, sp = runs[('plain', name)], runs[('spec', name)]
    mp, ms = statistics.median(x['tps'] for x in pl), statistics.median(x['tps'] for x in sp)
    dn, da = sum(x['draft_n'] for x in sp), sum(x['draft_n_accepted'] for x in sp)
    ref = pl[0]['tokens']
    plain_stable = all(x['tokens'] == ref for x in pl)
    firsts = [next((j for j, (a, b) in enumerate(zip(ref, x['tokens'])) if a != b), None if x['tokens'] == ref else min(len(ref), len(x['tokens'])))
              for x in sp]
    same = all(f is None for f in firsts); first = min((f for f in firsts if f is not None), default=None)
    print(f'{name:<8}{mp:8.1f}{ms:8.1f}{ms / mp:7.2f}  {100 * da / dn if dn else 0:5.0f} %   {same}'
          + ('' if same else f' ({sum(f is not None for f in firsts)} of {len(sp)} runs differ, first at {first})')
          + ('' if plain_stable else '; the plain runs differ among themselves'))
    out[name] = dict(plain=mp, spec=ms, accepted=(da / dn if dn else None), same=same, first_diff=first)
json.dump(dict(args=vars(A), summary=out, runs={f'{a}/{n}': v for (a, n), v in runs.items()}),
          open(os.path.join(A.out, 'llamacpp_spec_ab.json'), 'w'))
print('RESULT ' + ('PASS' if not bad else f'FAIL ({bad} replies short of {A.n} tokens)'))
sys.exit(1 if bad else 0)
