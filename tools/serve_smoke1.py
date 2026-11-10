"""The end-to-end smoke of the REAL caminus serve with a ONE-CARD model (qwen35moe: Qwen3.6-35B-A3B GSQ / HauhauCS; qwen35:
the dense qwen35 line, AREX-2): tools/caminus_serve.py as a child on one card (HIP_VISIBLE_DEVICES=--card), scratch cache /
flight directories only, then from outside, with a stdlib client (tools/serve_smoke.py's helpers):
  ids       /health 200, /props (the caminus engine marker, the model alias), /v1/models (alias and alias-nothink), /slots
  greedy    an OpenAI chat stream at temperature 0 (no thinking): content, finish, timings (decode tok/s)
  sampled   an OpenAI chat stream at temperature 0.7 with a tool (the device sampler's path: the two vocabulary shards on one
            card), then the tool's result back as the next turn (the prompt cache: the live line continued)
  anthropic an Anthropic /v1/messages stream: a text block
  long      a ~8K-token prompt (the batched prefill): TTFT and prompt tok/s from the timings
  needle    (--needle N) a passphrase at 37 % of a ~N-token prompt of synthetic records: the answer holds it; TTFT, prompt tok/s
  --block-imports gguf,tokenizers: the server runs under an import hook refusing those packages, and every code object it
            reports having loaded (/props caminus.code_objects, read after the requests) has no ggml / llama.cpp header in
            its source's include closure (the own-code gate)
  surface   /metrics (llamacpp: names and caminus: series), /caminus/status (one card, its name and guard line), no request
            text in the metrics / status / slots / flight files (a marker string), llm-top -once (--llm-top BIN)
  abort     a stream closed by the client after a few events, then a request: HTTP 200
  shutdown  SIGTERM: exit 0 and the shutdown park line; --restart: a second server on the same directories, the long prompt
            again: served from the disk tier (cached tokens > 0)
usage: tools/guard.sh LOG SECS python -I tools/serve_smoke1.py --model GGUF --scratch DIR [--card 1] [--max-ctx 32768]
           [--alias qwen3.6] [--llm-top BIN] [--restart]
Prints PASS / FAIL lines and RESULT; exit 0 only if every check passed. The scratch directory is refused under
~/.cache/caminus, $ARTIFACTS or /tmp (serve_smoke.safe_dir)."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import serve_smoke as S   # noqa: E402  (its request / SSE helpers, check, safe_dir, llm_top_check, the guard lines)


# --block-imports: run tools/caminus_serve.py under a meta-path hook that raises ImportError for the named packages (and
# prints the refusal), so a server that serves at all is proof none of them is needed on its path
BLOCKER = '''import sys, importlib.abc, runpy
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {block!r}:
            print('serve_smoke1: import refused: ' + name, flush=True); raise ImportError('blocked by serve_smoke1: ' + name)
sys.meta_path.insert(0, _Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


class Server1:
    """tools/caminus_serve.py on one card, its output followed (the port, the pid, 'ready on')"""
    def __init__(self, tag, a):
        self.tag, self.lines, self.port, self.ready = tag, [], None, threading.Event()
        cmd = [sys.executable, '-I', os.path.join(ROOT, 'tools', 'caminus_serve.py'), '--port', '0', '--model', a.model,
               '--max-ctx', str(a.max_ctx), '--cache-dir', os.path.join(a.scratch, 'kv'), '--flight-dir', os.path.join(a.scratch, 'flight')]
        if a.alias: cmd += ['--model-alias', a.alias]
        if a.block_imports:   # the server with an import hook that refuses these top-level packages (the own-code gate)
            cmd = cmd[:2] + ['-c', BLOCKER.format(block=tuple(a.block_imports.split(','))), cmd[2]] + cmd[3:]
        print(f'{tag}: HIP_VISIBLE_DEVICES={a.card} {" ".join(c for c in cmd[2:] if not c.startswith("import sys"))}', flush=True)
        self.t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  env=dict(os.environ, PYTHONUNBUFFERED='1', HIP_VISIBLE_DEVICES=str(a.card)))
        S.LIVE.append(self)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            line = line.rstrip('\n'); self.lines.append(line); print(f'  {self.tag}| {line}', flush=True)
            if self.port is None:
                mm = re.search(r'http://[^:]+:(\d+)', line)
                if mm: self.port = int(mm.group(1))
            if 'ready on' in line: self.ready.set()
        self.ready.set()

    def wait_ready(self, timeout=1200):
        self.ready.wait(timeout); self.load_s = time.time() - self.t0
        return self.ready.is_set() and self.p.poll() is None

    def stop(self, timeout=240):
        if self.p.poll() is None: self.p.send_signal(signal.SIGTERM)
        try: self.p.wait(timeout)
        except subprocess.TimeoutExpired: self.p.kill(); self.p.wait(60)
        return self.p.returncode


def chat(port, model, messages, stream=True, tools=None, **kw):
    body = dict(model=model, messages=messages, stream=stream, **kw)
    if tools: body['tools'] = tools
    r = S.request(port, 'POST', '/v1/chat/completions', body)
    return r, (S.openai_collect(r) if stream else r['json'])


TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'the current weather of a city',
                                            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--scratch', required=True)
    ap.add_argument('--card', default='1', help='HIP_VISIBLE_DEVICES of the server (1: the XT, 0: the XTX; 0,1: a two-card engine)')
    ap.add_argument('--cards', type=int, default=None, help='the cards /caminus/status must list (default: the --card count)')
    ap.add_argument('--max-ctx', default='auto');   # (auto: the server's default, the largest context that fits)
    ap.add_argument('--alias', default=None)
    ap.add_argument('--llm-top', default=None); ap.add_argument('--restart', action='store_true')
    ap.add_argument('--needle', type=int, default=0, help='also a needle at 37 %% of a prompt of about this many tokens')
    ap.add_argument('--block-imports', default='', help='comma list of packages the server must not import (e.g. gguf,tokenizers)')
    a = ap.parse_args()
    a.cards = a.cards or len(str(a.card).split(','))
    a.scratch = S.safe_dir(a.scratch); os.makedirs(a.scratch, mode=0o700, exist_ok=True)
    check = S.check
    srv = Server1('server', a)
    ok = srv.wait_ready()
    auto = [l.split('caminus serve: ')[-1] for l in srv.lines if 'max_ctx auto' in l]
    built = [l for l in srv.lines if 'engine built' in l]
    cache = re.search(r'cache (\d+) VRAM slots', built[-1]) if built else None
    check(ok, f'server ready (HIP_VISIBLE_DEVICES={a.card}) in {getattr(srv, "load_s", 0):.0f} s'
              + (f'; {auto[-1]}' if auto else f'; max_ctx {a.max_ctx}') + (f'; {cache.group(1)} VRAM cache slots' if cache else ''))
    if not ok: return finish()
    port = srv.port
    st, h = S.get(port, '/health'); st2, props = S.get(port, '/props'); st3, models = S.get(port, '/v1/models'); st4, slots = S.get(port, '/slots')
    ids = [x.get('id') for x in (models or {}).get('data', [])] if isinstance(models, dict) else []
    alias = (props or {}).get('model_alias') if isinstance(props, dict) else None
    check(st == 200 and st2 == 200 and st4 == 200 and isinstance(props, dict) and props.get('engine') == 'caminus' and alias in ids
          and f'{alias}-nothink' in ids, f'ids: /health {st}, /props {st2} (engine {props.get("engine") if isinstance(props, dict) else None}, '
                                         f'alias {alias}), /v1/models {ids}, /slots {st4}')
    marker = S.MARK
    # greedy, no thinking
    r, m = chat(port, f'{alias}-nothink', [{'role': 'user', 'content': f'In one sentence, what does a mutex do? ({marker})'}],
                temperature=0, max_tokens=64)
    tg = (m.get('timings') or {}).get('predicted_per_second')
    check(r['status'] == 200 and m['content'].strip() and m['finish'] in ('stop', 'length') and not m['error'],
          f'greedy: {len(m["content"])} chars, finish {m["finish"]}, decode {tg} tok/s')
    S.NUMBERS['greedy_decode_tps'] = tg
    # sampled with a tool, then the tool result as the next turn
    msgs = [{'role': 'system', 'content': f'You are a helpful assistant. Use tools when they help. ({marker})'},
            {'role': 'user', 'content': 'What is the weather in Lisbon right now? Use the tool.'}]
    r, m = chat(port, alias, msgs, tools=TOOLS, temperature=0.7, top_p=0.9, max_tokens=512)
    calls = m['calls']
    check(r['status'] == 200 and not m['error'] and (calls or m['content'].strip()),
          f'sampled (T 0.7, a tool): finish {m["finish"]}, {len(calls)} tool call(s) {[c["name"] for c in calls]}, '
          f'{len(m["content"])} chars, reasoning {len(m["reasoning"])} chars')
    if calls:
        c = calls[0]
        try: args = json.loads(c['arguments'] or '{}')
        except ValueError: args = None
        check(c['name'] == 'get_weather' and isinstance(args, dict) and 'city' in args, f'the tool call parsed: {c["name"]}({c["arguments"]})')
        msgs += [{'role': 'assistant', 'content': m['content'] or None, 'reasoning_content': m['reasoning'] or None,
                  'tool_calls': [{'id': c['id'] or 'call_0', 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}]},
                 {'role': 'tool', 'tool_call_id': c['id'] or 'call_0', 'content': json.dumps({'city': args.get('city') if args else 'Lisbon', 'temp_c': 19, 'sky': 'clear'})}]
        r2, m2 = chat(port, alias, msgs, tools=TOOLS, temperature=0.7, top_p=0.9, max_tokens=512)
        tm = m2.get('timings') or {}
        check(r2['status'] == 200 and not m2['error'] and (m2['content'].strip() or m2['calls']),
              f'the tool result turn: {len(m2["content"])} chars; prompt {tm.get("prompt_n")} new of {tm.get("cache_n", "?")} cached (the live line continued)')
    # anthropic
    r = S.request(port, 'POST', '/v1/messages', dict(model=f'{alias}-nothink', max_tokens=64, stream=True,
                                                     messages=[{'role': 'user', 'content': f'Name three prime numbers. ({marker})'}]),
                  headers={'anthropic-version': '2023-06-01', 'x-api-key': 'dummy'})
    am = S.anthropic_collect(r)
    txt = ''.join(b.get('text', '') for b in am['blocks'] if b.get('type') == 'text')
    check(r['status'] == 200 and txt.strip() and not am['error'], f'anthropic: {len(txt)} chars, stop {am["stop"]}')
    # long prompt: the batched prefill
    sysm = S.guidelines(7, 30000, 'Project guidelines') + f' ({marker})'
    lm = [{'role': 'system', 'content': sysm}, {'role': 'user', 'content': 'Summarize the guidelines in two sentences.'}]
    t0 = time.perf_counter(); r, m = chat(port, f'{alias}-nothink', lm, temperature=0, max_tokens=64); tt = (r['t_first'] or r['t_end']) - r['t_send']
    tm = m.get('timings') or {}
    check(r['status'] == 200 and m['content'].strip(), f'long: prompt {tm.get("prompt_n")} tokens at {tm.get("prompt_per_second")} tok/s, '
                                                       f'TTFT {tt * 1e3:.0f} ms (the first event), decode {tm.get("predicted_per_second")} tok/s')
    S.NUMBERS['long'] = dict(prompt_n=tm.get('prompt_n'), prompt_tps=tm.get('prompt_per_second'), ttft_ms=tt * 1e3)
    if a.needle:   # a needle at 37 % of a ~a.needle-token prompt (serve_smoke's synthetic records): retrieved, TTFT, prompt speed
        needle = 'amber-falcon-2291'
        probe = S.records(44, 400, -1, needle)
        cnt = S.request(port, 'POST', '/v1/messages/count_tokens', {'model': alias, 'messages': [{'role': 'user', 'content': probe}]},
                        {'anthropic-version': '2023-06-01', 'x-api-key': 'dummy'})
        per = (cnt.get('json') or {}).get('input_tokens', 12000) / 400.0
        n = int((a.needle - 200) / per); at = int(n * 0.37)
        text = S.records(44, n, at, needle) + f'\n\nWhich passphrase is stated in record {at:06d}? Answer with the passphrase only.'
        r, m = chat(port, f'{alias}-nothink', [{'role': 'user', 'content': text}], temperature=0, max_tokens=24)
        tm = m.get('timings') or {}; tt = (r['t_first'] or r['t_end']) - r['t_send']; ans = m.get('content', '')
        check(r['status'] == 200 and needle in ans, f'needle: prompt {tm.get("prompt_n")} tokens (record {at} of {n}) at '
              f'{tm.get("prompt_per_second")} tok/s, TTFT {tt * 1e3:.0f} ms, decode {tm.get("predicted_per_second")} tok/s; answer {ans[:40]!r}')
        S.NUMBERS['needle'] = dict(prompt_n=tm.get('prompt_n'), prompt_tps=tm.get('prompt_per_second'), ttft_ms=tt * 1e3,
                                   decode_tps=tm.get('predicted_per_second'), found=needle in ans, answer=ans[:60])
    if a.block_imports:   # the own-code gate's device half: what the process loaded, by its own record
        from own_closure import audit
        objs = ((S.get(port, '/props')[1] or {}).get('caminus') or {}).get('code_objects') or []
        rep = {o: audit(o + '.hip') for o in objs}
        bad = {o: (r[1], r[2]) for o, r in rep.items() if r[1] or r[2]}
        refused = [l for l in srv.lines if 'import refused' in l]
        check(objs and not bad and not refused, f'own code: the server loaded {objs}: no ggml / llama.cpp header in their include '
              f'closures ({bad or "none"}); imports refused by the hook: {len(refused)} ({a.block_imports} never asked for)')
    # the monitoring surface
    st, met = S.get(port, '/metrics'); st2, status = S.get(port, '/caminus/status'); st3, slots = S.get(port, '/slots')
    cards = (status or {}).get('cards') if isinstance(status, dict) else None
    met = met if isinstance(met, str) else json.dumps(met)
    check(st == 200 and 'llamacpp:prompt_tokens_total' in met and 'caminus:' in met,
          f'/metrics: llamacpp: names {"llamacpp:prompt_tokens_total" in met}, caminus: series {"caminus:" in met}')
    names = [c.get('name') for c in cards] if isinstance(cards, list) else None
    check(st2 == 200 and isinstance(cards, list) and len(cards) == a.cards, f'/caminus/status: {a.cards} card(s) {names} '
          f'(guard {[c.get("guard_mib") for c in cards] if isinstance(cards, list) else None} MiB)')
    leak = [n for n, t in (('metrics', met), ('status', json.dumps(status)), ('slots', json.dumps(slots))) if marker in t]
    fdir = os.path.join(a.scratch, 'flight')
    for root, _, files in os.walk(fdir):
        for f in files:
            try:
                if marker.encode() in open(os.path.join(root, f), 'rb').read(): leak.append(os.path.join(root, f))
            except OSError: pass
    check(not leak, f'no request text in the metrics / status / slots / flight files ({leak or "none"})')
    if a.llm_top: S.llm_top_check(a, port)
    # abort mid-generation, then a request (a long, unavoidable output; closed after the third event; one retry with a longer
    # request if the model still finishes first: a model that stops early makes the case moot, not the server wrong)
    asks = ['Write a Python module defining 150 functions f1 .. f150, each a one-line `def fN(): return N`, one per line. Output only the code.',
            'Write a Python module defining 400 functions g1 .. g400, each a one-line `def gN(x): return x + N`, one per line. Output only the code.']
    for ask in asks:
        r = S.request(port, 'POST', '/v1/chat/completions', dict(model=f'{alias}-nothink', stream=True, max_tokens=1200, temperature=0,
                                                                 messages=[{'role': 'user', 'content': ask}]),
                      close_when=lambda evs: len(evs) >= 3)
        if r['closed']: break
    time.sleep(1.0)
    r2, m2 = chat(port, f'{alias}-nothink', [{'role': 'user', 'content': 'Say OK.'}], temperature=0, max_tokens=8)
    check(r['closed'] and r2['status'] == 200 and not m2['error'], f'abort: closed after {len(r["events"])} events; the next request {r2["status"]}')
    # shutdown (+ restart from the disk tier)
    rc = srv.stop()
    park = [l for l in srv.lines if 'shutdown: park' in l]
    check(rc == 0 and park, f'SIGTERM: exit {rc}; {park[-1].split("caminus serve: ")[-1] if park else "no park line"}')
    if a.restart:
        srv2 = Server1('restart', a)
        ok = srv2.wait_ready()
        check(ok, f'restart on the same directories: ready in {getattr(srv2, "load_s", 0):.0f} s')
        if ok:
            r, m = chat(srv2.port, f'{alias}-nothink', lm, temperature=0, max_tokens=64)
            tm = m.get('timings') or {}
            # (from the disk tier: almost every position cached. A cold rerun means the entry was refused, e.g. the engine key
            # changed: it hashes the package sources and build/*.hsaco, so editing them during a run makes this fail)
            check(r['status'] == 200 and m['content'].strip() and (tm.get('cache_n') or 0) >= (tm.get('prompt_n') or 0),
                  f'restart: the long prompt again: prompt {tm.get("prompt_n")} new, '
                                                               f'cache {tm.get("cache_n")} (the disk tier: {[l for l in srv2.lines if "disk" in l][-1:]})')
            srv2.stop()
    return finish()


def finish():
    if S.NUMBERS: print('NUMBERS ' + json.dumps(S.NUMBERS))
    print('RESULT ' + ('PASS' if not S.FAILS else f'FAIL ({len(S.FAILS)})'))
    S.stop_servers()
    return 0 if not S.FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
