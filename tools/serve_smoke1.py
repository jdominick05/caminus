"""The end-to-end smoke of the REAL caminus serve with a ONE-CARD model (qwen35moe: Qwen3.6-35B-A3B GSQ / HauhauCS; qwen35:
the dense qwen35 line, AREX-2): tools/caminus_serve.py as a child on one card (HIP_VISIBLE_DEVICES=--card), scratch cache /
flight directories only, then from outside, with a stdlib client (tools/serve_smoke.py's helpers). With
CAMINUS_SMOKE_INSTALLED=1 it uses this interpreter's installed caminus CLI and packaged kernels (no editable install,
checkout fallback or kernel_dir override); the own-code gate requires matching checkout kernel sources:
  ids       /health 200, /props (the caminus engine marker, the model alias), /v1/models (alias and alias-nothink), /slots
  greedy    an OpenAI chat stream at temperature 0 (no thinking): content, finish, timings (decode tok/s)
  sampled   an OpenAI chat stream at temperature 0.7, then a separate weather tool round trip: a natural call first (one
            same-model tool_choice required retry if absent, reported), all calls / results replayed, then an answer using
            the synthetic observations with tool_choice none (the prompt cache: the live line continued)
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
import argparse, glob
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
    """Checkout or installed caminus on the selected cards, output followed through shutdown/restart."""
    def __init__(self, tag, a):
        self.tag, self.lines, self.port, self.ready = tag, [], None, threading.Event()
        cmd = server_command(a)
        print(f'{tag}: {"installed" if S.INSTALLED else "checkout"} HIP_VISIBLE_DEVICES={a.card} '
              f'{" ".join(c for c in cmd[2:] if not c.startswith("import sys"))}', flush=True)
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


def server_command(a):
    """Same serve arguments in both modes; installed helper never adds the checkout to sys.path."""
    argv = ['--port', '0', '--model', a.model, '--max-ctx', str(a.max_ctx),
            '--cache-dir', os.path.join(a.scratch, 'kv'), '--flight-dir', os.path.join(a.scratch, 'flight')]
    if a.alias: argv += ['--model-alias', a.alias]
    argv += ['--cards', str(a.cards)]
    if a.draft: argv += ['--draft', a.draft]
    if S.INSTALLED:
        cmd = [sys.executable, '-I', os.path.join(ROOT, 'tools', 'serve_smoke_installed.py')]
        if a.block_imports:
            # Parent only: this helper imports checkout code to hash its sources.
            # The isolated child receives just the digest, never that import path.
            import importlib.util
            spec = importlib.util.spec_from_file_location('smoke_kernel_sources', os.path.join(ROOT, 'tools', 'release', 'stage_kernels.py'))
            sources = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sources)
            cmd += ['--block-imports', a.block_imports, '--kernel-tree-sha256', sources.kernel_tree_sha256(ROOT)]
        return cmd + ['--'] + argv
    cmd = [sys.executable, '-I', os.path.join(ROOT, 'tools', 'caminus_serve.py')] + argv
    if a.block_imports:
        cmd = cmd[:2] + ['-c', BLOCKER.format(block=tuple(a.block_imports.split(','))), cmd[2]] + cmd[3:]
    return cmd


def chat(port, model, messages, stream=True, tools=None, **kw):
    body = dict(model=model, messages=messages, stream=stream, **kw)
    if tools: body['tools'] = tools
    r = S.request(port, 'POST', '/v1/chat/completions', body)
    return r, (S.openai_collect(r) if stream else r['json'])


TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'the current weather of a city',
                                            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


def _stream_ok(r, m):
    """Require a successful, complete chat stream before inspecting or replaying its contents."""
    return (r.get('status') == 200 and not r.get('closed') and isinstance(m, dict)
            and m.get('role') is True and m.get('done') is True and m.get('error') is None
            and m.get('finish') in ('stop', 'length', 'tool_calls')
            and isinstance(m.get('content'), str) and isinstance(m.get('reasoning'), str)
            and isinstance(m.get('calls'), list))


def _weather_args(calls):
    """Validate the whole call batch before replaying any of it; malformed output is a failed check."""
    def non_json_constant(value):
        raise ValueError(value)

    args, ids = [], set()
    for c in calls:
        if (not isinstance(c, dict) or c.get('name') != TOOLS[0]['function']['name']
                or not isinstance(c.get('id'), str) or not c['id'].strip() or c['id'] in ids
                or not isinstance(c.get('arguments'), str)):
            return None
        try:
            a = json.loads(c['arguments'], parse_constant=non_json_constant)
        except (ValueError, RecursionError):
            return None
        if not isinstance(a, dict) or not isinstance(a.get('city'), str) or not a['city'].strip():
            return None
        ids.add(c['id']); args.append(a)
    return args


def weather_round_trip(port, model, marker, *, chat_fn=chat, check_fn=S.check):
    """A bounded tool acceptance check, also driven by fake chat / request functions in CPU tests.

    Keep the model and thinking policy fixed. Text alone is not a pass: at most one forced retry must produce valid
    calls, and every result's observation code must appear in a completed final answer without further calls.
    """
    stats = dict(forced=0, calls=0, answered=False)
    msgs = [{'role': 'system', 'content': f'You are a helpful assistant. Use tools when they help. ({marker}) '
             'Give a short weather answer and include each tool result\'s observation_code verbatim.'},
            {'role': 'user', 'content': 'What is the weather in Lisbon right now? Use the tool.'}]
    kw = dict(tools=TOOLS, temperature=0.7, top_p=0.9, max_tokens=512)
    r, m = chat_fn(port, model, msgs, **kw)
    ok = _stream_ok(r, m)
    check_fn(ok, f'weather natural tool turn: HTTP {r.get("status")}, complete stream {ok}')
    if not ok:
        return stats
    if not m['calls']:
        stats['forced'] = 1
        print(f'weather: no natural tool call (finish {m["finish"]}); retrying the same model with tool_choice required', flush=True)
        r, m = chat_fn(port, model, msgs, tool_choice='required', **kw)
        ok = _stream_ok(r, m)
        check_fn(ok, f'weather required tool turn: HTTP {r.get("status")}, complete stream {ok}')
        if not ok:
            return stats
    calls = m['calls']
    args = _weather_args(calls)
    ok = bool(calls) and m['finish'] == 'tool_calls' and args is not None
    check_fn(ok, f'weather calls: {len(calls)} call(s), finish {m["finish"]}; '
             'declared name, JSON object with nonempty city string, nonempty unique IDs')
    if not ok:
        return stats
    stats['calls'] = len(calls)
    msgs = msgs + [{'role': 'assistant', 'content': m['content'] or None, 'reasoning_content': m['reasoning'] or None,
                   'tool_calls': [{'id': c['id'], 'type': 'function',
                                   'function': {'name': c['name'], 'arguments': c['arguments']}} for c in calls]}]
    codes = []
    for i, (c, a) in enumerate(zip(calls, args)):
        code = f'weather-observation-{i + 1:04d}'
        codes.append(code)
        msgs.append({'role': 'tool', 'tool_call_id': c['id'],
                     'content': json.dumps(dict(city=a['city'], temp_c=19, sky='clear', observation_code=code))})
    r, m = chat_fn(port, model, msgs, tool_choice='none', **kw)
    ok = (_stream_ok(r, m) and m['finish'] == 'stop' and not m['calls'] and bool(m['content'].strip())
          and bool(re.search(r'(?<!\d)19(?!\d)', m['content']))
          and all(re.search(r'(?<![\w-])' + re.escape(code) + r'(?![\w-])', m['content']) for code in codes))
    stats['answered'] = ok
    tm = m.get('timings') if isinstance(m, dict) else None
    tm = tm if isinstance(tm, dict) else {}
    check_fn(ok, f'weather result turn: HTTP {r.get("status")}, usable answer with all {len(codes)} observation code(s), '
             f'no further calls; prompt {tm.get("prompt_n")} new of {tm.get("cache_n", "?")} cached (the live line continued)')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--scratch', required=True)
    ap.add_argument('--card', default='1', help='HIP_VISIBLE_DEVICES of the server (1: the XT, 0: the XTX; 0,1: a two-card engine)')
    ap.add_argument('--cards', type=int, default=None, help='the cards the server runs the model on (caminus serve --cards) and '
                    '/caminus/status must list (default: the --card count)')
    ap.add_argument('--max-ctx', default='auto');   # (auto: the server's default, the largest context that fits)
    ap.add_argument('--alias', default=None)
    ap.add_argument('--llm-top', default=None); ap.add_argument('--restart', action='store_true')
    ap.add_argument('--needle', type=int, default=0, help='also a needle at 37 %% of a prompt of about this many tokens')
    ap.add_argument('--block-imports', default='', help='comma list of packages the server must not import (e.g. gguf,tokenizers)')
    ap.add_argument('--draft', choices=('off', 'mtp', 'dflash'), default=None,
                    help='the drafter kind the server runs (caminus serve --draft); the greedy reply\'s sha256 is in NUMBERS, '
                         'so two runs (off vs a drafter) with the same fixed prompts compare token for token')
    ap.add_argument('--side-question', action='store_true', help='also a side question (Qwen Code\'s /btw prompt) between two '
                    'turns of a chat: it runs in the side slot and the next turn continues the main slot')
    a = ap.parse_args()
    a.cards = a.cards or len(str(a.card).split(','))
    a.scratch = S.safe_dir(a.scratch); os.makedirs(a.scratch, mode=0o700, exist_ok=True)
    S.NUMBERS['installed_package'] = S.INSTALLED
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
    import hashlib
    S.NUMBERS['greedy_sha256'] = hashlib.sha256(m['content'].encode()).hexdigest()   # (the fixed MARK: same prompt every run)
    # sampled with a fixed seed, no thinking: the draws are indexed by position, so the text is the same with any drafter
    # (--draft mtp / dflash / off) - compare sampled_sha256 across serves, as greedy_sha256
    r, m = chat(port, f'{alias}-nothink', [{'role': 'user', 'content': f'Write a short Python function that reverses a linked list. ({marker})'}],
                temperature=0.7, top_p=0.9, seed=20261113, max_tokens=160)
    ts = (m.get('timings') or {}).get('predicted_per_second')
    check(r['status'] == 200 and m['content'].strip() and not m['error'],
          f'sampled (seed, T 0.7): {len(m["content"])} chars, finish {m["finish"]}, decode {ts} tok/s')
    S.NUMBERS['sampled_decode_tps'] = ts
    S.NUMBERS['sampled_sha256'] = hashlib.sha256(m['content'].encode()).hexdigest()
    S.NUMBERS['tool_round_trip'] = weather_round_trip(port, alias, marker)
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
    if a.side_question:   # a /btw between two turns: the side slot, then the next turn continues the main slot
        BTW = ('<system-reminder>\nThis is a side question from the user. Answer directly in a single response.\n\nCRITICAL '
               'CONSTRAINTS:\n- You have NO tools available.\n- You can ONLY use information already present in the conversation '
               'context.\n</system-reminder>\n\nWhich city did I ask about first?')
        # (a long system prompt, as a harness's: the chat outgrows every earlier request, so it is the main line)
        conv = [{'role': 'system', 'content': S.guidelines(11, 40000, 'Assistant rules') + f' ({marker})'},
                {'role': 'user', 'content': 'Name one landmark in Lisbon.'}]
        _, c1 = chat(port, f'{alias}-nothink', conv, temperature=0, max_tokens=48)
        conv += [{'role': 'assistant', 'content': c1['content']}, {'role': 'user', 'content': 'And one in Porto?'}]
        _, c2 = chat(port, f'{alias}-nothink', conv, temperature=0, max_tokens=48)
        conv += [{'role': 'assistant', 'content': c2['content']}]
        rq, q = chat(port, f'{alias}-nothink', conv + [{'role': 'user', 'content': BTW}], temperature=0, max_tokens=32000)
        rt_q = ((S.get(port, '/caminus/status')[1] or {}).get('cache') or {}).get('route') or {}
        rn, n3 = chat(port, f'{alias}-nothink', conv + [{'role': 'user', 'content': 'And one in Faro?'}], temperature=0, max_tokens=48)
        rt_n = ((S.get(port, '/caminus/status')[1] or {}).get('cache') or {}).get('route') or {}
        tq, tn = q.get('timings') or {}, n3.get('timings') or {}
        lq, ln = rt_q.get('last') or {}, rt_n.get('last') or {}
        check(rq['status'] == 200 and q['content'].strip() and (lq.get('route'), lq.get('reason')) == ('side', 'side_question')
              and (tq.get('cache_n') or 0) > 4000 and (tq.get('prompt_n') or 0) < 400
              and rn['status'] == 200 and n3['content'].strip() and (ln.get('route'), ln.get('reason')) == ('main', 'match')
              and (tn.get('prompt_n') or 0) < 200 and (tn.get('cache_n') or 0) > 4000,
              f'side question: route {lq.get("route")} / {lq.get("reason")}, prompt {tq.get("prompt_n")} new of {tq.get("cache_n")} '
              f'cached, answer {q["content"][:40]!r}; the next turn: route {ln.get("route")} / {ln.get("reason")}, prompt '
              f'{tn.get("prompt_n")} new of {tn.get("cache_n")} cached (the main slot continued)')
        # (a measurement, not gated: the side question's history against the stored conversation, as /caminus/status and the
        #  flight line give it; a thinking-mode chat too, whose replies carry reasoning the template may drop from history: its
        #  system prompt is longer than the first chat's, so it becomes the main line)
        tl = lq.get('tail') or {}
        fl = os.path.join(a.scratch, 'flight'); time.sleep(0.5)   # (the flight line is written as the request ends)
        rows = []
        for fn in sorted(glob.glob(os.path.join(fl, '*.jsonl'))):
            with open(fn, encoding='utf-8') as f: rows += [json.loads(x) for x in f if x.strip()]
        fq = [r_.get('cache', {}).get('side_question') for r_ in rows if (r_.get('cache') or {}).get('side_question')]
        # a long chat's side question as Qwen Code sends one (the opening, then only the last turns): asked against the stored
        # conversation when its history ends it (tail whole above), else as sent
        conv3 = conv + [{'role': 'user', 'content': 'And one in Faro?'}, {'role': 'assistant', 'content': n3['content']}]
        cut = [conv[0]] + conv3[-2:] + [{'role': 'user', 'content': BTW}]
        rc, qc = chat(port, f'{alias}-nothink', cut, temperature=0, max_tokens=64)
        tc = qc.get('timings') or {}
        time.sleep(0.5)
        rows = []
        for fn in sorted(glob.glob(os.path.join(fl, '*.jsonl'))):
            with open(fn, encoding='utf-8') as f: rows += [json.loads(x) for x in f if x.strip()]
        sent = [(r_.get('cache') or {}).get('side_question_sent') for r_ in rows if (r_.get('cache') or {}).get('side_question_sent')]
        want = bool(tl.get('whole'))
        check(rc['status'] == 200 and qc['content'].strip() and (len(sent) == 1) == want,
              f'a long chat\'s side question (history: the last turn only): {"asked against the stored conversation" if sent else "asked as sent"} '
              f'(expected {"stored" if want else "as sent"}: tail whole {want}; {sent[0] if sent else "-"} tokens sent), prompt '
              f'{tc.get("prompt_n")} new of {tc.get("cache_n")} cached, answer {qc["content"][:60]!r}')
        convt = [{'role': 'system', 'content': S.guidelines(12, 60000, 'Assistant rules') + f' ({marker})'},
                 {'role': 'user', 'content': 'Name one river in Spain.'}]
        _, k1 = chat(port, alias, convt, temperature=0, max_tokens=400)
        convt += [{'role': 'assistant', 'content': k1['content'], 'reasoning_content': k1['reasoning'] or None}]
        chat(port, f'{alias}-nothink', convt + [{'role': 'user', 'content': BTW}], temperature=0, max_tokens=64)
        tk = ((((S.get(port, '/caminus/status')[1] or {}).get('cache') or {}).get('route') or {}).get('last') or {}).get('tail') or {}
        print(f'NOTE side question tail (nothink chat): {tl}; flight lines with it: {len(fq)}; thinking chat (reasoning '
              f'{len(k1["reasoning"])} chars sent back): {tk}', flush=True)
        check(bool(tl) and len(fq) >= 1, f'side question tail measured: whole {tl.get("whole")}, matched {tl.get("matched")} of '
              f'{tl.get("tail")}, gap {tl.get("gap")}; thinking chat whole {tk.get("whole")}, matched {tk.get("matched")} of {tk.get("tail")}')
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
