"""The monitoring surface (docs/specs/spec-metrics.md) through the HTTP server on MockSession, CPU only:
  * llama-server's shapes: /props, /slots and the /metrics block against the keys, JSON types, names, HELP texts and
    label llama-server writes (tools/server: get_res_props, server_slot::to_json, task_params::to_json, to_metrics;
    the types as in llm-top's captured fixtures), /health 200 / 503 'Loading model';
  * what llm-top reads: /metrics through a port of its parseMetrics (src/server.go), live /slots progress (prompt
    tokens processed, then n_decoded, moving during a request; id_task per request; the client's n_predict unclamped),
    speculative.types draft-dflash with a drafter;
  * the caminus: series and /caminus/status: valid Prometheus text, every group, the per-card hardware data from a
    fixture sysfs tree, the agent counters (User-Agent classes, tool calls and failures, reasoning vs answer tokens,
    max_tokens hits, the loop detector), engine errors; every series and status field documented in the spec;
  * the process name: set_process_name in a child process (main thread and another thread), and tools/k3_serve.py
    started on port 0: /proc/<pid>/comm reads caminus.
No request text is ever in /slots or the status. Run:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import common
import srvutil as U
import synth
from caminus.serve.hw import HwReader, write_fixture
from caminus.serve.mock_engine import MockDrafter, MockEngine, ScriptedEngine, SynthProfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
PY = sys.executable
SPEC = os.path.join(REPO, 'docs', 'specs', 'spec-metrics.md')
SYS = {'role': 'system', 'content': 'You are a coding agent.'}

# llama-server's keys and JSON types (llm-top's captured testdata: props.json, slots.json)
PROPS_TYPES = {'default_generation_settings': dict, 'total_slots': int, 'model_alias': str, 'model_ftype': str,
               'model_path': str, 'modalities': dict, 'media_marker': str, 'endpoint_slots': bool,
               'endpoint_props': bool, 'endpoint_metrics': bool, 'ui': bool, 'ui_settings': dict,
               'chat_template': str, 'chat_template_caps': dict, 'bos_token': str, 'eos_token': str, 'build_info': str,
               'is_sleeping': bool, 'cors_proxy_enabled': bool}
PARAMS_TYPES = {'seed': int, 'temperature': float, 'dynatemp_range': float, 'dynatemp_exponent': float, 'top_k': int,
                'top_p': float, 'min_p': float, 'top_n_sigma': float, 'xtc_probability': float,
                'xtc_threshold': float, 'typical_p': float, 'repeat_last_n': int, 'repeat_penalty': float,
                'presence_penalty': float, 'frequency_penalty': float, 'dry_multiplier': float, 'dry_base': float,
                'dry_allowed_length': int, 'dry_penalty_last_n': int, 'mirostat': int, 'mirostat_tau': float,
                'mirostat_eta': float, 'adaptive_target': float, 'adaptive_decay': float, 'max_tokens': int,
                'n_predict': int, 'n_keep': int, 'n_discard': int, 'ignore_eos': bool, 'stream': bool,
                'n_probs': int, 'min_keep': int, 'chat_format': str, 'reasoning_format': str,
                'reasoning_in_content': bool, 'generation_prompt': str, 'samplers': list, 'speculative.types': str,
                'timings_per_token': bool, 'post_sampling_probs': bool, 'backend_sampling': bool, 'lora': list}
SLOT_TYPES = {'id': int, 'n_ctx': int, 'speculative': bool, 'is_processing': bool, 'id_task': int,
              'n_prompt_tokens': int, 'n_prompt_tokens_processed': int, 'n_prompt_tokens_cache': int, 'params': dict,
              'next_token': list}
NEXT_TYPES = {'has_next_token': bool, 'has_new_line': bool, 'n_remain': int, 'n_decoded': int}
# server-task.cpp to_metrics, in its order
LLAMA_SERIES = ['prompt_tokens_total', 'prompt_tokens_cached_total', 'prompt_seconds_total', 'tokens_predicted_total',
                'tokens_predicted_seconds_total', 'n_decode_total', 'n_tokens_max',
                'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total',
                'spec_decode_num_drafts_total', 'prompt_tokens_seconds', 'predicted_tokens_seconds',
                'requests_processing', 'requests_deferred', 'n_busy_slots_per_decode']
PER_POS = 'spec_decode_num_accepted_tokens_per_pos_total'


def llmtop_parse(text):
    """llm-top's parseMetrics (src/server.go), line for line: '#' lines skipped; labeled lines only for the
    per-position series, kept as 'name.N'; else exactly two fields, the llamacpp: prefix trimmed"""
    m = {}
    for line in text.split('\n'):
        if line.startswith('#'):
            continue
        i = line.find('{')
        if i >= 0:
            rest = line[i:]
            if rest.startswith('{position="'):
                n, sep, v = rest[len('{position="'):].partition('"}')
                f = v.split()
                if sep and len(f) == 1 and line[:i].removeprefix('llamacpp:') == PER_POS:
                    try:
                        m[PER_POS + '.' + n] = float(f[0])
                    except ValueError:
                        pass
            continue
        f = line.split()
        if len(f) == 2:
            try:
                m[f[0].removeprefix('llamacpp:')] = float(f[1])
            except ValueError:
                pass
    return m


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[a-zA-Z_][a-zA-Z0-9_]*="(?:\\.|[^"\\])*"'
                     r'(?:,[a-zA-Z_][a-zA-Z0-9_]*="(?:\\.|[^"\\])*")*\})? (\S+)$')


def prom_lint(tc, text):
    """the text format: HELP / TYPE once per metric and before its samples, valid sample lines and values"""
    types, helps, names = {}, set(), []
    for line in text.rstrip('\n').split('\n'):
        if line.startswith('# HELP '):
            name = line.split()[2]
            tc.assertNotIn(name, helps, f'HELP twice: {name}')
            helps.add(name)
        elif line.startswith('# TYPE '):
            _, _, name, typ = line.split()
            tc.assertNotIn(name, types, f'TYPE twice: {name}')
            tc.assertIn(typ, ('counter', 'gauge', 'summary', 'histogram'))
            types[name] = typ
        else:
            mt = _SAMPLE.match(line)
            tc.assertIsNotNone(mt, f'bad sample line {line!r}')
            name, val = mt.group(1), mt.group(2 + 1)
            base = name
            for suf in ('_sum', '_count'):
                if name.endswith(suf) and types.get(name[:-len(suf)]) == 'summary':
                    base = name[:-len(suf)]
            tc.assertIn(base, types, f'no TYPE before {name}')
            float(val)          # NaN / +Inf parse too
            names.append(base)
    return types


def get_json(srv, path, headers=None):
    st, h, data = U.request(srv, 'GET', path, headers=headers)
    return st, h, json.loads(data) if data else None


def status(srv):
    return get_json(srv, '/caminus/status')[2]


def check_types(tc, obj, types, where):
    tc.assertEqual(set(types) - set(obj), set(), f'{where}: llama-server keys missing')
    for k, t in types.items():
        tc.assertIsInstance(obj[k], t, f'{where}.{k}')
        if t is int:
            tc.assertNotIsInstance(obj[k], bool, f'{where}.{k}')


# ==================================================================================================== the surface
class TestSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = eng = ScriptedEngine(ntok=4)
        cls.hwroot = write_fixture(tempfile.mkdtemp(prefix='k3-hw-'))
        cls.srv = U.make_server(eng, drafter=MockDrafter(eng, 0.85), hw=HwReader(root=cls.hwroot), hw_interval=0,
                                session_kw={'synth': SynthProfile(sleep=False), 'disk_slots': 8})
        cls.tok = common.tokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def chat(self, msgs, reply, tools=None, ua=None, **body):
        p, r = U.script(self.eng, msgs, reply, tools)
        b = {'model': 'k3-coder', 'messages': msgs, 'temperature': 0}
        if tools:
            b['tools'] = tools
        b.update(body)
        st, h, j = U.post_json(self.srv, '/v1/chat/completions', b, headers={'User-Agent': ua} if ua else None)
        self.assertEqual(st, 200, j)
        return p, r, j

    def test_props(self):
        st, h, p = get_json(self.srv, '/props')
        self.assertEqual(st, 200)
        check_types(self, p, PROPS_TYPES, 'props')
        g = p['default_generation_settings']
        self.assertEqual(g['n_ctx'], self.srv.max_ctx)
        check_types(self, g['params'], PARAMS_TYPES, 'props.params')
        self.assertEqual(set(g['params']), set(PARAMS_TYPES), 'exactly llama-server\'s only_metrics params')
        self.assertEqual((g['params']['n_predict'], g['params']['speculative.types']), (-1, 'draft-dflash'))
        self.assertEqual((p['total_slots'], p['is_sleeping'], p['n_ctx']), (1, False, self.srv.max_ctx))
        self.assertEqual(p['engine'], 'caminus')
        self.assertEqual((p['caminus']['status'], p['caminus']['status_version']), ('/caminus/status', 1))
        self.assertTrue(p['build_info'].startswith('caminus-'))

    def test_slots_after_request(self):
        msgs = [SYS, {'role': 'user', 'content': 'Slots: say done.'}]
        p, r, j = self.chat(msgs, {'role': 'assistant', 'reasoning_content': 'Short.', 'content': 'done'},
                            max_tokens=10 ** 6)
        st, h, sl = get_json(self.srv, '/slots')
        self.assertEqual((st, len(sl)), (200, 1))
        s = sl[0]
        check_types(self, s, SLOT_TYPES, 'slot')
        check_types(self, s['next_token'][0], NEXT_TYPES, 'slot.next_token[0]')
        self.assertEqual(set(s['params']), set(PARAMS_TYPES))
        self.assertEqual((s['is_processing'], s['speculative'], s['n_prompt_tokens']), (False, True, len(p)))
        self.assertEqual(s['n_prompt_tokens_cache'] + s['n_prompt_tokens_processed'], len(p))
        self.assertEqual(s['next_token'][0]['n_decoded'], j['usage']['completion_tokens'])
        self.assertEqual(s['params']['n_predict'], 10 ** 6, 'the client\'s max_tokens, not the context clamp')
        self.assertEqual(s['params']['speculative.types'], 'draft-dflash')
        self.assertEqual(s['params']['temperature'], 0.0)
        self.assertNotIn('prompt', s)
        self.assertNotIn('generated', s)
        t0 = s['id_task']
        self.chat([SYS, {'role': 'user', 'content': 'Slots: again.'}], {'role': 'assistant', 'content': 'ok'})
        self.assertGreater(get_json(self.srv, '/slots')[2][0]['id_task'], t0)

    def test_metrics_as_llmtop_reads_them(self):
        st0 = status(self.srv)['llamacpp']
        msgs = [SYS, {'role': 'user', 'content': 'Metrics: write the function. ' + synth.CODE[0]}]
        reply = {'role': 'assistant', 'reasoning_content': 'Write it.', 'content': '```python\n' + synth.CODE[1] + '```'}
        p, r, j1 = self.chat(msgs, reply)
        msgs2 = msgs + [reply, {'role': 'user', 'content': 'And a test.'}]
        p2, r2, j2 = self.chat(msgs2, {'role': 'assistant', 'content': 'def test(): pass'})
        self.assertGreater(j2['usage']['prompt_tokens_details']['cached_tokens'], len(p))
        st, h, data = U.request(self.srv, 'GET', '/metrics')
        self.assertEqual((st, h['content-type']), (200, 'text/plain; version=0.0.4'))
        self.assertTrue(h['process-start-time-unix'].isdigit())
        text = data.decode()
        types = prom_lint(self, text)
        m = llmtop_parse(text)
        for k in LLAMA_SERIES:
            self.assertIn(k, m)
            self.assertEqual(types['llamacpp:' + k], 'counter' if LLAMA_SERIES.index(k) < 10 else 'gauge')
        self.assertIn(PER_POS + '.0', m)
        self.assertNotIn(PER_POS, m)
        for k in m:
            self.assertFalse('{' in k or k.startswith('llamacpp:'), k)
        u1, u2 = j1['usage'], j2['usage']
        self.assertEqual(m['prompt_tokens_total'] + m['prompt_tokens_cached_total'] - st0['prompt_tokens_total']
                         - st0['prompt_tokens_cached_total'], u1['prompt_tokens'] + u2['prompt_tokens'])
        self.assertEqual(m['prompt_tokens_cached_total'] - st0['prompt_tokens_cached_total'],
                         u1['prompt_tokens_details']['cached_tokens'] + u2['prompt_tokens_details']['cached_tokens'])
        self.assertEqual(m['tokens_predicted_total'] - st0['tokens_predicted_total'],
                         u1['completion_tokens'] + u2['completion_tokens'])
        self.assertEqual((m['requests_processing'], m['requests_deferred'], m['n_busy_slots_per_decode']), (0, 0, 1))
        self.assertGreater(m['spec_decode_num_drafts_total'], 0)
        self.assertGreaterEqual(m['spec_decode_num_drafts_total'], m[PER_POS + '.0'])
        self.assertEqual(sum(v for k, v in m.items() if k.startswith(PER_POS + '.')),
                         m['spec_decode_num_accepted_tokens_total'])
        # llama-server's own lines, in its order, with its HELP texts
        ll = [x for x in text.split('\n') if x.startswith('# HELP llamacpp:')]
        self.assertEqual([x.split()[2][len('llamacpp:'):] for x in ll], LLAMA_SERIES + [PER_POS])
        self.assertIn('# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens',
                      text)
        # the caminus: groups
        for name in ('caminus:physics_floor_ms', 'caminus:decode_pass_ms', 'caminus:ttft_ms',
                     'caminus:spec_accepted_per_pos_total', 'caminus:cache_tokens_total', 'caminus:context_position',
                     'caminus:bails_total', 'caminus:card_pass_ms', 'caminus:vram_used_mib', 'caminus:power_watts',
                     'caminus:clock_forced_high', 'caminus:vram_guard_margin_mib', 'caminus:energy_joules_total',
                     'caminus:requests_total', 'caminus:tokens_generated_total', 'caminus:loop_requests_total'):
            self.assertIn(name, types)
        self.assertIn('caminus:vram_used_mib{card="0",name="XTX"} 20205', text)
        self.assertIn('caminus:vram_guard_margin_mib{card="0",name="XTX"} 3083', text)     # 23,800 - 512 - 20,205
        self.assertIn('caminus:vram_guard_margin_mib{card="1",name="XT"} 4337', text)      # 19,850 - 15,513

    def test_status(self):
        self.chat([SYS, {'role': 'user', 'content': 'Status: one line.'}],
                  {'role': 'assistant', 'reasoning_content': 'Fine.', 'content': 'One line.'})
        s = status(self.srv)
        self.assertEqual((s['version'], s['engine']), (1, 'caminus'))
        for g in ('server', 'llamacpp', 'physics', 'decode', 'ttft', 'prefill', 'dflash', 'cache', 'context',
                  'reliability', 'cards', 'link', 'energy', 'agent', 'slot'):
            self.assertIn(g, s)
        self.assertEqual([(c['index'], c['name'], c['vram_used_mib'], c['margin_mib']) for c in s['cards']],
                         [(0, 'XTX', 20205.0, 3083.0), (1, 'XT', 15513.0, 4337.0)])
        self.assertEqual(s['reliability']['xtx_margin_mib'], 3083.0)
        self.assertEqual(s['physics']['floor_ms'], 4.0863)
        self.assertGreater(s['decode']['split_ms']['k3']['count'], 0)
        self.assertEqual(len(s['decode']['split_ms']['k3']['hist']['counts']),
                         len(s['decode']['split_ms']['k3']['hist']['le']))
        self.assertGreater(s['link']['bytes_total'], 0)
        self.assertEqual(s['slot']['id'], 0)
        blob = json.dumps(s)
        self.assertNotIn('One line', blob)          # no request text anywhere
        self.assertNotIn('Status: one line', blob)

    def test_agent_counters(self):
        a0 = status(self.srv)['agent']
        U.request(self.srv, 'GET', '/props', headers={'User-Agent': 'llm-top/0.9'})      # polls are not requests
        self.chat([SYS, {'role': 'user', 'content': 'Agent: hi from qwen.'}], {'role': 'assistant', 'content': 'hi'},
                  ua='QwenCode/0.0.14 (linux; x64)')
        # a tool call, then the same call cut by max_tokens
        tools = synth.TOOLS
        msgs = [SYS, {'role': 'user', 'content': 'Agent: read a.py.'}]
        reply = {'role': 'assistant', 'reasoning_content': 'Read it.', 'content': '',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                     'name': 'read_file', 'arguments': {'path': 'a.py'}}}]}
        p, r, j = self.chat(msgs, reply, tools, ua='claude-cli/2.1.3 (external, cli)')
        self.assertEqual(j['choices'][0]['finish_reason'], 'tool_calls')
        n = next(k for k in range(len(r)) if '<parameter=path>' in self.tok.decode(r[:k]))
        p, r, j = self.chat(msgs, reply, tools, max_tokens=n)
        self.assertEqual(j['choices'][0]['finish_reason'], 'length')
        # a reasoning loop
        line = 'Maybe the import path is wrong, check it again.'
        self.chat([SYS, {'role': 'user', 'content': 'Agent: why does it fail?'}],
                  {'role': 'assistant', 'reasoning_content': '\n'.join([line] * 4), 'content': 'Fixed the path.'})
        # (the server reports a request's tool and loop results in the handler's finally, after the reply is written: a
        # status read right after the reply can come first, so wait, bounded, for the last request's report)
        t0 = time.time()
        while True:
            a = status(self.srv)['agent']
            if a['loop']['requests_flagged'] - a0['loop']['requests_flagged'] >= 1 or time.time() - t0 > 5:
                break
            time.sleep(0.02)
        d = {k: a['harness'].get(k, 0) - a0['harness'].get(k, 0) for k in set(a['harness']) | set(a0['harness'])}
        self.assertEqual(d.get('qwen-code'), 1)
        self.assertEqual(d.get('claude-code'), 1)
        self.assertEqual(d.get('llm-top', 0), 0)
        self.assertEqual(a['tool_calls'] - a0['tool_calls'], 1)
        self.assertEqual(a['tool_call_failures']['cut'] - a0['tool_call_failures']['cut'], 1)
        self.assertEqual(a['max_tokens_hits'] - a0['max_tokens_hits'], 1)
        self.assertEqual(a['finish']['length'] - a0['finish']['length'], 1)
        self.assertEqual(a['loop']['requests_flagged'] - a0['loop']['requests_flagged'], 1)
        self.assertEqual(a['loop']['repeated_lines'] - a0['loop']['repeated_lines'], 3)
        self.assertEqual(a['loop']['live_max_repeat'], 0, 'reset when the request ends')
        s = status(self.srv)
        self.assertEqual(a['tokens']['reasoning'] + a['tokens']['answer'], s['llamacpp']['tokens_predicted_total'])

    def test_live_progress(self):
        """llm-top's READING / WRITING rows: /slots moves during a request (a second server, slower passes)"""
        eng = ScriptedEngine(ntok=4)
        srv = U.make_server(eng, pass_sleep=0.004)
        try:
            st, h, sl = get_json(srv, '/slots')
            self.assertEqual(sl, [{'id': 0, 'n_ctx': srv.max_ctx, 'speculative': False, 'is_processing': False}],
                             'idle before any request: llama-server\'s slot without a task')
            msgs = [SYS, {'role': 'user', 'content': 'Progress: ' + ' '.join(synth.CODE) * 2}]
            reply = {'role': 'assistant', 'reasoning_content': 'Long.', 'content': synth.CODE[2] * 3}
            p, r = U.script(eng, msgs, reply, None)
            seen = []
            done = threading.Event()

            def poll():
                while not done.is_set():
                    s = get_json(srv, '/slots')[2][0]
                    if s.get('is_processing'):
                        seen.append((s['n_prompt_tokens_processed'], s['next_token'][0]['n_decoded'],
                                     s['id_task']))
                    time.sleep(0.01)
            t = threading.Thread(target=poll)
            t.start()
            try:
                st, h, j = U.post_json(srv, '/v1/chat/completions', {'messages': msgs, 'temperature': 0})
            finally:
                done.set()
                t.join()
            reading = [x for x in seen if x[1] == 0]
            writing = [x for x in seen if x[1] > 0]
            self.assertGreater(len(reading), 3)
            self.assertGreater(len(writing), 3)
            self.assertLess(reading[0][0], reading[-1][0], 'prompt tokens processed grows')
            self.assertLess(writing[0][1], writing[-1][1], 'n_decoded grows')
            self.assertEqual(len({x[2] for x in seen}), 1)
            m = llmtop_parse(U.request(srv, 'GET', '/metrics')[2].decode())
            self.assertEqual(m['requests_processing'], 0)
            self.assertNotIn('spec_decode_num_accepted_tokens_per_pos_total.0', m)
            self.assertEqual(get_json(srv, '/props')[2]['default_generation_settings']['params']['speculative.types'],
                             'none')
            print(f'\n[metrics] live /slots during one request: {len(reading)} polls reading '
                  f'({reading[0][0]} -> {reading[-1][0]} of {len(p)} prompt tokens), {len(writing)} writing '
                  f'({writing[0][1]} -> {writing[-1][1]} tokens of {j["usage"]["completion_tokens"]})')
        finally:
            srv.stop()

    def test_loading_and_errors(self):
        eng = MockEngine(ntok=2, vocab=self.tok.n_vocab, eos_id=None, seed=7)
        srv = U.make_server(eng, fail_at={6})
        try:
            srv.set_loading(True)
            for path in ('/health', '/props', '/metrics', '/slots', '/caminus/status'):
                st, h, j = get_json(srv, path)
                self.assertEqual((st, j), (503, {'error': {'message': 'Loading model', 'type': 'unavailable_error',
                                                           'code': 503}}), path)
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': 'x', 'max_tokens': 2})
            self.assertEqual(st, 503)
            srv.set_loading(False)
            self.assertEqual(get_json(srv, '/health')[:3:2], (200, {'status': 'ok'}))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': [self.tok.bos_id, 50, 51], 'max_tokens': 40,
                                                            'temperature': 0})
            self.assertEqual(st, 503)
            s = status(srv)
            self.assertEqual((s['reliability']['errors']['engine'], s['reliability']['bails']['failed'],
                              s['agent']['finish']['error'], s['server']['requests_processing']), (1, 1, 1, 0))
            self.assertFalse(s['slot']['is_processing'])
        finally:
            srv.stop()


# ==================================================================================================== the spec
class TestSpecCoverage(unittest.TestCase):
    """every series /metrics writes and every status field is documented in docs/specs/spec-metrics.md"""

    def test_documented(self):
        with open(SPEC, encoding='utf-8') as f:
            doc = f.read()
        eng = ScriptedEngine(ntok=4)
        srv = U.make_server(eng, drafter=MockDrafter(eng, 0.85), hw=HwReader(root=write_fixture(
            tempfile.mkdtemp(prefix='k3-hw-'))), hw_interval=0, session_kw={'synth': SynthProfile(sleep=False,
                                                                                                bail_every=3)})
        try:
            msgs = [SYS, {'role': 'user', 'content': 'Doc: go.'}]
            p, r = U.script(eng, msgs, {'role': 'assistant', 'reasoning_content': 'Go.', 'content': synth.CODE[0]},
                            None)
            U.post_json(srv, '/v1/chat/completions', {'messages': msgs, 'temperature': 0})
            types = prom_lint(self, U.request(srv, 'GET', '/metrics')[2].decode())
            s = status(srv)
        finally:
            srv.stop()
        missing = [n for n in types if not re.search('`' + re.escape(n) + '[`{]', doc)]
        self.assertEqual(missing, [], 'series not in the spec')
        paths = []
        for k, v in s.items():
            if isinstance(v, dict):
                paths += [f'{k}.{k2}' for k2 in v]
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                paths += [f'{k}[].{k2}' for k2 in v[0]]
            else:
                paths.append(k)
        missing = [x for x in paths if f'`{x}`' not in doc]
        self.assertEqual(missing, [], 'status fields not in the spec')


# ==================================================================================================== process name
class TestProcessName(unittest.TestCase):
    def run_py(self, code):
        env = dict(os.environ, HIP_VISIBLE_DEVICES='')
        r = subprocess.run([PY, '-I', '-c', f'import sys; sys.path.insert(0, {os.path.join(REPO, "python")!r})\n'
                            + code], capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.split()

    def test_set_process_name(self):
        out = self.run_py('from caminus.serve.server import set_process_name\n'
                          'print(set_process_name("caminus"), open("/proc/self/comm").read().strip())')
        self.assertEqual(out, ['caminus', 'caminus'])
        out = self.run_py('import threading\nfrom caminus.serve.server import set_process_name\nr = []\n'
                          't = threading.Thread(target=lambda: r.append(set_process_name("caminus")))\n'
                          't.start(); t.join()\nprint(r[0], open("/proc/self/comm").read().strip())')
        self.assertEqual(out, ['caminus', 'caminus'])

    def test_launcher(self):
        """tools/k3_serve.py: names itself before any thread starts, serves the surface on a free port"""
        fx = tempfile.mkdtemp(prefix='k3-hw-launch-')
        env = dict(os.environ, HIP_VISIBLE_DEVICES='')
        proc = subprocess.Popen([PY, '-I', os.path.join(REPO, 'tools', 'k3_serve.py'), '--port', '0', '--log-dir', '',
                                 '--draft', '--max-ctx', '4096', '--pass-ms', '1', '--hw-fixture', fx,
                                 '--model', common.MODEL], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=env)
        try:
            line = proc.stdout.readline()
            mt = re.search(r'http://127\.0\.0\.1:(\d+) .*pid=(\d+)', line)
            self.assertIsNotNone(mt, line + proc.stderr.read() if proc.poll() is not None else line)
            port, pid = int(mt.group(1)), int(mt.group(2))
            self.assertEqual(pid, proc.pid)
            with open(f'/proc/{pid}/comm') as f:
                self.assertEqual(f.read().strip(), 'caminus')

            class S:
                pass
            srv = S()
            srv.port = port
            self.assertEqual(get_json(srv, '/health')[2], {'status': 'ok'})
            pr = get_json(srv, '/props')[2]
            self.assertEqual((pr['engine'], pr['default_generation_settings']['n_ctx']), ('caminus', 4096))
            st, h, j = U.post_json(srv, '/v1/completions', {'prompt': 'def f():', 'max_tokens': 24})
            self.assertEqual(st, 200)
            text = U.request(srv, 'GET', '/metrics')[2].decode()
            self.assertIn('caminus:vram_used_mib{card="0",name="XTX"}', text)
            self.assertIn('caminus:card_pass_ms{card="0",quantile="0.5"}', text)
            self.assertEqual(get_json(srv, '/slots')[2][0]['params']['speculative.types'], 'draft-dflash')
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            proc.stdout.close()
            proc.stderr.close()


if __name__ == '__main__':
    unittest.main()
