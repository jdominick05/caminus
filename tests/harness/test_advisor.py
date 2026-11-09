"""The Qwen Code advisor add-on (python/caminus/share/harness/qwen-code/advisor) and `caminus advisor`: CPU only, no
network, no API keys, nothing touches a GPU or the real home (every run uses a scratch HOME / XDG tree):

  python -I tests/harness/test_advisor.py --scratch <job scratch dir>/advisor

Detection, the brief, the hook as Qwen Code runs it (a subprocess, JSON in and out, exit 0 always), the advisor chain
(refusals move on), the provider adapters against fake SDKs and a stand-in OpenAI-compatible server, the setup's settings
merge, `caminus advisor` setup / status / uninstall, the package data. Ends RESULT PASS / RESULT FAIL."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(REPO, 'python')
ROOT = os.path.join(PKG, 'caminus', 'share', 'harness', 'qwen-code', 'advisor')   # the add-on
SETUP = os.path.join(ROOT, 'setup_advisor.py')
sys.path.insert(0, ROOT)
from qwen_advisor import advisor, config, detect, transcript  # noqa: E402

CFG = config._merge(config.DEFAULTS, {'min_interval_s': 0})   # (the time limit has its own test)


def shell(cmd, code=0):
    return dict(tool_name='run_shell_command', tool_input={'command': cmd}, status='success',
                tool_response={'response_parts': [{'text': f'Command: {cmd}\nOutput: ...\nExit Code: {code}'}]})


def edit(path, n=0):
    return dict(tool_name='edit', tool_input={'file_path': path, 'old_string': f'a{n}', 'new_string': f'b{n}'}, status='success',
                tool_response={'response_parts': [{'text': 'ok'}]})


class Detect(unittest.TestCase):
    def test_fail_streak_then_cooldown(self):
        st = detect.new_state('s')
        self.assertIsNone(detect.on_batch(st, [shell('make', 2)], CFG))
        self.assertIsNone(detect.on_batch(st, [shell('make test', 1)], CFG))
        sig = detect.on_batch(st, [shell('make check', 1)], CFG)
        self.assertEqual(sig[0], 'fail_streak')
        self.assertEqual(st['pending']['signal'], 'fail_streak')
        detect.after_advice(st)
        for i in range(3):    # still failing, inside the cooldown (6 batches): no new signal
            self.assertIsNone(detect.on_batch(st, [shell(f'make x{i}', 1)], CFG))
        for i in range(3):
            sig = detect.on_batch(st, [shell(f'make y{i}', 1)], CFG)
        self.assertEqual(sig[0], 'fail_streak')     # after the cooldown it may ask again

    def test_success_resets_streak(self):
        st = detect.new_state('s')
        detect.on_batch(st, [shell('a', 1)], CFG)
        detect.on_batch(st, [shell('b', 1)], CFG)
        detect.on_batch(st, [shell('c', 0)], CFG)
        self.assertIsNone(detect.on_batch(st, [shell('d', 1)], CFG))

    def test_repeat(self):
        st = detect.new_state('s')
        call = dict(tool_name='read_file', tool_input={'absolute_path': '/x/a.py'}, status='success', tool_response={})
        detect.on_batch(st, [call], CFG)
        detect.on_batch(st, [dict(call)], CFG)
        sig = detect.on_batch(st, [dict(call, tool_input={'absolute_path': '/x/a.py'})], CFG)
        self.assertEqual(sig[0], 'repeat')

    def test_edit_churn_needs_a_failure_after_the_last_edit(self):
        st = detect.new_state('s')
        for i in range(4):
            detect.on_batch(st, [edit('/p/f.py', i), shell(f'pytest {i}', 0)], CFG)
        self.assertIsNone(st['pending'])            # different edits that pass: no signal
        sig = detect.on_batch(st, [edit('/p/f.py', 9), shell('pytest z', 1)], CFG)
        self.assertEqual(sig[0], 'edit_churn')

    def test_long_run_and_user_reset(self):
        st = detect.new_state('s')
        for i in range(39):
            self.assertIsNone(detect.on_batch(st, [shell(f'ls {i}', 0)], CFG))
        self.assertEqual(detect.on_batch(st, [shell('ls z', 0)], CFG)[0], 'long_run')
        detect.on_user_prompt(st)
        self.assertEqual(st['batches_since_user'], 0)
        self.assertIsNone(st['pending'])

    def test_min_interval_holds_automatic_hints(self):
        cfg = dict(CFG, min_interval_s=600)
        st = detect.new_state('s')
        st['last_advice_ts'] = time.time() - 120          # a hint two minutes ago
        for c in ('a', 'b', 'c', 'd'):
            self.assertIsNone(detect.on_batch(st, [shell(c, 1)], cfg))
        self.assertEqual(st['suppressed'], 2)              # (fail streak reached at the 3rd and 4th call)
        st['last_advice_ts'] = time.time() - 601
        self.assertEqual(detect.on_batch(st, [shell('e', 1)], cfg)[0], 'fail_streak')
        detect.after_advice(st)
        self.assertEqual(st['suppressed'], 0)
        self.assertLess(time.time() - st['last_advice_ts'], 5)

    def test_session_limit_and_disabled(self):
        st = detect.new_state('s')
        st['calls_made'] = CFG['max_calls_per_session']
        for c in ('a', 'b', 'c'):
            self.assertIsNone(detect.on_batch(st, [shell(c, 1)], CFG))
        st2 = detect.new_state('s')
        off = dict(CFG, enabled=False)
        for c in ('a', 'b', 'c'):
            self.assertIsNone(detect.on_batch(st2, [shell(c, 1)], off))

    def test_cancelled_calls_ignored(self):
        st = detect.new_state('s')
        for c in ('a', 'b', 'c'):
            detect.on_batch(st, [dict(shell(c, 1), status='cancelled')], CFG)
        self.assertIsNone(st['pending'])


def write_transcript(path, records):
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')


def rec(typ, parts, **kw):
    return dict(uuid='u', parentUuid=None, sessionId='s', timestamp='t', type=typ, cwd='/w', version='1',
                message=dict(role='user' if typ != 'assistant' else 'model', parts=parts), **kw)


SAMPLE = [
    rec('user', [{'text': 'first task: add a CLI flag'}]),
    rec('assistant', [{'text': 'done'}]),
    rec('user', [{'text': 'Fix the failing date test. My token is sk-ant-' + 'api03-ABCDEFGHIJKLMNOPQRSTUV'}]),
    rec('user', [{'text': 'injected agent note'}], subtype='agent_message'),
    rec('assistant', [{'text': 'secret reasoning', 'thought': True}, {'text': 'Running the tests.'},
                      {'functionCall': {'id': 'c1', 'name': 'run_shell_command', 'args': {'command': 'pytest -q'}}}]),
    rec('tool_result', [{'functionResponse': {'id': 'c1', 'name': 'run_shell_command',
                                              'response': {'output': 'FAILED test_dates.py - ValueError: Invalid isoformat\nExit Code: 1'}}}]),
]


class Transcript(unittest.TestCase):
    def test_brief(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            write_transcript(p, SAMPLE)
            with open(p, 'a') as f:
                f.write('{not json\n')               # a broken line is skipped
            b = transcript.brief(p, 'several tool calls in a row have failed', '3 failed', cwd='/w')
        self.assertIn('Fix the failing date test', b)
        self.assertIn('first task: add a CLI flag', b)               # earlier request
        self.assertIn('CALL run_shell_command', b)
        self.assertIn('Invalid isoformat', b)
        self.assertIn('AGENT: Running the tests.', b)
        self.assertNotIn('secret reasoning', b)                      # thought parts never leave
        self.assertNotIn('injected agent note', b)                   # not a user prompt
        self.assertNotIn('sk-ant-api03', b)                          # redacted
        self.assertIn('[redacted key]', b)

    def test_redact(self):
        s = transcript.redact('api_key = ' + '"abcdefghijklmnop" ghp_' + 'a' * 36 + ' password: hunter22hunter22\n'
                              '-----BEGIN RSA ' + 'PRIVATE KEY-----\nAAAA\n-----END RSA ' + 'PRIVATE KEY-----')
        self.assertNotIn('abcdefghijklmnop', s)
        self.assertNotIn('ghp_' + 'a' * 36, s)
        self.assertNotIn('hunter22hunter22', s)
        self.assertNotIn('AAAA', s)

    def test_redact_structured_credentials(self):
        """credentials in JSON / YAML / env / header form, as tool output shows them, never reach a hosted advisor; ordinary
        fields that only contain the word stay readable"""
        pw, ak, tk = 'database-' + 'secret-value', 'k-' + '0123456789abcdef', 'tok-' + 'zyxwvutsrq987654'
        cases = [
            '{"password": "%s", "user": "app"}' % pw,
            "{'api_key': '%s'}" % ak,
            '{"apiKey":"%s","token": "%s"}' % (ak, tk),
            '{"client_secret": "%s", "db_password" : "a phrase with spaces"}' % pw,
            'password: %s' % pw,
            'DB_PASSWORD=%s' % pw,
            'export AWS_SECRET_ACCESS_KEY=%s' % ak,
            'GITHUB_TOKEN=%s' % tk,
            'Authorization: Bearer %s' % tk,
            'curl -H "Authorization: Basic %s"' % tk,
            '"refresh_token": "%s"' % tk,
        ]
        for c in cases:
            out = transcript.redact(c)
            for v in (pw, ak, tk, 'a phrase with spaces'):
                self.assertNotIn(v, out, c)
            self.assertIn('[redacted]', out, c)
        self.assertIn('"user": "app"', transcript.redact(cases[0]))
        out = transcript.redact('OPENAI_API_KEY=sk-' + 'proj-abcdefghijklmnopqrstuvwxyz\nPORT=8080')
        self.assertEqual(out, 'OPENAI_API_KEY=[redacted key]\nPORT=8080')      # (replaced once, not twice)
        for keep in ('max_tokens=8000', '"tokens": 512', 'token_count: 12', 'the token was rejected: expired',
                     'passwords must be 12 characters', 'secretary: Dana'):
            self.assertEqual(transcript.redact(keep), keep)

    def test_standing_goal_in_brief(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            goal_rec = dict(type='system', subtype='goal_state', systemPayload=dict(v=2, cause={}, snapshot=dict(
                v=2, goal=dict(goalId='g1', revision=1, objective='Move the build to the new toolchain', status='paused'))))
            write_transcript(p, [rec('user', [{'text': 'first'}]), goal_rec, rec('user', [{'text': 'check the MCP servers work'}])])
            b = transcript.brief(p, 'x')
        self.assertIn("THE USER'S STANDING GOAL (status: paused):\nMove the build to the new toolchain", b)
        self.assertIn("THE USER'S LATEST MESSAGE:\ncheck the MCP servers work", b)

    def test_cleared_goal_not_shown(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            set_ = dict(type='system', subtype='goal_state', systemPayload=dict(snapshot=dict(goal=dict(objective='old goal', status='active'))))
            clear = dict(type='system', subtype='goal_state', systemPayload=dict(snapshot=dict(goal=None)))
            write_transcript(p, [rec('user', [{'text': 'a'}]), set_, clear])
            self.assertNotIn('STANDING GOAL', transcript.brief(p, 'x'))

    def test_early_steps_kept_beyond_the_window(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            recs = [rec('user', [{'text': 'get the integration tests running'}]),
                    rec('tool_result', [{'functionResponse': {'name': 'run_shell_command', 'response': {'output': 'proj-db-1 Up 0.0.0.0:55432->5432/tcp'}}}])]
            for i in range(30):
                recs.append(rec('tool_result', [{'functionResponse': {'name': 'run_shell_command', 'response': {'output': f'refused {i}'}}}]))
            write_transcript(p, recs)
            b = transcript.brief(p, 'x')
        self.assertIn('FIRST STEPS AFTER THE REQUEST', b)
        self.assertIn('55432', b)
        self.assertIn('refused 29', b)
        self.assertIn('steps not shown', b)

    def test_short_session_has_no_early_section(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            write_transcript(p, SAMPLE)
            self.assertNotIn('FIRST STEPS', transcript.brief(p, 'x'))

    def test_clip_keeps_error_lines_from_the_middle(self):
        out = 'start\n' + 'x' * 600 + '\nsrc/log.cpp:3:10: fatal error: fmt/core.h: No such file\n' + 'warning: w\n' * 200 + 'make: Error 2'
        c = transcript._clip(out, 1100, tail=True)
        self.assertIn('fatal error: fmt/core.h', c)
        self.assertNotIn('warning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w\nwarning: w', c)   # (still bounded)
        self.assertLess(len(c), 1100 + 600)

    def test_missing_transcript(self):
        b = transcript.brief('/nonexistent/t.jsonl', 'x')
        self.assertIn('(not found in the transcript)', b)

    def test_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.jsonl')
            recs = [rec('user', [{'text': 'task'}])]
            for i in range(300):
                recs.append(rec('tool_result', [{'functionResponse': {'name': 'run_shell_command', 'response': {'output': 'x' * 5000}}}]))
            write_transcript(p, recs)
            self.assertLessEqual(len(transcript.brief(p, 'x')), transcript.BRIEF_CHARS + 200)


class HookProcess(unittest.TestCase):
    """the hook as Qwen Code runs it: a subprocess, JSON on stdin, JSON (or nothing) on stdout, exit 0 always"""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.env = dict(os.environ, HOME=t, XDG_CONFIG_HOME=os.path.join(t, 'cfg'), XDG_STATE_HOME=os.path.join(t, 'state'),
                        XDG_DATA_HOME=os.path.join(t, 'share'), QWEN_ADVISOR_FAKE='Check the timezone suffix: fromisoformat rejects Z.')
        self.env.pop('QWEN_ADVISOR_OFF', None)
        os.makedirs(os.path.join(self.env['XDG_CONFIG_HOME'], 'qwen-advisor'))
        with open(os.path.join(self.env['XDG_CONFIG_HOME'], 'qwen-advisor', 'config.toml'), 'w') as f:
            f.write('min_interval_s = 0\n')
        self.tr = os.path.join(t, 'tr.jsonl')
        write_transcript(self.tr, SAMPLE)
        self.proj = os.path.join(t, 'proj')
        os.makedirs(self.proj)

    def tearDown(self):
        self.tmp.cleanup()

    def run_hook(self, payload, env=None, raw=None):
        base = dict(session_id='sess-1', transcript_path=self.tr, cwd=self.proj, timestamp='t')
        data = raw if raw is not None else json.dumps(dict(base, **payload))
        p = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'qwen_advisor', 'hook.py')], input=data,
                           capture_output=True, text=True, env=env or self.env, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stderr, '')
        return json.loads(p.stdout) if p.stdout.strip() else None

    def batch(self, *calls):
        return self.run_hook(dict(hook_event_name='PostToolBatch', tool_calls=list(calls)))

    def tool_send(self):
        return self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='[tool results]'))

    def user(self, text):
        return self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt=text, submitted_prompt=text))

    def test_flow(self):
        self.assertIsNone(self.user('Fix the failing date test'))
        for i in range(2):
            self.assertIsNone(self.batch(shell(f'pytest {i}', 1)))
            self.assertIsNone(self.tool_send())
        self.assertIsNone(self.batch(shell('pytest 2', 1)))     # third failure: pending, nothing printed
        out = self.tool_send()
        ctx = out['hookSpecificOutput']['additionalContext']
        self.assertEqual(out['hookSpecificOutput']['hookEventName'], 'UserPromptSubmit')
        self.assertIn('fromisoformat rejects Z', ctx)
        self.assertIn('qwen-advisor note', ctx)
        self.assertIn('advice from fake', out['systemMessage'])
        self.assertIsNone(self.batch(shell('pytest 3', 1)))
        self.assertIsNone(self.tool_send())                      # delivered once; the cooldown holds
        log = [json.loads(l) for l in open(os.path.join(self.env['XDG_STATE_HOME'], 'qwen-advisor', 'advisor.log'))]
        self.assertTrue(any(r.get('verdict') == 'advice' and r.get('signal') == 'fail_streak' for r in log))
        st = os.path.join(self.env['XDG_STATE_HOME'], 'qwen-advisor', 'sessions', 'sess-1.json')
        self.assertEqual(oct(os.stat(st).st_mode & 0o777), '0o600')

    def test_rate_limit_in_process(self):
        cfgdir = os.path.join(self.env['XDG_CONFIG_HOME'], 'qwen-advisor')
        os.makedirs(cfgdir, exist_ok=True)
        with open(os.path.join(cfgdir, 'config.toml'), 'w') as f:
            f.write('min_interval_s = 600\ncooldown_batches = 1\n')
        self.user('Fix the failing date test')
        for i in range(3):
            self.batch(shell(f'pytest {i}', 1))
        self.assertIn('hookSpecificOutput', self.tool_send())          # first hint
        for i in range(6):                                              # still stuck: held back for 10 minutes
            self.batch(shell(f'pytest x{i}', 1))
            self.assertIsNone(self.tool_send())
        out = self.user('/advise what now?')                            # /advise is never limited
        self.assertIn('hookSpecificOutput', out)
        log = [json.loads(l) for l in open(os.path.join(self.env['XDG_STATE_HOME'], 'qwen-advisor', 'advisor.log'))]
        calls = [r for r in log if r.get('verdict')]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1]['held_back_since_last'], 4)

    def test_manual_advise(self):
        out = self.user('/advise why does the date test fail?')
        self.assertIn('fromisoformat', out['hookSpecificOutput']['additionalContext'])
        out = self.user('[qwen-advisor: requested] is my approach right?')   # the /advise command's expanded body
        self.assertIn('hookSpecificOutput', out)

    def test_on_track_adds_nothing(self):
        env = dict(self.env, QWEN_ADVISOR_FAKE='ON_TRACK')
        self.user('task')
        for i in range(3):
            self.run_hook(dict(hook_event_name='PostToolBatch', tool_calls=[shell(f'x{i}', 1)]), env=env)
        out = self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='r'), env=env)
        self.assertNotIn('hookSpecificOutput', out)
        self.assertIn('on track', out['systemMessage'])

    def test_off_switches(self):
        env = dict(self.env, QWEN_ADVISOR_OFF='1')
        self.assertIsNone(self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='/advise x', submitted_prompt='/advise x'), env=env))
        open(os.path.join(self.proj, '.qwen-advisor-off'), 'w').close()
        self.assertIsNone(self.user('/advise x'))

    def test_fails_open(self):
        self.assertIsNone(self.run_hook({}, raw='this is not json'))
        self.assertIsNone(self.run_hook({}, raw='[1, 2]'))
        self.assertIsNone(self.run_hook(dict(hook_event_name='SessionStart')))
        env = dict(self.env)
        env.pop('QWEN_ADVISOR_FAKE')                              # the real provider, but no key anywhere
        out = self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='/advise x', submitted_prompt='/advise x'), env=env)
        self.assertNotIn('hookSpecificOutput', out)
        self.assertIn('no advice', out['systemMessage'])

    def read_refusals(self):
        p = os.path.join(self.env['XDG_STATE_HOME'], 'qwen-advisor', 'refusals.jsonl')
        if not os.path.exists(p):
            return []
        self.assertEqual(oct(os.stat(p).st_mode & 0o777), '0o600')
        return [json.loads(l) for l in open(p)]

    def test_chain_moves_past_hard_and_soft_refusals(self):
        env = dict(self.env, QWEN_ADVISOR_FAKE="REFUSE:cyber||I can't help with that request.||Check the timezone suffix.")
        out = self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='/advise x', submitted_prompt='/advise x'), env=env)
        self.assertIn('Check the timezone suffix.', out['hookSpecificOutput']['additionalContext'])
        self.assertNotIn("can't help", out['hookSpecificOutput']['additionalContext'])
        recs = self.read_refusals()
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertTrue(r['delivered'])
        self.assertEqual([x['kind'] for x in r['refused']], ['hard', 'soft'])
        self.assertEqual(r['refused'][0]['category'], 'cyber')
        self.assertIn("can't help", r['refused'][1]['soft_text'])
        self.assertIn('advisor chain', r['outcome'])
        self.assertIn('Fix the failing date test', r['brief'])            # the brief, for replay
        self.assertNotIn('sk-ant-api03', r['brief'])                       # (redacted, as sent)

    def test_chain_moves_past_decline_and_policy_steering(self):
        env = dict(self.env, QWEN_ADVISOR_FAKE="DECLINE||That path is circumvention; stop and do not weaponize it.||Check the timezone suffix.")
        out = self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='/advise x', submitted_prompt='/advise x'), env=env)
        self.assertIn('Check the timezone suffix.', out['hookSpecificOutput']['additionalContext'])
        self.assertNotIn('weaponize', out['hookSpecificOutput']['additionalContext'])
        r = self.read_refusals()[0]
        self.assertEqual([x['kind'] for x in r['refused']], ['soft', 'soft'])

    def test_chain_all_refuse_delivers_nothing(self):
        env = dict(self.env, QWEN_ADVISOR_FAKE='REFUSE:cyber||I cannot assist with this.')
        out = self.run_hook(dict(hook_event_name='UserPromptSubmit', prompt='/advise x', submitted_prompt='/advise x'), env=env)
        self.assertNotIn('hookSpecificOutput', out)
        self.assertIn('no advice', out['systemMessage'])
        r = self.read_refusals()[0]
        self.assertFalse(r['delivered'])
        self.assertIn('refused by every advisor tried (2 of 2)', r['outcome'])

    def test_no_refusal_no_record(self):
        self.user('/advise x')
        self.assertEqual(self.read_refusals(), [])

    def test_stop_review(self):
        cfgdir = os.path.join(self.env['XDG_CONFIG_HOME'], 'qwen-advisor')
        os.makedirs(cfgdir, exist_ok=True)
        with open(os.path.join(cfgdir, 'config.toml'), 'w') as f:
            f.write('review_on_stop = true\nreview_min_batches = 2\nmin_interval_s = 0\n')
        self.user('task')
        self.assertIsNone(self.run_hook(dict(hook_event_name='Stop', stop_hook_active=False)))   # too little work yet
        for i in range(2):
            self.batch(shell(f'ok{i}', 0))
            self.tool_send()
        out = self.run_hook(dict(hook_event_name='Stop', stop_hook_active=False))
        self.assertEqual(out['decision'], 'block')
        self.assertIsNone(self.run_hook(dict(hook_event_name='Stop', stop_hook_active=True)))     # once per message
        env = dict(self.env, QWEN_ADVISOR_FAKE='DONE')
        self.user('next task')
        for i in range(2):
            self.batch(shell(f'k{i}', 0))
            self.tool_send()
        self.assertIsNone(self.run_hook(dict(hook_event_name='Stop', stop_hook_active=False), env=env))


class RefusalRecords(unittest.TestCase):
    def test_server_fallback_rescue(self):
        from qwen_advisor import refusal
        r = dict(ok=True, text='Use cyclic_find.', provider='anthropic', model='claude-opus-4-8', fallback_ran=True,
                 declined_by=['claude-opus-5-5'], tried=[])
        rec = refusal.record(r)
        self.assertIn('declined by claude-opus-5-5; answered by claude-opus-4-8 (server-side fallback)', rec['outcome'])
        self.assertEqual(rec['server_declined'], ['claude-opus-5-5'])

    def test_clean_answer_is_no_record(self):
        from qwen_advisor import refusal
        self.assertIsNone(refusal.record(dict(ok=True, text='Run pip install -e .', provider='mistral', model='m', tried=[])))

    def test_chain_budget_stops_early(self):
        os.environ['QWEN_ADVISOR_FAKE'] = 'REFUSE:cyber||REFUSE:cyber||REFUSE:cyber'
        try:
            r = advisor.ask(dict(CFG, chain_budget_s=1), 'brief')   # under 8 s left after the first try: the rest skipped
        finally:
            os.environ.pop('QWEN_ADVISOR_FAKE')
        self.assertFalse(r['ok'])
        self.assertEqual(r['skipped'], 2)


class FakeSDKs(unittest.TestCase):
    """the provider adapters against fake SDK modules: request shape, text extraction, refusals, errors"""
    def setUp(self):
        self.saved = {k: sys.modules.get(k) for k in ('anthropic', 'mistralai', 'mistralai.client')}
        self.tmp = tempfile.TemporaryDirectory()
        self.key = os.path.join(self.tmp.name, 'k')
        with open(self.key, 'w') as f:
            f.write('test-key\n')
        os.environ.pop('QWEN_ADVISOR_FAKE', None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        self.tmp.cleanup()

    def fake_anthropic(self, reply):
        seen = {}
        mod = types.ModuleType('anthropic')
        for n in ('APITimeoutError', 'RateLimitError', 'APIConnectionError'):
            setattr(mod, n, type(n, (Exception,), {}))
        mod.APIStatusError = type('APIStatusError', (Exception,), {'status_code': 400, 'message': 'bad'})
        mod.BadRequestError = type('BadRequestError', (mod.APIStatusError,), {})

        class Msgs:
            def create(self, **kw):
                seen.update(kw)
                seen.setdefault('calls', []).append(dict(kw))
                r = reply(mod) if callable(reply) else reply
                if isinstance(r, Exception):
                    raise r
                return r

        class Client:
            def __init__(self, **kw):
                seen['client'] = kw
                self.messages = Msgs()
                self.beta = types.SimpleNamespace(messages=Msgs())
        mod.Anthropic = Client
        sys.modules['anthropic'] = mod
        return seen, mod

    def test_anthropic_request_and_text(self):
        blk = lambda t, x: types.SimpleNamespace(type=t, text=x)
        reply = types.SimpleNamespace(stop_reason='end_turn', model='claude-opus-5-5', content=[types.SimpleNamespace(type='thinking'),
                                      blk('text', 'Look at the Z suffix.')], usage=types.SimpleNamespace(input_tokens=900, output_tokens=120))
        seen, _ = self.fake_anthropic(reply)
        cfg = config._merge(config.DEFAULTS, {'anthropic': {'key': self.key, 'effort': 'high'}})
        r = advisor.ask(cfg, 'brief')
        self.assertTrue(r['ok'])
        self.assertEqual(r['text'], 'Look at the Z suffix.')
        self.assertEqual(r['verdict'], 'advice')
        self.assertEqual(seen['client']['api_key'], 'test-key')
        self.assertEqual(seen['client']['max_retries'], 0)
        self.assertEqual(seen['thinking'], {'type': 'adaptive'})
        self.assertEqual(seen['output_config'], {'effort': 'high'})
        self.assertEqual(seen['fallbacks'], 'default')
        self.assertEqual(seen['betas'], ['server-side-fallback-2026-07-01'])
        self.assertNotIn('temperature', seen)

    def test_anthropic_haiku_no_fallbacks_and_refusal(self):
        reply = types.SimpleNamespace(stop_reason='refusal', model='claude-haiku-5-5', content=[],
                                      stop_details=types.SimpleNamespace(category='cyber'), usage=None)
        seen, _ = self.fake_anthropic(reply)
        cfg = config._merge(config.DEFAULTS, {'anthropic': {'key': self.key, 'model': 'claude-haiku-5-5'}})
        r = advisor.ask(cfg, 'brief')
        self.assertFalse(r['ok'])
        self.assertIn('refusal (cyber)', r['error'])
        self.assertNotIn('fallbacks', seen)

    def test_anthropic_error_then_mistral_fallback(self):
        self.fake_anthropic(lambda mod: mod.APITimeoutError('t'))   # (the class of the module that is installed)
        seen = {}

        class Chat:
            def complete(self, **kw):
                seen.update(kw)
                ch = [types.SimpleNamespace(type='thinking', thinking=[]), types.SimpleNamespace(type='text', text='Use dateutil.')]
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=ch))],
                                             usage=types.SimpleNamespace(prompt_tokens=800, completion_tokens=40))

        class Mistral:
            def __init__(self, **kw):
                seen['client'] = kw
                self.chat = Chat()
        pkg = types.ModuleType('mistralai')
        sub = types.ModuleType('mistralai.client')
        sub.Mistral = Mistral
        sys.modules['mistralai'] = pkg
        sys.modules['mistralai.client'] = sub
        cfg = config._merge(config.DEFAULTS, {'fallback_provider': 'mistral', 'anthropic': {'key': self.key},
                                              'mistral': {'key': self.key, 'model': 'mistral-large-latest', 'effort': 'medium'}})
        r = advisor.ask(cfg, 'brief')
        self.assertTrue(r['ok'])
        self.assertEqual(r['provider'], 'mistral')
        self.assertEqual(r['text'], 'Use dateutil.')
        self.assertEqual([(t['provider'], t['error']) for t in r['tried']], [('anthropic', 'timeout')])
        self.assertEqual(seen['reasoning_effort'], 'medium')
        self.assertEqual(seen['messages'][0]['role'], 'system')
        self.assertEqual(seen['client']['timeout_ms'], 55000)

    def test_anthropic_older_model_retried_without_thinking(self):
        ok = types.SimpleNamespace(stop_reason='end_turn', model='claude-haiku-4-5', usage=None,
                                   content=[types.SimpleNamespace(type='text', text='Read the traceback first.')])
        state = {'n': 0}

        def reply(mod):
            state['n'] += 1
            return mod.BadRequestError('thinking.type: adaptive is not supported') if state['n'] == 1 else ok
        seen, _ = self.fake_anthropic(reply)
        cfg = config._merge(config.DEFAULTS, {'anthropic': {'key': self.key, 'model': 'claude-haiku-4-5', 'effort': 'medium'}})
        r = advisor.ask(cfg, 'brief')
        self.assertTrue(r['ok'], r)
        self.assertEqual(len(seen['calls']), 2)
        self.assertIn('thinking', seen['calls'][0])
        self.assertNotIn('thinking', seen['calls'][1])
        self.assertNotIn('output_config', seen['calls'][1])

    def test_mistral_retried_without_reasoning_effort(self):
        calls = []

        class E(Exception):
            status_code = 422

        class Chat:
            def complete(self, **kw):
                calls.append(dict(kw))
                if 'reasoning_effort' in kw:
                    raise E('reasoning_effort not supported')
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='Pin the version.'))], usage=None)

        class Mistral:
            def __init__(self, **kw):
                self.chat = Chat()
        sub = types.ModuleType('mistralai.client')
        sub.Mistral = Mistral
        sys.modules['mistralai'] = types.ModuleType('mistralai')
        sys.modules['mistralai.client'] = sub
        cfg = config._merge(config.DEFAULTS, {'provider': 'mistral', 'mistral': {'key': self.key, 'model': 'open-mistral-nemo', 'effort': 'high'}})
        r = advisor.ask(cfg, 'brief')
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['text'], 'Pin the version.')
        self.assertEqual([('reasoning_effort' in c) for c in calls], [True, False])

    def test_on_track_verdict(self):
        os.environ['QWEN_ADVISOR_FAKE'] = 'ON_TRACK.'
        try:
            self.assertEqual(advisor.ask(CFG, 'b')['verdict'], 'on_track')
            os.environ['QWEN_ADVISOR_FAKE'] = 'DONE'
            self.assertEqual(advisor.ask(CFG, 'b', review=True)['verdict'], 'done')
        finally:
            os.environ.pop('QWEN_ADVISOR_FAKE')


class LocalProvider(unittest.TestCase):
    """the local provider against a stand-in OpenAI-compatible server (http.server on a free port)"""
    @classmethod
    def setUpClass(cls):
        import http.server, threading
        cls.seen = []
        outer = cls

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                out = json.dumps({'data': [{'id': 'arex-2'}, {'id': 'qwen3.6'}]}).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                outer.seen.append(dict(path=self.path, body=body, auth=self.headers.get('Authorization')))
                mode = outer.mode
                if mode == 'slow':
                    time.sleep(6)
                if mode == 'http500':
                    self.send_response(500); self.end_headers(); self.wfile.write(b'boom'); return
                if mode == 'length':
                    msg, fin = {'content': '<think>still thinking about the', 'reasoning_content': None}, 'length'
                elif mode == 'closing_only':   # (a template that opened the reasoning in the prompt)
                    msg, fin = {'content': 'the agent loops on pytest\n</think>\n\nPin the timezone first.'}, 'stop'
                else:
                    msg, fin = {'content': outer.reply}, 'stop'
                out = json.dumps({'model': 'local-test', 'choices': [{'message': msg, 'finish_reason': fin}],
                                  'usage': {'prompt_tokens': 321, 'completion_tokens': 45}}).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)

        cls.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        cls.srv.handle_error = lambda *a: None   # (a client that timed out on purpose: a broken pipe, not news)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.srv.server_port}/v1/'

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.seen.clear()
        type(self).mode, type(self).reply = 'ok', '<think>the agent reruns pytest</think>\nCheck the Z suffix: use fromisoformat on 3.11+.'
        self.env = os.environ.pop('QWEN_ADVISOR_FAKE', None)

    def tearDown(self):
        if self.env is not None:
            os.environ['QWEN_ADVISOR_FAKE'] = self.env

    def cfg(self, **local):
        return config._merge(CFG, {'provider': 'local', 'fallback_provider': '', 'chain': [],
                                   'local': dict({'url': self.url, 'model': 'qwen3.6'}, **local)})

    def test_request_and_reasoning_dropped(self):
        with tempfile.NamedTemporaryFile('w', suffix='.key', delete=False) as f:
            f.write('sk-local\n')
        try:
            r = advisor.ask(self.cfg(key=f.name), 'BRIEF TEXT')
        finally:
            os.unlink(f.name)
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['verdict'], 'advice')
        self.assertEqual(r['text'], 'Check the Z suffix: use fromisoformat on 3.11+.')
        self.assertEqual((r['provider'], r['model'], r['input_tokens'], r['output_tokens']), ('local', 'local-test', 321, 45))
        q = self.seen[0]
        self.assertEqual(q['path'], '/v1/chat/completions')
        self.assertEqual(q['auth'], 'Bearer sk-local')
        self.assertEqual(q['body']['model'], 'qwen3.6')
        self.assertEqual([m['role'] for m in q['body']['messages']], ['system', 'user'])
        self.assertEqual(q['body']['messages'][1]['content'], 'BRIEF TEXT')

    def test_no_key_no_model_sent(self):
        r = advisor.ask(self.cfg(model='', key=''), 'b')
        self.assertTrue(r['ok'])
        self.assertIsNone(self.seen[0]['auth'])
        self.assertNotIn('model', self.seen[0]['body'])

    def test_on_track(self):
        type(self).reply = 'ON_TRACK'
        self.assertEqual(advisor.ask(self.cfg(), 'b')['verdict'], 'on_track')

    def test_out_of_tokens_while_thinking(self):
        type(self).mode = 'length'
        r = advisor.ask(self.cfg(), 'b')
        self.assertFalse(r['ok'])
        self.assertIn('local.max_tokens', r['error'])

    def test_http_error_and_refused_connection(self):
        type(self).mode = 'http500'
        self.assertIn('HTTP 500', advisor.ask(self.cfg(), 'b')['error'])
        import socket
        s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
        r = advisor.ask(self.cfg(url=f'http://127.0.0.1:{port}/v1'), 'b')
        self.assertFalse(r['ok'])
        self.assertIn('connection error', r['error'])

    def test_chain_cloud_then_local(self):
        """an unconfigured cloud advisor fails; the chain's local entry answers, with the model from the entry"""
        c = config._merge(CFG, {'provider': 'mistral', 'fallback_provider': '', 'chain': ['local:arex-2'],
                                'mistral': {'model': 'mistral-large-4', 'key': ''}, 'local': {'url': self.url}})
        r = advisor.ask(c, 'b')
        self.assertTrue(r['ok'], r)
        self.assertEqual(r['provider'], 'local')
        self.assertEqual(self.seen[0]['body']['model'], 'arex-2')
        self.assertEqual(r['tried'][0]['provider'], 'mistral')

    def test_setup_configures_local(self):
        """setup.py, answered on stdin in a scratch HOME: plan 5 (local), the URL, no key, model 2 from the server's list"""
        with tempfile.TemporaryDirectory() as t:
            env = dict(os.environ, HOME=t, XDG_CONFIG_HOME=os.path.join(t, 'cfg'), XDG_STATE_HOME=os.path.join(t, 'st'),
                       XDG_DATA_HOME=os.path.join(t, 'share'))
            env.pop('QWEN_ADVISOR_CONFIG', None)
            sp = os.path.join(t, 'settings.json')
            with open(sp, 'w') as f:
                f.write('{}')
            p = subprocess.run([sys.executable, '-I', SETUP, '--settings', sp, '--no-live-test',
                                '--dry-run'], input=f'5\n{self.url}\n\n2\n6000\n200\n' + '\n' * 20, text=True,
                               capture_output=True, env=env, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr[-800:])
            self.assertIn('arex-2, qwen3.6', p.stdout)
            c = config.load(os.path.join(t, 'cfg', 'qwen-advisor', 'config.toml'))
        self.assertEqual((c['provider'], c['fallback_provider']), ('local', ''))
        self.assertEqual((c['local']['url'], c['local']['model'], c['local']['key'], c['local']['max_tokens'], c['local']['timeout_s']),
                         (self.url.strip(), 'qwen3.6', '', 6000, 200))
        self.assertGreaterEqual(c['chain_budget_s'], 210)          # the hooks' wait stays above the local reply limit

    def test_reasoning_with_only_the_closing_tag(self):
        type(self).mode = 'closing_only'
        r = advisor.ask(self.cfg(), 'b')
        self.assertEqual(r['text'], 'Pin the timezone first.')

    def test_local_reply_limit(self):
        """local.timeout_s is the local advisor's own limit, either way: a 6 s reply is dropped at 5 s (nothing added), and
        kept under a 30 s local limit although the hosted advisors' timeout_s is 5"""
        type(self).mode = 'slow'
        t0 = time.monotonic()
        r = advisor.ask(self.cfg(timeout_s=5), 'b')
        self.assertFalse(r['ok'])
        self.assertEqual(r['error'], 'timeout')
        self.assertLess(time.monotonic() - t0, 5.9)
        r = advisor.ask(dict(self.cfg(timeout_s=30), timeout_s=5), 'b')
        self.assertTrue(r['ok'], r)

    def caminus(self, env, *args, stdin=''):
        code = f'import sys; sys.path.insert(0, {PKG!r}); from caminus.__main__ import main; sys.exit(main(sys.argv[1:]))'
        return subprocess.run([sys.executable, '-I', '-c', code, *args], input=stdin, text=True, capture_output=True,
                              env=env, timeout=120)

    def test_caminus_advisor_local_only(self):
        """caminus advisor setup --local-only into Caminus's own Qwen home (no --settings), status, uninstall; ~/.qwen untouched"""
        with tempfile.TemporaryDirectory() as t:
            env = dict(os.environ, HOME=t, XDG_CONFIG_HOME=os.path.join(t, 'cfg'), XDG_STATE_HOME=os.path.join(t, 'st'),
                       XDG_DATA_HOME=os.path.join(t, 'share'))
            p = self.caminus(env, 'advisor', 'setup', '--local-only', '--no-live-test')
            self.assertEqual(p.returncode, 2)
            self.assertIn('caminus qwen-code setup', p.stdout)                     # no home yet: says how to make one
            home = os.path.join(t, 'cfg', 'caminus', 'qwen-home')
            os.makedirs(home)
            with open(os.path.join(home, 'settings.json'), 'w') as f:
                json.dump({'model': {'name': 'k3-coder'}}, f)
            dotqwen = os.path.join(t, '.qwen')
            os.makedirs(dotqwen)
            with open(os.path.join(dotqwen, 'settings.json'), 'w') as f:
                f.write('{}')
            # answers: the one plan (local), URL, no key, the model list's default, max tokens, wait, then the limits
            p = self.caminus(env, 'advisor', 'setup', '--local-only', '--no-live-test', stdin=f'\n{self.url}\n' + '\n' * 20)
            self.assertEqual(p.returncode, 0, p.stdout[-1500:] + p.stderr[-1500:])
            self.assertNotIn('Anthropic', p.stdout.split('advisors [')[0].split('steer the local model?')[-1])
            c = config.load(os.path.join(t, 'cfg', 'qwen-advisor', 'config.toml'))
            self.assertEqual((c['provider'], c['fallback_provider'], c['chain']), ('local', '', []))
            self.assertEqual(c['local']['model'], 'arex-2')
            s = json.load(open(os.path.join(home, 'settings.json')))
            self.assertEqual(s['model'], {'name': 'k3-coder'})
            self.assertIn('qwen_advisor', json.dumps(s['hooks']))
            self.assertTrue(os.path.isfile(os.path.join(home, 'commands', 'advise.md')))
            self.assertEqual(open(os.path.join(dotqwen, 'settings.json')).read(), '{}')   # only when named
            self.assertFalse(os.path.exists(os.path.join(t, 'share', 'qwen-advisor', 'venv')))   # no SDKs for local
            st = json.loads(self.caminus(env, 'advisor', 'status', '--json').stdout)
            self.assertTrue(st['installed'])
            self.assertEqual(st['advisors'], ['local'])
            self.assertEqual(st['homes'][home], {'hooks': True, 'advise_command': True})
            p = self.caminus(env, 'advisor', 'uninstall')
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            st = json.loads(self.caminus(env, 'advisor', 'status', '--json').stdout)
            self.assertEqual(st['homes'][home], {'hooks': False, 'advise_command': False})

    def test_local_soft_refusal_moves_on(self):
        type(self).reply = "I can't help with that request."
        c = self.cfg()
        c['chain'] = ['local:second']
        r = advisor.ask(c, 'b')
        self.assertFalse(r['ok'])
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(r['tried'][0]['error'], 'soft refusal')


class Setup(unittest.TestCase):
    """the setup's settings merge on a scratch settings.json (never the user's): others' hooks kept, idempotent, uninstall"""
    def test_merge_and_uninstall(self):
        with tempfile.TemporaryDirectory() as t:
            env = dict(os.environ, HOME=t, XDG_CONFIG_HOME=os.path.join(t, 'cfg'), XDG_STATE_HOME=os.path.join(t, 'st'),
                       XDG_DATA_HOME=os.path.join(t, 'share'))
            home = os.path.join(t, 'qhome')
            os.makedirs(home)
            sp = os.path.join(home, 'settings.json')
            mine = {'type': 'command', 'command': 'echo hi', 'name': 'my-hook'}
            with open(sp, 'w') as f:
                json.dump({'model': {'name': 'k3-coder'}, 'hooks': {'PostToolBatch': [{'hooks': [mine]}]}}, f)
            run = lambda *a: subprocess.run([sys.executable, '-I', SETUP, '--non-interactive',
                                             '--no-live-test', '--settings', sp, *a], capture_output=True, text=True, env=env, timeout=120)
            for _ in range(2):
                p = run()
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            s = json.load(open(sp))
            self.assertEqual(s['model'], {'name': 'k3-coder'})
            groups = s['hooks']['PostToolBatch']
            self.assertIn(mine, [h for g in groups for h in g['hooks']])
            ours = [h for ev in s['hooks'].values() for g in ev for h in g['hooks'] if h.get('name') == 'qwen-advisor']
            self.assertEqual(sorted(h['timeout'] for h in ours), [10, 165])         # PostToolBatch + UserPromptSubmit (chain budget + 15), once each
            self.assertTrue(all(os.path.join(t, 'share', 'qwen-advisor', 'app') in h['command'] for h in ours))
            self.assertTrue(os.path.exists(os.path.join(home, 'commands', 'advise.md')))
            cfgp = os.path.join(t, 'cfg', 'qwen-advisor', 'config.toml')
            self.assertEqual(oct(os.stat(cfgp).st_mode & 0o777), '0o600')
            self.assertEqual(config.load(cfgp)['provider'], 'anthropic')
            baks = [n for n in os.listdir(home) if n.startswith('settings.json.bak-qwen-advisor-')]
            self.assertTrue(baks)
            p = run('--uninstall')
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            s = json.load(open(sp))
            self.assertEqual(s['hooks'], {'PostToolBatch': [{'hooks': [mine]}]})
            self.assertFalse(os.path.exists(os.path.join(home, 'commands', 'advise.md')))


class CaminusSide(unittest.TestCase):
    def test_qwen_code_setup_keeps_the_hooks(self):
        """caminus qwen-code setup, run again later, swaps only its provider entries: the advisor's hooks stay"""
        sys.path.insert(0, PKG)
        try:
            from caminus.cmd import qwen_code
        finally:
            sys.path.remove(PKG)
        old = {'hooks': {'PostToolBatch': [{'hooks': [{'name': 'qwen-advisor', 'command': 'py -I app/qwen_advisor/hook.py'}]}]},
               'modelProviders': {'openai': [{'id': 'k3-coder', 'envKey': 'CAMINUS_API_KEY'}]}, 'model': {'name': 'k3-coder'}}
        new = qwen_code.merged(old, [{'id': 'k3-coder', 'envKey': 'CAMINUS_API_KEY', 'baseUrl': 'http://127.0.0.1:8092/v1'}],
                               'k3-coder', 'k3-coder-low')
        self.assertEqual(new['hooks'], old['hooks'])

    def test_package_data_ships_the_add_on(self):
        """every file of the add-on matches a package-data pattern (include-package-data is off: a missed file would be
        absent from the wheel, and `caminus advisor` would fail only after a pip install)"""
        import fnmatch, tomllib
        with open(os.path.join(REPO, 'pyproject.toml'), 'rb') as f:
            pats = tomllib.load(f)['tool']['setuptools']['package-data']['caminus']
        base = os.path.join(PKG, 'caminus')
        files = []
        for d, dirs, fs in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x != '__pycache__']
            files += [os.path.relpath(os.path.join(d, x), base) for x in fs if not x.endswith('.pyc')]
        self.assertGreaterEqual(len(files), 9)
        missing = [x for x in files if not any(fnmatch.fnmatchcase(x, pt) and x.count('/') == pt.count('/') for pt in pats)]
        self.assertEqual(missing, [])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True, help='a directory for temporary files (never /tmp)')
    a, rest = ap.parse_known_args()
    os.makedirs(a.scratch, exist_ok=True)
    tempfile.tempdir = os.environ['TMPDIR'] = os.path.abspath(a.scratch)   # (the children's temporary files too)
    for k in ('QWEN_ADVISOR_FAKE', 'QWEN_ADVISOR_OFF', 'QWEN_ADVISOR_CONFIG', 'CAMINUS_QWEN_HOME'):
        os.environ.pop(k, None)
    r = unittest.main(argv=[sys.argv[0], *rest], exit=False, verbosity=1).result
    ok = r.wasSuccessful()
    print('RESULT PASS' if ok else f'RESULT FAIL ({len(r.failures) + len(r.errors)})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
