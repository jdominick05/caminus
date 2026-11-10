"""The context note (python/caminus/share/harness/qwen-code/context_note.py) and its registration by `caminus qwen-code
setup`: CPU only, no network, no GPU, nothing in the real home (scratch transcripts and Qwen homes):

  python -I tests/harness/test_context_note.py --scratch <job scratch dir>/context_note

The hook as Qwen Code runs it (a subprocess, PostToolBatch JSON in, JSON or nothing out, exit 0 always): the numbers
against Qwen Code's own thresholds, the batch matched by its call ids, silence in subagents / without a record / switched
off; setup's settings merge (once, other hooks kept, the off choice remembered); the package data. Ends RESULT PASS / FAIL."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(REPO, 'python')
HOOK = os.path.join(PKG, 'caminus', 'share', 'harness', 'qwen-code', 'context_note.py')
sys.path.insert(0, os.path.dirname(HOOK))
import context_note as cn  # noqa: E402
sys.path.insert(0, PKG)
from caminus.cmd import qwen_code  # noqa: E402


def assistant(call_id, prompt, out, window, model='k3-coder'):
    return dict(type='assistant', model=model, contextWindowSize=window,
                usageMetadata=dict(promptTokenCount=prompt, candidatesTokenCount=out, totalTokenCount=prompt + out),
                message=dict(role='model', parts=[{'text': 'running it'}, {'functionCall': {'id': call_id, 'name': 'run_shell_command',
                                                                                           'args': {'command': 'ls'}}}]))


def batch(transcript, call_id, output='', **kw):
    return dict(session_id='s', transcript_path=transcript, cwd='/w', hook_event_name='PostToolBatch', permission_mode='default',
                tool_calls=[dict(tool_name='run_shell_command', tool_input={'command': 'ls'}, tool_use_id=call_id,
                                 tool_call_id=call_id, status='success',
                                 tool_response={'response_parts': [{'functionResponse': {'id': call_id, 'name': 'run_shell_command',
                                                                                         'response': {'output': output}}}]})], **kw)


def write(path, recs):
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def run_hook(inp, args=(), env=None, raw=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run([sys.executable, '-I', HOOK, *args], input=raw if raw is not None else json.dumps(inp),
                       capture_output=True, text=True, env=e, timeout=30)
    return p.returncode, p.stdout, p.stderr


class Numbers(unittest.TestCase):
    def test_thresholds_match_qwen_code(self):
        """computeThresholds' auto tier: min(pct x window, window - 33,000); the proportional value for tiny windows"""
        self.assertEqual(cn.auto_threshold(1_048_000, 0.85), 890_800)        # (the 1M mode: the percentage governs)
        self.assertEqual(cn.auto_threshold(262_000, 0.85), 222_700)          # (the compactions on record: ~223K)
        self.assertEqual(cn.auto_threshold(40_000, 0.85), 7_000)             # (a small window: the ceiling governs)
        self.assertEqual(cn.auto_threshold(30_000, 0.85), 25_500)            # (below the reserve: the percentage again)

    def test_estimate_is_high(self):
        """a third of a token per ASCII character (Qwen Code's own estimate is a quarter): the budget left is never overstated"""
        self.assertEqual(cn.est_tokens('x' * 3000), 1000)
        self.assertEqual(cn.est_tokens('é' * 10), 15)
        self.assertGreaterEqual(cn.est_tokens('a' * 4001), 4001 // 4)


class Hook(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.t = os.path.join(self.d, 't.jsonl')

    def test_note_after_a_batch(self):
        write(self.t, [dict(type='user', message=dict(parts=[{'text': 'task'}])), assistant('call_a', 100_000, 2_000, 262_000)])
        rc, out, err = run_hook(batch(self.t, 'call_a', 'x' * 3000))
        self.assertEqual(rc, 0, err)
        o = json.loads(out)
        self.assertEqual(o['hookSpecificOutput']['hookEventName'], 'PostToolBatch')
        used = 102_000 + cn.est_tokens(json.dumps([{'functionResponse': {'id': 'call_a', 'name': 'run_shell_command',
                                                                            'response': {'output': 'x' * 3000}}}]))
        self.assertEqual(o['hookSpecificOutput']['additionalContext'],
                         f'[context: ~{round(used / 1000)}K of 262K tokens used; ~{round((222_700 - used) / 1000)}K left before automatic compaction]')

    def test_the_batch_record_not_the_newest(self):
        """the record is the one holding this batch's call ids, not merely the newest"""
        write(self.t, [assistant('call_a', 50_000, 1_000, 262_000), assistant('call_b', 90_000, 1_000, 262_000)])
        line = cn.note(batch(self.t, 'call_a'), wait_s=0)
        self.assertIn('~51K of 262K tokens used', line)

    def test_threshold_from_settings_and_past_it(self):
        s = os.path.join(self.d, 'settings.json')
        with open(s, 'w') as f:
            json.dump({'context': {'autoCompactThreshold': 0.5}}, f)
        write(self.t, [assistant('call_a', 100_000, 0, 262_000)])
        self.assertIn('~31K left before automatic compaction', cn.note(batch(self.t, 'call_a'), s, wait_s=0))   # 131,000 - 100,000
        write(self.t, [assistant('call_a', 140_000, 0, 262_000)])
        self.assertEqual(cn.note(batch(self.t, 'call_a'), s, wait_s=0),
                         '[context: ~140K of 262K tokens used; automatic compaction comes before the next step]')

    def test_record_written_late(self):
        """Qwen Code writes the transcript asynchronously: the hook waits (up to a second) for this batch's record"""
        write(self.t, [assistant('call_old', 10_000, 0, 262_000)])
        p = subprocess.Popen([sys.executable, '-I', HOOK], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        p.stdin.write(json.dumps(batch(self.t, 'call_new')))
        p.stdin.close()
        time.sleep(0.3)
        write(self.t, [assistant('call_old', 10_000, 0, 262_000), assistant('call_new', 20_000, 0, 262_000)])
        out = p.stdout.read()
        p.stdout.close()
        self.assertEqual(p.wait(timeout=10), 0)
        self.assertIn('~20K of 262K', json.loads(out)['hookSpecificOutput']['additionalContext'])

    def test_silent_cases(self):
        write(self.t, [assistant('call_a', 100_000, 0, 262_000)])
        t0 = time.monotonic()
        for inp, env in ((batch(self.t, 'call_missing'), None),                 # the record never shows up (1 s)
                         (batch(self.t, 'call_a', agent_id='a1'), None),        # a subagent: not its transcript
                         (batch(self.t, 'call_a'), {'CAMINUS_CONTEXT_NOTE': '0'}),
                         (batch(os.path.join(self.d, 'none.jsonl'), 'call_a'), None),
                         (dict(batch(self.t, 'call_a'), tool_calls=[]), None),
                         (dict(batch(self.t, 'call_a'), hook_event_name='Stop'), None)):
            rc, out, _ = run_hook(inp, env=env)
            self.assertEqual((rc, out), (0, ''), inp.get('agent_id') or env or inp['tool_calls'][:1])
        self.assertLess(time.monotonic() - t0, 10)
        write(self.t, [dict(assistant('call_a', 1, 0, 262_000), usageMetadata=None)])        # no usage on the record
        self.assertEqual(run_hook(batch(self.t, 'call_a'))[:2], (0, ''))
        write(self.t, [dict(assistant('call_a', 100, 0, 262_000), contextWindowSize=None)])  # no window on it
        self.assertEqual(run_hook(batch(self.t, 'call_a'))[:2], (0, ''))

    def test_fails_open(self):
        for raw in ('{not json', '[]', '', '"x"'):
            self.assertEqual(run_hook(None, raw=raw)[:2], (0, ''), raw)
        with open(self.t, 'wb') as f:
            f.write(b'\xff\xfe{broken\n' + json.dumps(assistant('call_a', 5_000, 0, 32_000)).encode() + b'\n')
        rc, out, _ = run_hook(batch(self.t, 'call_a'), args=('--settings', os.path.join(self.d, 'missing.json')))
        self.assertEqual(rc, 0)
        self.assertIn('~5K of 32K', out)

    def test_long_transcript_tail_only(self):
        big = [dict(type='tool_result', message=dict(parts=[{'text': 'y' * 4000}])) for _ in range(2000)]   # ~8 MB
        write(self.t, big + [assistant('call_a', 500_000, 0, 1_048_000)])
        t0 = time.monotonic()
        self.assertIn('~500K of 1,048K tokens used; ~391K left', cn.note(batch(self.t, 'call_a'), wait_s=0))
        self.assertLess(time.monotonic() - t0, 2.0)


class Setup(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()

    def entries(self, st):
        return [h for g in (st.get('hooks') or {}).get('PostToolBatch', []) for h in g.get('hooks', []) if h.get('name') == qwen_code.NOTE]

    def test_with_note_once_others_kept(self):
        adv = {'hooks': [{'type': 'command', 'name': 'qwen-advisor', 'command': 'py -I app/qwen_advisor/hook.py', 'timeout': 10}]}
        old = {'hooks': {'PostToolBatch': [adv], 'UserPromptSubmit': [adv]}, 'model': {'name': 'k3-coder'}}
        st = qwen_code.with_note(json.loads(json.dumps(old)), self.home, True)
        st = qwen_code.with_note(st, self.home, True)                              # (setup run again: still one entry)
        self.assertEqual(len(self.entries(st)), 1)
        self.assertIn(adv, st['hooks']['PostToolBatch'])
        self.assertEqual(st['hooks']['UserPromptSubmit'], [adv])
        e = self.entries(st)[0]
        dst = os.path.join(self.home, 'hooks', 'context_note.py')
        self.assertTrue(os.path.isfile(dst))                                       # (a copy: never the package's file)
        self.assertIn(dst, e['command'])
        self.assertIn('--settings', e['command'])
        self.assertNotIn('qwen-advisor', e['command'] + e['name'])                 # (the advisor's uninstall never strips it)
        self.assertNotIn('qwen_advisor', json.dumps(e))
        off = qwen_code.with_note(st, self.home, False)
        self.assertEqual(self.entries(off), [])
        self.assertEqual(off['hooks'], old['hooks'])
        self.assertNotIn('hooks', qwen_code.with_note({'hooks': {'PostToolBatch': []}}, self.home, False))

    def test_installed_hook_runs(self):
        st = qwen_code.with_note({}, self.home, True)
        cmd = self.entries(st)[0]['command']
        t = os.path.join(self.home, 't.jsonl')
        write(t, [assistant('call_a', 30_000, 0, 262_000)])
        p = subprocess.run(cmd, shell=True, input=json.dumps(batch(t, 'call_a')), capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('~30K of 262K', p.stdout)

    def test_off_is_remembered(self):
        self.assertTrue(qwen_code.note_on(self.home))
        self.assertFalse(qwen_code.note_on(self.home, 'off'))
        self.assertFalse(qwen_code.note_on(self.home))                             # (caminus qwen later: still off)
        self.assertTrue(qwen_code.note_on(self.home, 'on'))
        self.assertTrue(qwen_code.note_on(self.home))

    def test_setup_writes_it(self):
        """setup() end to end on a scratch home, the running server stubbed: provider entries and the note, once"""
        srv = dict(name='coder', port=8092, model='coder', alias='k3-coder', guard_pid=1)
        saved = qwen_code.running, qwen_code.server_ids
        qwen_code.running = lambda: [srv]
        qwen_code.server_ids = lambda s, wait_s=0: (['k3-coder', 'k3-coder-nothink'], 262_144)
        try:
            qwen_code.setup(home=self.home, say=lambda s: None)
            qwen_code.setup(home=self.home, say=lambda s: None)
            st = qwen_code.read_settings(self.home)
            self.assertEqual(len(self.entries(st)), 1)
            self.assertEqual(st['modelProviders']['openai'][0]['generationConfig']['contextWindowSize'], 262_000)
            qwen_code.setup(home=self.home, say=lambda s: None, note='off')
            self.assertEqual(self.entries(qwen_code.read_settings(self.home)), [])
            qwen_code.setup(home=self.home, say=lambda s: None)
            self.assertEqual(self.entries(qwen_code.read_settings(self.home)), [])   # (remembered)
        finally:
            qwen_code.running, qwen_code.server_ids = saved

    def test_package_data(self):
        import fnmatch, tomllib
        with open(os.path.join(REPO, 'pyproject.toml'), 'rb') as f:
            pats = tomllib.load(f)['tool']['setuptools']['package-data']['caminus']
        rel = os.path.relpath(HOOK, os.path.join(PKG, 'caminus'))
        self.assertTrue(any(fnmatch.fnmatchcase(rel, p) and rel.count('/') == p.count('/') for p in pats), rel)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True, help='a directory for temporary files (never /tmp)')
    a, rest = ap.parse_known_args()
    os.makedirs(a.scratch, exist_ok=True)
    tempfile.tempdir = os.environ['TMPDIR'] = os.path.abspath(a.scratch)
    for k in ('CAMINUS_CONTEXT_NOTE', 'CAMINUS_QWEN_HOME'):
        os.environ.pop(k, None)
    r = unittest.main(argv=[sys.argv[0], *rest], exit=False, verbosity=1).result
    ok = r.wasSuccessful()
    print('RESULT', 'PASS' if ok else f'FAIL ({len(r.failures) + len(r.errors)})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
