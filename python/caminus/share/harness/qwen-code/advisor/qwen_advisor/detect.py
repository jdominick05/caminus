"""Signs that the agent is lost, from PostToolBatch payloads alone (no model calls, no transcript reads: this runs after
every batch and must stay in milliseconds). One small JSON state per session; detection only records a pending signal,
the UserPromptSubmit hook asks the advisor before the next model call."""
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time

from . import config

RECENT = 8                                            # calls kept for the repeat check
EXIT_RE = re.compile(r'Exit Code:\s*(-?\d+)', re.I)   # Qwen Code's shell tool output
EDIT_TOOLS = {'edit', 'write_file', 'replace', 'multi_edit', 'apply_patch', 'notebook_edit'}

SIGNALS = {
    'fail_streak': 'several tool calls in a row have failed',
    'repeat': 'the agent keeps making the same tool call with the same arguments',
    'edit_churn': 'the agent keeps editing the same file and the checks after the edits still fail',
    'long_run': 'the agent has worked through many tool calls since the user last spoke (a check-in)',
    'manual': 'the user asked for advice',
    'review': 'the agent says it is done; check that the user\'s request is actually complete',
}


def new_state(session_id):
    return dict(session_id=session_id, batches_total=0, batches_since_user=0, recent=[], fail_streak=0, edits={},
                last_edit_batch={}, last_fail_batch=-1, last_advice_batch=None, long_run_mark=0, calls_made=0,
                pending=None, awaiting_send=False, reviewed_this_turn=False, turn_batches=0, last_advice_ts=None, suppressed=0)


def _state_path(session_id):
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', session_id or 'unknown')[:120]
    return os.path.join(config.STATE_DIR, 'sessions', safe + '.json')


class State:
    """the session's state file, locked for the hook's lifetime (hooks of one session run one after another; the lock
    covers a stray overlap), written atomically"""
    def __init__(self, session_id):
        self.path = _state_path(session_id)
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        self._lock = open(self.path + '.lock', 'a')
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        try:
            with open(self.path) as f:
                self.data = {**new_state(session_id), **json.load(f)}
        except (OSError, ValueError):
            self.data = new_state(session_id)

    def save(self):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), prefix='.st-')
        with os.fdopen(fd, 'w') as f:
            json.dump(self.data, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def close(self):
        fcntl.flock(self._lock, fcntl.LOCK_UN)
        self._lock.close()


def _text(v, limit=20000):
    """a tool response (any JSON shape) flattened to text, bounded"""
    if v is None:
        return ''
    if isinstance(v, str):
        return v[:limit]
    try:
        return json.dumps(v, ensure_ascii=False, default=str)[:limit]
    except (TypeError, ValueError):
        return str(v)[:limit]


def call_failed(call):
    """a failed call: status error, or a shell command whose output reports a non-zero exit code"""
    if call.get('status') in ('error',):
        return True
    resp = call.get('tool_response') or {}
    if isinstance(resp, dict) and resp.get('error'):
        return True
    if call.get('tool_name') == 'run_shell_command':
        m = EXIT_RE.findall(_text(resp))
        if m and int(m[-1]) != 0:
            return True
    return False


def signature(call):
    """tool name + arguments, whitespace-normalised, hashed"""
    args = call.get('tool_input') or {}
    try:
        s = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(args)
    s = re.sub(r'\s+', ' ', s)
    return hashlib.sha1(f'{call.get("tool_name")}|{s}'.encode()).hexdigest()[:16]


def edited_path(call):
    name = call.get('tool_name') or ''
    if name not in EDIT_TOOLS and 'edit' not in name and 'write' not in name:
        return None
    a = call.get('tool_input') or {}
    for k in ('file_path', 'absolute_path', 'path', 'filePath'):
        if isinstance(a.get(k), str):
            return a[k]
    return None


def on_batch(st, calls, cfg):
    """update the state with one PostToolBatch's calls; returns the signal recorded as pending (or None)"""
    st['batches_total'] += 1
    st['batches_since_user'] += 1
    st['turn_batches'] += 1
    st['awaiting_send'] = True
    b = st['batches_total']
    for c in calls:
        if c.get('status') == 'cancelled':
            continue
        st['recent'] = (st['recent'] + [signature(c)])[-RECENT:]
        p = edited_path(c)
        if p:
            st['edits'][p] = st['edits'].get(p, 0) + 1
            st['last_edit_batch'][p] = b
        if call_failed(c):
            st['fail_streak'] += 1
            st['last_fail_batch'] = b
        else:
            st['fail_streak'] = 0
    if st['pending'] or not cfg.get('enabled', True):
        return None
    if st['calls_made'] >= int(cfg['max_calls_per_session']):
        return None
    if st['last_advice_batch'] is not None and b - st['last_advice_batch'] < int(cfg['cooldown_batches']):
        return None
    t = cfg['triggers']
    sig = None
    if t.get('fail_streak') and st['fail_streak'] >= int(t['fail_streak']):
        sig = ('fail_streak', f'{st["fail_streak"]} failed calls in a row')
    elif t.get('repeat') and st['recent']:
        top = max(set(st['recent']), key=st['recent'].count)
        n = st['recent'].count(top)
        if n >= int(t['repeat']):
            sig = ('repeat', f'the same call {n} times among the last {len(st["recent"])}')
    if sig is None and t.get('edit_churn'):
        for p, n in st['edits'].items():
            if n >= int(t['edit_churn']) and st['last_fail_batch'] >= st['last_edit_batch'].get(p, 1 << 30):
                sig = ('edit_churn', f'{os.path.basename(p)} edited {n} times, still failing after the last edit')
                break
    if sig is None and t.get('long_run') and st['batches_since_user'] - st['long_run_mark'] >= int(t['long_run']):
        sig = ('long_run', f'{st["batches_since_user"]} tool batches since the user\'s last message')
        st['long_run_mark'] = st['batches_since_user']
    if sig and not interval_ok(st, cfg):   # stuck, but a hint came less than min_interval_s ago: let the agent work
        st['suppressed'] = st.get('suppressed', 0) + 1
        return None
    if sig:
        st['pending'] = dict(signal=sig[0], detail=sig[1], batch=b)
    return sig


def interval_ok(st, cfg, now=None):
    """whether min_interval_s has passed since the last advisor call of this session"""
    last = st.get('last_advice_ts')
    return last is None or (now or time.time()) - last >= float(cfg.get('min_interval_s') or 0)


def after_advice(st):
    """the counters a delivered (or attempted) call resets, so one problem asks once per cooldown"""
    st['calls_made'] += 1
    st['last_advice_batch'] = st['batches_total']
    st['last_advice_ts'] = time.time()
    st['suppressed'] = 0
    st['pending'] = None
    st['fail_streak'] = 0
    st['recent'] = []
    st['edits'] = {}


def on_user_prompt(st):
    """a new message from the user: the per-task counters start over"""
    st.update(batches_since_user=0, long_run_mark=0, recent=[], fail_streak=0, edits={}, last_edit_batch={},
              pending=None, awaiting_send=False, reviewed_this_turn=False, turn_batches=0)
