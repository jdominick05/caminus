"""Qwen Code PostToolBatch hook (`caminus qwen-code setup` registers it in Caminus's Qwen homes): after each batch of tool
calls, one line on how much context is left, appended to the batch's last tool result:

  [context: ~498K of 1,048K tokens used; ~392K left before automatic compaction]

A model that can't see its context budget runs out of it mid-task; told how much is left after every step, it can plan
the rest of the work (finish, save notes) before Qwen Code compacts the conversation. The line becomes part of the tool
result Qwen Code records and sends, so it is in the history from then on: the server's prompt cache keeps its prefix.

The numbers: Qwen Code's own measure (its compaction gate): the last model call's prompt tokens plus its output tokens,
from the session transcript's assistant record for this batch (matched by the batch's call ids), plus this batch's tool
output (estimated high: a third of a token per ASCII character, so the budget left is never overstated). The window is
the record's contextWindowSize; automatic compaction comes at Qwen Code's threshold (computeThresholds in
chatCompressionService.ts: min(pct x window, window - 20,000 - 13,000), pct = context.autoCompactThreshold, default 0.85).

Nothing is added in a subagent (its transcript is not the one given), when the record for this batch is not on disk within
a second (Qwen Code writes the transcript asynchronously), when the record has no usage, or with CAMINUS_CONTEXT_NOTE=0.
Standard library only; always exits 0 and prints a JSON object or nothing (exit 2 would block the turn).
  python -I context_note.py --settings SETTINGS_JSON     (hook input on stdin, as Qwen Code runs it)"""
import json
import os
import sys
import time

TAIL_BYTES = 4 << 20          # the transcript's last 4 MiB: the newest assistant record is near the end
WAIT_S, POLL_S = 1.0, 0.05    # how long to wait for this batch's assistant record to reach the disk
SUMMARY_RESERVE = 20000       # Qwen Code's computeThresholds: the window minus the summary's output reserve ...
AUTOCOMPACT_BUFFER = 13000    # ... minus this buffer is the absolute ceiling of the auto tier
DEFAULT_PCT = 0.85            # ... and pct x window its proportional preference (the lower of the two applies)


def records(path):
    """the transcript's last records (dicts), oldest first; [] when unreadable"""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > TAIL_BYTES:
                f.seek(size - TAIL_BYTES)
                f.readline()          # (a partial first line)
            data = f.read()
    except (OSError, TypeError):
        return []
    out = []
    for line in data.decode('utf-8', 'replace').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def call_ids(r):
    parts = (r.get('message') or {}).get('parts') if isinstance(r.get('message'), dict) else None
    return {p['functionCall'].get('id') for p in parts or [] if isinstance(p, dict) and isinstance(p.get('functionCall'), dict)}


def batch_record(path, ids, wait_s=WAIT_S):
    """the assistant record whose calls this batch ran (None when it is not on disk within wait_s)"""
    t_end = time.monotonic() + wait_s
    while True:
        for r in reversed(records(path)):
            if r.get('type') == 'assistant' and call_ids(r) & ids:
                return r
        if time.monotonic() >= t_end:
            return None
        time.sleep(POLL_S)


def est_tokens(text):
    """a high estimate of text's tokens: a third of a token per ASCII character, 1.5 per other character"""
    a = len(text.encode('ascii', 'ignore'))          # (the ASCII characters, counted at C speed)
    return int((a + 2) // 3 + 1.5 * (len(text) - a))


def auto_threshold(window, pct):
    ceiling = window - SUMMARY_RESERVE - AUTOCOMPACT_BUFFER
    return min(pct * window, ceiling) if ceiling > 0 else pct * window


def settings_pct(path):
    try:
        with open(path) as f:
            v = (json.load(f).get('context') or {}).get('autoCompactThreshold')
    except (OSError, ValueError, TypeError, AttributeError):
        return DEFAULT_PCT
    return float(v) if isinstance(v, (int, float)) and 0 < v <= 1 else DEFAULT_PCT


def k(n):
    return f'{round(n / 1000):,}K'


def note(inp, settings_path=None, wait_s=WAIT_S):
    """the line for one PostToolBatch hook input, or None"""
    if os.environ.get('CAMINUS_CONTEXT_NOTE') == '0' or inp.get('agent_id'):
        return None
    calls = [c for c in inp.get('tool_calls') or [] if isinstance(c, dict)]
    ids = {c.get('tool_call_id') or c.get('tool_use_id') for c in calls} - {None}
    if not ids or not inp.get('transcript_path'):
        return None
    r = batch_record(inp['transcript_path'], ids, wait_s)
    u = (r or {}).get('usageMetadata')
    window = (r or {}).get('contextWindowSize')
    if not isinstance(u, dict) or not isinstance(window, int) or window <= 0:
        return None
    base = int(u.get('promptTokenCount') or 0) + int(u.get('candidatesTokenCount') or 0)
    if base <= 0:
        return None
    out = sum(est_tokens(json.dumps((c.get('tool_response') or {}).get('response_parts', c.get('tool_response')),
                                    ensure_ascii=False, default=str)) for c in calls)
    used = base + out
    left = auto_threshold(window, settings_pct(settings_path) if settings_path else DEFAULT_PCT) - used
    if left <= 0:
        return f'[context: ~{k(used)} of {k(window)} tokens used; automatic compaction comes before the next step]'
    return f'[context: ~{k(used)} of {k(window)} tokens used; ~{k(left)} left before automatic compaction]'


def main(argv, raw):
    settings_path = argv[argv.index('--settings') + 1] if '--settings' in argv[:-1] else None
    inp = json.loads(raw)
    if not isinstance(inp, dict) or inp.get('hook_event_name') not in (None, 'PostToolBatch'):
        return None
    line = note(inp, settings_path)
    return dict(hookSpecificOutput=dict(hookEventName='PostToolBatch', additionalContext=line)) if line else None


if __name__ == '__main__':
    out = None
    try:
        out = main(sys.argv[1:], sys.stdin.read(16 << 20))
    except BaseException:   # (fail open: never block the batch, never print a traceback into the context)
        out = None
    if out:
        sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    os._exit(0)
