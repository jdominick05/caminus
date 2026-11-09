"""The brief the advisor reads: the user's latest request and the agent's recent steps, condensed from Qwen Code's session
transcript (the JSONL file at the hook input's transcript_path: one record per line, `type` user / assistant /
tool_result / system, `message` in the model-API shape {role, parts: [{text}, {functionCall}, {functionResponse}, ...]}).
Secrets are redacted before anything leaves the machine."""
import json
import os
import re

TAIL_BYTES = 8 << 20          # read at most the last 8 MiB of a long transcript
MAX_STEPS = 18                # recent steps shown
EARLY_STEPS = 6               # ... plus the first steps after the request when they fall outside that window: early
                              # checks (ls, versions, docker ps, configs) often hold the clue a long session later needs
BRIEF_CHARS = 32000           # the whole brief, bounded

# obvious credentials: provider key prefixes, private key blocks, fields naming a secret (bare, env-style or quoted JSON /
# YAML keys: db_password=..., "client_secret": "...", AWS_SECRET_ACCESS_KEY=...), and Authorization header schemes
_KEY = (r'(?:[A-Za-z0-9]+[_.-])*(?:api[_-]?key|apikey|secret(?:[_-]?(?:access[_-]?)?key)?|(?:access|auth|refresh|session|id)?[_-]?token|'
        r'passw(?:or)?d|passphrase|pwd|authorization|private[_-]?key|access[_-]?key|credentials?)')
_SECRET_RES = [
    re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S),
    re.compile(r'\b(sk-ant-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
               r'xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20,}|'
               r'pypi-[A-Za-z0-9_-]{30,})\b'),
    re.compile(r'(?i)\b(Bearer|Basic|Token)(\s+)[A-Za-z0-9._~+/=-]{8,}'),
    # the field, its closing quote if quoted, the separator; then a quoted value (all of it, spaces included) or a bare one
    # (not one an earlier pattern already replaced: '[redacted ...]')
    re.compile(r'(?i)(?<![A-Za-z0-9])(["\']?' + _KEY + r'["\']?\s*[:=]\s*)(?:(["\'])((?:\\.|(?!\2).)+)\2|([^\s"\',;}\]\[][^\s"\',;}\]]{2,}))'),
]


def redact(s):
    s = _SECRET_RES[0].sub('[redacted private key]', s)
    s = _SECRET_RES[1].sub('[redacted key]', s)
    s = _SECRET_RES[2].sub(lambda m: m.group(1) + m.group(2) + '[redacted]', s)
    return _SECRET_RES[3].sub(lambda m: m.group(1) + (f'{m.group(2)}[redacted]{m.group(2)}' if m.group(2) else '[redacted]'), s)


ERR_LINE = re.compile(r'\b(?:error|fatal|exception|traceback|failed|failure|denied|refused|not found|no such file|undefined|segmentation fault|panicked)\b', re.I)


def _clip(s, n, tail=False):
    s = s.strip()
    if len(s) <= n:
        return s
    if tail:   # tool output: the end usually carries the error; error lines from the cut middle are kept too
        h = n // 3
        mid = s[h:len(s) - (n - h)]
        keep = [l.strip()[:220] for l in mid.splitlines() if ERR_LINE.search(l) and 'warning' not in l.lower()][:5]
        note = (f'\n[... {len(s) - n} chars cut; error lines from the cut part:\n' + '\n'.join(keep) + '\n...]\n') if keep else f'\n[... {len(s) - n} chars cut ...]\n'
        return s[:h] + note + s[-(n - h):]
    return s[:n] + f' [... {len(s) - n} chars cut]'


def _jdump(v):
    try:
        return json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(v)


def read_records(path):
    """the transcript's records (dicts), oldest first; [] when unreadable"""
    if not path or not os.path.exists(path):
        return []
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            if size > TAIL_BYTES:
                f.seek(size - TAIL_BYTES)
                f.readline()   # (a partial first line)
            data = f.read()
    except OSError:
        return []
    out = []
    for line in data.decode('utf-8', 'replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _parts(r):
    m = r.get('message') or {}
    p = m.get('parts') if isinstance(m, dict) else None
    return p if isinstance(p, list) else []


def _user_text(r):
    if r.get('type') != 'user' or r.get('subtype') not in (None, 'mid_turn_user_message'):
        return None
    t = '\n'.join(p['text'] for p in _parts(r) if isinstance(p, dict) and isinstance(p.get('text'), str))
    return t.strip() or None


def goal(records):
    """the session's standing goal from its latest goal_state record (Qwen Code's /goal): (objective, status) or None"""
    for r in reversed(records):
        if r.get('type') == 'system' and r.get('subtype') == 'goal_state':
            g = ((r.get('systemPayload') or {}).get('snapshot') or {}).get('goal')
            if isinstance(g, dict) and isinstance(g.get('objective'), str) and g['objective'].strip():
                return g['objective'].strip(), str(g.get('status') or 'unknown')
            return None   # (the latest state says no goal: cleared)
    return None


def steps(records):
    """(the user's latest request, earlier user requests (newest first, up to 2), recent steps as text lines)"""
    users = [(i, t) for i, r in enumerate(records) if (t := _user_text(r))]
    last_i, task = users[-1] if users else (-1, '')
    earlier = [t for _, t in users[:-1]][-2:][::-1]
    lines = []
    for r in records[last_i + 1:]:
        typ = r.get('type')
        for p in _parts(r):
            if not isinstance(p, dict):
                continue
            if typ == 'assistant' and isinstance(p.get('text'), str) and not p.get('thought'):
                if p['text'].strip():
                    lines.append('AGENT: ' + _clip(p['text'], 700))
            elif isinstance(p.get('functionCall'), dict):
                fc = p['functionCall']
                lines.append(f'CALL {fc.get("name")}: ' + _clip(_jdump(fc.get('args', {})), 500))
            elif isinstance(p.get('functionResponse'), dict):
                fr = p['functionResponse']
                resp = fr.get('response', {})
                body = resp.get('output', resp) if isinstance(resp, dict) else resp
                if isinstance(resp, dict) and resp.get('error'):
                    body = resp['error']
                lines.append(f'RESULT {fr.get("name")}: ' + _clip(body if isinstance(body, str) else _jdump(body), 1100, tail=True))
    return task, earlier, lines


def brief(path, signal_text, detail='', question='', cwd=''):
    """the advisor's input: why it was called, the task, the recent steps (newest last), redacted and bounded"""
    recs = read_records(path)
    task, earlier, lines = steps(recs)
    g = goal(recs)
    head = [f'Why you were called: {signal_text}' + (f' ({detail})' if detail else '') + '.']
    if question:
        head.append('The user\'s question for you: ' + _clip(question, 2000))
    if cwd:
        head.append(f'Working directory: {cwd}')
    if g:
        head.append(f'\nTHE USER\'S STANDING GOAL (status: {g[1]}):\n' + _clip(g[0], 3000))
    head.append('\nTHE USER\'S LATEST ' + ('MESSAGE' if g else 'REQUEST') + ':\n' + (_clip(task, 5000) or '(not found in the transcript)'))
    if earlier:
        head.append('\nEARLIER REQUESTS (newest first):\n' + '\n---\n'.join(_clip(t, 1200) for t in earlier))
    recent = lines[-MAX_STEPS:]
    early = lines[:min(EARLY_STEPS, max(0, len(lines) - MAX_STEPS))]
    gap = len(lines) - len(recent) - len(early)
    text = '\n'.join(head)
    if early:
        text += f'\n\nTHE AGENT\'S FIRST STEPS AFTER THE REQUEST:\n' + '\n'.join(_clip(l, 700, tail=True) for l in early)
        if gap > 0:
            text += f'\n[... {gap} steps not shown ...]'
    text += f'\n\nTHE AGENT\'S RECENT STEPS (oldest first, {len(recent)} of {len(lines)} since the request):\n' + '\n'.join(recent)
    if len(text) > BRIEF_CHARS:
        text = text[:BRIEF_CHARS // 3] + '\n[... cut ...]\n' + text[-(BRIEF_CHARS * 2 // 3):]
    return redact(text)
