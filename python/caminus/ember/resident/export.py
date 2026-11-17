"""/export: the goal's transcript as markdown, so a human can read the work outside the panel.

Each user / assistant message is a paragraph (steers marked), each tool call a fenced block with its
truncated result; the goal's prompt and its done check open the file. The default path is
~/exports/<goal8>-<date>.md in the agent's home.
"""
import json
from pathlib import Path

RESULT_LIMIT = 4000      # a tool result's characters in the export; the rest is cut with a note


def render(gid, g):
    """The goal transcript as markdown.

    Steers reach the transcript as user messages tagged '[steer #N, ...] text' (runtime's inbox
    pull), so those render under their own heading; other user messages are the human's words.
    """
    out = [f'# Goal {gid[:8]}', '', g['prompt'], '']
    if g.get('done_check'):
        out += [f"Done check: {g['done_check']}", '']
    if g.get('summary') and g['state'] == 'completed':
        out += [f"Finished: {g['summary']}", '']
    for m in g['messages']:
        role, content = m.get('role'), m.get('content') or ''
        if role == 'user':
            head = '## You (steer)' if content.startswith('[steer #') else '## You'
            out += [head, '', content, '']
        elif role == 'assistant':
            out += ['## Agent', '', content, ''] if content else []
            for call in m.get('tool_calls') or []:
                fn = call['function']
                out += [f'### tool: {fn["name"]}', '', '```json', fn['arguments'], '```', '']
        elif role == 'tool':
            result = m.get('content') or ''
            if len(result) > RESULT_LIMIT:
                result = result[:RESULT_LIMIT] + f'\n[{len(result) - RESULT_LIMIT} characters omitted]'
            out += ['<result>', '', '```', result, '```', '']
    return '\n'.join(out)


def path_for(home, gid):
    import datetime
    folder = home / 'exports'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (gid[:8] + '-' + datetime.date.today().isoformat() + '.md')


def export(store, gid, out=None):
    """Write the transcript to `out` (default under the agent home) and return (path, bytes)."""
    g = store.goal(gid)
    text = render(gid, g)
    path = Path(out).expanduser() if out else path_for(store.home, gid)
    if not path.is_absolute():
        path = store.home / path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode()
    path.write_bytes(data)
    return path, len(data)
