"""The audit log (docs/specs/spec-ember.md section 5): one JSON object per line, appended by the harness side only
(the file lives in the invoking user's state dir, which the agent user cannot read)."""
import json
import os
import time


def log(path, event):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps({'ts': time.time(), **event}) + '\n')


def tail(path, n=20):
    """the last n events as dicts (oldest first); a missing file is an empty log"""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
