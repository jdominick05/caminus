"""/diff: what the goal's file changes look like at their repos.

The goal's recorded edit / write / terminal calls name the places work happened: their paths,
resolved against the agent home, are walked up to the nearest .git root. Each repo gets
`git diff --stat` and the full `git diff`, capped at 200 KB with a truncation flag. With no
goal named, the most recent goal is used.
"""
import json
import re
import subprocess
from pathlib import Path

DIFF_CAP = 200 * 1024    # bytes of `git diff` reported per repo before it is cut


def _resolve(home, path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(home) / p


def touched_repos(store, gid):
    """The .git roots of the paths the goal's tool calls touched, first touched first."""
    roots = []
    for r in store.db.execute(
            "SELECT name,args FROM actions WHERE goal=? AND name IN ('edit','write','terminal') "
            "ORDER BY rowid", (gid,)):
        try:
            args = json.loads(r['args'])
        except ValueError:
            continue
        # edit/write touch a path; a terminal call offers where it ran (cwd) and the folders it cd's into
        if r['name'] in ('edit', 'write'):
            paths = [args.get('path')]
        else:
            paths = [args.get('cwd')] + CD.findall(args.get('command') or '')
        for path in paths:
            if path:
                _add_root(store.home, path, roots)
    return roots


CD = re.compile(r'(?:^|[;&|]|\bthen|\bdo)\s*cd\s+([^\s;&|]+)')


def _add_root(home, path, roots):
    p = _resolve(home, path.strip('"\''))
    if not p.exists():
        p = p.parent
    while True:
        if (p / '.git').exists():
            if p not in roots:
                roots.append(p)
            break
        if p.parent == p:
            break
        p = p.parent


def _run(args, timeout):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def diff(store, gid):
    """{"repos":[{"path","stat","diff","truncated"}]} for every repo the goal touched."""
    repos = []
    for root in touched_repos(store, gid):
        # Staged and unstaged changes against HEAD, plus new files (untracked: a file the agent wrote with write).
        stat = _run(['git', '-C', str(root), 'status', '--short'], 30)
        full = _run(['git', '-C', str(root), 'diff', 'HEAD'], 60)
        if stat is None or full is None:
            continue
        text = full.stdout if full.returncode == 0 else ''
        if full.returncode != 0:          # no commit yet: everything is new
            text = (_run(['git', '-C', str(root), 'diff', '--cached'], 60) or full).stdout
        new = _run(['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard'], 30)
        for name in (new.stdout.splitlines() if new else [])[:50]:
            if len(text.encode()) > DIFF_CAP:
                break
            one = _run(['git', '-C', str(root), 'diff', '--no-index', '--', '/dev/null', name], 30)
            if one is not None:
                text += one.stdout
        truncated = len(text.encode()) > DIFF_CAP
        if truncated:
            text = text.encode()[:DIFF_CAP].decode('utf-8', 'ignore')
        repos.append(dict(path=str(root), stat=(stat.stdout or stat.stderr).strip(),
                          diff=text, truncated=truncated))
    return dict(repos=repos)
