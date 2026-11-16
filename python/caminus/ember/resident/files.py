"""read, edit and write: file tools that do not go through the terminal.

The terminal is where an agent's edits went wrong: a PTY's line endings, quoting inside heredocs, a sed that matched
twice. These tools work on the file itself:

- read(path, offset, limit): numbered lines, a window at a time, never a binary file.
- edit(path, old, new, replace_all): an exact string replacement. old must occur exactly once (or replace_all), so an
  edit never lands somewhere the agent did not mean. A file with CRLF endings keeps them.
- write(path, content): the whole file, created or replaced.

Paths are relative to the agent's home unless absolute; the operating system's permissions decide what the account
may touch. Every edit and write returns a unified diff (bounded) for the chat's diff card, and is atomic (a temporary
file renamed into place, the old file's mode kept).
"""
import difflib
import os
import tempfile
from pathlib import Path

READ_DEFAULT = 400        # lines per read when no limit is given
READ_MAX = 2000           # lines one read may return
LINE_MAX = 2000           # characters of one line shown
WRITE_MAX = 2_000_000     # bytes one write or edit may produce
DIFF_MAX = 120            # diff lines returned to the model and the chat


def _path(home, path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError('a path is required')
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(home) / p


def _text(p, strict=False):
    data = p.read_bytes()
    if b'\0' in data[:8192]:
        raise ValueError(f'{p} looks binary; read it with the terminal (xxd, file) instead')
    if strict:
        # edit and write put the text back: a lossy decode would turn every non-UTF-8 byte into U+FFFD without a
        # line in the diff showing it
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f'{p} is not UTF-8 (byte {e.start}); edit it with the terminal (sed, python) so its '
                             'other bytes stay as they are') from None
    return data.decode('utf-8', errors='replace')


def _diff(path, before, after):
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), fromfile=str(path), tofile=str(path),
                                      lineterm='', n=2))
    added = sum(1 for ln in lines if ln.startswith('+') and not ln.startswith('+++'))
    removed = sum(1 for ln in lines if ln.startswith('-') and not ln.startswith('---'))
    body = lines[2:]                                   # drop the ---/+++ header: the path is in the result
    if len(body) > DIFF_MAX:
        body = body[:DIFF_MAX] + [f'… {len(body) - DIFF_MAX} more diff lines']
    return dict(added=added, removed=removed, diff='\n'.join(body))


def _atomic_write(p, text):
    data = text.encode()
    if len(data) > WRITE_MAX:
        raise ValueError(f'the result would be {len(data)} bytes; the file tools write at most {WRITE_MAX}')
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = p.stat().st_mode & 0o7777 if p.exists() else None
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix='.' + p.name + '.')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read(home, path, offset=None, limit=None):
    p = _path(home, path)
    if p.is_dir():
        names = sorted(e.name + ('/' if e.is_dir() else '') for e in p.iterdir())
        return dict(path=str(p), directory=True, entries=names[:500], more=max(0, len(names) - 500))
    text = _text(p)
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    start = max(1, int(offset or 1))
    count = max(1, min(int(limit or READ_DEFAULT), READ_MAX))
    shown = []
    for n in range(start, min(start + count, len(lines) + 1)):
        ln = lines[n - 1].rstrip('\r')
        if len(ln) > LINE_MAX:
            ln = ln[:LINE_MAX] + f' … [{len(ln) - LINE_MAX} more characters]'
        shown.append(f'{n:>6}\t{ln}')
    end = start + len(shown) - 1
    out = dict(path=str(p), lines=len(lines), shown=f'{start}-{end}' if shown else 'none', text='\n'.join(shown))
    if end < len(lines):
        out['next'] = f'read with offset={end + 1} for more'
    return out


def _miss_hint(before, old, new):
    """Why old is not in the file, as far as can be told: three misses in one real goal came back with no hint, one of
    them with quotes escaped as \\" that the file does not have."""
    if new and new != old and new in before:
        return ' (but new is already there: the edit seems applied; read the file to check)'
    if old.strip() and old.strip() in before:
        return ' (it occurs with different leading or trailing whitespace: copy it exactly from read)'
    unescaped = old.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
    if unescaped != old and unescaped in before:
        return ' (it matches once its backslash escapes are removed: send the text as it is in the file, unescaped)'
    import difflib
    lines = before.splitlines()
    want = next((ln for ln in old.splitlines() if ln.strip()), '')
    close = difflib.get_close_matches(want, lines, n=1, cutoff=0.6)
    if not close:
        return ' (no line looks like its first line: read the file again, it may have changed)'
    i = lines.index(close[0])
    span = lines[i:i + max(1, min(6, len(old.splitlines())))]
    return (f' (closest text, from line {i + 1}; copy from here exactly):\n' + '\n'.join(span))


def edit(home, path, old, new, replace_all=False):
    p = _path(home, path)
    if not isinstance(old, str) or old == '':
        raise ValueError('old must be the exact text to replace (to create a file, use write)')
    before = _text(p, strict=True)
    if old == new:
        if old in before:
            raise ValueError('old and new are the same, and the file already has this text: no edit is needed (an '
                             'earlier edit may have made it already)')
        raise ValueError('old and new are the same; nothing to change')
    crlf = '\r\n' in before
    if crlf and '\r\n' not in old:
        # The model writes \n; a CRLF file keeps its endings.
        old, new = old.replace('\n', '\r\n'), new.replace('\n', '\r\n')
    count = before.count(old)
    if count == 0:
        raise ValueError(f'old was not found in {p}' + _miss_hint(before, old, new))
    if count > 1 and not replace_all:
        raise ValueError(f'old occurs {count} times in {p}: add surrounding lines so it is unique, or set replace_all')
    after = before.replace(old, new) if replace_all else before.replace(old, new, 1)
    _atomic_write(p, after)
    return dict(path=str(p), replaced=count if replace_all else 1, **_diff(p, before, after))


def write(home, path, content):
    p = _path(home, path)
    if not isinstance(content, str):
        raise ValueError('content must be text')
    if p.is_dir():
        raise ValueError(f'{p} is a directory')
    existed = p.exists()
    before = _text(p, strict=True) if existed else ''
    _atomic_write(p, content)
    out = dict(path=str(p), created=not existed, **_diff(p, before, content))
    if not existed:
        out['diff'] = ''                               # a new file's diff is the file itself: say its size instead
        out['note'] = f'created, {content.count(chr(10)) + (0 if content.endswith(chr(10)) or not content else 1)} lines'
    return out


SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache', '.cache', 'build',
             'dist', '.tox'}
SEARCH_FILES = 20000      # files one search walks at most
SEARCH_BYTES = 2_000_000  # a file larger than this is skipped (logs, data)
SEARCH_HITS = 200         # matches returned at most


def search(home, pattern=None, glob=None, path='.', ignore_case=False, limit=None):
    """Find files (glob, e.g. '**/*.py' or '*test*') and lines in them (pattern, a regular expression), under path.
    A real agent ran ~20 grep / find commands through the terminal per goal for this; here the walk skips .git,
    node_modules, caches and binary or huge files, and every hit carries its line number for read's offset."""
    import fnmatch
    import re
    root = _path(home, path or '.')
    if not root.exists():
        raise ValueError(f'{root} does not exist')
    rx = None
    if pattern:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ValueError(f'pattern is not a valid regular expression: {e}') from None
    limit = max(1, min(int(limit or SEARCH_HITS), 1000))
    want = None
    if glob:
        g = glob[3:] if glob.startswith('**/') else glob
        want = lambda rel: fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(rel.rsplit('/', 1)[-1], g) or fnmatch.fnmatch(rel, g)
    files, hits, walked, more = [], [], 0, False
    paths = [root] if root.is_file() else None
    if paths is None:
        paths = []
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.git'))
            for n in sorted(names):
                paths.append(Path(d) / n)
                if len(paths) >= SEARCH_FILES:
                    break
            if len(paths) >= SEARCH_FILES:
                more = True
                break
    for p in paths:
        rel = str(p.relative_to(root)) if p != root else p.name
        if want and not want(rel):
            continue
        walked += 1
        if rx is None:
            files.append(rel)
            if len(files) >= limit:
                more = True
                break
            continue
        try:
            if p.stat().st_size > SEARCH_BYTES:
                continue
            data = p.read_bytes()
        except OSError:
            continue
        if b'\0' in data[:4096]:
            continue
        found = False
        for i, ln in enumerate(data.decode('utf-8', errors='replace').split('\n'), 1):
            if rx.search(ln):
                found = True
                ln = ln.rstrip('\r')
                hits.append(f'{rel}:{i}: {ln[:300]}' + (' …' if len(ln) > 300 else ''))
                if len(hits) >= limit:
                    more = True
                    break
        if found:
            files.append(rel)
        if more:
            break
    out = dict(root=str(root), files=len(files) if rx else files, searched=walked)
    if rx is not None:
        out['matches'] = hits
        out['note'] = 'read PATH with offset near a line number to see it in context'
    if more:
        out['truncated'] = f'stopped at {limit} results: narrow the glob, the path or the pattern'
    return out
