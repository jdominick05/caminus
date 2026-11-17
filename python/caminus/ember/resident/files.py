"""read, search, edit and write: file tools that do not go through the terminal.

The terminal is where an agent's edits went wrong: a PTY's line endings, quoting inside heredocs, a sed that matched
twice. These tools work on the file itself:

- read(path, start, lines): numbered lines, a window at a time, never a binary file. A PNG, JPEG, GIF or WebP
  file comes back as its size and format (image_info, from the header: no image library) with a sha256; the transport
  (model.py) sends the file itself as an image to a brain that can see, and a plain line to one that can't.
- search(path, glob, pattern, context): file names by glob, text by regex with context lines; .gitignore honoured in a
  git repository, binary files skipped, results capped with a count of the rest. ripgrep, when it is installed, only
  narrows the files to look in; the matching, the context and the caps are this module's, so the result is the same
  either way.
- edit(path, old, new, replace_all): an exact string replacement. old must occur exactly once (or replace_all), so an
  edit never lands somewhere the agent did not mean. A file with CRLF endings keeps them.
- write(path, content): the whole file, created or replaced.

Paths are relative to the agent's home unless absolute; the operating system's permissions decide what the account
may touch. Every edit and write returns a unified diff (bounded) for the chat's diff card, and is atomic (a temporary
file renamed into place, the old file's mode kept).
"""
import difflib
import hashlib
from fnmatch import fnmatchcase
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .diff import _run

READ_DEFAULT = 400        # lines per read when no limit is given
READ_MAX = 2000           # lines one read may return
LINE_MAX = 2000           # characters of one line shown
WRITE_MAX = 2_000_000     # bytes one write or edit may produce
DIFF_MAX = 120            # diff lines returned to the model and the chat
SEARCH_NAMES = 200        # file names one glob search lists
SEARCH_FILES = 20         # files with matches one search shows
SEARCH_MATCHES = 60       # matching lines one search shows
SEARCH_CHARS = 16_000     # characters of matches and context one search shows (the result stays below outputs' spill)
SEARCH_LINE = 300         # characters of one line shown (minified code is one huge line)
SEARCH_SCAN = 20_000      # matches counted before a search stops counting
SEARCH_FILE_MAX = 10_000_000   # bytes: a bigger file is skipped (the stdlib path reads each file whole)
SEARCH_SECONDS = 60       # a search stops looking after this long and says so
SKIP_DIRS = frozenset(('.git', 'node_modules', 'build'))   # skipped by the walk outside a git repository
# An image goes to the model whole (no resizing without an image library), so read refuses one past these: 5 MB once
# base64-encoded is the smallest per-image limit among Claude's platforms (Bedrock; the API's own is 10 MB), and past
# 20 images in one request a stricter size limit applies to every image (resent results count), which images of 3000 px
# a side or less stay under (platform.claude.com, build-with-claude/vision).
IMAGE_MAX_BYTES = 3_750_000
IMAGE_MAX_SIDE = 3000
IMAGE_HEAD = 1 << 20      # bytes of a file searched for its dimensions (a JPEG's frame header follows its EXIF)


def image_info(head):
    """(format, width, height) from an image file's first bytes, None when they are not a PNG, JPEG, GIF or WebP;
    ValueError when they start like one but the dimensions can't be read."""
    be16, le16 = (lambda i: int.from_bytes(head[i:i + 2], 'big')), (lambda i: int.from_bytes(head[i:i + 2], 'little'))
    if head[:8] == b'\x89PNG\r\n\x1a\n':
        if head[12:16] != b'IHDR' or len(head) < 24:
            raise ValueError('a PNG without its IHDR header')
        return 'png', int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
    if head[:6] in (b'GIF87a', b'GIF89a'):
        if len(head) < 10:
            raise ValueError('a GIF cut short in its header')
        return 'gif', le16(6), le16(8)
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        chunk = head[12:16]
        if chunk == b'VP8 ' and head[23:26] == b'\x9d\x01\x2a':
            return 'webp', le16(26) & 0x3fff, le16(28) & 0x3fff
        if chunk == b'VP8L' and head[20:21] == b'\x2f' and len(head) >= 25:
            bits = int.from_bytes(head[21:25], 'little')
            return 'webp', (bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1
        if chunk == b'VP8X' and len(head) >= 30:
            return 'webp', int.from_bytes(head[24:27], 'little') + 1, int.from_bytes(head[27:30], 'little') + 1
        raise ValueError('a WebP whose frame header is not VP8, VP8L or VP8X')
    if head[:3] == b'\xff\xd8\xff':
        i = 2
        while i + 4 <= len(head):
            if head[i] != 0xff:
                raise ValueError('a JPEG with a broken segment at byte %d' % i)
            m = head[i + 1]
            if m == 0xff:                                   # (fill byte)
                i += 1
                continue
            if m in (0x01, 0xd8) or 0xd0 <= m <= 0xd7:      # (markers without a length)
                i += 2
                continue
            if m in (0xd9, 0xda):
                break                                       # end of image / start of scan before any frame header
            if 0xc0 <= m <= 0xcf and m not in (0xc4, 0xc8, 0xcc):
                if i + 9 > len(head):
                    break
                return 'jpeg', be16(i + 7), be16(i + 5)
            i += 2 + be16(i + 2)
        raise ValueError('a JPEG whose frame header is not in its first %d bytes' % IMAGE_HEAD)
    return None


def image(p):
    """read's answer for an image file, or None when p is not one"""
    with open(p, 'rb') as f:
        head = f.read(16)
        if not head.startswith((b'\x89PNG', b'GIF8', b'RIFF', b'\xff\xd8\xff')):
            return None                                     # (a text file costs 16 bytes here)
        head += f.read(IMAGE_HEAD - 16)
    found = image_info(head)
    if found is None:
        return None
    fmt, w, h = found
    size = p.stat().st_size
    info = dict(format=fmt, width=w, height=h, bytes=size)
    if size > IMAGE_MAX_BYTES or max(w, h) > IMAGE_MAX_SIDE:
        return dict(path=str(p), image=info, note=(
            f'{w}x{h}, {size:,} bytes: past what read sends as an image ({IMAGE_MAX_SIDE} px a side, '
            f'{IMAGE_MAX_BYTES:,} bytes), so no image is sent. Make a smaller copy in the terminal (ImageMagick\'s '
            'convert -resize, where installed) and read that.'))
    info['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()   # (the transport sends the file only while it matches)
    return dict(path=str(p), image=info)


def _path(home, path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError('a path is required')
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(home) / p


def _binary(data):
    return b'\0' in data[:8192]


def _text(p, strict=False):
    data = p.read_bytes()
    if _binary(data):
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


def read(home, path, start=None, lines=None):
    p = _path(home, path)
    if p.is_dir():
        names = sorted(e.name + ('/' if e.is_dir() else '') for e in p.iterdir())
        return dict(path=str(p), directory=True, entries=names[:500], more=max(0, len(names) - 500))
    if (img := image(p)) is not None:
        return img
    text = _text(p)
    rows = _lines(text)
    start = max(1, int(start or 1))
    count = max(1, min(int(lines or READ_DEFAULT), READ_MAX))
    shown = [_numbered(n, rows[n - 1], LINE_MAX) for n in range(start, min(start + count, len(rows) + 1))]
    end = start + len(shown) - 1
    out = dict(path=str(p), shown=(f'{start}-{end}' if shown else 'none') + f' of {len(rows)}', text='\n'.join(shown))
    if end < len(rows):
        out['next'] = f'read with start={end + 1} for more'
    return out


def _lines(text):
    rows = [ln.rstrip('\r') for ln in text.split('\n')]
    if rows and rows[-1] == '':
        rows.pop()
    return rows


def _numbered(n, ln, width, mark='\t'):
    if len(ln) > width:
        ln = ln[:width] + f' … [{len(ln) - width} more characters]'
    return f'{n:>6}{mark}{ln}'


def _rg():
    return shutil.which('rg')


def _size(p):
    try:
        return p.stat().st_size
    except OSError:
        return -1                         # (gone since it was listed)


def _listing(root):
    """The files a search looks in, sorted: what git lists (tracked and untracked, not ignored) inside a repository,
    else a walk that skips SKIP_DIRS. A file git lists but that is gone, or a submodule, is left out. A folder git
    lists nothing in (one it ignores, asked for by name) is walked."""
    if root.is_file():
        return [root]
    r = _run(['git', '-C', str(root), 'ls-files', '-co', '--exclude-standard', '-z'], 30)
    files = [root / n for n in r.stdout.split('\0') if n] if r is not None and r.returncode == 0 else []
    if not files:
        for d, dirs, names in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            files += [Path(d) / n for n in names]
    return sorted((p for p in files if p.is_file()), key=lambda p: str(p.relative_to(root)))


def _rg_candidates(rg, pattern, files, deadline):
    """The files ripgrep finds a match in, or None when it cannot answer for this pattern (a lookaround or a
    backreference: its regex syntax is not Python's) and the stdlib reads every file instead."""
    found = set()
    for i in range(0, len(files), 500):
        left = deadline - time.monotonic()
        if left <= 0:
            break
        try:
            r = subprocess.run([rg, '--no-config', '--text', '--crlf', '--files-with-matches', '--null', '-e', pattern,
                                '--', *map(str, files[i:i + 500])], capture_output=True, timeout=left)
        except subprocess.TimeoutExpired:
            break
        except OSError:
            return None
        if r.returncode == 2 and not r.stdout:      # (2 with output: some file was unreadable, the rest were searched)
            return None
        found.update(os.fsdecode(n) for n in r.stdout.split(b'\0') if n)
    return [p for p in files if str(p) in found]


def _glob_match(p, root, g):
    """A glob without a slash matches the file's name at any depth; with one, its path under root (fnmatch's * also
    crosses folders, and a leading **/ may match none)."""
    if '/' not in g:
        return fnmatchcase(p.name, g)
    rel = p.name if p == root else str(p.relative_to(root))
    return fnmatchcase(rel, g) or (g.startswith('**/') and fnmatchcase(rel, g[3:]))


def search(home, path='.', glob=None, pattern=None, context=2):
    if not glob and not pattern:
        raise ValueError('give glob (file names), pattern (a regex to find in the text), or both')
    root = _path(home, path or '.')
    if root.is_dir():                     # refused as read refuses it: missing or unreadable
        os.scandir(root).close()
    else:
        open(root, 'rb').close()
    base = Path(home)
    show = lambda p: os.path.relpath(p, base) if p == base or base in p.parents else str(p)
    files = _listing(root)
    if glob:
        files = [p for p in files if _glob_match(p, root, glob.removeprefix('./'))]
    out = dict(path=show(root))
    if not pattern:
        out.update(files=len(files), text='\n'.join(show(p) for p in files[:SEARCH_NAMES]))
        if len(files) > SEARCH_NAMES:
            out['more'] = f'{len(files) - SEARCH_NAMES} more files not listed: narrow glob or path'
        return out
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f'pattern is not a valid regex ({e}); escape special characters such as ( [ . * with \\')
    big = [p for p in files if _size(p) > SEARCH_FILE_MAX]
    files = [p for p in files if 0 <= _size(p) <= SEARCH_FILE_MAX]
    deadline = time.monotonic() + SEARCH_SECONDS
    rg = _rg()
    look = (_rg_candidates(rg, pattern, files, deadline) if rg and files else None)
    if look is None:
        look = files
    context = max(0, min(int(context if context is not None else 2), 10))
    total = nfiles = shown_m = shown_f = chars = 0
    text, late = [], False
    for p in look:
        if total >= SEARCH_SCAN or time.monotonic() > deadline:
            late = total < SEARCH_SCAN
            break
        try:
            data = p.read_bytes()
        except OSError:
            continue                      # (unreadable: the account may not read it)
        if _binary(data):
            continue
        rows = _lines(data.decode('utf-8', errors='replace'))
        hits = [i for i, ln in enumerate(rows) if rx.search(ln)]
        if not hits:
            continue
        total += len(hits)
        nfiles += 1
        if shown_m >= SEARCH_MATCHES or shown_f >= SEARCH_FILES or chars >= SEARCH_CHARS:
            continue
        shown_f += 1
        keep = hits[:SEARCH_MATCHES - shown_m]
        hit, last = set(keep), None
        text.append(show(p))
        for i in sorted({j for h in keep for j in range(max(0, h - context), min(len(rows), h + context + 1))}):
            if i in hit and chars >= SEARCH_CHARS:
                break                             # (the text is full: this match and the rest are counted, not shown)
            if last is not None and i > last + 1:
                text.append('    --')
            text.append(_numbered(i + 1, rows[i], SEARCH_LINE, ':  ' if i in hit else '-  '))
            chars += len(text[-1]) + 1
            shown_m += i in hit
            last = i
    out.update(matches=total, files=nfiles, searched=len(files), text='\n'.join(text))
    if total > shown_m:
        where = f' in {nfiles - shown_f} more files' if nfiles > shown_f else ''
        out['more'] = (f'{"at least " if total >= SEARCH_SCAN else ""}{total - shown_m} more matches{where} not shown: '
                       'narrow pattern, glob or path')
    if late:
        out['stopped'] = f'stopped looking after {SEARCH_SECONDS} s: the counts are partial; narrow path or glob'
    if big:
        out['skipped'] = f'{len(big)} files over {SEARCH_FILE_MAX // 1_000_000} MB not searched'
    return out


def _miss_hint(before, old, new):
    """Why old is not in the file, as far as can be told: three misses in one real goal came back with no hint, one of
    them with quotes escaped as \\" that the file does not have."""
    # only a new text that carries the change (longer than a few characters, and containing what old did not) can
    # say "already applied": a one-letter new text occurs in any file (an audit's edit "x" claimed it was applied)
    whole_line = new.strip() and new.strip() in {ln.strip() for ln in before.splitlines()}
    if new and new != old and new in before and (whole_line or len(new.strip()) >= 12):
        return ' (but new is already there: the edit may have been applied before; read the file to check)'
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
