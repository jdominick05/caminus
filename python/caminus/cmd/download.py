"""caminus download: the model files from their public Hugging Face repositories (caminus.models: pinned revisions, sizes,
sha256), into the model directory (caminus.config.model_dir: CAMINUS_MODEL_DIR, model_dir in ~/.config/caminus/config.toml,
default ~/.local/share/caminus/models) as <repository name>/<path>, where caminus serve finds them by default.

  caminus download [--dir DIR] [--only coder|dflash] [--verify] [--dry-run]

  * resumable: a file arrives as NAME.part; an interrupted run (Ctrl-C, a dropped connection: up to 8 attempts with backoff)
    continues it with an HTTP Range request; a server that ignores Range restarts the file;
  * checked: the size from the server against the manifest before any byte is kept, then the size and the sha256 of the
    whole file; only a complete, matching file is renamed into place, with NAME.sha256 beside it (later runs and caminus
    doctor need not re-hash). A mismatching .part is removed (it can never complete);
  * a file already in place with the manifest's size is kept (--verify hashes it: slow, ~1 min per 30 GB); with another
    size it is reported and left alone;
  * free space is checked first (the missing bytes + 1 GiB);
  * HF_ENDPOINT (default https://huggingface.co) and HF_TOKEN, as huggingface_hub reads them; the token is sent to the
    endpoint's host only, never to a host a redirect leads to.
The Coder is 58.4 GB in two shards; the DFlash drafter 0.65 GB (its file was checked against this machine's copy only:
caminus.models). Network only: nothing touches a GPU."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .. import config, models

CHUNK = 8 << 20
ATTEMPTS = 8
MARGIN = 1 << 30


class DownloadError(Exception):
    pass


class _Redirect(urllib.request.HTTPRedirectHandler):
    """follows redirects, dropping Authorization when the host changes (a CDN never sees the token)"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header('Authorization')
        return new


def url_of(it, endpoint):
    q = urllib.parse.quote
    return f'{endpoint.rstrip("/")}/{it.repo}/resolve/{q(it.revision)}/{q(it.path)}'


def sha256_file(path, log=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h


def write_marker(final, sha):
    with open(final + '.sha256', 'w') as f:
        f.write(f'{sha}  {os.path.basename(final)}\n')


def marker_ok(final, sha):
    try:
        with open(final + '.sha256') as f:
            return f.read().split()[0] == sha
    except (OSError, IndexError):
        return False


def free_bytes(path):
    p = path
    while not os.path.isdir(p):
        p = os.path.dirname(p) or '.'
    return shutil.disk_usage(p).free


def _open(url, start, token, timeout):
    req = urllib.request.Request(url, headers={'User-Agent': 'caminus-download'})
    if start:
        req.add_header('Range', f'bytes={start}-')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    return urllib.request.build_opener(_Redirect()).open(req, timeout=timeout)


def fetch(it, final, endpoint, token, log=print, timeout=60, attempts=ATTEMPTS, sleep=time.sleep):
    """download one manifest item to final (via final.part); returns the sha256 hex. Raises DownloadError"""
    part = final + '.part'
    os.makedirs(os.path.dirname(final), exist_ok=True)
    url = url_of(it, endpoint)
    last = None
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > it.size:
            log(f'{it.key}: {part} is longer than the file ({have} > {it.size} bytes): starting over')
            os.unlink(part); have = 0
        h = sha256_file(part) if have else hashlib.sha256()
        clean = True                     # (h == the part's bytes; after a failed write the part is hashed again)
        if have < it.size:
            try:
                r = _open(url, have, token, timeout)
            except urllib.error.HTTPError as e:
                if e.code == 416 and have:
                    raise DownloadError(f'{it.key}: the server refuses the range from byte {have}: its file is shorter than the '
                                        f'manifest\'s {it.size} bytes ({url}); {part} is kept')
                if e.code in (401, 403):
                    raise DownloadError(f'{it.key}: HTTP {e.code} for {url}: the repository needs a login (set HF_TOKEN) or is not public')
                if e.code == 404:
                    raise DownloadError(f'{it.key}: HTTP 404 for {url}: no such file at revision {it.revision[:12]}'
                                        + (' (this file\'s presence in that repository was never checked: caminus.models)' if it.source == 'local-unverified' else ''))
                if e.code < 500:
                    raise DownloadError(f'{it.key}: HTTP {e.code} for {url}')
                last = f'HTTP {e.code}'
                r = None
            except (urllib.error.URLError, OSError) as e:
                last = f'{type(e).__name__}: {e}'
                r = None
            if r is not None:
                with r:
                    code = r.status
                    if code == 206:
                        cr = r.headers.get('Content-Range', '')
                        try:
                            start, total = int(cr.split()[1].split('-')[0]), int(cr.split('/')[1])
                        except (IndexError, ValueError):
                            raise DownloadError(f'{it.key}: a 206 answer without a usable Content-Range ({cr!r})')
                        if start != have:
                            raise DownloadError(f'{it.key}: asked for byte {have}, the server sent from byte {start}')
                    elif code == 200:
                        total = int(r.headers.get('Content-Length', -1))
                        if have:
                            log(f'{it.key}: the server ignored the range request: restarting the file')
                            have = 0; h = hashlib.sha256()
                    else:
                        raise DownloadError(f'{it.key}: HTTP {code} for {url}')
                    if total not in (-1, it.size):
                        raise DownloadError(f'{it.key}: the server\'s file is {total} bytes, the manifest says {it.size}: refused '
                                            f'(another file than the one this release was tested with)')
                    t0, done, tl = time.time(), have, time.time()
                    try:
                        with open(part, 'ab' if have else 'wb') as f:
                            while True:
                                b = r.read(CHUNK)
                                if not b:
                                    break
                                if done + len(b) > it.size:
                                    raise DownloadError(f'{it.key}: the server sent more than the manifest\'s {it.size} bytes')
                                f.write(b); h.update(b); done += len(b)
                                if time.time() - tl > 10:
                                    tl = time.time()
                                    log(f'{it.key}: {done / 1e9:.2f} / {it.size / 1e9:.2f} GB ({100 * done / it.size:.0f} %), '
                                        f'{(done - have) / max(1e-9, tl - t0) / 1e6:.0f} MB/s')
                            f.flush(); os.fsync(f.fileno())
                    except (urllib.error.URLError, OSError) as e:
                        last = f'{type(e).__name__}: {e}'; clean = False
                have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == it.size:
            got = h.hexdigest() if clean else sha256_file(part).hexdigest()
            if got != it.sha256:
                os.unlink(part)
                raise DownloadError(f'{it.key}: sha256 {got} != the manifest\'s {it.sha256}: the file was removed'
                                    + (' (this drafter file was checked against one machine\'s copy only: caminus.models)' if it.source == 'local-unverified' else ''))
            os.replace(part, final)
            write_marker(final, got)
            d = os.open(os.path.dirname(final), os.O_RDONLY)
            try: os.fsync(d)
            finally: os.close(d)
            return got
        if have < it.size and attempt < attempts:
            wait = min(30, 2 ** attempt)
            log(f'{it.key}: interrupted at {have} of {it.size} bytes ({last or "the connection ended early"}); attempt {attempt + 1} of {attempts} in {wait} s')
            sleep(wait)
    raise DownloadError(f'{it.key}: gave up after {attempts} attempts at {have} of {it.size} bytes ({last}); run again to resume')


def download(items, dest, endpoint=None, token=None, verify=False, dry_run=False, log=print, **kw):
    """every item into dest; returns the number of failures"""
    endpoint = endpoint or os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'
    token = os.environ.get('HF_TOKEN') if token is None else token
    bad = 0
    todo = []
    for it in items:
        final = os.path.join(dest, models.local_rel(it))
        if os.path.exists(final):
            sz = os.path.getsize(final)
            if sz != it.size:
                log(f'{it.key}: {final} exists with {sz} bytes, the manifest says {it.size}: left alone (move it away to download it)')
                bad += 1
            elif verify and not dry_run:
                log(f'{it.key}: hashing {final} ...')
                got = sha256_file(final).hexdigest()
                if got == it.sha256:
                    write_marker(final, got); log(f'{it.key}: present, sha256 verified')
                else:
                    log(f'{it.key}: present but sha256 {got} != {it.sha256}: left alone (move it away to download it)'); bad += 1
            else:
                log(f'{it.key}: present ({"sha256 verified earlier" if marker_ok(final, it.sha256) else "size matches; --verify hashes it"})')
            continue
        todo.append((it, final))
    need = sum(it.size - (os.path.getsize(f + '.part') if os.path.exists(f + '.part') else 0) for it, f in todo)
    for it, final in todo:
        log(f'{it.key}: {it.repo}@{it.revision[:12]} {it.path} ({it.size / 1e9:.2f} GB) -> {final}'
            + (' (resuming)' if os.path.exists(final + '.part') else ''))
    if dry_run or not todo:
        return bad
    if free_bytes(dest) < need + MARGIN:
        log(f'download: {free_bytes(dest) / 1e9:.1f} GB free under {dest}, {need / 1e9:.1f} GB (+1 GiB) needed: refused')
        return bad + len(todo)
    for it, final in todo:
        try:
            fetch(it, final, endpoint, token, log=log, **kw)
            log(f'{it.key}: done, sha256 verified')
        except DownloadError as e:
            log(f'download: {e}'); bad += 1
    return bad


def load_manifest(path):
    with open(path) as f:
        rows = json.load(f)
    return [models.Item(**r) for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus download', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None, help='the model directory (default: caminus.config model_dir)')
    ap.add_argument('--only', choices=('coder', 'dflash'), default=None)
    ap.add_argument('--verify', action='store_true', help='hash files already present against the manifest')
    ap.add_argument('--dry-run', action='store_true', help='say what would be fetched; fetch nothing')
    ap.add_argument('--manifest', default=None, help=argparse.SUPPRESS)   # (tests / mirrors: a JSON list of caminus.models.Item rows)
    a = ap.parse_args(argv)
    items = load_manifest(a.manifest) if a.manifest else list(models.ITEMS)
    if a.only:
        items = [it for it in items if it.role.startswith(a.only)]
    dest = os.path.abspath(os.path.expanduser(a.dir or config.model_dir()))
    try:
        bad = download(items, dest, verify=a.verify, dry_run=a.dry_run, log=lambda s: print(s, flush=True))
    except KeyboardInterrupt:
        print('download: interrupted; run it again to resume', flush=True)
        return 130
    if not bad and not a.dry_run:
        print(f'download: done in {dest}' + ('' if dest == config.model_dir() else
              f' (not the default model directory: set CAMINUS_MODEL_DIR={dest} or model_dir in {config.config_path()})'), flush=True)
    return 1 if bad else 0
