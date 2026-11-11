"""Check a release's (or a nightly's) wheel and sdist before they go to PyPI (.github/workflows/release.yml and nightly.yml;
tools/release/release_build.sh runs it too). CPU only, no network. Run it in a checkout of the release tag:

  python -I tools/release/verify_release.py DIST --tag vX.Y.Z      DIST: caminus-X.Y.Z-py3-none-any.whl, caminus-X.Y.Z.tar.gz, SHA256SUMS
  python -I tools/release/verify_release.py DIST --tag vX.Y.Z.devYYYYMMDD   a nightly's files at its tag (docs/dev/releasing.md)
  python -I tools/release/verify_release.py DIST --untagged        a rehearsal at an untagged commit (the version from the tree)

Checked: the tag's form and that it names HEAD; DIST holds exactly the two files and SHA256SUMS, whose hashes match; one version
everywhere (the tag, the file names, the wheel's METADATA, the sdist's PKG-INFO, python/caminus/__init__.py); the wheel's tag
(py3-none-any) and its RECORD hashes; every file of the package in the wheel is the checkout's file byte for byte
(only the code objects, their provenance.json and the panel are built); its code objects are exactly caminus.device.kernels.RUNTIME_SET, each one's sha256 and size equal
provenance.json's and the committed packaging/gfx1100-kernels.sha256's; provenance.json names this commit, a clean tree, gfx1100,
build.sh's sha256 and the kernel tree hash recomputed here (tools/release/stage_kernels.py's rule); the sdist carries the same code
objects and provenance.json byte for byte; no text file in either archive matches tools/release/privacy_scan.py; the archives are in
tools/release/repro_dist.py's reproducible form (every sdist member at one time, owner 0:0 with no user or group name, modes 0644 /
0755, a gzip header without a file name at that time; every wheel entry at one time with modes 0644 / 0755; that time ==
provenance.json's `built`). Exit 1 on any failure, every failure printed."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, HERE)
from caminus.device import kernels  # noqa: E402
import privacy_scan  # noqa: E402
import stage_kernels  # noqa: E402

HSACO = 'caminus/hsaco/gfx1100/'
PANEL = 'caminus/share/caminus-panel'
TAG_RE = re.compile(r'v(\d+\.\d+\.\d+(?:\.dev\d{8})?)')   # a release, or a nightly (tools/release/nightly_plan.py)
FAILS = []


def check(what, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"} {what}' + (f': {detail}' if detail and not ok else ''), flush=True)
    if not ok:
        FAILS.append(what)
    return ok


def sha(b):
    return hashlib.sha256(b).hexdigest()


def tree_version():
    txt = open(os.path.join(ROOT, 'python', 'caminus', '__init__.py')).read()
    m = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", txt, re.M)
    return m.group(1) if m else None


def git(*a):
    return subprocess.run(['git', '-C', ROOT, *a], capture_output=True, text=True)


def committed_hashes():
    p = os.path.join(ROOT, 'packaging', 'gfx1100-kernels.sha256')
    out = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                h, n = line.split()
                out[n] = h
    return out


def scan_text(label, name, data):
    if b'\0' in data[:8192]:
        return
    pats = [re.compile(p) for ps in privacy_scan.PATTERNS.values() for p in ps]
    for i, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
        if any(p.search(line) for p in pats) or privacy_scan.private_hits(line):
            check(f'{label}: no private or secret text', False, f'{name}:{i}')
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dist')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--tag')
    g.add_argument('--untagged', action='store_true')
    a = ap.parse_args(argv)

    head = git('rev-parse', 'HEAD').stdout.strip()
    check('a git checkout (HEAD resolves)', bool(re.fullmatch(r'[0-9a-f]{40}', head)), head)
    tv = tree_version()
    if a.tag:
        m = TAG_RE.fullmatch(a.tag)
        if not check(f'the tag {a.tag!r} has the form vX.Y.Z or vX.Y.Z.devYYYYMMDD', bool(m)):
            return 1
        ver = m.group(1)
        tc = git('rev-parse', '--verify', '--quiet', f'refs/tags/{a.tag}^{{commit}}').stdout.strip()
        check(f'the tag {a.tag} names HEAD', tc == head, f'tag {tc or "(not found)"} HEAD {head}')
    else:
        ver = tv
        print(f'NOTE untagged rehearsal at {head[:12]}: the version from the tree ({ver})', flush=True)
    check('python/caminus/__init__.py has the release version', tv == ver, f'{tv} != {ver}')
    status = git('status', '--porcelain', '--untracked-files=no').stdout.strip()
    check('the checkout has no changes to tracked files', not status, status[:300])

    whl_n, sd_n = f'caminus-{ver}-py3-none-any.whl', f'caminus-{ver}.tar.gz'
    names = sorted(os.listdir(a.dist))
    if not check('DIST holds the wheel, the sdist and SHA256SUMS, nothing else', names == sorted([whl_n, sd_n, 'SHA256SUMS']), str(names)):
        return 1
    blobs = {n: open(os.path.join(a.dist, n), 'rb').read() for n in (whl_n, sd_n)}
    sums = {}
    for line in open(os.path.join(a.dist, 'SHA256SUMS')).read().splitlines():
        if line.strip():
            h, n = line.split(maxsplit=1)
            sums[n.lstrip('*')] = h
    check('SHA256SUMS lists exactly the two files with their sha256', sums == {n: sha(b) for n, b in blobs.items()}, str(sums))

    # ---- the wheel
    zf = zipfile.ZipFile(io.BytesIO(blobs[whl_n]))
    files = {i.filename: zf.read(i) for i in zf.infolist() if not i.is_dir()}
    di = f'caminus-{ver}.dist-info/'
    meta = files.get(di + 'METADATA', b'').decode()
    check('wheel METADATA: Name caminus, the release version', re.search(r'^Name: caminus$', meta, re.M) is not None
          and re.search(rf'^Version: {re.escape(ver)}$', meta, re.M) is not None, meta[:200])
    check('wheel tag py3-none-any', re.search(r'^Tag: py3-none-any$', files.get(di + 'WHEEL', b'').decode(), re.M) is not None)
    bad = []
    for line in files.get(di + 'RECORD', b'').decode().splitlines():
        p, h, _n = line.rsplit(',', 2)
        if not h:
            continue
        algo, val = h.split('=', 1)
        got = base64.urlsafe_b64encode(hashlib.new(algo, files.get(p, b'')).digest()).rstrip(b'=').decode()
        if p not in files or got != val:
            bad.append(p)
    check('wheel RECORD: every file\'s hash', not bad and di + 'RECORD' in files, str(bad[:5]))
    tracked = set(git('ls-files', '-z', 'python/caminus').stdout.split('\0'))
    differ = []
    for p, b in sorted(files.items()):
        if p.startswith(di) or p.startswith(HSACO) or p == PANEL:
            continue
        src = os.path.join(ROOT, 'python', p)
        if 'python/' + p not in tracked or not os.path.isfile(src):
            differ.append(p + ' (not a tracked file)')
        elif open(src, 'rb').read() != b:
            differ.append(p)
    check('wheel: every package file is the checkout\'s tracked file, byte for byte (the code objects and the panel are the built ones)',
          not differ, str(differ[:5]))
    objs = {p[len(HSACO):]: b for p, b in files.items() if p.startswith(HSACO) and p.endswith('.hsaco')}
    want = sorted(n + '.hsaco' for n in kernels.RUNTIME_SET)
    check(f'wheel code objects == the runtime set ({len(want)})', sorted(objs) == want,
          f'missing {sorted(set(want) - set(objs))} extra {sorted(set(objs) - set(want))}')
    prov_b = files.get(HSACO + 'provenance.json')
    prov = json.loads(prov_b) if prov_b else {}
    check('wheel provenance.json present', bool(prov))
    po = prov.get('objects', {})
    check('each object\'s sha256 and size == provenance.json', all(n in po and po[n]['sha256'] == sha(b) and po[n]['bytes'] == len(b)
                                                                  for n, b in objs.items()) and sorted(po) == want)
    com = committed_hashes()
    check('each object\'s sha256 == packaging/gfx1100-kernels.sha256 (committed at this commit)',
          com == {n: sha(b) for n, b in objs.items()}, f'differ: {sorted(n for n in set(com) | set(objs) if com.get(n) != (sha(objs[n]) if n in objs else None))}')
    src = prov.get('source', {})
    check('provenance: built from this commit, a clean tree', src.get('commit') == head and src.get('dirty') is False,
          f'commit {src.get("commit")} dirty {src.get("dirty")}')
    check('provenance: arch gfx1100', prov.get('arch') == kernels.ARCH, str(prov.get('arch')))
    kt = stage_kernels.kernel_tree_sha256(ROOT)
    check('provenance: kernel_tree_sha256 == kernels/ + build.sh here', src.get('kernel_tree_sha256') == kt, f'{src.get("kernel_tree_sha256")} != {kt}')
    check('provenance: build_sh_sha256 == build.sh here', src.get('build_sh_sha256') == stage_kernels.sha256(os.path.join(ROOT, 'build.sh')))
    check('wheel carries the guard', 'caminus/share/guard.sh' in files)
    check('wheel carries the panel', 'caminus/share/caminus-panel' in files)
    for p, b in files.items():
        scan_text('wheel', p, b)

    # ---- the sdist
    tf = tarfile.open(fileobj=io.BytesIO(blobs[sd_n]), mode='r:gz')
    top = f'caminus-{ver}/'
    members = {m.name: m for m in tf.getmembers() if m.isfile()}
    check('sdist: every file under caminus-X.Y.Z/', all(n.startswith(top) for n in members), str([n for n in members if not n.startswith(top)][:3]))
    pkg = tf.extractfile(members[top + 'PKG-INFO']).read().decode() if top + 'PKG-INFO' in members else ''
    check('sdist PKG-INFO: the release version', re.search(rf'^Version: {re.escape(ver)}$', pkg, re.M) is not None, pkg[:200])
    for need in ('setup.py', 'pyproject.toml', 'LICENSE', 'THIRD_PARTY.md', 'python/caminus/share/guard.sh'):
        check(f'sdist carries {need}', top + need in members)
    sd_objs = {n[len(top + 'python/' + HSACO):]: tf.extractfile(m).read() for n, m in members.items()
               if n.startswith(top + 'python/' + HSACO) and n.endswith('.hsaco')}
    check('sdist code objects == the wheel\'s, byte for byte', sd_objs == objs, f'{len(sd_objs)} vs {len(objs)}')
    sp = top + 'python/' + HSACO + 'provenance.json'
    check('sdist provenance.json == the wheel\'s', sp in members and tf.extractfile(members[sp]).read() == prov_b)
    for n, m in members.items():
        scan_text('sdist', n, tf.extractfile(m).read())

    # ---- the reproducible form (tools/release/repro_dist.py; SOURCE_DATE_EPOCH)
    allm = tf.getmembers()
    times = {m.mtime for m in allm}
    epoch = next(iter(times)) if len(times) == 1 else None
    check('sdist: every member at one time (SOURCE_DATE_EPOCH), owner 0:0 with no user or group name, modes 0644 / 0755',
          epoch is not None and isinstance(epoch, int) and all(m.uid == 0 and m.gid == 0 and m.uname == '' and m.gname == ''
                                                               and m.mode in (0o644, 0o755) for m in allm),
          f'times {sorted(times)[:3]}, owners {sorted({(m.uid, m.gid, m.uname, m.gname) for m in allm})[:3]}, '
          f'modes {sorted({oct(m.mode) for m in allm})}')
    gz = blobs[sd_n][:10]
    check('sdist: the gzip header has no file name and the members\' time',
          gz[:3] == b'\x1f\x8b\x08' and not gz[3] & 0x08 and int.from_bytes(gz[4:8], 'little') == epoch, gz.hex())
    ztimes = {i.date_time for i in zf.infolist()}
    zmodes = {(i.external_attr >> 16) & 0o7777 for i in zf.infolist()}
    want_zt = time.gmtime(epoch)[:6] if epoch is not None else None
    check('wheel: every entry at that time (2-second resolution), modes 0644 / 0755',
          want_zt is not None and ztimes == {want_zt[:5] + (want_zt[5] // 2 * 2,)} and zmodes <= {0o644, 0o755},
          f'times {sorted(ztimes)[:3]} want {want_zt}, modes {sorted(oct(m) for m in zmodes)}')
    check('provenance: built == that time (SOURCE_DATE_EPOCH)',
          epoch is not None and prov.get('built') == time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))
          and prov.get('built_from') == 'SOURCE_DATE_EPOCH', f'built {prov.get("built")} from {prov.get("built_from")}, epoch {epoch}')

    print(f'verify_release: {whl_n} + {sd_n} at {head[:12]}: ' + ('RESULT PASS' if not FAILS else f'RESULT FAIL ({len(FAILS)}: {FAILS})'))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
