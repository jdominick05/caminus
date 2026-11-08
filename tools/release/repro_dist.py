"""Make a release's wheel and sdist depend only on their files' contents and SOURCE_DATE_EPOCH, so two builds of the same
commit give the same bytes (tools/release/release_build.sh runs it between the build and SHA256SUMS; docs/dev/releasing.md).

  python -I tools/release/repro_dist.py DIST [--epoch N]       N: default $SOURCE_DATE_EPOCH (one of the two is required)

What the build tools leave machine- or time-dependent, and what is rewritten:
  - the sdist (setuptools' build_sdist): each member's mtime is its file's in the checkout (PAX headers with fractions of a
    second), its uid / gid and user / group names are the build account's, its mode follows the checkout's umask, and the
    gzip header carries the clock's time. Rewritten in the same member order with the same contents: mtime N, uid / gid 0,
    no user or group name, mode 0644 (0755 for directories and files with an executable bit), the PAX format; gzip at level
    9 with mtime N and no file name.
  - the wheel (setuptools' bdist_wheel): its entry times already come from SOURCE_DATE_EPOCH; each entry's mode is its file's
    in the checkout. Rewritten with each mode 0644 / 0755 (by the executable bit); names, order, times, compression and the
    RECORD unchanged (RECORD hashes contents only). A wheel whose modes are already normal comes out byte-identical.
Each archive goes to a temporary name in DIST, is read back and compared with the original (the same names, types and
bytes), then renamed over it. Members other than regular files and directories are refused. Exit 1 on any difference or
refusal, nothing renamed. CPU only."""
import argparse
import gzip
import io
import os
import sys
import tarfile
import zipfile


def norm_mode(mode, isdir=False):
    return 0o755 if isdir or mode & 0o111 else 0o644


def tar_entries(data):
    """[(name, type, mode-class, bytes or None)] of a tar.gz, in member order"""
    out = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        for m in t.getmembers():
            if not (m.isfile() or m.isdir()):
                raise ValueError(f'{m.name}: a member that is not a regular file or a directory (type {m.type!r})')
            out.append((m.name, m.isdir(), norm_mode(m.mode, m.isdir()), None if m.isdir() else t.extractfile(m).read()))
    return out


def repack_sdist(data, epoch):
    raw = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as src, \
            gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=epoch) as gz, \
            tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as out:
        for m in src.getmembers():
            if not (m.isfile() or m.isdir()):
                raise ValueError(f'{m.name}: a member that is not a regular file or a directory (type {m.type!r})')
            n = tarfile.TarInfo(m.name)
            n.type, n.mtime, n.uid, n.gid, n.uname, n.gname = m.type, epoch, 0, 0, '', ''
            n.mode = norm_mode(m.mode, m.isdir())
            if m.isdir():
                out.addfile(n)
            else:
                n.size = m.size
                out.addfile(n, src.extractfile(m))
    return raw.getvalue()


def zip_entries(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return [(i.filename, i.date_time, i.compress_type, norm_mode(i.external_attr >> 16), z.read(i)) for i in z.infolist()]


def repack_wheel(data):
    raw = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(raw, 'w', allowZip64=True) as out:
        for i in src.infolist():
            if i.is_dir():
                raise ValueError(f'{i.filename}: a directory entry (bdist_wheel writes none)')
            n = zipfile.ZipInfo(i.filename, date_time=i.date_time)
            n.compress_type, n.create_system = i.compress_type, i.create_system
            kind = (i.external_attr >> 16) & 0o170000
            n.external_attr = ((kind or 0o100000) | norm_mode(i.external_attr >> 16)) << 16
            out.writestr(n, src.read(i))
    return raw.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dist')
    ap.add_argument('--epoch', type=int, default=None, help='the time every entry gets (default $SOURCE_DATE_EPOCH)')
    a = ap.parse_args(argv)
    epoch = a.epoch
    if epoch is None:
        sde = os.environ.get('SOURCE_DATE_EPOCH', '').strip()
        if not sde.isdigit():
            print(f'repro_dist: set SOURCE_DATE_EPOCH (whole seconds) or pass --epoch (got {sde!r})', file=sys.stderr)
            return 2
        epoch = int(sde)
    names = sorted(os.listdir(a.dist))
    sdists = [n for n in names if n.endswith('.tar.gz')]
    wheels = [n for n in names if n.endswith('.whl')]
    if len(sdists) != 1 or len(wheels) != 1:
        print(f'repro_dist: {a.dist} must hold one .tar.gz and one .whl: {names}', file=sys.stderr)
        return 2
    done = []
    for n, repack, entries in ((sdists[0], lambda d: repack_sdist(d, epoch), tar_entries),
                               (wheels[0], repack_wheel, zip_entries)):
        p = os.path.join(a.dist, n)
        data = open(p, 'rb').read()
        try:
            new = repack(data)
            same = entries(new) == entries(data)
        except (ValueError, tarfile.TarError, zipfile.BadZipFile, OSError) as e:
            print(f'repro_dist: {n}: refused: {e}', file=sys.stderr)
            return 1
        if not same:
            print(f'repro_dist: {n}: the rewritten archive differs from the original (names, types or contents)', file=sys.stderr)
            return 1
        done.append((n, p, new, new == data, 'every member at' if n.endswith('.tar.gz') else 'modes 0644 / 0755, times'))
    for n, p, new, unchanged, what in done:
        tmp = p + '.repro-tmp'
        with open(tmp, 'wb') as f:
            f.write(new)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        when = epoch if n.endswith('.tar.gz') else 'as built (SOURCE_DATE_EPOCH)'
        print(f'repro_dist: {n}: {len(new):,} bytes, {what} {when}{" (unchanged)" if unchanged else ""}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
