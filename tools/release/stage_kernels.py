"""Stage the gfx1100 runtime code objects into the package with their provenance (docs/dev/packaging.md, "Code objects").

  ./build.sh && python -I tools/release/stage_kernels.py [--build build] [--out python/caminus/hsaco/gfx1100] [--allow-dirty]

Copies caminus.device.kernels.RUNTIME_SET from build/ into the package directory (gitignored: build artefacts) and writes
provenance.json: build.sh's build/BUILD-INFO (toolchain name, version directory and compiler, the architecture, the source
commit and dirty flag of kernels/ + build.sh, their content hash, the ggml headers' commit and dirty flag, the build time) and
build/BUILD-CMDS.tsv (each object's compile command as build.sh ran it, paths normalized), plus every staged object's sha256 and
size. `built` is SOURCE_DATE_EPOCH when it is set (UTC; tools/release/release_build.sh sets it to the commit's committer time, so a
rebuild of the same commit writes the same provenance.json), else BUILD-INFO's build time; `built_from` says which. Refused: a kernel tree whose content hash differs from the one recorded at build time (the objects are not from these
sources: rebuild), a dirty kernel tree at build time unless --allow-dirty, a missing object or command, an output directory
holding other code objects (stage into an empty one). pip wheel refuses to build without a complete staged set (setup.py).
CPU only."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.device import kernels   # noqa: E402


def sha256(p):
    return kernels.sha256(p)


def kernel_tree_sha256(root):
    """build.sh's kernel_tree_sha256, recomputed: sha256 of `sha256sum` lines ("<hex>  <path>") of every regular file under
    kernels/ plus build.sh, paths relative to the root, in byte order"""
    paths = ['build.sh']
    for d, _, fs in os.walk(os.path.join(root, 'kernels')):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.isfile(p) and not os.path.islink(p):
                paths.append(os.path.relpath(p, root))
    lines = ''.join(f'{sha256(os.path.join(root, p))}  {p}\n' for p in sorted(paths, key=lambda s: s.encode()))
    return hashlib.sha256(lines.encode()).hexdigest()


def built_time(info):
    """(built, built_from): SOURCE_DATE_EPOCH as an ISO UTC time when set (reproducible builds), else build.sh's own time"""
    sde = os.environ.get('SOURCE_DATE_EPOCH', '').strip()
    if not sde:
        return info.get('built'), 'build.sh'
    if not sde.isdigit():
        raise ValueError(f'SOURCE_DATE_EPOCH must be whole seconds since 1970, not {sde!r}')
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(sde))), 'SOURCE_DATE_EPOCH'


def read_tsv(p):
    out = {}
    with open(p) as f:
        for line in f:
            if line.strip():
                k, v = line.rstrip('\n').split('\t', 1)
                out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build', default=os.path.join(ROOT, 'build'))
    ap.add_argument('--out', default=kernels.PACKAGE_DIR)
    ap.add_argument('--allow-dirty', action='store_true', help='stage objects built from uncommitted kernel sources (recorded)')
    a = ap.parse_args(argv)
    info_p, cmds_p = os.path.join(a.build, 'BUILD-INFO'), os.path.join(a.build, 'BUILD-CMDS.tsv')
    if not (os.path.isfile(info_p) and os.path.isfile(cmds_p)):
        print(f'stage_kernels: {a.build} has no BUILD-INFO / BUILD-CMDS.tsv: run ./build.sh first', file=sys.stderr); return 2
    info, cmds = read_tsv(info_p), read_tsv(cmds_p)
    try:
        built_at, built_from = built_time(info)
    except ValueError as e:
        print(f'stage_kernels: refused: {e}', file=sys.stderr); return 2
    now = kernel_tree_sha256(ROOT)
    if info.get('kernel_tree_sha256') != now:
        print(f'stage_kernels: refused: kernels/ + build.sh hash {now[:12]} now, {info.get("kernel_tree_sha256", "?")[:12]} at the build '
              f'({info.get("built")}): the objects are not from these sources; run ./build.sh', file=sys.stderr); return 2
    dirty = info.get('source_dirty')
    if dirty != '0' and not a.allow_dirty:
        print(f'stage_kernels: refused: the kernel sources were {"dirty" if dirty == "1" else "of unknown state"} at the build '
              f'(commit {info.get("source_commit")}): commit them and rebuild, or --allow-dirty (recorded)', file=sys.stderr); return 2
    if info.get('arch') != kernels.ARCH:
        print(f'stage_kernels: refused: built for {info.get("arch")}, the package ships {kernels.ARCH}', file=sys.stderr); return 2
    names = [n + '.hsaco' for n in kernels.RUNTIME_SET]
    miss = [n for n in names if not os.path.isfile(os.path.join(a.build, n)) or n not in cmds]
    if miss:
        print(f'stage_kernels: refused: missing objects or commands: {miss}', file=sys.stderr); return 2
    os.makedirs(a.out, exist_ok=True)
    other = sorted(f for f in os.listdir(a.out) if f.endswith('.hsaco') and f not in names)
    if other:
        print(f'stage_kernels: refused: {a.out} holds code objects outside the runtime set {other}: stage into an empty directory', file=sys.stderr)
        return 2
    objs = {}
    for n in names:
        src, dst = os.path.join(a.build, n), os.path.join(a.out, n)
        tmp = dst + '.tmp'
        shutil.copyfile(src, tmp); os.replace(tmp, dst)
        h = sha256(dst)
        if h != sha256(src):
            print(f'stage_kernels: copy of {n} differs from {src}', file=sys.stderr); return 1
        objs[n] = dict(sha256=h, bytes=os.path.getsize(dst), command=cmds[n])
    built = sorted(f for f in os.listdir(a.build) if f.endswith('.hsaco'))
    prov = dict(
        format=1, arch=info['arch'], built=built_at, built_from=built_from,
        toolchain=dict(name=info.get('toolchain'), version=info.get('toolchain_version'), compiler=info.get('compiler')),
        source=dict(commit=info.get('source_commit'), dirty=dirty == '1' if dirty in ('0', '1') else None,
                    kernel_tree_sha256=info.get('kernel_tree_sha256'), build_sh_sha256=sha256(os.path.join(ROOT, 'build.sh')),
                    note='kernel_tree_sha256: sha256 over `sha256sum` lines of kernels/** and build.sh (history-independent)'),
        ggml_headers=dict(repo='ignite-llama (https://github.com/jdominick05/ignite-llama: the llama.cpp fork build.sh reads ggml headers from, IGNITE)',
                          commit=info.get('ggml_commit'),
                          dirty=info.get('ggml_dirty') == '1' if info.get('ggml_dirty') in ('0', '1') else None),
        objects=objs, not_shipped=[f for f in built if f not in names])
    tmp = os.path.join(a.out, kernels.PROVENANCE + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(prov, f, indent=1, sort_keys=True); f.write('\n')
    os.replace(tmp, os.path.join(a.out, kernels.PROVENANCE))
    total = sum(o['bytes'] for o in objs.values())
    print(f'stage_kernels: {len(objs)} objects ({total / 2**20:.1f} MiB) -> {a.out}; built {prov["built"]} from {prov["source"]["commit"][:12]}'
          f'{" (dirty)" if prov["source"]["dirty"] else ""}, kernel tree {now[:12]}; not shipped: {len(prov["not_shipped"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
