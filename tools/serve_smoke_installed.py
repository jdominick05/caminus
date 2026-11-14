"""Isolated installed-package entry point for serve_smoke1; no checkout path injection.

The parent invokes this file with python -I, under its outer GPU guard. Only the
test harness lives in the checkout: caminus, its CLI and kernels must come from
the installed distribution. Preflight imports no torch and starts no server.
"""
import argparse
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys


class PreflightError(RuntimeError):
    pass


def blocked_names(value):
    names = tuple(n.strip() for n in value.split(',') if n.strip())
    if value and (not names or any(not n.isidentifier() for n in names)):
        raise PreflightError('--block-imports requires comma-separated top-level package names')
    return names


class ImportBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, names):
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.names:
            print('serve_smoke1: import refused: ' + fullname, flush=True)
            raise ImportError('blocked by serve_smoke1: ' + fullname)


def preflight(expected_tree=None):
    """Refuse source/editable/shadowed imports or non-package/unverified kernels.

    Called again in every child, including a disk-restore restart. The outer
    guard keeps __main__ from execing another process and losing our import hook.
    """
    if not os.environ.get('CAMINUS_GUARD'):
        raise PreflightError('run the smoke under tools/guard.sh (an outer guard is required)')
    if expected_tree is not None and not re.fullmatch('[0-9a-f]{64}', expected_tree):
        raise PreflightError('expected kernel-tree digest must be a lowercase SHA256')
    try:
        import caminus
        dist = importlib.metadata.distribution('caminus')
    except ImportError as exc:
        raise PreflightError('caminus must be installed for this isolated interpreter') from exc
    direct = dist.read_text('direct_url.json')
    if direct:
        try:
            editable = json.loads(direct).get('dir_info', {}).get('editable', False)
        except (ValueError, AttributeError, TypeError) as exc:
            raise PreflightError('invalid installed direct_url.json') from exc
        if editable:
            raise PreflightError('editable installations cannot satisfy an installed-wheel smoke')
    origin = Path(caminus.__file__).resolve()
    recorded = Path(dist.locate_file('caminus/__init__.py')).resolve()
    if origin != recorded:
        raise PreflightError('imported caminus does not match its installed distribution')
    from caminus.device import kernels
    if kernels.checkout_build() is not None:
        raise PreflightError('caminus imports from a source checkout')
    kdir, where = kernels.kernel_dir()
    if where != 'package' or Path(kdir).resolve() != Path(kernels.PACKAGE_DIR).resolve():
        raise PreflightError('installed smoke requires packaged kernels; remove kernel_dir overrides from environment/config')
    prov, checks = kernels.verify(kdir)
    if prov is None or not checks or any(not ok for _, ok, _ in checks):
        raise PreflightError('packaged kernel provenance is missing or fails verification')
    manifest = set(prov.get('objects', {}))
    physical = {p.name for p in Path(kdir).glob('*.hsaco')}
    runtime = {name + '.hsaco' for name in kernels.RUNTIME_SET}
    if manifest != runtime or physical != manifest:
        raise PreflightError('packaged kernel files, provenance and runtime set must match exactly (no unlisted objects)')
    tree = prov.get('source', {}).get('kernel_tree_sha256')
    if expected_tree is not None and tree != expected_tree:
        raise PreflightError('installed kernels do not match the checkout sources used by the own-code include audit')
    print(f'serve_smoke1: installed caminus {dist.version} from {origin}; '
          f'{len(checks)} packaged kernel objects verified, kernel tree {tree}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--block-imports', default='')
    ap.add_argument('--kernel-tree-sha256')
    ap.add_argument('server_args', nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    server_args = args.server_args
    if server_args[:1] == ['--']:
        server_args = server_args[1:]
    try:
        names = blocked_names(args.block_imports)
        if names and args.kernel_tree_sha256 is None:
            raise PreflightError('the own-code import gate requires the checkout kernel-tree digest')
        if names:
            loaded = sorted(n for n in sys.modules if n.split('.')[0] in names)
            if loaded:
                raise PreflightError('blocked packages were already imported before the smoke hook: ' + ', '.join(loaded))
            sys.meta_path.insert(0, ImportBlocker(names))
        preflight(args.kernel_tree_sha256)
    except (PreflightError, OSError, ValueError) as exc:
        print(f'serve_smoke1: installed preflight refused: {exc}', file=sys.stderr, flush=True)
        return 2
    from caminus.__main__ import main as serve
    return serve(['serve', *server_args])


if __name__ == '__main__':
    raise SystemExit(main())
