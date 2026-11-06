"""caminus purge: empty the prompt cache's disk tier and the flight recorder (CPU only).

  caminus purge [--kv-only | --flight-only] [--older-than-days D] [--kv-dir DIR] [--flight-dir DIR] [--yes]

Without --yes it only lists what would go (the entries of each directory). The directories default to caminus.config's
(~/.cache/caminus/kv and ~/.cache/caminus/flight). Both refuse while a server holds the directory (exit 2): stop the server
first, or purge the flight recorder through it (caminus flight purge --url ...). Flight bundles hold the token ids of your
conversations; the purge removes them."""
import argparse
import os
import sys

from .. import config
from . import flight, kv_cache


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus purge', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--kv-only', action='store_true')
    g.add_argument('--flight-only', action='store_true')
    ap.add_argument('--older-than-days', type=float, default=None)
    ap.add_argument('--kv-dir', default=None)
    ap.add_argument('--flight-dir', default=None)
    ap.add_argument('--yes', action='store_true', help='delete (without it: list only)')
    a = ap.parse_args(argv)
    age = [] if a.older_than_days is None else ['--older-than-days', str(a.older_than_days)]
    rc = 0
    if not a.flight_only:
        d = a.kv_dir or config.kv_dir()
        print(f'== the prompt cache\'s disk tier: {d}', flush=True)
        if not os.path.isdir(d):
            print(f'kv_cache: no directory {d}')
        else:
            rc = max(rc, kv_cache.main(['purge' if a.yes else 'list', '--dir', d] + (age if a.yes else [])))
    if not a.kv_only:
        d = a.flight_dir or config.flight_dir()
        print(f'== the flight recorder: {d}', flush=True)
        rc = max(rc, flight.main(['purge' if a.yes else 'list', '--dir', d] + (age if a.yes else [])))
    if not a.yes:
        print('(nothing deleted: add --yes to purge' + (f' entries older than {a.older_than_days} days' if age else ' everything listed') + ')')
    return rc


if __name__ == '__main__':
    sys.exit(main())
