"""The flight recorder from the command line (M7 ops; CPU only, no GPU): list / purge the flight directory, or ask a running
server to save a replay bundle.

  caminus flight list  [--dir DIR]                                     (from a checkout: python -I tools/flight.py ...)                          the JSONL days and the bundles (name, bytes, age)
  caminus flight purge [--dir DIR] [--older-than-days D]    delete them (all by default); refused while a server
                                                                       holds the directory (use --url: the server purges)
  caminus flight purge --url http://127.0.0.1:8092 [--older-than-days D]
  caminus flight bundle [--url http://127.0.0.1:8092] [--request ID]
                                                                       a replay bundle of the most recent request (or ID, one of
                                                                       the last 8), saved by the server on its engine thread
DIR defaults to ~/.cache/caminus/flight (the server's default; tests pass a scratch directory). Replay a bundle with
`caminus replay BUNDLE` (GPU, under the guard; from a checkout tools/replay.py)."""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from ..serve import flight as F   # noqa: E402


def post(url, path, body):
    req = urllib.request.Request(url.rstrip('/') + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            return r.status, json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus flight', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('list', 'purge', 'bundle'))
    ap.add_argument('--dir', default=F.DEFAULT_DIR)
    ap.add_argument('--url', default=None)
    ap.add_argument('--older-than-days', type=float, default=None)
    ap.add_argument('--request', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'bundle' or (a.cmd == 'purge' and a.url):
        url = a.url or 'http://127.0.0.1:8092'
        if a.cmd == 'bundle':
            st, body = post(url, '/caminus/flight/bundle', {} if a.request is None else {'request_id': a.request})
        else:
            st, body = post(url, '/caminus/flight/purge', {} if a.older_than_days is None else {'older_than_days': a.older_than_days})
        print(json.dumps(body, indent=1))
        return 0 if st == 200 else 1
    root = os.path.abspath(os.path.expanduser(a.dir))
    if not os.path.isdir(root):
        print(f'flight: no directory {root}')
        return 0
    if a.cmd == 'purge' and F.held(root):
        print(f'flight: {root} is held by a running server: purge through it (--url http://127.0.0.1:8092)', file=sys.stderr)
        return 2
    fr = F.FlightRecorder(root, cap_gb=1e6, retention_days=1e6, lock=(a.cmd == 'purge'))   # (no eviction from a listing; list beside a server)
    try:
        if a.cmd == 'list':
            now = time.time()
            ents = fr.entries()
            for kind, p, nb, mt in ents:
                print(f'{kind:6s} {nb / 1e6:10.1f} MB  {(now - mt) / 86400:6.2f} d  {os.path.basename(p)}')
            print(f'{len(ents)} entries, {sum(e[2] for e in ents) / 1e9:.3f} GB in {root}')
        else:
            out = fr.purge(all=a.older_than_days is None, older_than_days=a.older_than_days)
            print(f'flight: removed {len(out)}: ' + ' '.join(out))
    finally:
        fr.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
