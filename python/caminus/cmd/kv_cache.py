"""The prompt cache's disk tier from the command line (caminus.promptcache.DiskStore; CPU only, nothing launched).

  caminus kv-cache list  [--dir DIR]                                     (from a checkout: python -I tools/kv_cache.py ...)
  caminus kv-cache purge [--dir DIR] [--older-than-days D]     (no age: every entry)

DIR defaults to the cache's own (~/.cache/caminus/kv). The store takes its directory's lock: with a server running on it,
both commands refuse with a one-line message and exit 2 (stop the server first: one owner process at a time). Opening the
store already applies its retention (7 days) and removes refused entries (bad header, truncated) and stale temp / spill
files, as the server does at its start."""
import argparse, sys, time


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus kv-cache', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('list', 'purge')); ap.add_argument('--dir', default=None)
    ap.add_argument('--older-than-days', type=float, default=None)
    a = ap.parse_args(argv)
    try:
        from .. import promptcache as pc          # (imports torch: the disk tier's staging buffers are torch tensors, CPU only)
    except ImportError as e:
        if e.name != 'torch' and not (e.name or '').startswith('torch.'):
            raise
        from ..torchcheck import status
        print(f'kv_cache: {a.cmd} refused: {status()[2]}', file=sys.stderr)
        return 2
    try:
        st = pc.DiskStore(a.dir, staging=pc.Staging(piece=pc.PIECE, nbuf=1, pin=False))
    except pc.CacheError as e:
        # (the directory's lock is held: a running caminus serve owns it; a refusal, not a crash: no traceback)
        print(f'kv_cache: {a.cmd} refused: {e}. Stop the server first (one owner process at a time).', file=sys.stderr)
        return 2
    try:
        for what, nm, why in st.events: print(f'{what}: {nm} ({why})')
        if a.cmd == 'list':
            now = time.time()
            for nm, h in sorted(st.index.items(), key=lambda x: -x[1]['_mtime']):
                print(f'{nm}  {h["kind"]:5s}  {h["n"]:7d} tokens  {h["file_bytes"] / 2**20:8.1f} MiB  {len(h.get("ckpts", []))} checkpoints  '
                      f'last used {(now - h["_mtime"]) / 3600:.1f} h ago')
            print(f'{len(st.index)} entries, {st.total_bytes() / 1e9:.2f} GB of the {st.cap / 1e9:.0f} GB cap in {st.root}')
        else:
            out = st.purge(all=a.older_than_days is None, older_than_s=None if a.older_than_days is None else a.older_than_days * 86400)
            print(f'purged {len(out)} entries from {st.root}')
    finally:
        st.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
