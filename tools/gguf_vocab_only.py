"""A vocab-only GGUF: a GGUF's metadata (every key / value, verbatim) and no tensors, for the CPU suites that need only the
tokenizer and the chat template (tests/serve, through CODER). CI fetches the first bytes of the Coder's shard 1 with an HTTP
Range request (the metadata ends near 10.5 MiB of a 29.6 GB file) and runs this on them. CPU only, no network.

  python -I tools/gguf_vocab_only.py IN OUT      IN: a GGUF, or its first bytes (it may end anywhere after the metadata)
  python -I tools/gguf_vocab_only.py --url coder-1   the Hugging Face URL of a manifest entry (caminus.models; HF_ENDPOINT)

OUT holds the same version, the same key / value pairs byte for byte and a tensor count of 0; GGUFReader reads its fields as
it reads the original's (tests/serve/common.py loads the tokenizer from it). Refused: a file that is not GGUF v2 / v3, and an IN
that ends inside the metadata (the message says how many bytes are needed)."""
import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # GGUF value types of fixed size
STRING, ARRAY = 8, 9


class Short(Exception):
    pass


def kv_end(buf):
    """(version, n_kv, offset where the key / value section ends) of a GGUF header in buf; Short when buf ends inside it"""
    def need(o, n):
        if o + n > len(buf):
            raise Short(o + n)

    need(0, 24)
    if buf[:4] != b'GGUF':
        raise ValueError('not a GGUF file (no GGUF magic)')
    version, _n_tensors, n_kv = struct.unpack_from('<IQQ', buf, 4)
    if version not in (2, 3):
        raise ValueError(f'GGUF version {version} (2 or 3 expected)')
    o = 24

    def string(o):
        need(o, 8)
        n = struct.unpack_from('<Q', buf, o)[0]
        need(o + 8, n)
        return o + 8 + n

    def value(o, t):
        if t in SIZES:
            need(o, SIZES[t])
            return o + SIZES[t]
        if t == STRING:
            return string(o)
        if t == ARRAY:
            need(o, 12)
            et, n = struct.unpack_from('<IQ', buf, o)
            o += 12
            if et in SIZES:
                need(o, n * SIZES[et])
                return o + n * SIZES[et]
            for _ in range(n):
                o = value(o, et)
            return o
        raise ValueError(f'unknown GGUF value type {t} at byte {o}')

    for _ in range(n_kv):
        o = string(o)
        need(o, 4)
        t = struct.unpack_from('<I', buf, o)[0]
        o = value(o + 4, t)
    return version, n_kv, o


def vocab_only(src, dst):
    with open(src, 'rb') as f:
        buf = f.read()
    try:
        version, n_kv, end = kv_end(buf)
    except Short as e:
        raise SystemExit(f'gguf_vocab_only: {src} ends at byte {len(buf)}, inside the metadata (at least {e.args[0]} bytes are needed: '
                         f'fetch more)')
    out = b'GGUF' + struct.pack('<IQQ', version, 0, n_kv) + buf[24:end]
    out += b'\0' * (-len(out) % 32)          # (the default alignment; no tensor data follows)
    tmp = dst + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(out)
    os.replace(tmp, dst)
    print(f'gguf_vocab_only: {dst}: {n_kv} keys, {len(out)} bytes (the metadata of {src}, no tensors)')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', metavar='IN OUT')
    ap.add_argument('--url', metavar='KEY', help='print the download URL of a manifest entry (caminus.models) and exit')
    a = ap.parse_args(argv)
    if a.url:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))
        from caminus import models
        from caminus.cmd.download import url_of
        print(url_of(models.BY_KEY[a.url], os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'))
        return 0
    if len(a.files) != 2:
        ap.error('IN and OUT are required')
    return vocab_only(*a.files)


if __name__ == '__main__':
    sys.exit(main())
