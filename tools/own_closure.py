"""The include closure of a code object's source and the ggml / llama.cpp headers in it: the own-code gate's detector, shared
by tests/test_own_code.py (the objects named there) and tools/serve_smoke1.py --block-imports (the code objects a running
server reports in /props caminus.code_objects); and the reader of kernels/objects.tsv (the objects build.sh compiles from
another source with defines)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGML = re.compile(r'(common|vecdotq|quantize|mmq|mmvq|convert|ggml[\w-]*)\.(cuh|h)$')
TABLE = os.path.join(ROOT, 'kernels', 'objects.tsv')


def rows():
    """kernels/objects.tsv as {object: (source, [(NAME, VALUE), ...])} (VALUE '' for an empty define), as build.sh reads it"""
    out = {}
    for ln in open(TABLE):
        if not ln.strip() or ln.startswith('#'): continue
        obj, src, defs = ln.rstrip('\n').split('\t')
        out[obj] = (src, [tuple(d.split('=', 1)) if '=' in d else (d, '') for d in defs.split(';')])
    return out


def source(obj):
    """(the kernels/ file build.sh compiles for code object obj, the defines it adds in build.sh's order: the row's, then
    its SOURCE row's): a table row's chain, else kernels/<obj>.hip; a _w4 object is its base object's with K3_PER_WGP=4"""
    table, defs, w4 = rows(), [], obj.endswith('_w4')
    if w4: obj = obj[:-len('_w4')]
    while obj in table:
        obj, d = table[obj]; defs += d
    return obj + '.hip', defs + [('K3_PER_WGP', '4')] * w4


def closure(name, seen=None, macros=None):
    """the kernels/ files a kernel file includes, transitively: #include "x", and #include NAME with NAME #defined as a
    "file" earlier on the path (dec_core.hip's K3_EXTRA_OPS_HEADER); <system> headers are the toolchain's"""
    seen = set() if seen is None else seen
    macros = {} if macros is None else macros
    p = os.path.join(ROOT, 'kernels', name)
    if name in seen: return seen
    seen.add(name)
    if not os.path.isfile(p): return seen
    for m in re.finditer(r'^\s*#\s*(?:define\s+(\w+)\s+"([^"]+)"|include\s+"([^"]+)"|include\s+(\w+))', open(p).read(), re.M):
        if m.group(1): macros[m.group(1)] = m.group(2)
        elif m.group(3): closure(m.group(3), seen, macros)
        elif m.group(4) in macros: closure(macros[m.group(4)], seen, macros)
        else: seen.add('?' + m.group(4))     # (an include through an undefined macro: reported as missing)
    return seen


def audit(obj):
    """(closure, the names in it that are not files of kernels/, the ggml / llama.cpp headers in it) of code object obj's
    source (source above)"""
    cl = closure(source(obj)[0])
    missing = sorted(f for f in cl if not os.path.isfile(os.path.join(ROOT, 'kernels', f)))
    bad = sorted(f for f in cl if GGML.search(f) or 'ggml' in f)
    return cl, missing, bad
