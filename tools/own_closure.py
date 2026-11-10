"""The include closure of a kernel source and the ggml / llama.cpp headers in it: the own-code gate's detector, shared by
tests/test_own_code.py (the sources named there) and tools/serve_smoke1.py --block-imports (the code objects a running
server reports in /props caminus.code_objects)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGML = re.compile(r'(common|vecdotq|quantize|mmq|mmvq|convert|ggml[\w-]*)\.(cuh|h)$')


def closure(name, seen=None, macros=None):
    """the kernels/ files a kernel file includes, transitively: #include "x", and #include NAME with NAME #defined as a
    "file" earlier on the path (k3.hip's K3_EXTRA_OPS_HEADER); <system> headers are the toolchain's"""
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


def audit(top):
    """(closure, the names in it that are not files of kernels/, the ggml / llama.cpp headers in it) of kernels/<top>"""
    cl = closure(top)
    missing = sorted(f for f in cl if not os.path.isfile(os.path.join(ROOT, 'kernels', f)))
    bad = sorted(f for f in cl if GGML.search(f) or 'ggml' in f)
    return cl, missing, bad
