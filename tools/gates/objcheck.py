"""What a kernel change did to the built code objects (CPU only, standard library only; reads build/*.hsaco, runs nothing).

  python -I tools/gates/objcheck.py snap DIR [-o FILE]            one line per code object: code and data digests, registers
  python -I tools/gates/objcheck.py check BEFORE DIR [--expect PATTERN ...] [--allow-missing] [--allow-sgpr-spill-up]
                                                                 BEFORE: a snap file, or a directory of objects

check fails (exit 1, RESULT FAIL) when
- an object's code or data changed and no --expect pattern names it (an edit leaked into a module it must not touch);
- an --expect pattern matches objects that are in both sets and none of them changed (a stale build, or the wrong
  directory: the comparison would have proved nothing);
- an object is gone (unless --allow-missing);
- the persistent kernel (--kernel, default k3_run) of a changed or new object is over the register budget (--max-vgprs),
  spills a VGPR, or spills more SGPRs than before (unless --allow-sgpr-spill-up, with the measured cost in the pull
  request). Unchanged objects are listed when they are outside the budget, never judged: this change did not build them.

What is compared: the bytes of the sections that hold instructions (code) and of the loaded data sections (data). The
whole file is NOT compared: the symbol and string tables carry names derived from the source path, so two builds of the
same source in two directories differ there and nowhere else. Registers come from the object's AMDGPU metadata note
(the fields llvm-readelf --notes prints). Patterns are shell globs on the object's name without .hsaco."""
import argparse
import fnmatch
import hashlib
import os
import struct
import sys

SHT_PROGBITS, SHT_NOTE = 1, 7
SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
NT_AMDGPU_METADATA = 32


def sections(buf):
    """(name, type, flags, bytes) of each section of a little-endian ELF64 file"""
    if buf[:4] != b'\x7fELF' or buf[4] != 2 or buf[5] != 1: raise ValueError('not a little-endian ELF64 file')
    shoff, = struct.unpack_from('<Q', buf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', buf, 0x3A)
    hdr = [struct.unpack_from('<IIQQQQ', buf, shoff + i * shentsize) for i in range(shnum)]
    stroff, strsize = hdr[shstrndx][4], hdr[shstrndx][5]
    names = buf[stroff:stroff + strsize]
    out = []
    for name, typ, flags, _addr, off, size in hdr:
        out.append((names[name:names.index(b'\0', name)].decode(), typ, flags, buf[off:off + size] if typ != 8 else b''))
    return out


def unpack(b, i=0):
    """a minimal MessagePack reader: (value, next offset)"""
    t = b[i]
    if t <= 0x7f: return t, i + 1
    if t >= 0xe0: return t - 0x100, i + 1
    if 0xa0 <= t <= 0xbf: return b[i + 1:i + 1 + (t & 0x1f)].decode(), i + 1 + (t & 0x1f)
    if 0x80 <= t <= 0x8f: return unpack_map(b, i + 1, t & 0x0f)
    if 0x90 <= t <= 0x9f: return unpack_list(b, i + 1, t & 0x0f)
    if t == 0xc0: return None, i + 1
    if t in (0xc2, 0xc3): return t == 0xc3, i + 1
    fixed = {0xcc: '>B', 0xcd: '>H', 0xce: '>I', 0xcf: '>Q', 0xd0: '>b', 0xd1: '>h', 0xd2: '>i', 0xd3: '>q', 0xca: '>f', 0xcb: '>d'}
    if t in fixed: return struct.unpack_from(fixed[t], b, i + 1)[0], i + 1 + struct.calcsize(fixed[t])
    sized = {0xd9: 1, 0xda: 2, 0xdb: 4, 0xc4: 1, 0xc5: 2, 0xc6: 4, 0xdc: 2, 0xdd: 4, 0xde: 2, 0xdf: 4}
    if t in sized:
        w = sized[t]; n = int.from_bytes(b[i + 1:i + 1 + w], 'big'); j = i + 1 + w
        if t in (0xd9, 0xda, 0xdb): return b[j:j + n].decode(), j + n
        if t in (0xc4, 0xc5, 0xc6): return b[j:j + n], j + n
        return unpack_list(b, j, n) if t in (0xdc, 0xdd) else unpack_map(b, j, n)
    raise ValueError(f'MessagePack type 0x{t:02x} is not handled')


def unpack_list(b, i, n):
    out = []
    for _ in range(n):
        v, i = unpack(b, i); out.append(v)
    return out, i


def unpack_map(b, i, n):
    out = {}
    for _ in range(n):
        k, i = unpack(b, i); v, i = unpack(b, i); out[k] = v
    return out, i


def kernels(note):
    """{kernel name: its metadata map} from a .note section's AMDGPU metadata entry"""
    i = 0
    while i + 12 <= len(note):
        namesz, descsz, typ = struct.unpack_from('<III', note, i)
        name = note[i + 12:i + 12 + namesz].rstrip(b'\0')
        d = i + 12 + (namesz + 3) // 4 * 4
        if name == b'AMDGPU' and typ == NT_AMDGPU_METADATA:
            meta, _ = unpack(note[d:d + descsz])
            return {k.get('.name'): k for k in meta.get('amdhsa.kernels', [])}
        i = d + (descsz + 3) // 4 * 4
    return {}


def describe(path, kernel):
    """(code digest, data digest, registers of `kernel` or None) of one code object"""
    buf = open(path, 'rb').read()
    code, data, regs = hashlib.sha256(), hashlib.sha256(), None
    for name, typ, flags, body in sections(buf):
        if typ == SHT_PROGBITS and flags & SHF_EXECINSTR: code.update(body)
        elif typ == SHT_PROGBITS and flags & SHF_ALLOC: data.update(name.encode() + b'\0' + body)
        elif typ == SHT_NOTE and regs is None:
            k = kernels(body).get(kernel)
            if k: regs = tuple(int(k.get(f, 0)) for f in ('.vgpr_count', '.vgpr_spill_count', '.sgpr_spill_count', '.private_segment_fixed_size'))
    return code.hexdigest(), data.hexdigest(), regs


def snap(d, kernel):
    """{object name: (code, data, regs)} of every *.hsaco in a directory"""
    names = sorted(f for f in os.listdir(d) if f.endswith('.hsaco'))
    if not names: raise SystemExit(f'objcheck: no *.hsaco in {d}')
    objs = {}
    for f in names:
        try: objs[f[:-6]] = describe(os.path.join(d, f), kernel)
        except (ValueError, IndexError, struct.error) as e:
            raise SystemExit(f'objcheck: {os.path.join(d, f)}: not a code object this reads ({e})')
    return objs


def write(objs, out):
    for name, (code, data, regs) in objs.items():
        r = 'vgpr=%d vspill=%d sspill=%d priv=%d' % regs if regs else '-'
        out.write(f'{name}\t{code}\t{data}\t{r}\n')
    whole = hashlib.sha256(''.join(f'{n}{c}{d}' for n, (c, d, _) in objs.items()).encode()).hexdigest()
    out.write(f'# {len(objs)} objects, set digest {whole[:16]}\n')


def read(path):
    objs = {}
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip(): continue
        name, code, data, r = line.rstrip('\n').split('\t')
        regs = None if r == '-' else tuple(int(x.split('=')[1]) for x in r.split())
        objs[name] = (code, data, regs)
    return objs


def check(a):
    before = snap(a.before, a.kernel) if os.path.isdir(a.before) else read(a.before)
    after = snap(a.dir, a.kernel)
    fails = []
    named = lambda n: any(fnmatch.fnmatch(n, p) for p in a.expect)
    changed = [n for n in after if n in before and after[n][:2] != before[n][:2]]
    new = [n for n in after if n not in before]
    gone = [n for n in before if n not in after]
    same = len(after) - len(changed) - len(new)
    print(f'objects: {len(after)} now, {same} with the same code and data sections as before, {len(changed)} changed, {len(new)} new, {len(gone)} gone')
    for n in changed:
        what = ' and '.join(w for w, i in (('code', 0), ('data', 1)) if after[n][i] != before[n][i])
        ok = named(n)
        print(f'  {"PASS" if ok else "FAIL"} changed ({what}): {n}' + ('' if ok else '  (no --expect pattern names it)'))
        if not ok: fails.append(n)
    for n in new: print(f'  new: {n}')
    for n in gone:
        print(f'  {"gone" if a.allow_missing else "FAIL gone"}: {n}')
        if not a.allow_missing: fails.append(n)
    for p in a.expect:
        both = [n for n in after if n in before and fnmatch.fnmatch(n, p)]
        if both and not any(n in changed for n in both):
            print(f'  FAIL --expect {p}: {len(both)} object(s) in both sets, none changed (a stale build or the wrong directory?)')
            fails.append(p)
        if not both and not any(fnmatch.fnmatch(n, p) for n in new):
            print(f'  FAIL --expect {p}: matches no object'); fails.append(p)
    print(f'registers of {a.kernel} (budget {a.max_vgprs} VGPRs, 0 VGPR spills, SGPR spills not up):')
    old = []
    for n, (_, _, regs) in after.items():
        if not regs: continue
        v, vs, ss, priv = regs
        was = before[n][2] if n in before else None
        if n not in changed and n not in new:              # this change did not build it: reported, not judged
            if v > a.max_vgprs or vs: old.append(f'{n} ({v} VGPRs, {vs} spilled)')
            continue
        bad = []
        if v > a.max_vgprs: bad.append(f'{v} VGPRs > {a.max_vgprs}')
        if vs: bad.append(f'{vs} spilled VGPRs')
        if was and ss > was[2] and not a.allow_sgpr_spill_up: bad.append(f'spilled SGPRs {was[2]} -> {ss}')
        moved = was is None or was != regs
        if bad or moved or a.verbose:
            prev = f' (was {was[0]} / {was[1]} / {was[2]})' if was and was != regs else (' (new)' if was is None else '')
            print(f'  {"FAIL" if bad else "PASS"} {n}: {v} VGPRs, {vs} spilled, {ss} spilled SGPRs, private segment {priv} B{prev}' + (': ' + '; '.join(bad) if bad else ''))
        if bad: fails.append(n)
    if old: print(f'  unchanged objects already outside the budget (not judged here): {", ".join(old)}')
    print(f'RESULT {"PASS" if not fails else "FAIL"}')
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snap'); s.add_argument('dir'); s.add_argument('-o', '--out')
    c = sub.add_parser('check'); c.add_argument('before'); c.add_argument('dir')
    c.add_argument('--expect', action='append', default=[], metavar='PATTERN')
    c.add_argument('--allow-missing', action='store_true'); c.add_argument('--allow-sgpr-spill-up', action='store_true')
    c.add_argument('--max-vgprs', type=int, default=96); c.add_argument('-v', '--verbose', action='store_true')
    for p in (s, c): p.add_argument('--kernel', default='k3_run')
    a = ap.parse_args(argv)
    if a.cmd == 'snap':
        objs = snap(a.dir, a.kernel)
        if a.out:
            if os.path.exists(a.out): raise SystemExit(f'objcheck: {a.out} exists; a snapshot goes to a new file')
            with open(a.out, 'w', encoding='utf-8') as f: write(objs, f)
            print(f'{len(objs)} objects -> {a.out}')
        else: write(objs, sys.stdout)
        return 0
    return check(a)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
