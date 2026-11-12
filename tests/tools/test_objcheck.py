"""tools/gates/objcheck.py on small code objects made here (CPU, no toolchain): which objects a change touched, and the
persistent kernel's registers, read from the files alone.

  python -I tests/tools/test_objcheck.py --scratch <a new directory>

The objects are little-endian ELF64 files with the sections objcheck reads: instructions (.text), loaded data (.rodata),
the AMDGPU metadata note (MessagePack, the k3_run entry's register counts) and a string table that stands for the names
a build derives from its source path.
Cases: a snapshot's lines and its refusal to overwrite; the same sources built in another directory (string table
differs, whole files differ) count as unchanged; a changed object that no --expect names fails and a named one passes;
a named object that did not change fails (a stale build); a data-only change; a missing object and --allow-missing; an
--expect that matches nothing; a changed kernel over the VGPR budget, with a VGPR spill, with more SGPR spills (and
--allow-sgpr-spill-up); an unchanged object outside the budget listed but not judged; a new object over the budget; a
snapshot file as BEFORE gives the directory's result; a file that is not ELF is refused by name."""
import argparse
import hashlib
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
OBJCHECK = os.path.join(ROOT, 'tools', 'gates', 'objcheck.py')

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
S = os.path.join(A.scratch, 'objcheck')
if os.path.exists(S):
    raise SystemExit(f'test_objcheck: {S} exists (a new --scratch directory)')
os.makedirs(S)
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def pack(v):
    """a minimal MessagePack writer (what the metadata note needs)"""
    if isinstance(v, int):
        return bytes([v]) if v < 128 else b'\xcc' + bytes([v]) if v < 256 else b'\xcd' + struct.pack('>H', v)
    if isinstance(v, str): return bytes([0xa0 | len(v)]) + v.encode()
    if isinstance(v, list): return bytes([0x90 | len(v)]) + b''.join(pack(x) for x in v)
    return bytes([0x80 | len(v)]) + b''.join(pack(k) + pack(x) for k, x in v.items())


def note(regs):
    v, vs, ss, priv = regs
    desc = pack({'amdhsa.kernels': [{'.name': 'other', '.vgpr_count': 3},
                                    {'.name': 'k3_run', '.vgpr_count': v, '.vgpr_spill_count': vs, '.sgpr_spill_count': ss,
                                     '.private_segment_fixed_size': priv}], 'amdhsa.version': [1, 2]})
    return struct.pack('<III', 7, len(desc), 32) + b'AMDGPU\0\0' + desc + bytes(-len(desc) % 4)


def elf(code, data=b'\x01\x02', regs=None, strings=b'\0src/k.hip\0'):
    secs = [('', 0, 0, b''), ('.text', 1, 0x6, code), ('.rodata', 1, 0x2, data)]
    if regs: secs.append(('.note', 7, 0x2, note(regs)))
    secs += [('.dynstr', 3, 0x2, strings), ('.shstrtab', 3, 0, b'')]
    names, offs = b'\0', []
    for n, *_ in secs:
        offs.append(len(names) if n else 0)
        if n: names += n.encode() + b'\0'
    secs[-1] = ('.shstrtab', 3, 0, names)
    body, hdrs = b'', b''
    for (_, typ, flags, b), no in zip(secs, offs):
        hdrs += struct.pack('<IIQQQQIIQQ', no, typ, flags, 0, 64 + len(body), len(b), 0, 0, 1, 0)
        body += b
    head = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8) + struct.pack('<HHIQQQIHHHHHH', 3, 224, 1, 0, 0, 64 + len(body), 0, 64, 0, 0, 64, len(secs), len(secs) - 1)
    return head + body + hdrs


def build(name, objs):
    d = os.path.join(S, name); os.makedirs(d)
    for n, b in objs.items():
        with open(os.path.join(d, n + '.hsaco'), 'wb') as f: f.write(b)
    return d


def run(*a):
    r = subprocess.run([sys.executable, '-I', OBJCHECK, *a], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


BASE = {'k3_a': elf(b'A' * 40, regs=(90, 0, 80, 0)), 'k3_b': elf(b'B' * 40, regs=(94, 0, 86, 0)), 'attn': elf(b'C' * 24),
        'k3_old': elf(b'D' * 40, regs=(109, 0, 0, 0))}
base = build('base', BASE)
with_b = lambda **kw: dict(BASE, k3_b=elf(kw.pop('code', b'b' * 44), regs=kw.pop('regs', (94, 0, 86, 0)), **kw))

# ---- snap
snapf = os.path.join(S, 'before.txt')
rc, out = run('snap', base, '-o', snapf)
lines = open(snapf).read().splitlines() if os.path.exists(snapf) else []
check(rc == 0 and len(lines) == 5 and lines[-1].startswith('# 4 objects, set digest '), f'snap: one line per object and the set line ({out.strip()})')
row = {l.split('\t')[0]: l.split('\t') for l in lines[:-1]}
check(row.get('k3_b', [''])[-1] == 'vgpr=94 vspill=0 sspill=86 priv=0' and row.get('attn', [''])[-1] == '-',
      'snap: k3_run\'s registers from the metadata note; "-" for an object without that kernel')
check(row.get('k3_b', ['', ''])[1] == hashlib.sha256(b'B' * 40).hexdigest(), 'snap: the code digest is the sha256 of the instruction sections')
rc, out = run('snap', base, '-o', snapf)
check(rc != 0 and 'exists' in out, 'snap: an existing snapshot file is not overwritten')

# ---- the same sources in another directory
moved = build('moved', {n: elf(b[64:64 + (40 if n != 'attn' else 24)], regs={'k3_a': (90, 0, 80, 0), 'k3_b': (94, 0, 86, 0), 'k3_old': (109, 0, 0, 0)}.get(n),
                               strings=b'\0another/place/k.hip\0') for n, b in BASE.items()})
rc, out = run('check', base, moved)
check(open(os.path.join(base, 'k3_a.hsaco'), 'rb').read() != open(os.path.join(moved, 'k3_a.hsaco'), 'rb').read()
      and rc == 0 and '4 with the same code and data sections as before, 0 changed' in out and out.strip().endswith('RESULT PASS'),
      'check: objects that differ only in their string table are unchanged')
check('not judged' in out and 'k3_old (109 VGPRs' in out, 'check: an unchanged object outside the budget is listed, not judged')

# ---- a change, named or not
edit = build('edit', with_b())
rc, out = run('check', base, edit)
check(rc == 1 and 'FAIL changed (code): k3_b' in out and 'no --expect pattern names it' in out and out.strip().endswith('RESULT FAIL'),
      'check: a changed object that no --expect names fails')
rc, out = run('check', base, edit, '--expect', 'k3_b')
check(rc == 0 and 'PASS changed (code): k3_b' in out and out.strip().endswith('RESULT PASS'), 'check: the same change passes when named')
rc2, out2 = run('check', snapf, edit, '--expect', 'k3_b')
check((rc2, out2) == (rc, out), 'check: a snapshot file as BEFORE gives the directory\'s result')
rc, out = run('check', base, edit, '--expect', 'k3_b', '--expect', 'k3_a')
check(rc == 1 and 'FAIL --expect k3_a: 1 object(s) in both sets, none changed' in out, 'check: a named object that did not change fails (a stale build)')
rc, out = run('check', base, base, '--expect', 'k3_*')
check(rc == 1 and 'none changed' in out, 'check: the same directory on both sides with a change expected fails')
rc, out = run('check', base, edit, '--expect', 'k3_b', '--expect', 'nomatch*')
check(rc == 1 and 'FAIL --expect nomatch*: matches no object' in out, 'check: an --expect that matches no object fails')
rc, out = run('check', base, build('data', with_b(code=b'B' * 40, data=b'\x09\x09')), '--expect', 'k3_b')
check(rc == 0 and 'PASS changed (data): k3_b' in out, 'check: a data-only change is a change')

# ---- a missing object
less = build('less', {n: b for n, b in BASE.items() if n != 'attn'})
rc, out = run('check', base, less)
check(rc == 1 and 'FAIL gone: attn' in out, 'check: a missing object fails')
rc, out = run('check', base, less, '--allow-missing')
check(rc == 0 and 'gone: attn' in out and 'FAIL' not in out, 'check: --allow-missing accepts it')

# ---- the changed kernel's registers
rc, out = run('check', base, build('wide', with_b(regs=(97, 0, 86, 0))), '--expect', 'k3_b')
check(rc == 1 and 'FAIL k3_b: 97 VGPRs' in out and '97 VGPRs > 96' in out, 'check: a changed kernel over the VGPR budget fails')
rc, out = run('check', base, build('wide2', with_b(regs=(97, 0, 86, 0))), '--expect', 'k3_b', '--max-vgprs', '100')
check(rc == 0 and 'PASS k3_b: 97 VGPRs' in out and '(was 94 / 0 / 86)' in out, 'check: --max-vgprs moves the budget; the earlier counts are shown')
rc, out = run('check', base, build('spill', with_b(regs=(94, 1, 86, 0))), '--expect', 'k3_b')
check(rc == 1 and '1 spilled VGPRs' in out, 'check: a VGPR spill fails')
up = build('sgpr', with_b(regs=(94, 0, 88, 0)))
rc, out = run('check', base, up, '--expect', 'k3_b')
check(rc == 1 and 'spilled SGPRs 86 -> 88' in out, 'check: more spilled SGPRs than before fails')
rc, out = run('check', base, up, '--expect', 'k3_b', '--allow-sgpr-spill-up')
check(rc == 0 and 'PASS k3_b: 94 VGPRs, 0 spilled, 88 spilled SGPRs' in out, 'check: --allow-sgpr-spill-up accepts it')
rc, out = run('check', base, build('new', dict(BASE, k3_new=elf(b'N' * 40, regs=(100, 0, 0, 0)))))
check(rc == 1 and 'new: k3_new' in out and 'FAIL k3_new: 100 VGPRs' in out, 'check: a new object over the budget fails')
rc, out = run('check', base, build('new2', dict(BASE, k3_new=elf(b'N' * 40, regs=(60, 0, 0, 0)))), '--expect', 'k3_new')
check(rc == 0 and 'PASS k3_new: 60 VGPRs' in out and '(new)' in out, 'check: a new object inside the budget passes, named or not')

# ---- not a code object
junk = build('junk', {'k3_a': b'not an object file'})
rc, out = run('snap', junk)
check(rc != 0 and 'k3_a.hsaco' in out and 'Traceback' not in out, 'snap: a file that is not ELF64 is refused by name')
rc, out = run('snap', os.path.join(S, 'data'), '--kernel', 'other')
check(rc == 0 and 'vgpr=3 vspill=0 sspill=0 priv=0' in out, 'snap: --kernel reads another kernel\'s entry')

print('RESULT', 'FAIL' if FAILS else 'PASS')
sys.exit(1 if FAILS else 0)
