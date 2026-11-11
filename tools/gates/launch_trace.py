"""Which GPU kernels a served request launches: the own-code launch gate (a ROCm kernel trace, rocprofv3 --kernel-trace).

  tools/guard.sh LOG 300  python -I tools/gates/launch_trace.py probe --scratch DIR [--card 1]
  tools/guard.sh LOG 2400 python -I tools/gates/launch_trace.py serve --scratch DIR --model GGUF [--card 1] [--max-ctx 8192]
                                                                    [--root CHECKOUT]

probe: the instrument first. A child launches the `smoke` object's saxpy between two readings of CLOCK_MONOTONIC and
CLOCK_BOOTTIME, under rocprofv3; the trace must hold that kernel, and its timestamps tell which clock the trace uses
(written to DIR/clock.txt for `serve`). Exit 0 only if the trace names the kernel inside one clock's bracket.

serve: tools/caminus_serve.py on one card under rocprofv3. After the server is ready and one warm-up request has run, the
window opens; inside it a greedy chat stream and a sampled one (temperature 0.7, top_k 20: the device sampler's path) run;
then the window closes and the server stops. Every kernel that started inside the window is sorted into:
  own      a kernel of a code object Caminus loaded (the server's /props caminus.code_objects, symbol `<name>.kd`)
  runtime  the HIP runtime's copy and fill kernels (`__amd_rocclr_*`: hipMemcpy*, hipMemset* inside the runtime)
  other    anything else: PyTorch's kernels, rocBLAS, ... (each one is a finding to report, not to whitelist)
Kernels before the window (model load, warm-up) are listed for information. RESULT PASS when `other` is empty.
Prints PASS / FAIL lines and RESULT; the trace files stay in DIR.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, os.path.join(ROOT, 'python'))
import serve_smoke as S   # noqa: E402  (request / SSE helpers, check, safe_dir)

PROBE_CHILD = '''import json, os, sys, time, ctypes
sys.path.insert(0, {py!r})
import torch
from caminus.device.hip import Module
m = Module('smoke'); n = 1 << 16
x = torch.ones(n, device='cuda'); y = torch.ones(n, device='cuda'); torch.cuda.synchronize()
c0 = dict(mono=time.clock_gettime_ns(time.CLOCK_MONOTONIC), boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
m.saxpy(((n + 255) // 256,), (256,), y, x, ctypes.c_float(2.0), ctypes.c_int(n)); torch.cuda.synchronize()
c1 = dict(mono=time.clock_gettime_ns(time.CLOCK_MONOTONIC), boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
print('PROBE ' + json.dumps(dict(c0=c0, c1=c1)), flush=True)
'''


def rocprof():
    p = shutil.which('rocprofv3') or os.path.join(os.environ.get('ROCM_PATH', ''), 'bin', 'rocprofv3')
    if not os.path.isfile(p):
        raise SystemExit('launch_trace: rocprofv3 not found (PATH or $ROCM_PATH/bin)')
    return p


def traced(out, cmd):
    return [rocprof(), '--kernel-trace', '-f', 'csv', '-d', out, '-o', 'trace', '--'] + cmd


def kernels(out):
    """[(name, start ns, end ns)] of every kernel_trace csv under out"""
    rows = []
    for f in glob.glob(os.path.join(out, '**', '*kernel_trace.csv'), recursive=True):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                rows.append((r.get('Kernel_Name', ''), int(r.get('Start_Timestamp', 0)), int(r.get('End_Timestamp', 0))))
    return rows


def probe(a):
    out = os.path.join(a.scratch, 'probe')
    cmd = traced(out, [sys.executable, '-I', '-c', PROBE_CHILD.format(py=os.path.join(ROOT, 'python'))])
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=str(a.card)), timeout=240)
    line = [l for l in r.stdout.splitlines() if l.startswith('PROBE ')]
    S.check(r.returncode == 0 and bool(line), f'probe child under rocprofv3: exit {r.returncode}'
            + ('' if line else f'; output tail: {(r.stdout + r.stderr)[-600:]!r}'))
    if not line:
        return 1
    c = json.loads(line[0][6:])
    ks = [k for k in kernels(out) if 'saxpy' in k[0]]
    S.check(len(ks) == 1, f'the trace holds the probe kernel once ({len(ks)} rows; {len(kernels(out))} kernels in all)')
    if len(ks) != 1:
        return 1
    _, t0, t1 = ks[0]
    clock = next((n for n in ('mono', 'boot') if c['c0'][n] <= t0 <= t1 <= c['c1'][n]), None)
    S.check(clock is not None, f'the kernel lies inside one clock\'s bracket: {clock} (kernel {t0} .. {t1}; mono {c["c0"]["mono"]} '
                               f'.. {c["c1"]["mono"]}; boot {c["c0"]["boot"]} .. {c["c1"]["boot"]})')
    if clock:
        open(os.path.join(a.scratch, 'clock.txt'), 'w').write(clock + '\n')
    return 0 if clock else 1


def now(clock):
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC if clock == 'mono' else time.CLOCK_BOOTTIME)


def symbols(names, root):
    """{kernel symbol name} of the code objects Caminus loaded (their `.kd` descriptors, read from the files in root's
    build/, or $CAMINUS_KERNEL_DIR when set: where the server took them from)"""
    d = os.environ.get('CAMINUS_KERNEL_DIR') or os.path.join(root, 'build')
    syms = set()
    for n in names:
        p = os.path.join(d, n + '.hsaco')
        data = open(p, 'rb').read()
        syms.update(m.decode() for m in re.findall(rb'([A-Za-z_][A-Za-z0-9_]*)\.kd\x00', data))
    return syms


def serve(a):
    ck = os.path.join(a.scratch, 'clock.txt')
    if not os.path.isfile(ck):
        raise SystemExit('launch_trace serve: run `probe` first (it writes DIR/clock.txt)')
    clock = open(ck).read().strip()
    out = os.path.join(a.scratch, 'serve')
    cmd = [sys.executable, '-I', os.path.join(a.root, 'tools', 'caminus_serve.py'), '--port', '0', '--model', a.model,
           '--max-ctx', str(a.max_ctx), '--cache-dir', os.path.join(a.scratch, 'kv'), '--flight-dir', os.path.join(a.scratch, 'flight')]
    p = subprocess.Popen(traced(out, cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=a.root,
                         env=dict(os.environ, PYTHONUNBUFFERED='1', HIP_VISIBLE_DEVICES=str(a.card)))
    lines, port, ready = [], [None], threading.Event()

    def rd():
        for line in p.stdout:
            line = line.rstrip('\n'); lines.append(line); print(f'  server| {line}', flush=True)
            mm = re.search(r'http://[^:]+:(\d+)', line)
            if mm and port[0] is None: port[0] = int(mm.group(1))
            if 'ready on' in line: ready.set()
        ready.set()
    threading.Thread(target=rd, daemon=True).start()
    ok = ready.wait(1500) and p.poll() is None and port[0]
    S.check(bool(ok), 'server ready under rocprofv3')
    if not ok:
        p.kill(); p.wait(60); return 1
    alias = 'm'
    st, models = S.get(port[0], '/v1/models')
    if st == 200 and isinstance(models, dict) and models.get('data'):
        alias = models['data'][0]['id']
    msg = [{'role': 'user', 'content': 'Name three prime numbers and say why each is prime, briefly.'}]

    def chat(**kw):
        r = S.request(port[0], 'POST', '/v1/chat/completions', dict(model=alias, messages=msg, stream=True, max_tokens=48, **kw))
        return r, S.openai_collect(r)
    chat(temperature=0)                                     # (warm-up: outside the window)
    t0 = now(clock)
    r1, o1 = chat(temperature=0)
    r2, o2 = chat(temperature=0.7, top_k=20, seed=5)
    t1 = now(clock)
    S.check(r1['status'] == 200 and r2['status'] == 200, f'two requests inside the window: HTTP {r1["status"]} / {r2["status"]}')
    st, props = S.get(port[0], '/props')
    loaded = ((props.get('caminus') or {}).get('code_objects') or []) if isinstance(props, dict) else []
    p.send_signal(signal.SIGTERM)
    try: p.wait(300)
    except subprocess.TimeoutExpired: p.kill(); p.wait(60)
    time.sleep(1)
    own = symbols(loaded, a.root)
    ks = kernels(out)
    inside = [k for k in ks if t0 <= k[1] <= t1]
    cat = {'own': Counter(), 'runtime': Counter(), 'other': Counter()}
    for name, _, _ in inside:
        base = name.split('(')[0].strip()
        c = 'own' if base in own else 'runtime' if base.startswith('__amd_rocclr_') else 'other'
        cat[c][name] += 1
    before = Counter(k[0] for k in ks if k[1] < t0)
    S.check(bool(ks), f'the trace holds {len(ks)} kernels ({len(inside)} inside the window); code objects loaded: {", ".join(loaded)}')
    for c in ('own', 'runtime', 'other'):
        print(f'NOTE {c}: {sum(cat[c].values())} launches of {len(cat[c])} kernels', flush=True)
        for name, n in cat[c].most_common():
            print(f'    {n:6d}  {name[:160]}', flush=True)
    print(f'NOTE before the window (load, warm-up): {sum(before.values())} launches of {len(before)} kernels', flush=True)
    for name, n in before.most_common(40):
        print(f'    {n:6d}  {name[:160]}', flush=True)
    S.check(not cat['other'], f'every kernel inside the window is Caminus\'s or the runtime\'s copy / fill ({len(cat["other"])} other kernels)')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('probe', 'serve'))
    ap.add_argument('--scratch', required=True); ap.add_argument('--card', default='1')
    ap.add_argument('--model'); ap.add_argument('--max-ctx', default='8192')
    ap.add_argument('--root', default=ROOT, help='the checkout whose server runs (default: this one; another one gives a baseline)')
    a = ap.parse_args()
    a.scratch = S.safe_dir(a.scratch); os.makedirs(a.scratch, mode=0o700, exist_ok=True)
    if a.mode == 'serve' and not a.model:
        ap.error('serve needs --model')
    (probe if a.mode == 'probe' else serve)(a)
    print('RESULT ' + ('PASS' if not S.FAILS else f'FAIL ({len(S.FAILS)})'), flush=True)
    return 1 if S.FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
