"""Which GPU kernels a served request launches: the own-code launch gate, from the HIP runtime's own launch log.

  tools/guard.sh LOG 300  python -I tools/gates/launch_trace.py probe --scratch DIR [--card 1]
  tools/guard.sh LOG 2400 python -I tools/gates/launch_trace.py serve --scratch DIR --model GGUF [--card 1] [--max-ctx 8192]
                                                                    [--root CHECKOUT]

The HIP runtime the server runs on (the one in the PyTorch wheel) logs every dispatch's kernel name ("ShaderName : NAME",
"Graph ShaderName : NAME" for a HIP graph's nodes) when AMD_LOG_LEVEL is set; the log goes to the process's stderr. A
profiler outside the runtime is not used: TheRock's rocprofv3 traced a small probe but aborted inside the server
(free(): invalid size in its finalization thread) and wrote nothing.

probe: the instrument first. A child launches the `smoke` object's saxpy with the log on, trying a narrow mask first
(AMD_LOG_MASK=0x80, the kernel-dispatch lines) and then wider settings; the first setting whose log names saxpy is written
to DIR/log-env.json for `serve`. Exit 0 only if one does.

serve: tools/caminus_serve.py on one card with that log setting. After the server is ready and one warm-up request has
run, the window opens; inside it three chat streams run: greedy on a new prompt (its prefill inside the window), sampled
on the warm-up's prompt (temperature 0.7, top_k 20, seed 5: the device sampler's path), and sampled with penalties on
another new prompt (presence 0.5, repeat 1.1: the penalty ids' values too); then the window closes and the server stops.
Each reply's sha256 is printed: two checkouts with the same seed must print the same ones. Every dispatch logged inside
the window is sorted into:
  own      a kernel of a code object Caminus loaded (the server's /props caminus.code_objects, symbol `<name>.kd`)
  runtime  the HIP runtime's copy and fill kernels (`__amd_rocclr_*`: hipMemcpy*, hipMemset* inside the runtime)
  other    anything else: PyTorch's kernels, rocBLAS, ... (each one is a finding to report, not to whitelist)
The window is bounded by the server's log lines read before and after the two requests (each bound read after the log
has been quiet for a second). Dispatches before the window are listed for information, model load and warm-up apart. RESULT PASS when
`other` is empty.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools')); sys.path.insert(0, os.path.join(ROOT, 'python'))
import serve_smoke as S   # noqa: E402  (request / SSE helpers, check, safe_dir, FAILS)

SHADER = re.compile(r'(?:Graph )?ShaderName : (.+?)(?:, device id : \d+)?[ \t]*$', re.M)   # (the whole name: C++ ones hold spaces)
PROBE_CHILD = '''import sys, ctypes
sys.path.insert(0, {py!r})
import torch
from caminus.device.hip import Module
m = Module('smoke'); n = 1 << 16
x = torch.ones(n, device='cuda'); y = torch.ones(n, device='cuda')
m.saxpy(((n + 255) // 256,), (256,), y, x, ctypes.c_float(2.0), ctypes.c_int(n)); torch.cuda.synchronize()
print('PROBE DONE', flush=True)
'''
SETTINGS = ({'AMD_LOG_LEVEL': '3', 'AMD_LOG_MASK': '0x80'}, {'AMD_LOG_LEVEL': '3'}, {'AMD_LOG_LEVEL': '4'})


def probe(a):
    for env in SETTINGS:
        r = subprocess.run([sys.executable, '-I', '-c', PROBE_CHILD.format(py=os.path.join(ROOT, 'python'))], capture_output=True,
                           text=True, timeout=240, env=dict(os.environ, HIP_VISIBLE_DEVICES=str(a.card), **env))
        out = r.stdout + r.stderr
        names = SHADER.findall(out)
        ok = r.returncode == 0 and 'PROBE DONE' in r.stdout and 'saxpy' in names
        print(f'NOTE {env}: exit {r.returncode}, {len(out.splitlines())} log lines, dispatches named: {sorted(set(names))[:8]}', flush=True)
        if ok:
            json.dump(env, open(os.path.join(a.scratch, 'log-env.json'), 'w'))
            S.check(True, f'the runtime log names the probe kernel with {env}')
            return 0
    S.check(False, 'no log setting named the probe kernel')
    return 1


def symbols(names, root):
    """{kernel symbol name} of the code objects Caminus loaded (their `.kd` descriptors, read from the files in root's
    build/, or $CAMINUS_KERNEL_DIR when set: where the server took them from)"""
    d = os.environ.get('CAMINUS_KERNEL_DIR') or os.path.join(root, 'build')
    syms = set()
    for n in names:
        data = open(os.path.join(d, n + '.hsaco'), 'rb').read()
        syms.update(m.decode() for m in re.findall(rb'([A-Za-z_][A-Za-z0-9_]*)\.kd\x00', data))
    return syms


def serve(a):
    ef = os.path.join(a.scratch, 'log-env.json')
    if not os.path.isfile(ef):
        raise SystemExit('launch_trace serve: run `probe` first (it writes DIR/log-env.json)')
    logenv = json.load(open(ef))
    cmd = [sys.executable, '-I', os.path.join(a.root, 'tools', 'caminus_serve.py'), '--port', '0', '--model', a.model,
           '--max-ctx', str(a.max_ctx), '--cache-dir', os.path.join(a.scratch, 'kv'), '--flight-dir', os.path.join(a.scratch, 'flight')]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=a.root,
                         env=dict(os.environ, PYTHONUNBUFFERED='1', HIP_VISIBLE_DEVICES=str(a.card), **logenv))
    lines, port, ready, last = [], [None], threading.Event(), [time.time()]
    log = open(os.path.join(a.scratch, 'server.log'), 'w')

    def rd():
        for line in p.stdout:
            line = line.rstrip('\n'); lines.append(line); log.write(line + '\n'); last[0] = time.time()
            if not SHADER.search(line) and not line.startswith(':'):
                print(f'  server| {line}', flush=True)
            mm = re.search(r'caminus serve: http://[^:]+:(\d+)', line)
            if mm and port[0] is None: port[0] = int(mm.group(1))
            if 'ready on' in line: ready.set()
        ready.set()
    threading.Thread(target=rd, daemon=True).start()

    def quiet():
        """the number of log lines once the log has been quiet for a second (a bound of the window)"""
        t = time.time()
        while time.time() - last[0] < 1.0 and time.time() - t < 30:
            time.sleep(0.1)
        return len(lines)
    ok = ready.wait(1500) and p.poll() is None and port[0]
    S.check(bool(ok), f'server ready with {logenv}')
    if not ok:
        p.kill(); p.wait(60); return 1
    st, models = S.get(port[0], '/v1/models')
    alias = models['data'][0]['id'] if st == 200 and isinstance(models, dict) and models.get('data') else 'm'
    A = 'Name three prime numbers and say why each is prime, briefly.'
    B = 'Write a short Python function that reverses a linked list, with a one-line docstring.'
    C = 'List four uses of a hash map in everyday programs, one sentence each.'

    def chat(text, **kw):
        r = S.request(port[0], 'POST', '/v1/chat/completions',
                      dict(model=alias, messages=[{'role': 'user', 'content': text}], stream=True, max_tokens=48, **kw))
        return r, S.openai_collect(r)
    il = quiet()                                            # (model load ends here)
    chat(A, temperature=0)                                  # (warm-up: outside the window)
    i0 = quiet()
    runs = [('greedy, new prompt', B, dict(temperature=0)), ('sampled, cached prompt', A, dict(temperature=0.7, top_k=20, seed=5)),
            ('sampled + penalties, new prompt', C, dict(temperature=0.7, top_k=20, seed=7, presence_penalty=0.5, repeat_penalty=1.1))]
    out = [(what, *chat(text, **kw)) for what, text, kw in runs]
    i1 = quiet()
    S.check(all(r['status'] == 200 for _, r, _ in out), 'three requests inside the window: HTTP ' + ' / '.join(str(r['status']) for _, r, _ in out))
    import hashlib
    for what, r, m in out:
        print(f'REPLY {what}: sha256 {hashlib.sha256((m.get("content") or "").encode()).hexdigest()[:16]}, '
              f'{len(m.get("content") or "")} characters, finish {m.get("finish")}', flush=True)
    st, props = S.get(port[0], '/props')
    loaded = ((props.get('caminus') or {}).get('code_objects') or []) if isinstance(props, dict) else []
    p.send_signal(signal.SIGTERM)
    try: p.wait(300)
    except subprocess.TimeoutExpired: p.kill(); p.wait(60)
    log.close()
    json.dump(dict(load=il, window=[i0, i1], lines=len(lines)), open(os.path.join(a.scratch, 'window.json'), 'w'))   # (server.log's lines)
    own = symbols(loaded, a.root)
    inside = [m for line in lines[i0:i1] for m in SHADER.findall(line)]
    load = Counter(m for line in lines[:il] for m in SHADER.findall(line))
    warm = Counter(m for line in lines[il:i0] for m in SHADER.findall(line))
    before = load + warm
    cat = {'own': Counter(), 'runtime': Counter(), 'other': Counter()}
    for name in inside:
        c = 'own' if name in own else 'runtime' if name.startswith('__amd_rocclr_') else 'other'
        cat[c][name] += 1
    S.check(bool(inside), f'{len(inside)} dispatches logged inside the window ({sum(before.values())} before it); code objects '
                          f'loaded: {", ".join(loaded)}')
    for c in ('own', 'runtime', 'other'):
        print(f'NOTE {c}: {sum(cat[c].values())} dispatches of {len(cat[c])} kernels', flush=True)
        for name, n in cat[c].most_common():
            print(f'    {n:6d}  {name[:160]}', flush=True)
    for what, cnt in (('model load', load), ('warm-up request', warm)):
        print(f'NOTE before the window, {what}: {sum(cnt.values())} dispatches of {len(cnt)} kernels', flush=True)
        for name, n in cnt.most_common(40):
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
