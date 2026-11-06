"""The installable package's own parts (docs/RELEASE-PLAN.md), CPU only, nothing touches a GPU:

  HIP_VISIBLE_DEVICES= python -I tests/test_packaging.py --scratch <job scratch dir>/pkg [STAGE ...]

Stages (default: all):
  config    caminus.config: environment > config.toml > XDG defaults, legacy names, refusals of a bad file
  guard     guard.sh --lines (bash) == caminus.machine (Python) on synthetic sysfs trees (this machine's layout, a headless
            card, card10 ordering, a display on the second card, an iGPU, a non-AMD card) and on this machine's own sysfs
            (XTX 23,800 / XT 19,850, the display on the XTX); capacity's lines and the cache reserve from it
  kernels   caminus.kernels: the directory order (CAMINUS_KERNEL_DIR, a checkout's build/, the package's), provenance
            verify on a staged scratch copy (a changed byte and a missing object found); provenance's `built` from
            SOURCE_DATE_EPOCH when set (a bad value refused), else build.sh's time
  repro     tools/repro_dist.py on synthetic archives: two sdists and wheels of the same files with different times, owners,
            user names, modes and gzip times come out byte-identical; a second run changes nothing; a symlink member and a
            missing SOURCE_DATE_EPOCH are refused
  download  caminus download against a local fake HTTP server: fresh, resume (Range), a server that ignores Range, a dropped
            connection, a corrupt file (sha256), a short file, 404, an existing verified file, --verify of a present file
  doctor    caminus doctor's CPU path: runs, PASS / WARN / FAIL lines and a RESULT, no HIP initialized (torch.cuda never
            touched), the JSON form; a missing model file is a FAIL with the download hint
  cli       python -I -m caminus (help, unknown command), caminus kv-cache / flight / purge on scratch directories
  torch     without PyTorch (blocked in the child: sys.modules['torch'] = None): serve / replay / kv-cache / purge print the
            ROCm install line and exit 2, never a traceback, never starting the guard; with a CUDA build (a stub torch whose
            version.hip is None) serve does the same (serve / replay check the build; kv-cache / purge only need torch to
            import); --help and doctor still work
The scratch directory is required (never /tmp); it must not exist or be empty."""
import argparse
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
PY = sys.executable
FAILS = []


def cam(*args):
    """`python -I -m caminus ARGS` against this checkout's package (-I drops the working directory from sys.path)"""
    return [PY, '-I', '-c', f'import sys; sys.path.insert(0, {os.path.join(ROOT, "python")!r}); from caminus.__main__ import main; '
            f'sys.exit(main({list(args)!r}))']


def check(name, ok, detail=''):
    d = ' '.join(str(detail).split())
    print(f'{"PASS" if ok else "FAIL"} {name}' + (f': {d if not ok else d[:240]}' if d else ''), flush=True)
    if not ok:
        FAILS.append(name)


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


# ---------------------------------------------------------------------------------------------------------------- config
def stage_config(S):
    from caminus import config as C
    names = ['CAMINUS_CONFIG', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME'] + [n for ns in C.KEYS.values() for n in ns]
    keep = {k: os.environ.get(k) for k in names}   # (every variable the layer reads: the run's own CAMINUS_* must not leak in)
    try:
        for k in keep: os.environ.pop(k, None)
        home = os.path.join(S, 'cfg-home')
        os.environ.update(XDG_CONFIG_HOME=os.path.join(home, 'config'), XDG_CACHE_HOME=os.path.join(home, 'cache'),
                          XDG_DATA_HOME=os.path.join(home, 'data'), XDG_STATE_HOME=os.path.join(home, 'state'))
        C._FILE.clear()
        d = C.summary()
        check('config: XDG defaults (no file, no environment)',
              d['kv_dir'] == (os.path.join(home, 'cache', 'caminus', 'kv'), 'default')
              and d['model_dir'][0] == os.path.join(home, 'data', 'caminus', 'models')
              and d['coder'][0].startswith(os.path.join(home, 'data', 'caminus', 'models', 'Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF', 'IQ1_M'))
              and d['state_dir'][0] == os.path.join(home, 'state', 'caminus') and d['mtp'] == (None, 'default')
              and d['gguf_py'][0] == C.VENDOR, str(d['kv_dir']))
        w(os.path.join(home, 'config', 'caminus', 'config.toml'), 'model_dir = "~/m"\ncoder = "/x/coder.gguf"\n')
        C._FILE.clear()
        d = C.summary()
        check('config: config.toml values (~ expanded), the coder default follows model_dir only when coder is unset',
              d['model_dir'] == (os.path.expanduser('~/m'), 'file ' + C.config_path()) and d['coder'][0] == '/x/coder.gguf'
              and d['dflash'][0].startswith(os.path.expanduser('~/m') + '/'), f"{d['model_dir']} {d['coder']}")
        os.environ['K3_GGUF'] = '/legacy.gguf'
        check('config: the old K3_GGUF overrides the file', C.coder() == '/legacy.gguf')
        os.environ['CAMINUS_CODER'] = '/new.gguf'
        check('config: CAMINUS_CODER wins over K3_GGUF', C.resolve('coder') == ('/new.gguf', 'env CAMINUS_CODER'))
        os.environ['XDG_CACHE_HOME'] = 'relative/path'
        check('config: a relative XDG_CACHE_HOME is ignored (the spec)', C.kv_dir() == os.path.expanduser('~/.cache/caminus/kv'))
        w(os.path.join(home, 'config', 'caminus', 'config.toml'), 'cache = "/x"\n')
        C._FILE.clear()
        try:
            C.kv_dir(); ok = False
        except C.ConfigError as e:
            ok = 'unknown key' in str(e)
        check('config: an unknown key is refused', ok)
        w(os.path.join(home, 'config', 'caminus', 'config.toml'), 'coder = 3\n')
        C._FILE.clear()
        try:
            C.dflash(); ok = False
        except C.ConfigError as e:
            ok = 'must be a string' in str(e)
        check('config: a non-string value is refused', ok)
        w(os.path.join(home, 'config', 'caminus', 'config.toml'), 'coder = \n')
        C._FILE.clear()
        try:
            C.dflash(); ok = False
        except C.ConfigError:
            ok = True
        check('config: bad TOML is refused', ok)
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        C._FILE.clear()


# ----------------------------------------------------------------------------------------------------------------- guard
def fixture(root, cards, kfd):
    """cards: (name, total MiB, PCI device id, render minor, boot_vga, connectors {name: status}, vendor, pci slot);
    kfd: (render minor, gfx_target_version)"""
    for n, total, did, minor, bvga, conns, vendor, slot in cards:
        d = os.path.join(root, 'sys', 'class', 'drm', n, 'device')
        w(os.path.join(d, 'mem_info_vram_total'), f'{total << 20}\n')
        w(os.path.join(d, 'mem_info_vram_used'), f'{100 << 20}\n')
        w(os.path.join(d, 'device'), did + '\n')
        w(os.path.join(d, 'vendor'), vendor + '\n')
        w(os.path.join(d, 'boot_vga'), f'{bvga}\n')
        w(os.path.join(d, 'uevent'), f'DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n')
        os.makedirs(os.path.join(d, 'drm', f'renderD{minor}'), exist_ok=True)
        for c, st in conns.items():
            w(os.path.join(root, 'sys', 'class', 'drm', f'{n}-{c}', 'status'), st + '\n')
    for i, (minor, gfx) in enumerate(kfd):
        w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', str(i + 1), 'properties'),
          f'cpu_cores_count 0\nsimd_count 96\ngfx_target_version {gfx}\ndrm_render_minor {minor}\n')
    w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', '0', 'properties'),
      'cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\ndrm_render_minor 0\n')


def guard_lines(root):
    r = subprocess.run(['bash', os.path.join(ROOT, 'tools', 'guard.sh'), '--lines', root], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        x = line.split()
        out.append(dict(card=x[0], label=x[1], total_mib=int(x[3]), line_mib=int(x[5]), display=x[7] == '1', reserve_mib=int(x[9])))
    return r.returncode, out


def stage_guard(S):
    from caminus import machine, capacity
    A = '0x1002'
    trees = {
        'this machine': ([('card1', 24560, '0x744c', 128, 1, {'DP-1': 'disconnected', 'HDMI-A-1': 'connected'}, A, '0000:03:00.0'),
                          ('card2', 20464, '0x744c', 129, 0, {'DP-4': 'disconnected', 'HDMI-A-2': 'disconnected'}, A, '0000:06:00.0'),
                          ('card3', 512, '0x164e', 130, 0, {'DP-7': 'disconnected'}, A, '0000:76:00.0')],
                         [(128, 110000), (129, 110000), (130, 100306)],
                         [('card1', 'XTX', 23800, True, 512), ('card2', 'XT', 19850, False, 0)]),
        'headless one card': ([('card0', 24560, '0x744c', 128, 0, {}, A, '0000:03:00.0')], [(128, 110000)],
                              [('card0', 'XTX', 23800, False, 0)]),
        'card10 ordering, display on the second card': (
            [('card10', 20464, '0x744c', 138, 0, {}, A, '0000:0a:00.0'), ('card2', 24560, '0x744c', 129, 0, {'DP-1': 'connected'}, A, '0000:06:00.0'),
             ('card9', 16368, '0x7480', 137, 0, {}, A, '0000:09:00.0')],
            [(138, 110000), (129, 110000), (137, 110001)],
            [('card2', 'XTX', 23800, True, 512), ('card10', 'XT', 19850, False, 0)]),
        'a non-AMD card and a card without KFD': (
            [('card0', 24560, '0x2684', 128, 1, {'DP-1': 'connected'}, '0x10de', '0000:01:00.0'), ('card1', 20464, '0x744c', 129, 0, {}, A, '0000:02:00.0'),
             ('card3', 24560, '0x744c', 131, 0, {}, A, '0000:04:00.0')],
            [(128, 110000), (129, 110000)],
            [('card1', 'XT', 19850, False, 0)]),
        'an odd total': ([('card1', 16368, '0x73bf', 128, 1, {}, A, '0000:03:00.0')], [(128, 110000)], [('card1', 'card1', 15850, True, 512)]),
    }
    for name, (cards, kfd, want) in trees.items():
        root = os.path.join(S, 'sysfs', name.replace(' ', '_').replace(',', ''))
        fixture(root, cards, kfd)
        rc, got = guard_lines(root)
        py = [(c['card'], c['label'], c['line_mib'], c['display'], c['reserve_mib']) for c in machine.watched(root)]
        sh = [(c['card'], c['label'], c['line_mib'], c['display'], c['reserve_mib']) for c in got]
        check(f'guard: {name}: guard.sh --lines == caminus.machine == expected', rc == 0 and sh == py == want, f'sh {sh} py {py}')
    # this machine, live (read only)
    rc, got = guard_lines('/')
    live = [(c['card'], c['label'], c['line_mib'], c['display']) for c in got]
    py = [(c['card'], c['label'], c['line_mib'], c['display']) for c in machine.watched('/')]
    check('guard: this machine: guard.sh --lines == caminus.machine', rc == 0 and live == py, str(live))
    if [x[:3] for x in live] == [('card1', 'XTX', 23800), ('card2', 'XT', 19850)]:
        check('guard: this machine: the old fixed lines reproduced (XTX 23,800 / XT 19,850, the display on the XTX)',
              live[0][3] is True and capacity.GUARD_LINE_MIB == {'card1': 23800, 'card2': 19850})
    else:
        print(f'NOTE guard: not the development machine ({live}): the fixed-line check is skipped', flush=True)
    check('guard: line_mib rule (0.97, 50 MiB steps)', [machine.line_mib(t) for t in (24560, 20464, 16368, 8176, 49136)] == [23800, 19850, 15850, 7900, 47650])
    txt = open(os.path.join(ROOT, 'python', 'caminus', 'share', 'guard.sh')).read()
    check('guard: tools/guard.sh is the shipped guard', os.path.realpath(os.path.join(ROOT, 'tools', 'guard.sh'))
          == os.path.realpath(os.path.join(ROOT, 'python', 'caminus', 'share', 'guard.sh')))
    check('guard: no machine path in the guard', not any(s in txt for s in ('/home/', '/run/' 'media', 'gpu' '-fan')))


# --------------------------------------------------------------------------------------------------------------- kernels
def stage_kernels(S):
    from caminus import kernels as K
    keep = os.environ.pop('CAMINUS_KERNEL_DIR', None)
    try:
        d, where = K.kernel_dir()
        check('kernels: a checkout reads its own build/', (d, where) == (os.path.join(ROOT, 'build'), 'checkout'), f'{d} {where}')
        os.environ['CAMINUS_KERNEL_DIR'] = os.path.join(S, 'kdir')
        d, where = K.kernel_dir()
        check('kernels: CAMINUS_KERNEL_DIR first', (d, where) == (os.path.join(S, 'kdir'), 'env CAMINUS_KERNEL_DIR'), f'{d} {where}')
    finally:
        if keep is None: os.environ.pop('CAMINUS_KERNEL_DIR', None)
        else: os.environ['CAMINUS_KERNEL_DIR'] = keep
    names = set(K.RUNTIME_SET)
    check('kernels: the runtime set has no trace / swiglu / test-reference builds', not any(n.endswith(('_trace', '_swiglu')) or n == 'gg_mmvf_ref' for n in names)
          and len(names) == len(K.RUNTIME_SET) == 36, str(len(names)))
    if not os.path.isfile(os.path.join(ROOT, 'build', 'BUILD-INFO')):
        print('NOTE kernels: no build/BUILD-INFO (./build.sh not run): the staging checks are skipped', flush=True); return
    st = os.path.join(S, 'staged')
    noepoch = {k: v for k, v in os.environ.items() if k != 'SOURCE_DATE_EPOCH'}
    r = subprocess.run([PY, '-I', os.path.join(ROOT, 'tools', 'stage_kernels.py'), '--out', st, '--allow-dirty'], capture_output=True, text=True,
                       env=noepoch)
    prov, res = K.verify(st)
    check('kernels: a staged copy verifies against its provenance', r.returncode == 0 and prov is not None and len(res) == 36 and all(ok for _, ok, _ in res),
          (r.stdout + r.stderr).strip()[-300:])
    check('kernels: provenance carries the toolchain, the commands, the sources and no home path',
          prov is not None and prov['toolchain']['version'] and all(o['command'] for o in prov['objects'].values())
          and len(prov['source']['kernel_tree_sha256']) == 64 and '/home/' not in json.dumps(prov), '')
    info0 = dict(l.split('\t', 1) for l in open(os.path.join(ROOT, 'build', 'BUILD-INFO')).read().splitlines() if l)
    check('kernels: without SOURCE_DATE_EPOCH, built is build.sh\'s time', prov is not None and prov.get('built') == info0.get('built')
          and prov.get('built_from') == 'build.sh', f'{prov and prov.get("built")} {prov and prov.get("built_from")}')
    st3 = os.path.join(S, 'staged-epoch')
    r = subprocess.run([PY, '-I', os.path.join(ROOT, 'tools', 'stage_kernels.py'), '--out', st3, '--allow-dirty'], capture_output=True, text=True,
                       env=dict(noepoch, SOURCE_DATE_EPOCH='1793976692'))
    p3 = json.load(open(os.path.join(st3, K.PROVENANCE))) if r.returncode == 0 else {}
    check('kernels: SOURCE_DATE_EPOCH sets built (UTC) and built_from', p3.get('built') == '2026-11-06T14:51:32Z' and p3.get('built_from') == 'SOURCE_DATE_EPOCH'
          and {k: v for k, v in p3.items() if k not in ('built', 'built_from')} == {k: v for k, v in prov.items() if k not in ('built', 'built_from')},
          (r.stdout + r.stderr).strip()[-300:])
    r = subprocess.run([PY, '-I', os.path.join(ROOT, 'tools', 'stage_kernels.py'), '--out', os.path.join(S, 'staged-bad'), '--allow-dirty'],
                       capture_output=True, text=True, env=dict(noepoch, SOURCE_DATE_EPOCH='yesterday'))
    check('kernels: a SOURCE_DATE_EPOCH that is not whole seconds is refused', r.returncode == 2 and 'SOURCE_DATE_EPOCH' in r.stderr
          and not os.path.exists(os.path.join(S, 'staged-bad')), r.stderr.strip()[-200:])
    p = os.path.join(st, 'k3_guard.hsaco')
    b = bytearray(open(p, 'rb').read()); b[-1] ^= 1; open(p, 'wb').write(bytes(b))
    os.rename(os.path.join(st, 'smoke.hsaco'), os.path.join(st, 'smoke.hsaco.moved'))
    _, res = K.verify(st)
    bad = {n: dt for n, ok, dt in res if not ok}
    check('kernels: verify finds a changed byte and a missing object', set(bad) == {'k3_guard.hsaco', 'smoke.hsaco'}, str(bad))
    r = subprocess.run([PY, '-I', os.path.join(ROOT, 'tools', 'stage_kernels.py'), '--out', os.path.join(S, 'staged2')], capture_output=True, text=True)
    info = dict(l.split('\t', 1) for l in open(os.path.join(ROOT, 'build', 'BUILD-INFO')).read().splitlines() if l)
    if info.get('source_dirty') == '1':
        check('kernels: a dirty build is refused without --allow-dirty', r.returncode == 2 and 'dirty' in r.stderr, r.stderr.strip()[-200:])


# -------------------------------------------------------------------------------------------------------------- download
class FakeHF:
    """a local stand-in for huggingface.co's /<repo>/resolve/<rev>/<path> (127.0.0.1, a free port): Range requests, and per path
    a mode: 'ok', 'ignore-range' (always 200 + the whole file), 'drop' (the first GET closes the connection after half the
    bytes), 'redirect' (a 302 to another path on the same server), 'gone' (404). Records each request's Range and
    Authorization headers"""

    def __init__(self):
        self.files, self.modes, self.log, self.dropped = {}, {}, [], set()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                outer.log.append((self.path, self.headers.get('Range'), self.headers.get('Authorization')))
                path = self.path.split('?')[0]
                if path.startswith('/cdn/'):
                    path = path[4:]
                mode = outer.modes.get(path, 'ok')
                if path not in outer.files or mode == 'gone':
                    self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
                if mode == 'redirect' and not self.path.startswith('/cdn/'):
                    self.send_response(302); self.send_header('Location', '/cdn' + path); self.send_header('Content-Length', '0'); self.end_headers(); return
                data = outer.files[path]
                rng = self.headers.get('Range')
                start = 0
                if rng and mode != 'ignore-range':
                    start = int(rng.split('=')[1].split('-')[0])
                    if start >= len(data):
                        self.send_response(416); self.send_header('Content-Length', '0'); self.end_headers(); return
                    self.send_response(206); self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header('Content-Length', str(len(body))); self.end_headers()
                if mode == 'drop' and path not in outer.dropped:
                    outer.dropped.add(path)
                    self.wfile.write(body[:len(body) // 2]); self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                self.wfile.write(body)

        self.srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        self.url = f'http://127.0.0.1:{self.srv.server_address[1]}'
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def add(self, repo, rev, path, data, mode='ok'):
        p = f'/{repo}/resolve/{rev}/{path}'
        self.files[p], self.modes[p] = data, mode
        return p

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


def stage_download(S):
    from caminus import models
    from caminus.cmd import download as D
    import random
    srv = FakeHF()
    try:
        rnd = random.Random(7)
        def mk(key, n, mode='ok', serve=None, role='coder'):
            data = bytes(rnd.getrandbits(8) for _ in range(n))
            it = models.Item(key, role, f'org/repo-{key}', 'a' * 40, f'sub/{key}.gguf', n, hashlib.sha256(data).hexdigest(), 'hf-lfs')
            srv.add(it.repo, it.revision, it.path, data if serve is None else serve, mode)
            return it, data
        logs = []
        log = logs.append
        kw = dict(endpoint=srv.url, token='', log=log, sleep=lambda s: None)
        dest = os.path.join(S, 'models')
        a, da = mk('fresh', 300_000)
        bad = D.download([a], dest, **kw)
        fa = os.path.join(dest, models.local_rel(a))
        check('download: a fresh file, verified and in place with its marker', bad == 0 and open(fa, 'rb').read() == da
              and D.marker_ok(fa, a.sha256) and not os.path.exists(fa + '.part'), str(logs[-2:]))
        # resume: a partial .part, the server honours Range
        b, db = mk('resume', 400_000)
        fb = os.path.join(dest, models.local_rel(b)); os.makedirs(os.path.dirname(fb), exist_ok=True)
        open(fb + '.part', 'wb').write(db[:123_456])
        n0 = len(srv.log)
        bad = D.download([b], dest, **kw)
        rq = srv.log[n0:]
        check('download: resume from a .part with a Range request', bad == 0 and open(fb, 'rb').read() == db
              and rq and rq[0][1] == 'bytes=123456-', str(rq))
        # a server that ignores Range: the file restarts and still verifies
        c, dc = mk('norange', 200_000, 'ignore-range')
        fc = os.path.join(dest, models.local_rel(c)); os.makedirs(os.path.dirname(fc), exist_ok=True)
        open(fc + '.part', 'wb').write(dc[:50_000])
        logs.clear()
        bad = D.download([c], dest, **kw)
        check('download: a server that ignores Range restarts the file', bad == 0 and open(fc, 'rb').read() == dc
              and any('ignored the range' in x for x in logs), str(logs[-3:]))
        # a dropped connection mid-file: the next attempt resumes
        d_, dd = mk('drop', 500_000, 'drop')
        n0 = len(srv.log); logs.clear()
        bad = D.download([d_], dest, **kw)
        fd = os.path.join(dest, models.local_rel(d_))
        rq = srv.log[n0:]
        check('download: a dropped connection resumes on the next attempt', bad == 0 and open(fd, 'rb').read() == dd and len(rq) >= 2
              and rq[1][1] is not None and rq[1][1].startswith('bytes='), f'{rq} {logs[-2:]}')
        # corrupt bytes of the right size: refused, the .part removed, nothing in place
        e, de = mk('corrupt', 100_000, serve=bytes(100_000))
        logs.clear()
        bad = D.download([e], dest, **kw)
        fe = os.path.join(dest, models.local_rel(e))
        check('download: a corrupt file (sha256) is refused and removed', bad == 1 and not os.path.exists(fe) and not os.path.exists(fe + '.part')
              and any('sha256' in x for x in logs), str(logs[-1:]))
        # a server file of another size: refused before any byte is kept
        f_, df = mk('short', 100_000, serve=os.urandom(90_000))
        logs.clear()
        bad = D.download([f_], dest, **kw)
        ff = os.path.join(dest, models.local_rel(f_))
        check('download: a server file of another size is refused', bad == 1 and not os.path.exists(ff)
              and any('90000 bytes' in x for x in logs), str(logs[-1:]))
        # 404
        g, _ = mk('gone', 1000, 'gone')
        logs.clear()
        bad = D.download([g], dest, **kw)
        check('download: 404 names the revision', bad == 1 and any('HTTP 404' in x and 'aaaaaaaaaaaa' in x for x in logs), str(logs[-1:]))
        # a redirect to another host drops the token; to the same host it keeps it (here: same host, so it is kept)
        h, dh = mk('redir', 70_000, 'redirect')
        n0 = len(srv.log)
        bad = D.download([h], dest, **dict(kw, token='tok-test'))
        rq = srv.log[n0:]
        check('download: a redirect is followed (the token kept on the same host)', bad == 0 and len(rq) == 2 and rq[1][0].startswith('/cdn/')
              and rq[0][2] == 'Bearer tok-test' and rq[1][2] == 'Bearer tok-test', str(rq))
        req = D.urllib.request.Request('http://a.example/x', headers={'Authorization': 'Bearer t'})
        new = D._Redirect().redirect_request(req, None, 302, 'Found', {}, 'http://cdn.example/y')
        check('download: a redirect to another host drops the Authorization header', new is not None and not new.has_header('Authorization'))
        # present files: kept (size), --verify hashes, a wrong size is left alone
        logs.clear()
        bad = D.download([a], dest, **kw)
        check('download: a present, verified file is kept', bad == 0 and any('present (sha256 verified earlier)' in x for x in logs), str(logs))
        os.unlink(fa + '.sha256')
        logs.clear()
        bad = D.download([a], dest, verify=True, **kw)
        check('download: --verify hashes a present file and writes the marker', bad == 0 and D.marker_ok(fa, a.sha256), str(logs))
        open(fb, 'ab').write(b'x')
        logs.clear()
        bad = D.download([b], dest, **kw)
        check('download: a present file of another size is reported and left alone', bad == 1 and os.path.getsize(fb) == b.size + 1, str(logs))
        # the command line with a manifest; --dry-run fetches nothing
        man = os.path.join(S, 'manifest.json')
        i2, d2 = mk('cli', 50_000)
        json.dump([i2._asdict()], open(man, 'w'))
        env = dict(os.environ, HF_ENDPOINT=srv.url, HF_TOKEN='', HIP_VISIBLE_DEVICES='')
        dest2 = os.path.join(S, 'models-cli')
        r0 = subprocess.run(cam('download', '--manifest', man, '--dir', dest2, '--dry-run'), capture_output=True, text=True, env=env)
        r1 = subprocess.run(cam('download', '--manifest', man, '--dir', dest2), capture_output=True, text=True, env=env)
        f2 = os.path.join(dest2, models.local_rel(i2))
        check('download: the command line (--dry-run fetches nothing; then the file)', r0.returncode == 0 and r1.returncode == 0
              and open(f2, 'rb').read() == d2 and 'done' in r1.stdout, (r0.stdout + r0.stderr + r1.stdout + r1.stderr)[-300:])
        check('download: the real manifest is well formed', all(len(it.sha256) == 64 and len(it.revision) == 40 and it.size > 0 for it in models.ITEMS)
              and [it.key for it in models.ITEMS] == ['coder-1', 'coder-2', 'dflash'])
    finally:
        srv.close()


# ---------------------------------------------------------------------------------------------------------------- doctor
def stage_doctor(S):
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', CAMINUS_MODEL_DIR=os.path.join(S, 'no-models'), CAMINUS_CODER='', K3_GGUF='',
               CAMINUS_DFLASH='', XDG_CACHE_HOME=os.path.join(S, 'cache'), CAMINUS_CONFIG=os.path.join(S, 'none.toml'))
    probe = (f'import sys; sys.path.insert(0, {os.path.join(ROOT, "python")!r}); import caminus.__main__ as M; rc = M.main(["doctor"])\n'
             'try:\n import torch\nexcept ImportError:\n torch = None\n'
             'print("HIP_INITIALIZED", torch is not None and torch.cuda.is_initialized()); sys.exit(rc)')
    r = subprocess.run([PY, '-I', '-c', probe], capture_output=True, text=True, env=env, timeout=300)
    out = r.stdout
    lines = [l for l in out.splitlines() if l[:4] in ('PASS', 'INFO', 'WARN', 'FAIL')]
    check('doctor: the CPU path runs and ends with a RESULT', 'RESULT ' in out and len(lines) > 15, (out + r.stderr)[-400:])
    check('doctor: no HIP initialized by the CPU path', 'HIP_INITIALIZED False' in out, out[-200:])
    check('doctor: a missing model file is a FAIL with the download hint', r.returncode == 1 and 'RESULT FAIL' in out
          and any(l.startswith('FAIL') and 'coder-1' in l and 'caminus download' in l for l in lines))
    for area in ('python', 'kernels', 'deps', 'gguf', 'torch', 'gpu', 'guard', 'host', 'models', 'disk', 'port', 'config'):
        check(f'doctor: reports {area}', any(l[6:].split()[0] == area for l in lines))
    rj = subprocess.run(cam('doctor', '--json'), capture_output=True, text=True, env=env, timeout=300)
    try:
        js = json.loads(rj.stdout); ok = js['result'] == 'FAIL' and all(set(c) == {'level', 'area', 'message'} for c in js['checks'])
    except (ValueError, KeyError):
        ok = False
    check('doctor: --json', ok, rj.stdout[-200:])


# ------------------------------------------------------------------------------------------------------------------- cli
def stage_cli(S):
    env = dict(os.environ, HIP_VISIBLE_DEVICES='')
    run = lambda *a: subprocess.run(cam(*a), capture_output=True, text=True, env=env, timeout=120)
    r = run('--help')
    check('cli: --help lists every command', r.returncode == 0 and all(c in r.stdout for c in ('serve', 'download', 'doctor', 'purge', 'replay', 'flight', 'kv-cache', 'guard')))
    r = run('nonsense')
    check('cli: an unknown command is refused (exit 2)', r.returncode == 2 and 'unknown command' in r.stderr)
    r = run('version')
    check('cli: version', r.returncode == 0 and r.stdout.startswith('caminus 0.1.0'), r.stdout)
    for c in ('serve', 'replay', 'download', 'doctor', 'purge', 'flight', 'kv-cache'):
        r = run(c, '--help')
        check(f'cli: {c} --help (no guard, nothing started)', r.returncode == 0 and 'usage:' in r.stdout, (r.stdout + r.stderr)[-200:])
    from caminus import machine, torchcheck
    r = run('guard', '--lines')
    if machine.watched('/'):
        check('cli: guard --lines', r.returncode == 0 and ' line ' in r.stdout, r.stdout + r.stderr)
    else:
        check('cli: guard --lines on a machine without a gfx1100 card (nothing listed, exit 0)', r.returncode == 0 and not r.stdout.strip(),
              r.stdout + r.stderr)
    kv, fl = os.path.join(S, 'kv'), os.path.join(S, 'flight')
    r = run('kv-cache', 'list', '--dir', kv)
    check('cli: kv-cache list on a fresh scratch directory', r.returncode == 0 and '0 entries' in r.stdout, r.stdout + r.stderr)
    os.makedirs(fl, exist_ok=True)
    open(os.path.join(fl, 'requests-20261106.jsonl'), 'w').write('{}\n')
    r = run('purge', '--kv-dir', kv, '--flight-dir', fl)
    check('cli: purge without --yes lists only', r.returncode == 0 and 'nothing deleted' in r.stdout and os.path.exists(os.path.join(fl, 'requests-20261106.jsonl')), r.stdout + r.stderr)
    r = run('purge', '--kv-dir', kv, '--flight-dir', fl, '--yes')
    check('cli: purge --yes empties both', r.returncode == 0 and not os.path.exists(os.path.join(fl, 'requests-20261106.jsonl')), r.stdout + r.stderr)
    r = subprocess.run([PY, '-I', os.path.join(ROOT, 'tools', 'kv_cache.py'), 'list', '--dir', kv], capture_output=True, text=True, env=env)
    check('cli: tools/kv_cache.py (the checkout wrapper) runs the same code', r.returncode == 0 and '0 entries' in r.stdout, r.stdout + r.stderr)
    # the guard wrapper: `caminus serve` outside a guard execs the shipped guard, which runs `python -m caminus serve` with
    # CAMINUS_GUARD set; a missing model ends it at once (exit 2), nothing touches a GPU (a stub gpu-high: no clock change)
    stub = os.path.join(S, 'gpu-high-stub')
    w(stub, '#!/bin/sh\nexec "$@"\n'); os.chmod(stub, 0o755)
    genv = dict(env, CAMINUS_GPU_HIGH=stub, CAMINUS_STATE_DIR=os.path.join(S, 'state'))
    genv.pop('CAMINUS_GUARD', None)
    r = subprocess.run(cam('serve', '--model', os.path.join(S, 'no-such.gguf')), capture_output=True, text=True, env=genv, timeout=120)
    log = os.path.join(S, 'state', 'serve.log')
    txt = open(log).read() if os.path.exists(log) else ''
    rocm_torch = torchcheck.status()[0]
    if not rocm_torch:
        print('NOTE cli: no ROCm PyTorch here: serve / replay refuse before the guard (the torch stage checks that); the guard '
              'wrapper checks are skipped', flush=True)
        check('cli: caminus serve without the ROCm PyTorch refuses before the guard', r.returncode == 2 and torchcheck.INSTALL in r.stderr
              and 'under the guard' not in r.stderr, (r.stdout + r.stderr)[-300:])
    elif r.returncode in (97, 98):
        print(f'NOTE cli: the guard refused here (exit {r.returncode}: no gfx1100 card / a recorded fault): the wrapper check is skipped', flush=True)
    else:
        check('cli: caminus serve outside a guard runs under the shipped guard (its log in state_dir; the inner serve refuses the '
              'missing model, exit 2)', r.returncode == 2 and 'under the guard' in r.stderr and 'does not exist' in txt
              and 'rc=2' in r.stdout, (r.stdout + r.stderr + txt)[-400:])
    r = subprocess.run(cam('replay', os.path.join(S, 'no-bundle')), capture_output=True, text=True, env=genv, timeout=120)
    if rocm_torch and r.returncode not in (97, 98):
        check('cli: caminus replay runs under the guard as a child and prints its log', r.returncode == 1 and 'under the guard' in r.stderr
              and 'bundle.json' in r.stdout, (r.stdout + r.stderr)[-300:])
    bad = dict(env, CAMINUS_CONFIG=os.path.join(S, 'bad.toml'))
    open(bad['CAMINUS_CONFIG'], 'w').write('nope = "x"\n')
    r = subprocess.run(cam('download', '--dry-run'), capture_output=True, text=True, env=bad)
    check('cli: a bad config file is one line and exit 2, no traceback', r.returncode == 2 and 'unknown key' in r.stderr and 'Traceback' not in r.stderr, r.stderr[-300:])


# ----------------------------------------------------------------------------------------------------------------- torch
def cam_torch(args, block=True, stub=None):
    """`python -I -m caminus ARGS` with torch blocked (block) or a stub torch package first on sys.path (stub: its directory)"""
    pre = "sys.modules['torch'] = None; " if block else ''
    if stub:
        pre += f'sys.path.insert(0, {stub!r}); '
    return [PY, '-I', '-c', f'import sys; {pre}sys.path.insert(0, {os.path.join(ROOT, "python")!r}); from caminus.__main__ import main; '
            f'sys.exit(main({list(args)!r}))']


def stage_torch(S):
    from caminus import torchcheck
    os.makedirs(S, exist_ok=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', CAMINUS_STATE_DIR=os.path.join(S, 'state'))
    env.pop('CAMINUS_GUARD', None)
    stub = os.path.join(S, 'cuda-torch')
    w(os.path.join(stub, 'torch', '__init__.py'), "__version__ = '2.14.1+cu130'\n\n\nclass version:\n    hip = None\n    cuda = '13.0'\n")
    run = lambda args, e=env, **k: subprocess.run(cam_torch(args, **k), capture_output=True, text=True, env=e, timeout=120)
    clean = lambda r: 'Traceback' not in r.stdout + r.stderr
    nolog = lambda name: not os.path.exists(os.path.join(S, 'state', f'{name}.log'))
    gguf = os.path.join(S, 'no-such.gguf')
    r = run(['serve', '--model', gguf])
    check('torch: caminus serve without torch: the install line, exit 2, the guard never started',
          r.returncode == 2 and torchcheck.INSTALL in r.stderr and 'not installed' in r.stderr and clean(r) and 'under the guard' not in r.stderr
          and nolog('serve'), (r.stdout + r.stderr)[-300:])
    r = run(['serve', '--model', gguf], e=dict(env, CAMINUS_GUARD='1'))
    check('torch: caminus serve under a guard without torch (serve.cli checks without importing torch)',
          r.returncode == 2 and torchcheck.INSTALL in r.stderr and clean(r), (r.stdout + r.stderr)[-300:])
    r = run(['serve', '--model', gguf], block=False, stub=stub)
    check('torch: caminus serve with a CUDA build of torch: refused with the install line, exit 2',
          r.returncode == 2 and 'not a ROCm build' in r.stderr and torchcheck.INSTALL in r.stderr and clean(r) and nolog('serve'),
          (r.stdout + r.stderr)[-300:])
    r = run(['serve', '--help'])
    check('torch: caminus serve --help without torch', r.returncode == 0 and 'usage:' in r.stdout, (r.stdout + r.stderr)[-200:])
    r = run(['replay', os.path.join(S, 'no-bundle')])
    check('torch: caminus replay without torch: the install line, exit 2, no guard',
          r.returncode == 2 and torchcheck.INSTALL in r.stderr and clean(r) and nolog('replay'), (r.stdout + r.stderr)[-300:])
    kv, fl = os.path.join(S, 'kv'), os.path.join(S, 'flight')
    os.makedirs(kv, exist_ok=True); os.makedirs(fl, exist_ok=True)
    r = run(['kv-cache', 'list', '--dir', kv])
    check('torch: caminus kv-cache list without torch: the install line, exit 2',
          r.returncode == 2 and torchcheck.INSTALL in r.stderr and clean(r), (r.stdout + r.stderr)[-300:])
    r = run(['purge', '--kv-dir', kv, '--flight-dir', fl])
    check('torch: caminus purge without torch: the kv part refused with the install line, the flight part still listed, exit 2',
          r.returncode == 2 and torchcheck.INSTALL in r.stderr and 'the flight recorder' in r.stdout and clean(r), (r.stdout + r.stderr)[-300:])
    r = run(['doctor'])
    check('torch: caminus doctor without torch: a FAIL line with the install command, no traceback',
          r.returncode != 0 and 'FAIL' in r.stdout and torchcheck.INSTALL in r.stdout and clean(r), (r.stdout + r.stderr)[-300:])


# ----------------------------------------------------------------------------------------------------------------- repro
def stage_repro(S):
    import gzip, io, tarfile, time, zipfile
    tool = os.path.join(ROOT, 'tools', 'repro_dist.py')
    files = [('pkg-0.1/', None, 0o775), ('pkg-0.1/PKG-INFO', b'Version: 0.1\n', 0o664), ('pkg-0.1/run.sh', b'#!/bin/sh\n', 0o775),
             ('pkg-0.1/' + 'd' * 90 + '/long-name.txt', b'x' * 3000, 0o600)]

    def make(d, mtime, owner, umask_mode, gz_time):
        os.makedirs(d)
        raw = io.BytesIO()
        with gzip.GzipFile(filename='pkg-0.1.tar', mode='wb', fileobj=raw, mtime=gz_time) as g, tarfile.open(fileobj=g, mode='w', format=tarfile.PAX_FORMAT) as t:
            for name, data, mode in files:
                ti = tarfile.TarInfo(name.rstrip('/')); ti.mtime = mtime; ti.uid = ti.gid = owner[0]; ti.uname = ti.gname = owner[1]
                ti.mode = mode & umask_mode
                if data is None:
                    ti.type = tarfile.DIRTYPE; t.addfile(ti)
                else:
                    ti.size = len(data); t.addfile(ti, io.BytesIO(data))
        open(os.path.join(d, 'pkg-0.1.tar.gz'), 'wb').write(raw.getvalue())
        with zipfile.ZipFile(os.path.join(d, 'pkg-0.1-py3-none-any.whl'), 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data, mode in files:
                if data is None:
                    continue
                zi = zipfile.ZipInfo(name.split('/', 1)[1], date_time=time.gmtime(1793976692)[:6]); zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (0o100000 | (mode & umask_mode)) << 16
                z.writestr(zi, data)

    a, b = os.path.join(S, 'a'), os.path.join(S, 'b')
    make(a, 1793976000.25, (1000, 'someone'), 0o775, 1793979999)
    make(b, 1793977777.5, (1234, 'another'), 0o755, 1793988888)
    env = {k: v for k, v in os.environ.items() if k != 'SOURCE_DATE_EPOCH'}
    ra = subprocess.run([PY, '-I', tool, a], capture_output=True, text=True, env=dict(env, SOURCE_DATE_EPOCH='1793976692'))
    rb = subprocess.run([PY, '-I', tool, b, '--epoch', '1793976692'], capture_output=True, text=True, env=env)
    same = lambda n: open(os.path.join(a, n), 'rb').read() == open(os.path.join(b, n), 'rb').read()
    check('repro: two builds with different times, owners, modes and gzip times come out byte-identical',
          ra.returncode == 0 and rb.returncode == 0 and same('pkg-0.1.tar.gz') and same('pkg-0.1-py3-none-any.whl'), (ra.stdout + ra.stderr + rb.stderr)[-300:])
    with tarfile.open(os.path.join(a, 'pkg-0.1.tar.gz')) as t:
        ms = t.getmembers()
        ok = all(m.mtime == 1793976692 and (m.uid, m.gid, m.uname, m.gname) == (0, 0, '', '') and m.mode == (0o755 if m.isdir() or m.name.endswith('.sh') else 0o644)
                 for m in ms) and [m.name for m in ms] == [n.rstrip('/') for n, _, _ in files] and t.extractfile(ms[3]).read() == b'x' * 3000
    gzh = open(os.path.join(a, 'pkg-0.1.tar.gz'), 'rb').read(10)
    check('repro: the sdist\'s headers (one time, 0:0 without names, 0644 / 0755), order and contents; gzip without a name at that time',
          ok and not gzh[3] & 0x08 and int.from_bytes(gzh[4:8], 'little') == 1793976692, gzh.hex())
    before = [open(os.path.join(a, n), 'rb').read() for n in sorted(os.listdir(a))]
    r = subprocess.run([PY, '-I', tool, a, '--epoch', '1793976692'], capture_output=True, text=True, env=env)
    check('repro: a second run changes nothing', r.returncode == 0 and r.stdout.count('(unchanged)') == 2
          and before == [open(os.path.join(a, n), 'rb').read() for n in sorted(os.listdir(a))], r.stdout[-300:])
    r = subprocess.run([PY, '-I', tool, a], capture_output=True, text=True, env=env)
    check('repro: no SOURCE_DATE_EPOCH and no --epoch: refused', r.returncode == 2 and 'SOURCE_DATE_EPOCH' in r.stderr, r.stderr[-200:])
    c = os.path.join(S, 'c'); make(c, 1793976000, (0, ''), 0o755, 0)
    raw = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as g, tarfile.open(fileobj=g, mode='w') as t:
        ti = tarfile.TarInfo('pkg-0.1/link'); ti.type = tarfile.SYMTYPE; ti.linkname = '/etc/passwd'; t.addfile(ti)
    open(os.path.join(c, 'pkg-0.1.tar.gz'), 'wb').write(raw.getvalue())
    before = open(os.path.join(c, 'pkg-0.1-py3-none-any.whl'), 'rb').read()
    r = subprocess.run([PY, '-I', tool, c, '--epoch', '1793976692'], capture_output=True, text=True, env=env)
    check('repro: a symlink member refused, nothing renamed', r.returncode == 1 and 'not a regular file' in r.stderr
          and open(os.path.join(c, 'pkg-0.1.tar.gz'), 'rb').read() == raw.getvalue()
          and open(os.path.join(c, 'pkg-0.1-py3-none-any.whl'), 'rb').read() == before and sorted(os.listdir(c)) == ['pkg-0.1-py3-none-any.whl', 'pkg-0.1.tar.gz'],
          r.stderr[-200:])


STAGES = ('config', 'guard', 'kernels', 'repro', 'download', 'doctor', 'cli', 'torch')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='*', default=None)
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.scratch)
    if S.startswith('/tmp'):
        raise SystemExit('test_packaging: --scratch under /tmp refused (use the job scratch dir)')
    if os.path.exists(S) and os.listdir(S):
        raise SystemExit(f'test_packaging: --scratch {S} is not empty')
    os.makedirs(S, exist_ok=True)
    for st in (a.stages or STAGES):
        print(f'---- {st}', flush=True)
        globals()['stage_' + st](os.path.join(S, st))
    print(f'RESULT {"PASS" if not FAILS else "FAIL"}' + (f' ({len(FAILS)} failed: {FAILS})' if FAILS else ''), flush=True)
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
