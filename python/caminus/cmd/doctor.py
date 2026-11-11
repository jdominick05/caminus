"""caminus doctor: is this machine ready to run Caminus? PASS / WARN / FAIL lines and a RESULT (exit 1 on any FAIL).

  caminus doctor [--verify] [--json] [--gpu]

CPU-safe by default: it reads sysfs and the KFD topology and imports PyTorch, but never initializes HIP (no torch.cuda call),
so it runs beside a server. Checked: Python and the platform, the package and every code object against its provenance, the
dependencies and the vendored gguf-py reader, PyTorch (a ROCm build, else the install command), the gfx1100 cards (count, VRAM
total and in use, the guard's lines, the display-driving card, other GPU processes, the PCIe links), clocks and runtime PM,
the guard (present, its lines == caminus.device.machine, gpu-high, the STOP file, the kernel log readable for the fault watch), host
RAM, the model files (sizes; --verify: sha256, slow), disk space for the caches, port 8092, the configuration in effect.
--gpu (runs itself under the guard): loads the `smoke` code object on each gfx1100 card, runs one bounded saxpy and checks
the result, then peer access between the first two cards."""
import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys

from .. import config, models
from ..device import host, kernels, machine

LEVELS = ('PASS', 'INFO', 'WARN', 'FAIL')


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, area, msg):
        assert level in LEVELS
        self.rows.append((level, area, msg))
        return level

    def result(self):
        lv = [r[0] for r in self.rows]
        return 'FAIL' if 'FAIL' in lv else 'WARN' if 'WARN' in lv else 'PASS'


def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def check_python(R):
    v = sys.version_info
    R.add('PASS' if v >= (3, 12) else 'FAIL', 'python', f'Python {platform.python_version()} ({sys.executable})'
          + ('' if v >= (3, 12) else ': Caminus needs 3.12 or newer'))
    R.add('PASS' if sys.platform.startswith('linux') else 'FAIL', 'platform', f'{platform.system()} {platform.release()} {platform.machine()}'
          + ('' if sys.platform.startswith('linux') else ': Linux only (ROCm, sysfs)'))


def check_package(R):
    import caminus
    R.add('INFO', 'package', f'caminus {getattr(caminus, "__version__", "?")} at {os.path.dirname(caminus.__file__)}')
    d, where = kernels.kernel_dir()
    prov, res = kernels.verify(d)
    if prov is None:
        miss = [n for n in kernels.RUNTIME_SET if not os.path.isfile(os.path.join(d, n + '.hsaco'))]
        R.add('FAIL' if miss else 'WARN', 'kernels', f'{d} ({where}) has no provenance.json'
              + (f'; missing {len(miss)} of the runtime set: {miss[:6]} (./build.sh)' if miss else '; the runtime set is present (built in place)'))
        return
    bad = [(n, dt) for n, ok, dt in res if not ok]
    R.add('FAIL' if bad else 'PASS', 'kernels', f'{len(res) - len(bad)} / {len(res)} code objects match provenance.json in {d} ({where})'
          + (f'; bad: {bad[:4]}' if bad else ''))
    R.add('PASS' if prov.get('arch') == kernels.ARCH else 'FAIL', 'kernels', kernels.describe())
    if prov.get('source', {}).get('dirty'):
        R.add('WARN', 'kernels', 'built from uncommitted kernel sources (provenance source.dirty)')


def check_deps(R):
    for mod in ('numpy', 'jinja2', 'tokenizers'):
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            if mod == 'jinja2':
                from ..serve.template import JINJA2_MIN, jinja2_version_ok
                if not jinja2_version_ok(v):
                    R.add('FAIL', 'deps', f'jinja2 {v}: before {".".join(map(str, JINJA2_MIN))} its sandbox (which renders the model\'s '
                                          f'chat template) has known escapes: pip install "jinja2>={".".join(map(str, JINJA2_MIN))}"')
                    continue
            R.add('PASS', 'deps', f'{mod} {v}')
        except ImportError as e:
            R.add('FAIL', 'deps', f'{mod} not importable ({e}): pip install {mod}')
    try:
        p = config.ensure_gguf()
        import gguf
        ok = hasattr(gguf.GGMLQuantizationType, 'Q2_0') and hasattr(gguf, 'GGUFReader')
        R.add('PASS' if ok else 'FAIL', 'gguf', f'gguf-py reader from {p}' + ('' if ok else ': no Q2_0 type (the Coder needs it): use the vendored copy'))
    except Exception as e:   # noqa: BLE001
        R.add('FAIL', 'gguf', f'the gguf-py reader does not import: {type(e).__name__}: {e}')


def check_torch(R):
    from ..device.torchcheck import status
    ok, ver, detail = status()
    R.add('PASS' if ok else 'FAIL', 'torch', detail)


def kfd_procs():
    root = '/sys/class/kfd/kfd/proc'
    try:
        return sorted(int(p) for p in os.listdir(root) if p.isdigit() and int(p) != os.getpid())
    except OSError:
        return None


def check_gpus(R, root='/'):
    """the host's GPUs (caminus.device.host, D-292): one supported card runs the one-card models, two the two-card ones (the
    pair placement would pick, on adequate links); unsupported cards are listed and not used"""
    gs = host.gpus(root)
    w = [c for c in gs if c['watched']]
    u = host.usable(gs)
    if not gs:
        R.add('FAIL', 'gpu', 'no GPU in /sys/class/drm (the amdgpu driver, ROCm)')
        return w
    if not os.path.isdir(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology')):
        R.add('FAIL', 'gpu', 'no KFD topology (/sys/class/kfd): the amdgpu compute stack is not available')
    R.add('PASS' if u else 'FAIL', 'gpu', f'{len(u)} supported card(s)' + ('' if u else
          ': Caminus runs on AMD RX 7900 XTX / XT (gfx1100) cards; other AMD hardware: please send a hardware report'))
    for c in u:
        used = _read(os.path.join(c['dev'], 'mem_info_vram_used'))
        used = int(used) >> 20 if used else None
        lvl = 'PASS' if used is not None and used < 1024 else 'WARN'
        R.add(lvl, 'gpu', f'{host.describe(c)}: {c["total_mib"]} MiB, {used} MiB in use'
              + (' (something holds VRAM: another engine or test? Caminus needs the cards idle)' if lvl == 'WARN' else '')
              + f'; guard line {c["line_mib"]} MiB' + ('; drives a display (512 MiB kept free below its line)' if c['display'] else '')
              + f'; {host.link_text(c["link"])}' + (f' (slowest hop at {c["link"]["at"]})' if c['link'] and c['link']['at'] != c['pci'] else '')
              + (f'; idle now at {host.link_text(dict(gts=c["link"]["now_gts"], width=c["link"]["now_width"]))} (the driver\'s link power saving)'
                 if host.link_lowered(c['link']) else ''))
    if len(u) == 1:
        R.add('PASS', 'gpu', 'one card: the one-card models run (Qwen3.6, AREX-2); the two-card models (the Coder, Qwen3-Coder-Next) need two')
    elif len(u) >= 2:
        try:
            pr = host.pair(gs)
            from .. import arch
            cs = arch.CODER.split(pr)
            R.add('PASS', 'gpu', f'two-card models on {" + ".join(host.describe(c) for c in pr)} (card 0 first); the Coder\'s split '
                  f'{cs["gdn_keys"]} / {cs["moe_blocks"]}')
            if pr[0]['total_mib'] < 22 * 1024:
                R.add('WARN', 'gpu', 'the Coder needs a 24 GB RX 7900 XTX as card 0 (it holds the bigger share of its split); '
                      'Qwen3-Coder-Next and the one-card models run on this pair')
        except host.PlacementError as e:
            R.add('WARN', 'gpu', f'two-card models: {e}')
    for lv, _, text in host.problems(gs):
        if lv == 'warn' and 'link' not in text:          # (the link rule is on the pair line above)
            R.add('WARN', 'gpu', text)
    for c in gs:
        if c['status'] == 'unsupported' and c['integrated']:
            R.add('INFO', 'gpu', f'{host.describe(c)}: integrated GPU ({c["total_mib"]} MiB), not used')
    ps = kfd_procs()
    if ps:
        names = []
        for p in ps[:6]:
            names.append(f'{p} ({_read(f"/proc/{p}/comm") or "?"})')
        R.add('WARN', 'gpu', f'{len(ps)} other process(es) hold a GPU context: {", ".join(names)} (only one engine at a time)')
    elif ps is not None:
        R.add('PASS', 'gpu', 'no other process holds a GPU context')
    return w


def gpu_high():
    p = os.environ.get('CAMINUS_GPU_HIGH')
    if p:
        return p
    import shutil
    p = shutil.which('gpu-high')
    if p:
        return p
    p = os.path.expanduser('~/.local/bin/gpu-high')
    return p if os.access(p, os.X_OK) else None


def check_clocks(R, w):
    from ..serve.hw import clock_check
    cs = []
    for c in w:
        cs.append(dict(index=len(cs), name=c['label'], pci=c['pci'],
                       perf_level=_read(os.path.join(c['dev'], 'power_dpm_force_performance_level')),
                       runtime_pm=_read(os.path.join(c['dev'], 'power', 'control')),
                       runtime_status=_read(os.path.join(c['dev'], 'power', 'runtime_status'))))
    if not cs:
        return
    chk = clock_check(cs)
    gh = gpu_high()
    for x in chk['cards']:
        R.add('INFO', 'clocks', f"{x['name']}: DPM level {x['perf_level']}, runtime PM {x['runtime_pm']} / {x['runtime_status']}"
              + (' (as gpu-high sets it for a run)' if x['ok'] else ' (outside a run; the guard\'s gpu-high sets high / on for a run)'))
    R.add('PASS' if gh else 'WARN', 'clocks', f'gpu-high: {gh}' if gh else
          'no gpu-high (CAMINUS_GPU_HIGH, PATH, ~/.local/bin): the guard runs at the current clocks; decode is ~13 % slower at DPM '
          'auto, and a runtime suspend drops a forced high level (power/control should be on during a run)')


def klog_readable():
    try:
        r = subprocess.run(['journalctl', '-k', '-b', '-n', '1', '--no-pager', '-q'], capture_output=True, text=True, timeout=20)
        if r.stdout.strip():
            return 'journalctl -k'
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if subprocess.run(['dmesg'], capture_output=True, timeout=20).returncode == 0:
            return 'dmesg'
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def check_guard(R, w):
    from .guarded import GUARD
    if not os.path.isfile(GUARD):
        R.add('FAIL', 'guard', f'{GUARD} missing (a broken install)')
        return
    r = subprocess.run(['bash', GUARD, '--lines'], capture_output=True, text=True, timeout=60)
    got = [x.split()[0] + ':' + x.split()[5] for x in r.stdout.splitlines() if x.strip()]
    want = [f'{c["card"]}:{c["line_mib"]}' for c in w]
    R.add('PASS' if r.returncode == 0 and got == want and want else 'FAIL', 'guard',
          f'{GUARD}: lines {", ".join(got) or "none"}' + ('' if got == want else f' != caminus.device.machine {want}'))
    stop = os.environ.get('K3_GUARD_STOP', os.path.expanduser('~/.local/state/k3/GPU_FAULT_STOP'))
    if os.path.exists(stop):
        R.add('FAIL', 'guard', f'a GPU fault was recorded in {stop}: every guarded run refuses (exit 98); read the kernel-log lines in '
              'it, inspect the fault, then remove the file by hand')
    else:
        R.add('PASS', 'guard', f'no recorded GPU fault ({stop})')
    k = klog_readable()
    R.add('PASS' if k else 'WARN', 'guard', f'the kernel log is readable ({k}): the amdgpu fault watch works' if k else
          'the kernel log is not readable by this user (journalctl -k printed nothing, dmesg is restricted): the guard\'s amdgpu fault '
          'watch would be OFF. Add the user to the systemd-journal (or adm) group, or set kernel.dmesg_restrict=0')


def check_host(R):
    mi = {}
    for line in (_read('/proc/meminfo') or '').splitlines():
        k, _, v = line.partition(':')
        if v.strip().endswith('kB'): mi[k] = int(v.split()[0]) // 1024
    tot, av = mi.get('MemTotal'), mi.get('MemAvailable')
    if tot is None:
        R.add('WARN', 'host', '/proc/meminfo unreadable'); return
    R.add('PASS' if tot >= 28 * 1024 else 'WARN', 'host', f'RAM {tot / 1024:.1f} GiB, {av / 1024:.1f} GiB available'
          + ('' if tot >= 28 * 1024 else ' (tested with 29 GiB; the guard kills below 1,500 MiB available)'))


def shard2(p):
    return p.replace('-00001-of-00002.gguf', '-00002-of-00002.gguf') if '-00001-of-00002.gguf' in p else None


def check_models(R, verify):
    from .download import marker_ok, sha256_file
    paths = [('coder-1', config.resolve('coder')), ('coder-2', None), ('dflash', config.resolve('dflash'))]
    c1 = paths[0][1][0]
    paths[1] = ('coder-2', (shard2(c1), 'next to shard 1') if c1 and shard2(c1) else (None, None))
    for key, (p, where) in paths:
        it = models.BY_KEY[key]
        lvl_missing = 'WARN' if key == 'dflash' else 'FAIL'
        if not p:
            R.add(lvl_missing, 'models', f'{key}: no path'); continue
        if not os.path.isfile(p):
            R.add(lvl_missing, 'models', f'{key}: {p} ({where}) does not exist: `caminus download`'
                  + (' (or set CAMINUS_CODER / coder in config.toml)' if key == 'coder-1' else ' (or set CAMINUS_DFLASH; without it: --no-drafter)' if key == 'dflash' else ''))
            continue
        sz = os.path.getsize(p)
        same_name = os.path.basename(p) == os.path.basename(it.path)
        if sz != it.size:
            R.add('WARN' if not same_name else 'FAIL', 'models', f'{key}: {p}: {sz} bytes, the release was tested with {it.size}'
                  + ('' if same_name else ' (another file name: a file of your own?)'))
            continue
        if verify:
            got = sha256_file(p).hexdigest()
            R.add('PASS' if got == it.sha256 else 'FAIL', 'models', f'{key}: {p}: sha256 ' + ('matches' if got == it.sha256 else f'{got} != {it.sha256}'))
        else:
            R.add('PASS', 'models', f'{key}: {p} ({where}): size matches' + (', sha256 verified by caminus download' if marker_ok(p, it.sha256) else ' (--verify hashes it)'))


def check_disk(R):
    from .download import free_bytes
    for key in ('kv_dir', 'flight_dir', 'model_dir'):
        p = config.get(key)
        try:
            fr = free_bytes(p) / 1e9
        except OSError as e:
            R.add('WARN', 'disk', f'{key} {p}: {e}'); continue
        if key == 'model_dir':   # (the files caminus download would still fetch into it)
            where = {'coder-1': config.coder(), 'coder-2': shard2(config.coder() or ''), 'dflash': config.dflash()}
            miss = [models.BY_KEY[k] for k in models.BY_NAME['coder'].items if not (where.get(k) and os.path.isfile(where[k]))]   # (the default download: the Coder's files)
            need = sum(it.size for it in miss) / 1e9 + 1
            if not miss:
                R.add('PASS', 'disk', f'{key} {p}: {fr:.0f} GB free (every model file present)'); continue
        else:
            need = 10
        R.add('PASS' if fr >= need else 'WARN', 'disk', f'{key} {p}: {fr:.0f} GB free'
              + ('' if fr >= need else f' (below {need:.0f} GB: ' + ('the disk tier keeps a 10 GB floor' if key == 'kv_dir' else
                 'caminus download needs the missing files\' size + 1 GB there' if key == 'model_dir' else 'the flight recorder caps at 10 GB') + ')'))


def check_port(R, port=8092):
    s = socket.socket()
    try:
        s.bind(('127.0.0.1', port))
        R.add('PASS', 'port', f'127.0.0.1:{port} is free')
    except OSError:
        R.add('WARN', 'port', f'127.0.0.1:{port} is in use (a caminus serve already running? or pass --port)')
    finally:
        s.close()


def check_config(R):
    p = config.config_path()
    R.add('INFO', 'config', f'{p} ' + ('(in use)' if os.path.isfile(p) else '(none: defaults and the environment)'))
    for k, (v, wh) in config.summary().items():
        R.add('INFO', 'config', f'{k} = {v} ({wh})')


def gpu_probe(R):
    """the GPU path (under the guard): the smoke object on each gfx1100 card, one saxpy, then peer access"""
    u = [c for c in host.usable(host.gpus()) if c['hip'] is not None]
    if 'HIP_VISIBLE_DEVICES' not in os.environ:
        os.environ['HIP_VISIBLE_DEVICES'] = ','.join(str(c['hip']) for c in u)   # (no context on another GPU, e.g. the iGPU)
    import ctypes
    import time
    import torch
    from ..device.hip import Module
    n = 1 << 20
    devs = [d for d in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(d).gcnArchName.startswith('gfx1100')]
    if not devs:
        R.add('FAIL', 'gpu-run', 'no gfx1100 device visible to HIP'); return
    for d in devs:
        with torch.cuda.device(d):
            t0 = time.perf_counter()
            m = Module('smoke')
            x = torch.arange(n, dtype=torch.float32, device=f'cuda:{d}'); y = torch.ones(n, device=f'cuda:{d}')
            m.saxpy(((n + 255) // 256,), (256,), y, x, ctypes.c_float(2.0), ctypes.c_int(n))
            torch.cuda.synchronize(d)
            ok = bool(torch.equal(y.cpu(), 2.0 * torch.arange(n, dtype=torch.float32) + 1.0))
            R.add('PASS' if ok else 'FAIL', 'gpu-run', f'cuda:{d} {torch.cuda.get_device_name(d)}: smoke.hsaco loaded, saxpy of {n} floats '
                  + ('exact' if ok else 'WRONG') + f' ({(time.perf_counter() - t0) * 1e3:.0f} ms with the load)')
            del x, y, m
    for i, a in enumerate(devs):          # every pair a two-card model could get (caminus.device.host.pair picks one of them)
        for b in devs[i + 1:]:
            p = torch.cuda.can_device_access_peer(a, b) and torch.cuda.can_device_access_peer(b, a)
            R.add('PASS' if p else 'WARN', 'gpu-run', f'peer access cuda:{a} <-> cuda:{b}: {"yes" if p else "no: two-card models cannot use this pair (the TP exchange needs P2P; another slot or the BIOS IOMMU / ACS settings)"}')


def run(verify=False, gpu=False):
    R = Report()
    steps = [check_python, check_package, check_deps, check_torch]
    for f in steps:
        try: f(R)
        except Exception as e:   # noqa: BLE001 - a broken check is a finding, not a crash
            R.add('FAIL', f.__name__[6:], f'the check failed: {type(e).__name__}: {e}')
    try:
        w = check_gpus(R)
    except Exception as e:   # noqa: BLE001
        R.add('FAIL', 'gpu', f'the check failed: {type(e).__name__}: {e}'); w = []
    for f, args in ((check_clocks, (R, w)), (check_guard, (R, w)), (check_host, (R,)), (check_models, (R, verify)),
                    (check_disk, (R,)), (check_port, (R,)), (check_config, (R,))):
        try: f(*args)
        except Exception as e:   # noqa: BLE001
            R.add('FAIL', f.__name__[6:], f'the check failed: {type(e).__name__}: {e}')
    if gpu:
        try: gpu_probe(R)
        except Exception as e:   # noqa: BLE001
            R.add('FAIL', 'gpu-run', f'{type(e).__name__}: {e}')
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus doctor', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verify', action='store_true', help='hash the model files (slow: ~1 min per 30 GB)')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--gpu', action='store_true', help='also launch the smoke kernel on each gfx1100 card (under the guard)')
    a = ap.parse_args(argv)
    if a.gpu:
        from .guarded import under_guard, run_guarded
        if not under_guard():
            return run_guarded('doctor', list(argv if argv is not None else sys.argv[1:]), 300)
    R = run(verify=a.verify, gpu=a.gpu)
    res = R.result()
    if a.json:
        print(json.dumps(dict(result=res, checks=[dict(level=l, area=ar, message=m) for l, ar, m in R.rows]), indent=1))
    else:
        for l, ar, m in R.rows:
            print(f'{l:4s}  {ar:8s} {m}')
        print(f'RESULT {res}')
    return 1 if res == 'FAIL' else 0
