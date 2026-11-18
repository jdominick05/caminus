"""CPU tests of caminus wizard (caminus.cmd.wizard) on fake hosts: nothing touches a GPU, the network or a real server.

  python -I tests/cmd/test_wizard.py --scratch DIR     RESULT PASS

The five setups the wizard must handle (one XTX, one XT, XTX + XT, an untested gfx1100 card, an unsupported card) are
synthetic sysfs trees; `--yes --dry-run` runs on each, then the interactive paths with answers fed in. Every caminus
command the wizard runs goes through one seam (wizard.run_cmd), replaced here by a recorder that runs nothing; process
launches, downloads and exec are replaced by refusers besides. State, models, config and the Qwen home are scratch
directories (never ~/.config/caminus, ~/.qwen or the model directory)."""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, '..', '..', 'python')
sys.path.insert(0, PY)
os.environ.pop('CAMINUS_ALLOW_PAUSED', None)      # (the catalog as users see it: the paused models left out)
FAILS = []
GIB = 1 << 30


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


# a card: (PCI device id, VRAM GiB, gfx_target_version)
XTX, XT = ('0x744c', 24, 110000), ('0x744c', 20, 110000)
W7900, GRE = ('0x7448', 48, 110000), ('0x744c', 16, 110000)
RX7800 = ('0x747e', 16, 110001)


def host_tree(root, cards, slow=()):
    """a minimal sysfs: each card's endpoint on a PCIe 4.0 x16 hop (a card index in slow: PCIe 1.0 x1), its DRM node, its
    KFD node (card 1 drives the display)"""
    if os.path.isdir(root):
        shutil.rmtree(root)
    w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', '0', 'properties'), 'simd_count 0\n')
    for i, (did, gib, gfx) in enumerate(cards):
        slot, minor, name = f'0000:{3 + 3 * i:02x}:00.0', 128 + i, f'card{i + 1}'
        ep = os.path.join(root, 'sys', 'devices', 'pci0000:00', f'0000:00:01.{i + 1}', slot)
        gts, width = ('2.5', '1') if i in slow else ('16.0', '16')
        for k, v in (('current_link_speed', f'{gts} GT/s PCIe'), ('current_link_width', width), ('max_link_speed', '16.0 GT/s PCIe'),
                     ('max_link_width', '16'), ('vendor', '0x1002'), ('device', did), ('mem_info_vram_total', str(gib * GIB)),
                     ('mem_info_vram_used', str(100 << 20)), ('boot_vga', '1' if i == 0 else '0'),
                     ('uevent', f'DRIVER=amdgpu\nPCI_SLOT_NAME={slot}')):
            w(os.path.join(ep, k), v + '\n')
        os.makedirs(os.path.join(ep, 'drm', f'renderD{minor}'), exist_ok=True)
        os.makedirs(os.path.join(root, 'sys', 'class', 'drm', name), exist_ok=True)
        os.symlink(ep, os.path.join(root, 'sys', 'class', 'drm', name, 'device'))
        w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', str(i + 1), 'properties'),
          f'simd_count 96\ngfx_target_version {gfx}\ndrm_render_minor {minor}\n')
    return root


class Runs:
    """the seam: records each caminus command the wizard runs; rc per command name (default 0); runs nothing"""

    def __init__(self, rc=None):
        self.calls, self.rc = [], rc or {}

    def __call__(self, argv):
        self.calls.append(list(argv))
        return self.rc.get(' '.join(argv), self.rc.get(argv[0], 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.scratch)
    for bad in (os.path.expanduser('~/.config'), os.path.expanduser('~/.qwen'), os.path.expanduser('~/.local')):
        if S.startswith(bad):
            raise SystemExit(f'--scratch must not be under {bad}')
    os.makedirs(S, exist_ok=True)
    md, st = os.path.join(S, 'models'), os.path.join(S, 'state')
    os.environ.update(CAMINUS_MODEL_DIR=md, CAMINUS_STATE_DIR=st, CAMINUS_KV_DIR=os.path.join(S, 'kv'),
                      CAMINUS_FLIGHT_DIR=os.path.join(S, 'flight'), CAMINUS_QWEN_HOME=os.path.join(S, 'qwen-home'),
                      CAMINUS_CONFIG=os.path.join(S, 'none.toml'), XDG_DATA_HOME=os.path.join(S, 'data'),
                      XDG_CONFIG_HOME=os.path.join(S, 'config'), HF_TOKEN='never-logged-sentinel')
    for k in ('CAMINUS_CODER', 'K3_GGUF', 'CAMINUS_DFLASH', 'CAMINUS_MODEL_SEARCH', 'NO_COLOR'):
        os.environ.pop(k, None)
    from caminus import models
    from caminus.cmd import doctor, download, manage, qwen_code, wizard

    # ---- nothing may launch, download or exec: refusers under the seam
    def refuse(what):
        def f(*a, **k):
            raise RuntimeError(f'test_wizard: {what} was attempted')
        return f

    class NoLaunch:
        DEVNULL, STDOUT, PIPE = subprocess.DEVNULL, subprocess.STDOUT, subprocess.PIPE
        Popen = staticmethod(refuse('a server launch'))
        run = staticmethod(refuse('a subprocess'))
        TimeoutExpired = subprocess.TimeoutExpired
    manage.subprocess = qwen_code.subprocess = NoLaunch
    download.fetch = refuse('a download')
    real_run_cmd = wizard.run_cmd
    os.execve = refuse('an exec')
    manage.unmanaged_servers = lambda proc='/proc': []     # (no /proc scan, no request to a live server on this machine)
    doctor.kfd_procs = lambda: []
    download.free_bytes = lambda p: 10 ** 15
    qwen_code.binary = lambda: None
    wizard.node_missing = lambda: ''

    def go(cards, tag, yes=True, dry=True, answers=(), rc=None, slow=()):
        manage.ROOT = host_tree(os.path.join(S, 'hosts', tag), cards, slow)
        runs = Runs(rc)
        wizard.run_cmd = runs
        it = iter(answers)

        def ask(prompt):
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None
        out = io.StringIO()
        code = wizard.Wizard(yes=yes, dry=dry, out=out, ask=ask).main()
        return code, runs.calls, out.getvalue()

    # ---- the setup table: every name a catalog model, two-card forms only where the registry has one
    names = {m.name for m in models.CATALOG}
    two = {m.name for m in models.CATALOG if any(x['cards'] == 2 for x in manage.serving_modes(m)) and m.cards == 1}
    bad = [(s['row'], r) for s in models.SETUPS for r in s['runs']
           if r.split(':')[0] not in names or (r.endswith(':2') and r[:-2] not in two)]
    check(not bad and all(s['start'] in s['runs'] for s in models.SETUPS)
          and all(set(s['cards']) <= {'XTX', 'XT'} for s in models.SETUPS),
          f'SETUPS: catalog names, two-card forms the registry has, the start model among what runs ({bad})')
    cls = lambda *mib: models.setup_of([dict(total_mib=x) for x in mib])
    check(cls(24560)['start'] == 'swift-1.5' and cls(20464)['start'] == 'thinkingcap' and cls(24560, 20464)['start'] == 'coder'
          and cls(20464, 20464)['row'] == 'Two XT' and cls(49136)['row'] == 'One XTX' and cls(24560, 16368)['row'] == 'One XTX'
          and cls(16368) is None and cls(24560, 20464, 20464)['row'] == 'XTX + XT',
          'setup_of: by VRAM class (a W7900 as a 24 GB card, a 16 GB card no class, three cards: the first two)')

    # ---- --yes --dry-run on the five setups: read-only commands run, the rest printed
    for cards, tag, start in (([XTX], 'xtx', 'swift-1.5'), ([XT], 'xt', 'thinkingcap'), ([XTX, XT], 'xtx-xt', 'coder')):
        code, calls, out = go(cards, tag)
        check(code == 0 and calls == [['doctor'], ['pull', start, '--dry-run']]
              and f'$ caminus start {start}\n  (dry run: not run)' in out and '$ caminus doctor --gpu\n  (dry run: not run)' in out
              and f'$ caminus qwen-code setup --server {start} --install' in out and f'$ caminus qwen --server {start}' in out
              and f'* {start} ' in out and 'Step 6/6' in out and '\033[' not in out,
              f'--yes --dry-run, {tag}: {start} preselected; doctor and pull --dry-run run, the rest printed ({calls})')
    check('coder: 2 cards (XTX card1 (0000:03:00.0) + XT card2 (0000:06:00.0))' in go([XTX, XT], 'xtx-xt')[2],
          'XTX + XT: the summary line names the pair, card 0 first')
    code, calls, out = go([W7900], 'w7900')
    check(code == 0 and 'W7900 card1 (0000:03:00.0): not tested with Caminus yet' in out and wizard.REPORT in out
          and 'Your cards: W7900' in out and 'gfx1100, 49,152 MiB' in out and '* swift-1.5' in out,
          'an untested gfx1100 card (W7900): used, the hardware report asked for with a pasteable summary, a 24 GB setup row')
    code, calls, out = go([RX7800], 'rx7800')
    check(code == 1 and calls == [['doctor']] and 'no supported GPU' in out and wizard.REPORT in out and 'Step 2/6' not in out,
          f'an unsupported card (gfx1101): stops after caminus doctor, the report link given ({calls})')
    code, calls, out = go([GRE], 'gre')
    check(code == 1 and 'no measured setup' in out and 'without --yes and choose one' in out,
          'a 16 GB card (no setup row): every model offered as untested; --yes stops instead of guessing')

    code, calls, out = go([XTX, XT], 'slow', slow=(1,))
    check(code == 0 and '* swift-1.5' in out and 'two-card models: ' in out and 'PCIe 1.0 x1' in out
          and 'does not run on these cards' in out and calls[-1] == ['pull', 'swift-1.5', '--dry-run'],
          'XTX + XT with the XT on a PCIe 1.0 x1 link: no pair, so the one-XTX setup applies (Swift 1.5), said why')
    kfd0 = wizard.kfd_closed
    wizard.kfd_closed = lambda root: '/dev/kfd'
    code, calls, out = go([XTX], 'kfd', dry=False)
    check(code == 1 and calls == [['doctor']] and wizard.GROUPS in out and 'Step 2/6' not in out,
          '/dev/kfd closed to the user: the render-group command printed (not run), the wizard stops')
    wizard.kfd_closed = kfd0

    # ---- the real flow (the seam records, nothing runs): XTX + XT, defaults taken, Qwen Code not opened
    code, calls, out = go([XTX, XT], 'run', yes=False, dry=False, answers=['', 'n'])
    check(code == 0 and calls == [['doctor'], ['doctor', '--gpu'], ['pull', 'coder'], ['start', 'coder'],
                                  ['qwen-code', 'setup', '--server', 'coder', '--install']] and 'Later: caminus qwen --server coder' in out,
          f'a run: doctor, doctor --gpu, pull, start, qwen-code setup --install, in that order ({calls})')
    code, calls, _ = go([XTX, XT], 'open', yes=False, dry=False, answers=['', 'y'])
    check(calls[-1] == ['qwen', '--server', 'coder'], 'Open Qwen Code now? y: caminus qwen on that server')
    code, calls, out = go([XT], 'pick', yes=False, dry=False, answers=['coder', '4', 'n'])
    check(code == 0 and ['start', 'qwen3.8-27b'] in calls and "'coder': not one of the choices" in out,
          'one XT: a model that does not run there is refused, a number picks from the list')
    code, calls, _ = go([XT, XT], 'two-xt', yes=False, dry=False, answers=['swift-1.5', 'n'])
    check(['start', 'swift-1.5', '--cards', '2'] in calls, 'two XT: Swift 1.5 runs only in its two-card form: --cards 2')
    code, calls, out = go([XTX, XT], 'eof', yes=False, dry=False, answers=[])
    check(code == 1 and 'no answer' in out and ['pull', 'coder'] not in calls, 'end of input at a question: stopped cleanly')

    # ---- refusals and skips
    code, calls, out = go([XTX], 'doctor-fail', dry=False, rc={'doctor': 1})
    check(code == 1 and calls == [['doctor']] and 'found a FAIL' in out, 'a caminus doctor FAIL stops the wizard (a dry run too)')
    check(go([XTX], 'doctor-fail-dry', rc={'doctor': 1})[0] == 1, 'a FAIL stops a dry run as well')
    code, calls, out = go([XTX], 'gpu-fail', dry=False, rc={'doctor --gpu': 1})
    check(code == 1 and calls[-1] == ['doctor', '--gpu'], 'a caminus doctor --gpu failure stops the wizard')
    doctor.kfd_procs = lambda: [4242]
    code, calls, out = go([XTX], 'held', dry=False)
    check(['doctor', '--gpu'] not in calls and 'waits for idle cards: 1 process(es)' in out and 'skipped' in out,
          'another process holds a GPU context: doctor --gpu is skipped, said so')
    doctor.kfd_procs = lambda: []
    download.free_bytes = lambda p: 5 * 10 ** 9
    code, calls, out = go([XTX, XT], 'disk', dry=False)
    check(code == 1 and ['pull', 'coder'] not in calls and 'not enough disk' in out and 'GB to download into ' + md in out,
          'a download that does not fit the disk: refused before caminus pull, with the size, the folder and the free space')
    download.free_bytes = lambda p: 10 ** 15
    for k in models.BY_NAME['thinkingcap'].items:          # (sparse files of the manifest's sizes: on disk)
        it = models.BY_KEY[k]
        p = os.path.join(md, models.local_rel(it))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.truncate(it.size)
    code, calls, out = go([XT], 'present', dry=False)
    tc = os.path.join(md, models.local_rel(models.BY_KEY['thinkingcap']))
    dfl = os.path.join(md, models.local_rel(models.BY_KEY['dflash2-27b']))
    check(code == 0 and not any(c[0] == 'pull' for c in calls) and 'on disk already' in out
          and calls[2:4] == [['fit', '--model', tc, '--cards', '1', '--drafter', dfl], ['start', 'thinkingcap']],
          f'files on disk: no pull (re-runnable); caminus fit on the files start serves, then the start ({calls[2:4]})')
    code, calls, out = go([XT], 'fit-fail', dry=False, rc={'fit': 1})
    check(code == 0 and ['start', 'thinkingcap'] in calls and 'caminus start checks again' in out,
          'a caminus fit that cannot project is said and does not stop the start (start checks again)')

    # a running server: kept unless the user switches (manage.alive: the guard pid 4242 counts as alive)
    os.makedirs(os.path.join(st, 'servers', 'thinkingcap'), exist_ok=True)
    with open(os.path.join(st, 'servers', 'thinkingcap', 'server.json'), 'w') as f:
        json.dump(dict(name='thinkingcap', model='thinkingcap', port=8092, guard_pid=4242, card_names=['XT'],
                       log=os.path.join(S, 'x.log'), devices=[1]), f)
    alive0 = manage.alive
    manage.alive = lambda pid: pid == 4242
    code, calls, out = go([XTX, XT], 'keep', dry=False)
    check(code == 0 and not any(c[0] in ('start', 'stop') for c in calls) and ['doctor', '--gpu'] not in calls
          and calls[-1] == ['qwen-code', 'setup', '--server', 'thinkingcap', '--install'] and 'K/s] k (--yes)' in out,
          '--yes with another model running: kept, no start, Qwen Code set up for the running server')
    code, calls, _ = go([XTX, XT], 'switch', yes=False, dry=False, answers=['', 's', 'n'])
    check(calls[-3:] == [['stop', 'thinkingcap'], ['start', 'coder'], ['qwen-code', 'setup', '--server', 'coder', '--install']],
          f'switch: caminus stop, then caminus start ({calls[-3:]})')
    code, calls, out = go([XT], 'same', dry=False)
    check(code == 0 and not any(c[0] == 'start' for c in calls) and 'runs thinkingcap already' in out,
          'the chosen model runs already: kept, no start')
    qwen_code.binary, qs0 = (lambda: '/opt/qwen'), qwen_code.status
    qwen_code.status = lambda: dict(target='thinkingcap')
    code, calls, out = go([XT], 'qwen-done', dry=False)
    check(not any(c[0] == 'qwen-code' for c in calls) and 'set up for thinkingcap already' in out,
          'Qwen Code set up for the server already: no second setup (each writes a backup)')
    qwen_code.binary, qwen_code.status = (lambda: None), qs0
    manage.alive = alive0
    manage.unmanaged_servers = lambda proc='/proc': [dict(name='observed-8092', url='http://127.0.0.1:8092', model='x')]
    code, calls, out = go([XTX, XT], 'outside', dry=False)
    check(code == 0 and not any(c[0] in ('start', 'stop', 'qwen-code', 'qwen') for c in calls) and 'started outside caminus start' in out
          and 'step 4' in out and 'step 5' in out, 'a server started outside caminus start: kept, steps 4 and 5 skipped with the reason')
    manage.unmanaged_servers = lambda proc='/proc': []
    wizard.node_missing = lambda: 'Node.js 18'
    code, calls, out = go([XTX, XT], 'no-node', dry=False)
    check(code == 0 and not any(c[0] in ('qwen-code', 'qwen') for c in calls) and wizard.NODE_URL in out
          and 'caminus qwen-code setup --server coder --install' in out and 'sudo' not in out,
          'no Node.js 22: the step is skipped with the install link and the command to run after (no sudo run or printed)')
    wizard.node_missing = lambda: ''

    # ---- the log: the lines and exit codes, never a key
    log = open(os.path.join(st, 'wizard.log')).read()
    check('$ caminus doctor' in log and '-> exit 0' in log and 'never-logged-sentinel' not in log and '\033[' not in log,
          'the log: commands and exit codes, no colour codes, no token')

    # ---- the real seam (caminus.__main__.dispatch): a read-only command runs, a bad one returns its error code
    check(real_run_cmd(['pull', 'thinkingcap', '--dry-run']) == 0 and real_run_cmd(['pull', 'no-such-model']) == 2,
          'run_cmd: caminus pull --dry-run through dispatch exits 0; an unknown model, argparse\'s 2')

    # ---- the CLI knows the command (no torch import for its help)
    r = subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); from caminus.__main__ import main;'
                        ' sys.exit(main(["wizard", "--help"]))', PY], capture_output=True, text=True, env=os.environ, timeout=120)
    check(r.returncode == 0 and '--dry-run' in r.stdout and '--yes' in r.stdout, 'caminus wizard --help')

    print('RESULT PASS' if not FAILS else f'RESULT FAIL ({len(FAILS)})')
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
