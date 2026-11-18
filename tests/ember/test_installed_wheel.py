"""caminus ember from an installed wheel, with no source checkout: what `pip install caminus` gives a new user.

  python -I tests/ember/test_installed_wheel.py --scratch DIR

Builds a wheel from a copy of this tree (no build isolation: needs setuptools; SKIP without it), checks it is the x86-64
platform wheel setup.py's WHEEL_TAG sets (never any / purelib: the panel is an x86-64 binary), installs it into a fresh
virtual environment with no dependencies, and from a folder outside the checkout checks that `caminus ember` works:
the commands parse, ROBOTS.md ships as package data, the setup script is generated, and the resident's zipapp builds
from the installed copy, runs, and holds only the resident package. Nothing here needs an agent account, a network,
a GPU or root. The wheel build refuses to package without the GPU code objects (a wheel without them would fail at
the first launch), so the copy stages synthetic, non-executable ones that match their provenance, as
tests/tools/test_serve_smoke_installed.py does: the env side never launches one.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []


def check(ok, what, detail=''):
    print(('PASS ' if ok else 'FAIL ') + what + (': ' + detail if detail and not ok else ''), flush=True)
    if not ok:
        FAILS.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    if importlib.util.find_spec('setuptools') is None:
        print('SKIP: setuptools is not installed (needed to build the wheel without isolation)')
        print('RESULT PASS')
        return 0
    S = os.path.abspath(a.scratch)
    os.makedirs(S, exist_ok=True)
    wheels, venv, away = os.path.join(S, 'wheels'), os.path.join(S, 'venv'), os.path.join(S, 'elsewhere')
    os.makedirs(away, exist_ok=True)
    tree = os.path.join(S, 'tree')
    shutil.rmtree(tree, ignore_errors=True)
    files = subprocess.run(['git', '-C', ROOT, 'ls-files', 'pyproject.toml', 'setup.py', 'README.md', 'LICENSE*',
                            'MANIFEST.in', 'python'], capture_output=True, text=True).stdout.split()
    for f in files:
        if '/hsaco/' in f or not os.path.isfile(os.path.join(ROOT, f)):
            continue
        os.makedirs(os.path.dirname(os.path.join(tree, f)) or tree, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), os.path.join(tree, f))
    for f in ('python/caminus/share/harness/ember/ROBOTS.md',):      # package data the copy must carry
        check(os.path.isfile(os.path.join(tree, f)), 'the tree copy has ' + f)
    sys.path.insert(0, os.path.join(tree, 'python'))
    from caminus.device import kernels
    sys.path.pop(0)
    staged = os.path.join(tree, 'python', 'caminus', 'hsaco', 'gfx1100')
    os.makedirs(staged, exist_ok=True)
    objects = {}
    for n in kernels.RUNTIME_SET:
        data = ('synthetic, not executable: ' + n).encode()
        with open(os.path.join(staged, n + '.hsaco'), 'wb') as fh:
            fh.write(data)
        objects[n + '.hsaco'] = dict(sha256=hashlib.sha256(data).hexdigest())
    with open(os.path.join(staged, 'provenance.json'), 'w') as fh:
        json.dump(dict(objects=objects), fh)
    r = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '-q', '-w', wheels,
                        tree], capture_output=True, text=True, timeout=900)
    built = [f for f in os.listdir(wheels) if f.endswith('.whl')] if os.path.isdir(wheels) else []
    check(r.returncode == 0 and len(built) == 1, 'a wheel builds from the tree', (r.stderr or r.stdout)[-800:])
    if not built:
        print('RESULT FAIL (%d failed)' % len(FAILS))
        return 1
    wheel = os.path.join(wheels, built[0])
    with zipfile.ZipFile(wheel) as z:
        names = set(z.namelist())
        wn = [n for n in names if n.endswith('.dist-info/WHEEL')]
        wf = z.read(wn[0]).decode() if wn else ''
    # a platform wheel: the panel it ships is an x86-64 binary, so pip must not offer it to aarch64 or macOS
    plat = 'py3-none-manylinux_2_17_x86_64'
    check(built[0].endswith('-' + plat + '.whl'), 'the wheel is tagged ' + plat, built[0])
    check('Root-Is-Purelib: false' in wf.splitlines() and [l for l in wf.splitlines() if l.startswith('Tag:')] == ['Tag: ' + plat],
          'its WHEEL says Root-Is-Purelib: false and the one platform tag (never any / purelib)', wf)
    check(not any('.data/' in n for n in names), 'the package sits at the wheel\'s root (no .data/purelib or platlib tree)')
    check('caminus/share/harness/ember/ROBOTS.md' in names, 'the wheel ships ROBOTS.md')
    check(all(f'caminus/ember/{m}.py' in names for m in ('provision', 'runner', 'relay', 'agent_host')) and
          'caminus/ember/resident/runtime.py' in names, 'the wheel ships the owner side and the resident')
    subprocess.run([sys.executable, '-m', 'venv', venv], check=True, timeout=300)
    py = os.path.join(venv, 'bin', 'python')
    r = subprocess.run([py, '-m', 'pip', 'install', '--no-deps', '-q', wheel], capture_output=True, text=True, timeout=600)
    check(r.returncode == 0, 'the wheel installs into a fresh venv', (r.stderr or '')[-800:])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CAMINUS_', 'PYTHON'))}
    env.update(HOME=os.path.join(S, 'home'), XDG_DATA_HOME=os.path.join(S, 'home', 'data'),
               XDG_STATE_HOME=os.path.join(S, 'home', 'state'), XDG_CONFIG_HOME=os.path.join(S, 'home', 'config'),
               CAMINUS_ENV_KEY=os.path.join(S, 'home', 'key'))
    os.makedirs(env['HOME'], exist_ok=True)

    def run(*args, **kw):
        return subprocess.run([py, '-I', *args], cwd=away, env=env, capture_output=True, text=True, timeout=120, **kw)

    r = run('-c', 'import caminus, os; print(os.path.realpath(caminus.__file__))')
    where = r.stdout.strip()
    check(r.returncode == 0 and where.startswith(os.path.realpath(venv)) and not where.startswith(os.path.realpath(ROOT) + '/python'),
          'caminus imports from the venv, not the checkout', where or r.stderr[-300:])
    r = run('-m', 'caminus', 'env', '--help')
    check(r.returncode == 0 and 'setup' in r.stdout and 'agent' in r.stdout, 'caminus ember --help', r.stderr[-300:])
    r = run('-m', 'caminus', 'env', 'prompt')
    check(r.returncode == 0 and len(r.stdout) > 500, 'caminus ember prompt prints ROBOTS.md from the package', r.stderr[-300:])
    if subprocess.run(['sh', '-c', 'command -v ssh-keygen'], capture_output=True).returncode == 0:
        r = run('-m', 'caminus', 'env', 'setup', '--print')
        check(r.returncode == 0 and 'set -eu' in r.stdout and 'leases' in r.stdout and os.path.exists(env['CAMINUS_ENV_KEY']),
              'caminus ember setup --print: the key made, the root script printed', r.stderr[-300:])
        if r.returncode == 0:
            check(subprocess.run(['sh', '-n'], input=r.stdout, text=True).returncode == 0, 'the printed script parses')
    else:
        print('SKIP setup --print: ssh-keygen is not installed')
    pyz = os.path.join(S, 'resident.pyz')
    r = run('-c', 'import sys; from caminus.ember import agent_host; open(sys.argv[1], "wb").write(agent_host.bundle())', pyz)
    check(r.returncode == 0 and os.path.exists(pyz), 'the resident zipapp builds from the installed copy', r.stderr[-300:])
    if os.path.exists(pyz):
        with zipfile.ZipFile(pyz) as z:
            inside = z.namelist()
        check(all(n == '__main__.py' or n in ('caminus/__init__.py', 'caminus/ember/__init__.py') or
                  n.startswith('caminus/ember/resident/') for n in inside), 'the zipapp holds only the resident package')
        r = subprocess.run([sys.executable, '-I', pyz, '--help'], cwd=away, env=env, capture_output=True, text=True,
                           timeout=60)
        check(r.returncode == 0 and 'once' in r.stdout, 'the zipapp runs (--help)', r.stderr[-300:])
    print('RESULT ' + ('PASS' if not FAILS else 'FAIL (%d failed)' % len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
