"""The agent-side half of the environment (docs/specs/spec-ember.md sections 4 and 10): standalone stdlib-only
scripts that `caminus ember setup --agent-side` installs into the agent user's ~/bin. They talk only to the bridge
directory. BRIDGE is baked in at install time.

  ask-root CMD...        ask the human to run CMD as root (exit 0/rc, 126 denied, 124 timeout)
  env-ask TEXT CHOICE... ask the human a question, optionally with choices (prints the answer)

Setup also fills the agent's own venv (fill_venv: venv_command run as the agent, read by venv_report): its package
set, and CPU torch with --test-venv.
"""
import base64
import json
import os
import subprocess

from . import runner
from .resident.discovery import AGENT_PACKAGES, AGENT_VENV

_COMMON = '''
import json, os, sys, time, uuid

BRIDGE = {bridge!r}
TIMEOUT = float(os.environ.get('CAMINUS_ENV_TIMEOUT', '3600'))


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.' + uuid.uuid4().hex
    try:
        with open(tmp, 'x') as f:
            os.fchmod(f.fileno(), 0o660)
            os.fchown(f.fileno(), -1, os.stat(os.path.dirname(path)).st_gid)
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _ask(kind, extra):
    os.umask(0o007)
    rid = uuid.uuid4().hex[:16]
    req = {'id': rid, 'pid': os.getpid(), 'user': os.environ.get('USER', ''),
           'cwd': os.getcwd(), 'kind': kind, 'ts': time.time()}
    req.update(extra)
    _write(os.path.join(BRIDGE, 'req', rid + '.json'), req)
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        p = os.path.join(BRIDGE, 'res', rid + '.json')
        if os.path.exists(p):
            with open(p) as f:
                result = json.load(f)
            if result.get('decision') != 'executing':
                return result
        time.sleep(0.2)
    print(f'no answer within {TIMEOUT:g}s (request {rid})', file=sys.stderr)
    return None
'''

ASK_ROOT_SOURCE = '''#!/usr/bin/env python3
"""ask-root CMD... : ask the human to run CMD as root (caminus ember ui / watch answers)."""
''' + _COMMON + '''

def main():
    cmd = ' '.join(sys.argv[1:])
    if not cmd:
        print('ask-root: no command given', file=sys.stderr)
        return 2
    res = _ask('root', {'cmd': cmd})
    if res is None:
        return 124
    if res['decision'] in ('denied','withdrawn'):
        print(f"ask-root: denied by {res.get('approved_by', '?')}: {cmd}", file=sys.stderr)
        return 126
    sys.stdout.write(res.get('out') or '')
    sys.stderr.write(res.get('err') or '')
    return res.get('rc') if res.get('rc') is not None else 0


if __name__ == '__main__':
    sys.exit(main())
'''

ENV_ASK_SOURCE = '''#!/usr/bin/env python3
"""env-ask TEXT CHOICE... : ask the human a question; prints the answer (or the chosen choice)."""
''' + _COMMON + '''

def main():
    if len(sys.argv) < 2:
        print('env-ask: TEXT [CHOICE...]', file=sys.stderr)
        return 2
    text, choices = sys.argv[1], sys.argv[2:]
    res = _ask('question', {'text': text, 'choices': choices})
    if res is None:
        return 124
    if res['decision'] in ('denied','withdrawn'):
        print(f"env-ask: denied by {res.get('approved_by', '?')}", file=sys.stderr)
        return 126
    ans = res.get('answer') or ''
    if choices and ans.strip().isdigit() and 1 <= int(ans) <= len(choices):
        ans = choices[int(ans) - 1]
    print(ans)
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''

def _baked(source, bridge):
    return source.replace('{bridge!r}', repr(bridge))


def ask_root_source(bridge):
    return _baked(ASK_ROOT_SOURCE, bridge)


def install_command(bridge, bin_path, robots_source, robots_path, memory_source, memory_path):
    """the shell string that installs the agent-side scripts, ROBOTS.md and MEMORY.md over SSH (base64: no quoting
    crosses the wire). bin_path is ~/bin/ask-root; env-ask lands beside it, and an env-note left by an older
    install (the steering-notes inbox, replaced by /steer) is removed."""
    d = os.path.dirname(bin_path)
    parts = [f'mkdir -p {d} {os.path.dirname(robots_path)}']
    for name, src in (('ask-root', ask_root_source(bridge)),
                      ('env-ask', _baked(ENV_ASK_SOURCE, bridge))):
        payload = base64.b64encode(src.encode()).decode()
        target = os.path.join(d, name)
        parts.append(f"printf %s {payload} | base64 -d > {target} && chmod 700 {target}")
    parts.append(f"rm -f {os.path.join(d, 'env-note')}")
    robots = base64.b64encode(robots_source.encode()).decode()
    parts.append(f"printf %s {robots} | base64 -d > {robots_path} && chmod 600 {robots_path}")
    mem = base64.b64encode(memory_source.encode()).decode()
    parts.append(f"[ -e {memory_path} ] || printf %s {mem} | base64 -d > {memory_path}")
    return ' && '.join(parts)


# The agent's venv (resident/discovery.py holds the package set: the prompt names it from there). Run by the account's
# python3 over SSH; one pip call per missing package, so one failure is named and the others still install. Prints one
# JSON line; exit 1 only when no venv with a working pip could be made or torch was asked for and failed.
VENV_TIMEOUT = 3600
WEASYPRINT_APT = 'sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libharfbuzz-subset0'
_VENV_FILL = r'''
import json, os, re, subprocess, sys
P = PARAMS
v = os.path.expanduser(P['venv']); py = os.path.join(v, 'bin', 'python'); pip = os.path.join(v, 'bin', 'pip')
out = dict(made=False, repaired=False, present=[], installed=[], failed={}, skipped=[], offline=None, torch=None, weasyprint=None)
NET = re.compile(r'NewConnectionError|Failed to establish a new connection|Temporary failure in name resolution|'
                 r'Name or service not known|Network is unreachable|ConnectTimeout|ProxyError|No route to host')
def tail(text): return ' '.join((text or '').strip().splitlines()[-2:])[-300:]
def run(argv, timeout=1800):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired: return 124, 'timed out after %d s' % timeout
    except OSError as e: return 127, str(e)
def done(code): print(json.dumps(out)); sys.exit(code)
def ver(text):
    # the release part with trailing zeros dropped, so an installed 5 meets a floor of 5.0
    m = re.match(r'\d+(?:\.\d+)*', text.strip())
    t = [int(x) for x in m.group().split('.')] if m else [0]
    while len(t) > 1 and t[-1] == 0: t.pop()
    return tuple(t)
def pip_works(): return os.access(pip, os.X_OK) and run([pip, '--version'], 120)[0] == 0
if not pip_works():
    # Without python3-venv, python3 -m venv leaves bin/python and no pip. Running it again on the same directory adds
    # pip and keeps what is installed (never --clear: that would drop torch and the agent's own packages).
    had = os.access(py, os.X_OK)
    rc, text = run([sys.executable, '-m', 'venv', v])
    if rc or not pip_works():
        out['error'] = ('python3 -m venv %s: %s (Debian / Ubuntu: sudo apt install python3-venv, then run caminus '
                        'ember setup again)' % ('left it without a working pip' if had else 'failed',
                                                tail(text) or 'bin/pip does not run'))
        done(1)
    out['repaired' if had else 'made'] = True
code = 0
if P['torch']:
    if run([py, '-c', 'import torch, numpy'])[0] == 0: out['torch'] = 'present'
    else:
        rc, text = run([pip, 'install', '-q', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch'])
        if rc == 0: rc, text = run([pip, 'install', '-q', 'numpy'])
        out['torch'] = 'installed' if rc == 0 else 'failed: ' + tail(text); code = int(rc != 0)
specs = [re.fullmatch(r'([A-Za-z0-9_.-]+)>=(\S+)', s).groups() + (s,) for s, _ in P['packages']]   # (name, low, spec)
VERSIONS = ('import json,sys,importlib.metadata as m\nr={}\nfor n in sys.argv[1:]:\n'
            ' try:r[n]=m.version(n)\n except m.PackageNotFoundError:r[n]=None\nprint(json.dumps(r))')
rc, text = run([py, '-c', VERSIONS] + [n for n, _, _ in specs])
have = json.loads(text) if rc == 0 else {}
for name, low, spec in specs:
    if have.get(name) and ver(have[name]) >= ver(low): out['present'].append(name); continue
    if out['offline']: out['skipped'].append(name); continue
    rc, text = run([pip, 'install', '-q', '--disable-pip-version-check', '--only-binary', ':all:', spec])
    if rc == 0: out['installed'].append(name); continue
    no_index = '(from versions: none)' in text and (os.environ.get('PIP_NO_INDEX') or
                                                   'no-index' in run([pip, 'config', 'list'], 30)[1])
    net = NET.search(text)
    if net or no_index:
        out['offline'] = net.group(0) if net else 'pip is set to --no-index'; out['skipped'].append(name)
    else: out['failed'][name] = tail(text)
if 'weasyprint' in out['present'] + out['installed']:
    rc, text = run([py, '-c', 'import weasyprint'], 120)
    out['weasyprint'] = 'ok' if rc == 0 else ('libs' if 'cannot load library' in text or 'OSError' in text
                                             else 'import failed: ' + tail(text))
done(code)
'''


def venv_command(torch=False, venv=None):
    """the shell string that makes the agent's venv if missing and installs what it lacks of the package set (torch and
    numpy too when asked: the repository's CPU tests); safe to rerun, and quick when nothing is missing."""
    params = dict(venv=venv or AGENT_VENV, torch=bool(torch), packages=AGENT_PACKAGES)
    return "python3 - <<'CAMINUS_VENV'\n" + _VENV_FILL.replace('PARAMS', repr(params)) + 'CAMINUS_VENV\n'


def fill_venv(transport, torch=False, venv=None):
    """venv_command run as the agent: (lines to show the owner, ok) as venv_report gives them; a fill that runs past
    its hour (a very slow link) is a line too, never a traceback"""
    try:
        r = runner.run_shell(venv_command(torch, venv), transport=transport, timeout=VENV_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [f'agent venv {venv or AGENT_VENV}: no answer in {VENV_TIMEOUT} s (a slow link?); what installed '
                'stays: run `caminus ember setup` again'], False
    return venv_report(r, torch, venv)


def venv_report(r, torch=False, venv=None):
    """(lines to show the owner, ok) from venv_command's CompletedProcess. ok is False when there is no venv with a
    working pip (python3-venv missing: the line says what to install) or torch was asked for and is missing. The
    package set is optional, so a failed package, no index (offline) or missing weasyprint libraries are reported
    with what to run, and setup still passes."""
    v = venv or AGENT_VENV
    try:
        res = json.loads((r.stdout or '').strip().splitlines()[-1])
    except (IndexError, ValueError):
        why = (r.stderr or r.stdout or '').strip()[-300:] or f'exit {r.returncode}'
        return [f'agent venv {v}: failed: {why}'], False
    if res.get('error'):
        return [f"agent venv {v}: {res['error']}"], False
    lines = []
    if res['torch']:
        lines.append(f"agent venv {v}: CPU torch and numpy {res['torch']}")
    done = ', '.join(res['installed']) or 'nothing'
    made = 'made; ' if res['made'] else 'repaired (it had no working pip); ' if res.get('repaired') else ''
    lines.append(f"agent venv {v}: {made}installed {done}"
                 f"{'; already there: ' + ', '.join(res['present']) if res['present'] else ''}")
    for name, err in res['failed'].items():
        rest = ' (the rest are fine)' if res['present'] or res['installed'] else ''
        lines.append(f'agent venv {v}: {name} did not install{rest}: {err}')
    if res['offline']:
        lines.append(f"agent venv {v}: no package index reachable ({res['offline']}); not installed: "
                     f"{', '.join(res['skipped'])}. Run `caminus ember setup` again when online.")
    if res['weasyprint'] == 'libs':
        lines.append(f'agent venv {v}: weasyprint (pandoc\'s PDF engine) needs system libraries this machine lacks: '
                     f'{WEASYPRINT_APT}')
    elif res['weasyprint'] not in (None, 'ok'):
        lines.append(f"agent venv {v}: weasyprint: {res['weasyprint']}")
    return lines, not (res['torch'] or '').startswith('failed')
