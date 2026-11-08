"""caminus qwen-code status | setup, caminus qwen: Qwen Code against a Caminus server, in a Qwen home of Caminus's own.

  caminus qwen-code status [--json]                      installed? which version, which home, set up for which server
  caminus qwen-code setup [--server NAME] [--install] [--home DIR] [--json]
  caminus qwen [--server NAME] [--model NAME] [qwen's own arguments...]

The home is CAMINUS_QWEN_HOME, else ~/.config/caminus/qwen-home (the brain / hands team: ~/.config/caminus/qwen-home-bh):
the same homes packaging/harness/qwen-code's qwen-k3 / qwen-bh use, so both see one configuration. Qwen Code reads its global
settings, sessions and memory from QWEN_HOME (its Storage.getGlobalQwenDir), so ~/.qwen is never read or written here.

setup writes the provider entries of the running Caminus server(s) into the home's settings.json: the server's own model ids
(from its /v1/models; the Coder's k3-coder / -nothink / -low / -xhigh, the others' alias / alias-nothink), its URL, the
context window from its /props (n_ctx rounded down to a thousand: 262,144 -> 262,000, the shipped config's value), no
max_tokens (the side calls send their own). An existing settings.json is merged, not replaced: only the provider entries
that point at Caminus (envKey CAMINUS_API_KEY) are swapped, model.name / fastModel only when they named a Caminus id or are
unset; everything else stays, and the old file is kept as settings.json.bak-<time> first. The team setup also writes the
`hands` subagent (agents/hands.md) and the brain's policy (BRAIN.md) when they are missing.

--install puts Qwen Code PIN (the version these configs were written against) into Caminus's own npm prefix
($XDG_DATA_HOME/caminus/harness/qwen-code): your own Qwen Code install, if any, is not touched. It needs Node.js / npm."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from .. import config
from . import manage

PIN = '0.26.9'
PKG = '@qwen-code/qwen-code'
SHARE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share', 'harness', 'qwen-code')


def prefix():
    return os.path.join(config.xdg('XDG_DATA_HOME', os.path.join('.local', 'share')), 'caminus', 'harness', 'qwen-code')


def binary():
    own = os.path.join(prefix(), 'node_modules', '.bin', 'qwen')
    if os.path.isfile(own) and os.access(own, os.X_OK):
        return own
    return shutil.which('qwen')


def version(b):
    try:
        r = subprocess.run([b, '--version'], capture_output=True, text=True, timeout=20)
        return (r.stdout.strip().splitlines() or [''])[-1]
    except (OSError, subprocess.TimeoutExpired):
        return ''


def base_home():
    return os.path.expanduser(os.environ.get('CAMINUS_QWEN_HOME') or os.path.join(config.xdg('XDG_CONFIG_HOME', '.config'),
                                                                                   'caminus', 'qwen-home'))


def home_for(s):
    """the Qwen home a server's sessions live in (the team's members: the team home)"""
    if s and s.get('name') in ('brain', 'hands'):
        return base_home() + '-bh'
    return base_home()


def running():
    return [s for s in manage.all_servers() if manage.alive(s.get('guard_pid'))]


WAIT_CTX_S = 120          # how long setup waits for a loading server's /props to give its context
FLOOR_CTX = 32000         # the window when the server never says: below every context Caminus starts a model with (arch.MIN_CTX)


def server_ids(s, wait_s=WAIT_CTX_S):
    """(model ids, n_ctx) the server answers to: /v1/models and /props (waiting up to wait_s for a server still loading to
    report its context: max_ctx auto sizes it to the cards, so no fixed value is right), else its catalog alias"""
    url = f"http://127.0.0.1:{s['port']}"
    t_end = time.time() + wait_s
    while True:
        _, props = manage.http_json(url + '/props')
        n_ctx = manage.find_key(props, 'n_ctx') if isinstance(props, dict) else None
        if n_ctx or time.time() >= t_end:
            break
        time.sleep(2)
    _, models = manage.http_json(url + '/v1/models')
    ids = [m.get('id') for m in (models or {}).get('data', [])] if isinstance(models, dict) else []
    if not ids:
        a = s.get('alias') or s['model']
        ids = [a, a + '-nothink']
    return ids, n_ctx


def providers(s):
    ids, n_ctx = server_ids(s)
    window = (int(n_ctx) // 1000) * 1000 if n_ctx else FLOOR_CTX     # (Qwen Code compacts against this: never above the server's)
    url = f"http://127.0.0.1:{s['port']}/v1"
    out = []
    for i in ids:
        kind = 'no thinking' if i.endswith('-nothink') else ('thinking, low effort' if i.endswith('-low') else
                                                            'thinking, extra-high effort' if i.endswith('-xhigh') else 'thinking')
        out.append({'id': i, 'name': f'Caminus {i} ({kind})', 'envKey': 'CAMINUS_API_KEY', 'baseUrl': url,
                    'generationConfig': {'timeout': 900000, 'contextWindowSize': window}})
    return out, ids


def merged(old, provs, main_id, fast_id, team=None):
    """old settings with Caminus's provider entries replaced (see the module doc)"""
    new = json.loads(json.dumps(old or {}))
    mp = new.setdefault('modelProviders', {})
    keep = [p for p in mp.get('openai', []) if p.get('envKey') != 'CAMINUS_API_KEY']
    gone = {p.get('id') for p in mp.get('openai', []) if p.get('envKey') == 'CAMINUS_API_KEY'}
    mp['openai'] = keep + provs
    new.setdefault('security', {}).setdefault('auth', {}).setdefault('selectedType', 'openai')
    model = new.setdefault('model', {})
    if not model.get('name') or model.get('name') in gone:
        model['name'] = main_id
    if not new.get('fastModel') or new.get('fastModel') in gone:
        new['fastModel'] = fast_id
    if team:
        new.setdefault('context', {}).setdefault('fileName', ['QWEN.md', 'BRAIN.md'])
    return new


def write_settings(home, settings, say):
    os.makedirs(home, mode=0o700, exist_ok=True)
    os.chmod(home, 0o700)
    p = os.path.join(home, 'settings.json')
    if os.path.exists(p):
        bak = f'{p}.bak-{time.strftime("%Y%m%d-%H%M%S")}'
        shutil.copy2(p, bak)
        say(f'qwen-code: kept the old settings as {bak}')
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return p


def setup(server=None, home=None, say=print):
    rs = running()
    if not rs:
        raise manage.ManageError('no Caminus server is running (caminus start MODEL): setup reads its model ids and context')
    team = {s['name']: s for s in rs if s['name'] in ('brain', 'hands')}
    if server is None and set(team) == {'brain', 'hands'}:
        provs, ids_b = providers(team['brain'])
        p_h, ids_h = providers(team['hands'])
        h = home or (base_home() + '-bh')
        old = read_settings(h)
        quick = next((i for i in ids_h if i.endswith('-nothink')), ids_h[0])
        st = merged(old, provs + p_h, ids_b[0], quick, team=True)
        # the team's agent rules (packaging/harness/qwen-code/brain-hands/settings.json): the hands subagent's grades on the
        # hands server's own ids, Explore on the hands, and ONE agent at a time per hands model (its server has one slot: a
        # second parallel subagent would evict the first one's conversation from its cache); the user's own values are kept
        ag = st.setdefault('agents', {})
        ag.pop('exploreModel', None)                    # (an earlier setup wrote it at the wrong level)
        ag.setdefault('builtin', {}).setdefault('exploreModel', ids_h[0])
        ag.setdefault('modelGrades', {'hands': ids_h[0], 'hands-quick': quick})
        ag.setdefault('allowedGrades', ['hands', 'hands-quick'])
        mp = ag.setdefault('maxParallelAgentsByModel', {})
        for i in ids_h:
            mp.setdefault(i, 1)
        p = write_settings(h, st, say)
        for rel in ('agents/hands.md', 'BRAIN.md'):
            src, dst = os.path.join(SHARE, 'brain-hands', rel), os.path.join(h, rel)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        return f'set up for the brain / hands team in {p}', h
    s = next((x for x in rs if x['name'] == server), None) if server else (rs[0] if len(rs) == 1 else None)
    if s is None:
        raise manage.ManageError(f'{"no running server " + repr(server) if server else "several servers run"}: '
                                 f'--server NAME (caminus ps)')
    provs, ids = providers(s)
    h = home or home_for(s)
    st = merged(read_settings(h), provs, ids[0], next((i for i in ids if i.endswith('-nothink')), ids[0]))
    return f"set up for {s['name']} ({', '.join(ids)}) in {write_settings(h, st, say)}", h


def read_settings(h):
    try:
        with open(os.path.join(h, 'settings.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def install(say):
    npm = shutil.which('npm')
    if not npm:
        raise manage.ManageError('npm is not installed (Node.js >= 20): install it, then caminus qwen-code setup --install')
    os.makedirs(prefix(), exist_ok=True)
    say(f'qwen-code: npm install {PKG}@{PIN} into {prefix()}')
    r = subprocess.run([npm, 'install', '--prefix', prefix(), '--no-fund', '--no-audit', f'{PKG}@{PIN}'],
                       stdout=sys.stderr, stderr=sys.stderr)
    if r.returncode != 0:
        raise manage.ManageError(f'npm install failed (rc {r.returncode})')


def status():
    b = binary()
    rs = running()
    target = None
    for h in {home_for(s) for s in rs} | {base_home()}:
        st = read_settings(h) or {}
        urls = {p.get('baseUrl') for p in st.get('modelProviders', {}).get('openai', []) if p.get('envKey') == 'CAMINUS_API_KEY'}
        for s in rs:
            if f"http://127.0.0.1:{s['port']}/v1" in urls:
                target = target or s['name']
    return dict(installed=bool(b), binary=b or '', version=version(b) if b else '', pinned=PIN, home=base_home(),
                configured=target is not None or os.path.isfile(os.path.join(base_home(), 'settings.json')),
                npm=bool(shutil.which('npm')), target=target or '')


def main(argv):
    ap = argparse.ArgumentParser(prog='caminus qwen-code')
    ap.add_argument('what', choices=('status', 'setup'))
    ap.add_argument('--server')
    ap.add_argument('--home')
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.what == 'status':
        st = status()
        if a.json:
            print(json.dumps(st))
        else:
            print(f"Qwen Code: {'installed ' + st['version'] + ' (' + st['binary'] + ')' if st['installed'] else 'not installed'}")
            print(f"home: {st['home']} ({'set up' + (' for ' + st['target'] if st['target'] else '') if st['configured'] else 'not set up'})")
        return 0

    def go(say):
        if a.install:
            install(say)
        note, _ = setup(a.server, os.path.expanduser(a.home) if a.home else None, say)
        return note
    return manage.run_action(a, go)


def launch(argv):
    """caminus qwen [--server NAME] [--model NAME] [qwen args...]: start a server when none runs (--model), set the home up
    for it, then exec Qwen Code"""
    ap = argparse.ArgumentParser(prog='caminus qwen', add_help=False)
    ap.add_argument('--server')
    ap.add_argument('--model')
    a, rest = ap.parse_known_args(argv)
    b = binary()
    if not b:
        print('caminus qwen: Qwen Code is not installed (caminus qwen-code setup --install, or npm i -g @qwen-code/qwen-code)',
              file=sys.stderr)
        return 1
    try:
        rs = running()
        if not rs:
            if not a.model:
                raise manage.ManageError('no Caminus server is running: caminus start MODEL, or caminus qwen --model MODEL')
            manage.start_one(a.model, log=lambda s: print(s, file=sys.stderr, flush=True))
        note, home = setup(a.server, None, say=lambda s: print(s, file=sys.stderr))
    except manage.ManageError as e:
        print(f'caminus qwen: {e}', file=sys.stderr)
        return 1
    print(f'caminus qwen: {note}', file=sys.stderr)
    env = dict(os.environ, QWEN_HOME=home, CAMINUS_API_KEY=os.environ.get('CAMINUS_API_KEY') or 'caminus-local')
    os.execve(b, [b, *rest], env)
