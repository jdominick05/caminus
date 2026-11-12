"""CPU tests of the management commands (caminus.cmd.manage, caminus.cmd.qwen_code): nothing touches a GPU or the network.

  python -I tests/cmd/test_manage.py --scratch DIR     RESULT PASS

Model files are sparse files of the manifest's sizes under a scratch model directory; servers are server.json files with
dead or live (this process's child) guard pids; the Qwen homes are scratch directories (never ~/.config/caminus or ~/.qwen)."""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
# the paused models (caminus.models.PAUSED) stay in these fixtures; paused_checks() runs a child without the variable
PAUSED_ENV = {k: v for k, v in os.environ.items() if k != 'CAMINUS_ALLOW_PAUSED'}
os.environ['CAMINUS_ALLOW_PAUSED'] = '1'

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


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
                      XDG_CONFIG_HOME=os.path.join(S, 'config'))
    for k in ('CAMINUS_CODER', 'K3_GGUF', 'CAMINUS_DFLASH', 'CAMINUS_MODEL_SEARCH', 'LLMTOP_CONFIG'):
        os.environ.pop(k, None)
    from caminus import models
    from caminus.cmd import manage, qwen_code

    # nothing here may ever start a process that could reach a GPU: manage's own Popen is replaced by one that refuses
    class NoLaunch:
        DEVNULL = subprocess.DEVNULL
        STDOUT = subprocess.STDOUT

        @staticmethod
        def Popen(*a, **k):
            raise RuntimeError('test_manage: a launch was attempted (the test must refuse before it)')

        run = staticmethod(subprocess.run)
    manage.subprocess = NoLaunch

    # ---- models: present only with the manifest's size; model_search finds files kept elsewhere
    it = models.BY_KEY['qwen3.6']
    e = manage.model_entry(models.BY_NAME['qwen3.6'])
    check(not e['present'] and e['path'] is None, 'models: qwen3.6 absent before its file exists')
    p = os.path.join(md, models.local_rel(it))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.truncate(it.size - 1)                                   # one byte short: a partial download
    check(not manage.model_entry(models.BY_NAME['qwen3.6'])['present'], 'models: a file of the wrong size is not the model')
    with open(p, 'r+b') as f:
        f.truncate(it.size)
    e = manage.model_entry(models.BY_NAME['qwen3.6'])
    check(e['present'] and e['path'] == p, f'models: qwen3.6 found at the model_dir layout ({e["path"]})')
    other = os.path.join(S, 'elsewhere', 'a', 'b')
    os.makedirs(other, exist_ok=True)
    ax = models.BY_KEY['arex-2']
    with open(os.path.join(other, os.path.basename(ax.path)), 'wb') as f:
        f.truncate(ax.size)
    check(not manage.model_entry(models.BY_NAME['arex-2'])['present'], 'models: arex-2 not found outside the search path')
    os.environ['CAMINUS_MODEL_SEARCH'] = os.path.join(S, 'elsewhere')
    from caminus import config
    config._FILE.clear()
    check(manage.model_entry(models.BY_NAME['arex-2'])['present'], 'models: arex-2 found three levels under model_search')
    out = subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); from caminus.__main__ import main;'
                          ' sys.exit(main(["models", "--json"]))', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python')],
                         capture_output=True, text=True, env=os.environ)
    try:
        doc = json.loads(out.stdout)
        names = [m['name'] for m in doc['models']]
        check(out.returncode == 0 and names == [m.name for m in models.CATALOG] and
              {m['name']: m['present'] for m in doc['models']} == {'coder': False, 'qwen3.6': True, 'qwen3.6-hauhau': False,
                                                                   'arex-2': True, 'qwen3-coder-next': False, 'swift-1.5': False,
                                                                   'thinkingcap': False, 'qwen3-30b-a3b': False},
              f'caminus models --json: {names}')
    except ValueError:
        check(False, f'caminus models --json printed no JSON: {out.stdout[:200]} {out.stderr[-300:]}')

    # ---- the download default stays the Coder's files (a bare `caminus download` must not fetch every model)
    from caminus.cmd import download
    calls = []
    real = download.download
    download.download = lambda items, dest, **kw: calls.append([i.key for i in items]) or []
    try:
        download.main(['--dry-run']); download.main(['--model', 'qwen3.6', '--dry-run'])
    finally:
        download.download = real
    check(calls == [['coder-1', 'coder-2', 'dflash'], ['qwen3.6']], f'download: default = the Coder, --model NAME = its files ({calls})')

    # ---- cards: a big one-card model takes the biggest free card, a small one leaves it; two-card models need both
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'device'))
    import test_host as TH                       # (its sysfs tree builder: this machine's layout, the XTX driving the display)
    manage.ROOT = TH.tree(os.path.join(S, 'sysfs'), [TH.card('card1', '0000:03:00.0', TH.XTX, 128, display=True),
                                                     TH.card('card2', '0000:06:00.0', TH.XT, 129, link=(16.0, 4))])
    cv = manage.card_view()
    check([(c['index'], c['name'], c['pci'], c['hip']) for c in cv] == [(0, 'XTX', '0000:03:00.0', 0), (1, 'XT', '0000:06:00.0', 1)],
          f'cards: the fixture host\'s cards by position, PCI address and HIP index ({cv})')
    pick = lambda name, card=None, held=(): manage.pick_cards(models.BY_NAME[name], manage.model_entry(models.BY_NAME[name]), card, set(held))
    check(pick('arex-2') == [0], 'cards: AREX-2 (17 GB) goes to the 24 GB card')
    check(pick('qwen3.6') == [1], 'cards: Qwen3.6 GSQ (12 GB) goes to the 20 GB card')
    check(pick('qwen3.6', held={1}) == [0], 'cards: ... or the other one when that is held')
    check(pick('qwen3.6', card='0') == [0], 'cards: --card 0 is taken as given')
    for args, what in (((('qwen3.6', '0', {0})), 'a held card'), (('coder', None, {1}), 'a two-card model with one held'),
                       (('qwen3.6', '7', ()), 'a card that does not exist')):
        try:
            pick(*args)
            check(False, f'cards: {what} refused')
        except manage.ManageError as err:
            check(True, f'cards: {what} refused ({err})')
    check(pick('coder') == [0, 1], 'cards: the Coder takes both')
    keep = manage.ROOT
    manage.ROOT = TH.tree(os.path.join(S, 'sysfs-swapped'), [TH.card('card1', '0000:03:00.0', TH.XT, 128),
                                                             TH.card('card2', '0000:06:00.0', TH.XTX, 129, display=True)])
    check(pick('coder') == [1, 0] and pick('arex-2') == [1], 'cards: swapped DRM order: the Coder\'s pair in engine order (the XTX, '
          'position 1, first); AREX-2 on the XTX')
    manage.ROOT = TH.tree(os.path.join(S, 'sysfs-xtx2'), [TH.card('card1', '0000:03:00.0', TH.XTX, 128, display=True),
                                                          TH.card('card2', '0000:06:00.0', TH.XTX, 129)])
    check(pick('coder') == [0, 1] and pick('qwen3-coder-next') == [0, 1], 'cards: XTX + XTX: the Coder and Qwen3-Coder-Next placed')
    from caminus import arch
    from caminus.device import host as H
    xx = H.pair(H.gpus(manage.ROOT))
    check(arch.CODER.split(xx) == dict(gdn_keys=10, moe_blocks=12) and arch.lookup('qwen3next').split(xx) == dict(gdn_keys=8, moe_blocks=10),
          'split: XTX + XTX: the Coder at its measured 10 / 12 (128-row expert shards), Qwen3-Coder-Next at the even 8 / 10')
    cm = models.BY_NAME['coder']; ce = manage.model_entry(cm)
    for var, item in (('CODER', cm.items[0]), ('DFLASH', models.drafter_item(cm))):   # (the developer's files: CLAUDE.md's exports)
        fp = os.environ.get(var)
        if item and fp and os.path.exists(fp):
            ce = dict(ce, paths=dict(ce['paths'], **{item: fp}), **({'path': fp} if var == 'CODER' else {}))
    def projection_runs():   # (CI's $CODER is a vocab-only GGUF and the catalog holds stubs: no tensors to project)
        try:
            arch.of(ce['path']).auto_need(dict(model=ce['path'], drafter=None, no_drafter=True, chunk=2048, max_ctx='auto',
                                               auto_cache_slots=4), 4, cards=xx)
            return True
        except Exception as e:
            print(f'NOTE cards: the Coder\'s projection can\'t run here ({type(e).__name__}: {str(e)[:80]})', flush=True)
            return False
    real = projection_runs()
    if not real or manage.pair_fits(cm, ce) is None:
        print('NOTE cards: the Coder\'s projection checks are skipped', flush=True)
    else:
        dr = 'with its drafter' if (ce['paths'].get(models.drafter_item(cm)) or '') else 'without its drafter'
        check(manage.pick_cards(cm, ce, None, set()) == [0, 1], f'cards: XTX + XTX: the Coder fits by its projection ({dr})')
    manage.ROOT = TH.tree(os.path.join(S, 'sysfs-xt2'), [TH.card('card1', '0000:03:00.0', TH.XT, 128, display=True),
                                                         TH.card('card2', '0000:06:00.0', TH.XT, 129)])
    if real and manage.pair_fits(cm, ce) is not None:
        try:
            manage.pick_cards(cm, ce, None, set()); check(False, 'cards: XT + XT: the Coder refused (card 0 must be a 24 GB XTX)')
        except manage.ManageError as err:
            check('24 GB RX 7900 XTX as card 0' in str(err), f'cards: XT + XT: the Coder refused by its projection ({str(err)[:120]})')
    check(pick('qwen3-coder-next') == [0, 1], 'cards: XT + XT: Qwen3-Coder-Next placed')
    manage.ROOT = TH.tree(os.path.join(S, 'sysfs-one'), [TH.card('card0', '0000:03:00.0', TH.XT, 128)])
    try:
        pick('coder'); check(False, 'cards: one card: the Coder refused')
    except manage.ManageError as err:
        check('two supported cards' in str(err), f'cards: one card: the Coder refused ({err})')
    check(pick('qwen3.6') == [0], 'cards: one card: Qwen3.6 runs')
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        manage.cmd_ps(['--json'])
    js = json.loads(buf.getvalue())
    check(set(js) >= {'servers', 'cards', 'gpus', 'problems'} and js['problems'][0]['level'] == 'info' and js['cards'][0]['pci'] == '0000:03:00.0',
          f"ps --json: servers, cards (with pci / hip / link), gpus, problems ({js['problems']})")
    manage.ROOT = keep

    # ---- servers: state from server.json + the guard pid; failed vs stopped from the log
    def put(name, pid, log_text, port):
        d = manage.server_dir(name)
        os.makedirs(d, exist_ok=True)
        lp = os.path.join(d, 'serve.log')
        with open(lp, 'w') as f:
            f.write(log_text)
        s = dict(name=name, model='qwen3.6', path=p, port=port, devices=[1], card_names=['XT'], argv=['--model', p],
                 guard_pid=pid, log=lp, started=time.time() - 65, alias='qwen3.6')
        with open(os.path.join(d, 'server.json'), 'w') as f:
            json.dump(s, f)
        return s
    dead = subprocess.Popen(['true']); dead.wait()
    put('crashed', dead.pid, 'caminus serve: http://127.0.0.1:1 (loading)  pid=1\nRuntimeError: something broke\n', 18092)
    put('clean', dead.pid, 'caminus serve: http://127.0.0.1:1 (loading)  pid=1\ncaminus serve: stopped in 0.2 s\n', 18093)
    guard_refusal = 'GUARD REFUSED: a GPU fault was recorded ($HOME/.local/state/k3/GPU_FAULT_STOP). Stop all GPU work and report to the lead.'
    put('guard-refused', dead.pid, guard_refusal + '\n', 18095)
    v = {x['name']: x for x in map(manage.server_view, manage.all_servers())}
    check(v['crashed']['state'] == 'failed' and 'something broke' in v['crashed']['error'], f"ps: a crashed server is failed ({v['crashed']['error']})")
    check(v['clean']['state'] == 'stopped', 'ps: a cleanly stopped server is stopped')
    check(v['guard-refused']['state'] == 'failed' and v['guard-refused']['error'] == guard_refusal,
          'ps: a guard refusal preserves the startup failure reason')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        manage.cmd_ps(['--json'])
    refused = next(s for s in json.loads(buf.getvalue())['servers'] if s['name'] == 'guard-refused')
    check(refused['error'] == guard_refusal, 'ps --json: the panel receives the guard refusal reason')
    # a live 'guard' (a sleeping child whose command line holds guard.sh) not answering HTTP -> loading
    g = os.path.join(S, 'guard.sh')
    with open(g, 'w') as f:
        f.write('#!/bin/sh\nsleep 30\n')       # (no exec: the shell, whose command line holds guard.sh, stays the pid)
    os.chmod(g, 0o755)
    live = subprocess.Popen(['/bin/sh', g])
    time.sleep(0.3)
    try:
        put('loading', live.pid, 'caminus serve: http://127.0.0.1:1 (loading)  pid=%d\n' % live.pid, 18094)
        check(manage.server_view(manage.load_server('loading'))['state'] == 'loading', 'ps: a live guard with no HTTP answer is loading')
        held = {d for s in manage.all_servers() if manage.alive(s.get('guard_pid')) for d in s.get('devices', [])}
        check(held == {1}, f'ps: only live servers hold cards ({held})')
        check(manage.pick_port(None, {8092}) != 8092, 'ports: a taken port is skipped')
        try:
            manage.start_one('qwen3.6', name='loading')
            check(False, 'start: a name already running is refused')
        except manage.ManageError as err:
            check('already running' in str(err), f'start: a name already running is refused ({err})')
        try:
            manage.start_one('qwen3-coder-next')
            check(False, 'start: a model not on disk is refused')
        except manage.ManageError as err:
            check('not downloaded' in str(err), f'start: a model not on disk is refused ({err})')
    finally:
        live.kill(); live.wait()

    # ---- monitor: preserve global RGB/display preferences while routing each server to its own sessions and port
    prefs_dir = os.path.join(S, 'config', 'caminus')
    os.makedirs(prefs_dir, exist_ok=True)
    prefs_path = os.path.join(prefs_dir, 'top.json')
    prefs = {'openrgb': {'enabled': True, 'fps': 10, 'speed': .5, 'zones': {'Fan': 'ring'}},
             'web': {'enabled': False, 'listen': '127.0.0.1:8765'}, 'hold_file': 'maintenance.hold',
             'server_url': 'http://old-server:1', 'sessions_dir': 'old-sessions'}
    original = b'\xef\xbb\xbf' + json.dumps(prefs).encode()
    with open(prefs_path, 'wb') as f:
        f.write(original)
    monitor = dict(name='monitor', model='qwen3.6', port=18096)
    generated = manage.top_config(monitor)
    with open(generated) as f:
        top = json.load(f)
    check(top['openrgb'] == prefs['openrgb'] and top['web'] == prefs['web'],
          'top: RGB and display preferences survive the generated server config')
    check(top['server_url'] == 'http://127.0.0.1:18096' and
          top['sessions_dir'] == os.path.join(qwen_code.home_for(monitor), 'projects') and
          top['hold_file'] == os.path.join(prefs_dir, 'maintenance.hold'),
          'top: server routing wins; relative preference paths still belong to the source directory')
    with open(prefs_path, 'rb') as f:
        check(f.read() == original, 'top: user preferences left byte-identical')
    check(os.stat(generated).st_mode & 0o777 == 0o600, 'top: generated preferences are private (0600)')
    custom = os.path.join(S, 'custom-top.json')
    with open(custom, 'w') as f:
        json.dump({'openrgb': {'enabled': True, 'brightness': .25}}, f)
    os.environ['LLMTOP_CONFIG'] = custom
    with open(manage.top_config(monitor)) as f:
        check(json.load(f)['openrgb']['brightness'] == .25, 'top: LLMTOP_CONFIG chooses the preference source')
    with open(generated, 'rb') as f:
        previous = f.read()
    for bad in ('{broken', '[]', 'null'):
        with open(custom, 'w') as f:
            f.write(bad)
        try:
            manage.top_config(monitor)
            check(False, 'top: invalid preferences refused')
        except manage.ManageError:
            with open(generated, 'rb') as f:
                check(f.read() == previous, 'top: invalid preferences refused without replacing the runtime config')
    # Explicit configs bypass the generated template, with or without a positional server name.
    before_servers, before_alive, before_exec = manage.all_servers, manage.alive, manage.exec_panel
    launched = []
    manage.all_servers, manage.alive = lambda: [monitor], lambda pid: True
    manage.exec_panel = lambda args: launched.append(args) or 0
    try:
        for argv in (['-config', prefs_path], ['monitor', '--config', prefs_path]):
            check(manage.cmd_top(argv) == 0 and launched[-1] == ['top', '-config', prefs_path],
                  'top: explicit -config reaches the monitor even when global preferences are broken')
    finally:
        manage.all_servers, manage.alive, manage.exec_panel = before_servers, before_alive, before_exec
        os.environ.pop('LLMTOP_CONFIG')
    from unittest.mock import patch
    with patch.object(manage.os, 'replace', side_effect=OSError('simulated publish failure')):
        try:
            manage.top_config(monitor)
            check(False, 'top: config publication failure surfaced')
        except OSError:
            with open(generated, 'rb') as f:
                check(f.read() == previous, 'top: failed publication leaves the previous runtime config intact')
    check(not [n for n in os.listdir(os.path.dirname(generated)) if n.startswith('.top-')],
          'top: failed publication cleans up only its own temporary file')

    # ---- Qwen Code: the merge keeps the user's own providers and choices, swaps only Caminus's entries
    user = {'modelProviders': {'openai': [{'id': 'my-own', 'baseUrl': 'https://example.invalid/v1', 'envKey': 'MY_KEY'},
                                          {'id': 'k3-coder', 'baseUrl': 'http://127.0.0.1:8092/v1', 'envKey': 'CAMINUS_API_KEY'}]},
            'model': {'name': 'k3-coder'}, 'fastModel': 'my-own', 'mcpServers': {'x': {'command': 'y'}}}
    provs = [{'id': 'qwen3.6', 'envKey': 'CAMINUS_API_KEY', 'baseUrl': 'http://127.0.0.1:8093/v1'},
             {'id': 'qwen3.6-nothink', 'envKey': 'CAMINUS_API_KEY', 'baseUrl': 'http://127.0.0.1:8093/v1'}]
    m = qwen_code.merged(user, provs, 'qwen3.6', 'qwen3.6-nothink')
    ids = [x['id'] for x in m['modelProviders']['openai']]
    check(ids == ['my-own', 'qwen3.6', 'qwen3.6-nothink'], f'qwen-code: the user\'s provider kept, the old Caminus one replaced ({ids})')
    check(m['model']['name'] == 'qwen3.6' and m['fastModel'] == 'my-own' and m['mcpServers'] == user['mcpServers'],
          'qwen-code: model.name moves off the gone Caminus id; the user\'s fastModel and MCP servers stay')
    check(user['modelProviders']['openai'][1]['id'] == 'k3-coder', 'qwen-code: merged() leaves its input unchanged')
    # setup() against a fake running server, into the scratch home: settings written 0600, the old file kept as a backup
    srv = dict(name='qwen3.6', model='qwen3.6', port=8093, alias='qwen3.6', guard_pid=1)
    qwen_code.running = lambda: [srv]
    qwen_code.server_ids = lambda s: (['qwen3.6', 'qwen3.6-nothink'], 262144)
    home = os.environ['CAMINUS_QWEN_HOME']
    os.makedirs(home, exist_ok=True)
    with open(os.path.join(home, 'settings.json'), 'w') as f:
        json.dump(user, f)
    note, h = qwen_code.setup(say=lambda s: None)
    sp = os.path.join(h, 'settings.json')
    with open(sp) as f:
        got = json.load(f)
    baks = [x for x in os.listdir(h) if x.startswith('settings.json.bak-')]
    win = {x['id']: x['generationConfig']['contextWindowSize'] for x in got['modelProviders']['openai'] if x.get('envKey') == 'CAMINUS_API_KEY'}
    check(h == home and oct(os.stat(sp).st_mode & 0o777) == '0o600' and len(baks) == 1,
          f'qwen-code setup: written to the scratch home (0600), the old settings kept as {baks}')
    check(win == {'qwen3.6': 262000, 'qwen3.6-nothink': 262000} and got['model']['name'] == 'qwen3.6',
          f'qwen-code setup: the server\'s ids with its context ({win})')
    # the brain / hands team: the shipped config's agent rules (one hands agent at a time, grades, Explore on the hands)
    team = [dict(name='brain', model='arex-2', port=8092, alias='arex-2', guard_pid=1), dict(name='hands', model='qwen3.6', port=8093, alias='qwen3.6', guard_pid=1)]
    qwen_code.running = lambda: team
    qwen_code.server_ids = lambda s: ([s['alias'], s['alias'] + '-nothink'], 90112 if s['name'] == 'brain' else 262144)
    note, h = qwen_code.setup(say=lambda s: None)
    with open(os.path.join(h, 'settings.json')) as f:
        ts = json.load(f)
    ag = ts.get('agents', {})
    check(h == home + '-bh' and ts['model']['name'] == 'arex-2' and ts['fastModel'] == 'qwen3.6-nothink'
          and ag.get('maxParallelAgentsByModel') == {'qwen3.6': 1, 'qwen3.6-nothink': 1}
          and ag.get('modelGrades') == {'hands': 'qwen3.6', 'hands-quick': 'qwen3.6-nothink'}
          and ag.get('builtin', {}).get('exploreModel') == 'qwen3.6' and 'exploreModel' not in ag
          and ts.get('context', {}).get('fileName') == ['QWEN.md', 'BRAIN.md']
          and os.path.isfile(os.path.join(h, 'agents', 'hands.md')) and os.path.isfile(os.path.join(h, 'BRAIN.md')),
          f'qwen-code setup (team): the shipped agent rules, the hands agent and BRAIN.md in {h} ({ag})')
    qwen_code.running = lambda: [srv]
    qwen_code.server_ids = lambda s: (['qwen3.6', 'qwen3.6-nothink'], 262144)
    st_ = qwen_code.status()
    check(st_['home'] == home and st_['target'] == 'qwen3.6', f"qwen-code status: home {st_['home']}, set up for {st_['target']}")
    check(not os.path.exists(os.path.expanduser('~/.qwen/settings.json.bak-x')), 'qwen-code: nothing written under ~/.qwen')

    paused_checks(S)
    print(f'RESULT {"PASS" if not FAILS else "FAIL (" + str(len(FAILS)) + ")"}')
    return 0 if not FAILS else 1


PAUSED_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from caminus import arch, models
from caminus.cmd import download, manage
out = dict(catalog=[m.name for m in models.CATALOG], all=[m.name for m in models.ALL])
out['pull'] = manage.cmd_pull(['qwen3.6', '--dry-run'])
out['download'] = download.main(['--model', 'qwen3-coder-next', '--dry-run'])
try:
    manage.start_one('qwen3.6-hauhau', wait=False); out['start'] = 'started'
except manage.ManageError as e:
    out['start'] = str(e)
out['team'] = manage.cmd_start(['--team', 'brain-hands', '--json'])
out['serve'] = {a: arch.paused(arch.ARCHS[a]) for a in ('qwen35moe', 'qwen3next', 'qwen35', 'qwen35-own', 'qwen3moe', 'qwen4exp')}
print(json.dumps(out))
'''


def paused_checks(S):
    """without CAMINUS_ALLOW_PAUSED: the paused models are left out of the catalog and refused by pull, download, start and
    caminus serve's architecture check, each with the reason; the five shipped models stay"""
    env = dict(PAUSED_ENV, CAMINUS_CONFIG=os.path.join(S, 'paused-none.toml'), CAMINUS_MODEL_DIR=os.path.join(S, 'paused-models'))
    r = subprocess.run([sys.executable, '-I', '-c', PAUSED_CHILD, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python')],
                       capture_output=True, text=True, env=env)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        check(False, f'paused: the child printed no JSON (rc {r.returncode}): {r.stderr[-400:]}'); return
    check(o['catalog'] == ['coder', 'arex-2', 'swift-1.5', 'thinkingcap', 'qwen3-30b-a3b'] and len(o['all']) == 8,
          f'paused: the catalog offers the five shipped models ({o["catalog"]}), the tree keeps all eight')
    check(o['pull'] == 2 and o['download'] == 2 and 'paused' in r.stderr and 'Qwen3-Coder-Next is paused' in r.stderr,
          f'paused: caminus pull / caminus download refuse a paused model with the reason (pull {o["pull"]}, download {o["download"]})')
    check('paused' in o['start'], f'paused: caminus start refuses a paused model ({o["start"]})')
    check(o['team'] not in (0, None) and 'brain-hands team is paused' in r.stdout + r.stderr,
          f'paused: caminus start --team brain-hands is refused before any server starts (rc {o["team"]})')
    sv = o['serve']
    check(all(sv[a] and 'CAMINUS_ALLOW_PAUSED=1' in sv[a] for a in ('qwen35moe', 'qwen3next'))
          and not any(sv[a] for a in ('qwen35', 'qwen35-own', 'qwen3moe', 'qwen4exp')),
          'paused: caminus serve refuses the qwen35moe and qwen3next architectures only (AREX-2\'s qwen35 is served)')


if __name__ == '__main__':
    sys.exit(main())
