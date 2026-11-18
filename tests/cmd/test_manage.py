"""CPU tests of the management commands (caminus.cmd.manage, caminus.cmd.qwen_code): nothing touches a GPU or the network.

  python -I tests/cmd/test_manage.py --scratch DIR     RESULT PASS

Model files are sparse files of the manifest's sizes under a scratch model directory; servers are server.json files with
dead or live (this process's child) guard pids; the Qwen homes are scratch directories (never ~/.config/caminus or ~/.qwen)."""
import argparse
import contextlib
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

    # The basic panel's model picker gets capabilities from the serving registry, without importing the GPU runtime.
    swift_modes = manage.serving_modes(models.BY_NAME['swift-1.5'])
    both = [dict(cards=1, drafts=['off', 'mtp', 'dflash']), dict(cards=2, drafts=['off', 'mtp', 'dflash'])]
    check(swift_modes == both and manage.serving_modes(models.BY_NAME['thinkingcap']) == both
          # (the plain 27B has no two-card form: one card only, so no two-card drafting is offered)
          and manage.serving_modes(models.BY_NAME['qwen3.8-27b']) == [dict(cards=1, drafts=['off', 'mtp', 'dflash'])]
          # (Qwen3-30B-A3B: two cards offered, one card first: the catalog's card count, the picker's default)
          and manage.serving_modes(models.BY_NAME['qwen3-30b-a3b']) == [dict(cards=n, drafts=['off'], kv=['f16', 'q8'], long_context=True) for n in (1, 2)]
          and models.BY_NAME['qwen3-30b-a3b'].cards == 1
          and manage.serving_modes(models.BY_NAME['coder']) == [dict(cards=2, drafts=['off', 'dflash'])]
          and 'torch' not in sys.modules,
          'model picker: supported card counts and drafters from the registry, no torch import; auto stays server-owned')

    # nothing here may ever start a process that could reach a GPU: manage's own Popen is replaced by one that refuses
    class NoLaunch:
        DEVNULL = subprocess.DEVNULL
        STDOUT = subprocess.STDOUT

        @staticmethod
        def Popen(*a, **k):
            raise RuntimeError('test_manage: a launch was attempted (the test must refuse before it)')

        run = staticmethod(subprocess.run)
    manage.subprocess = NoLaunch

    # ---- the 27B models' optional drafter: start passes its file when it is there and adds nothing when it is not (a
    # --no-drafter would turn the server's --draft off, swift-1.5's MTP block with it); the Coder keeps --no-drafter
    for nm in ('swift-1.5', 'thinkingcap'):
        mm = models.BY_NAME[nm]; dk = models.drafter_item(mm)
        base = dict(path='/m/model.gguf', paths={mm.items[0]: '/m/model.gguf', dk: None})
        a0 = manage.serve_argv(mm, base, 'x', 1234)
        a1 = manage.serve_argv(mm, dict(base, paths={mm.items[0]: '/m/model.gguf', dk: '/m/drafter.gguf'}), 'x', 1234)
        en = manage.model_entry(mm)
        check(dk == 'dflash2-27b' and '--no-drafter' not in a0 and '--drafter' not in a0 and a1[a1.index('--drafter') + 1] == '/m/drafter.gguf'
              and en['need_gb'] < en['size_gb'] and en['missing_drafter'] == 'dflash' and en['missing_drafter_gb'] > 0,
              f'{nm}: no drafter flag without its file, --drafter with it; the fit rule sizes the model without the drafter '
              f'({en["need_gb"]} of {en["size_gb"]} GB)')
    cm_ = models.BY_NAME['coder']; dkc = models.drafter_item(cm_)
    check('--no-drafter' in manage.serve_argv(cm_, dict(path='/m/c.gguf', paths={cm_.items[0]: '/m/c.gguf', dkc: None}), 'x', 1234),
          'the Coder without its DFlash file: --no-drafter (its own drafter is DFlash)')
    ce = dict(path='/m/c.gguf', paths={cm_.items[0]: '/m/c.gguf', dkc: None})
    for opts in (['--draft', 'dflash', '--drafter', '/m/custom.gguf'], ['--draft=dflash', '--drafter=/m/custom.gguf'],
                 ['--drafter', '/m/custom.gguf'], ['--draft', 'dflash']):
        got = manage.serve_argv(cm_, ce, 'x', 1234, extra=opts)
        check('--no-drafter' not in got and got[-len(opts):] == opts,
              'the Coder preserves an explicit drafter file or DFlash choice instead of adding --no-drafter')

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
                                                                   'thinkingcap': False, 'qwen3.8-27b': False, 'qwen3-30b-a3b': False},
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
    q3m, q3e = models.BY_NAME['qwen3-30b-a3b'], manage.model_entry(models.BY_NAME['qwen3-30b-a3b'])
    check(manage.pick_cards(q3m, q3e, None, set(), cards=2) == [1, 0],
          'cards: Qwen3-30B-A3B --cards 2 takes the pair in engine order (the bigger XTX first: its 512-row expert share)')
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
    fault = os.path.join(S, 'k3', 'GPU_FAULT_STOP')         # (a scratch stop file: never the real one)
    os.makedirs(os.path.dirname(fault), exist_ok=True)
    with open(fault, 'w') as f:
        f.write('fault\n')
    guard_refusal = f'GUARD REFUSED: a GPU fault was recorded ({fault}). Stop all GPU work and report to the lead.'
    put('guard-refused', dead.pid, guard_refusal + '\n', 18095)
    gone = os.path.join(S, 'k3', 'GPU_FAULT_STOP.cleared')
    put('guard-cleared', dead.pid, guard_refusal.replace(fault, gone) + '\n', 18097)
    v = {x['name']: x for x in map(manage.server_view, manage.all_servers())}
    check(v['crashed']['state'] == 'failed' and 'something broke' in v['crashed']['error'], f"ps: a crashed server is failed ({v['crashed']['error']})")
    check(v['clean']['state'] == 'stopped', 'ps: a cleanly stopped server is stopped')
    check(v['guard-refused']['state'] == 'failed' and v['guard-refused']['error'] == guard_refusal,
          'ps: a guard refusal on a fault file that is still there preserves the startup failure reason')
    check(v['guard-cleared']['state'] == 'stopped' and v['guard-cleared']['error'] == '' and 'cleared' in v['guard-cleared']['note'],
          f"ps: a guard refusal whose fault file is gone is re-checked: stopped, with a note ({v['guard-cleared']})")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        manage.cmd_ps(['--json'])
    listed = {s['name']: s for s in json.loads(buf.getvalue())['servers']}
    refused = listed['guard-refused']
    check(refused['error'] == guard_refusal, 'ps --json: the panel receives the guard refusal reason')
    # a dead record of a paused model (AREX-2's old brain) is left out of caminus ps; with CAMINUS_ALLOW_PAUSED it shows
    put('old-brain', dead.pid, guard_refusal + '\n', 18098)
    put('kept', dead.pid, guard_refusal + '\n', 18099)
    for n, model in (('old-brain', 'arex-2'), ('kept', 'coder')):
        rec = manage.load_server(n); rec['model'] = model
        with open(os.path.join(manage.server_dir(n), 'server.json'), 'w') as f:
            json.dump(rec, f)
    shown_allowed = {s['name'] for s in manage.shown_servers()}
    os.environ.pop('CAMINUS_ALLOW_PAUSED')
    try:
        shown_paused = {s['name'] for s in manage.shown_servers()}
    finally:
        os.environ['CAMINUS_ALLOW_PAUSED'] = '1'
    check('old-brain' in shown_allowed and 'old-brain' not in shown_paused and 'kept' in shown_paused,
          f'ps: a dead record of a paused model is hidden ({sorted(shown_paused)})')
    for n in ('old-brain', 'kept'):
        with contextlib.suppress(OSError):
            os.remove(os.path.join(manage.server_dir(n), 'server.json'))
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

    # ---- --kv (caminus start --kv, or `-- --kv`): checked against the model's entry before any card is picked (only
    # qwen3-30b-a3b takes q8, one card or two), threaded into the card fit's projection, shown by ps from /props
    q3 = models.BY_NAME['qwen3-30b-a3b']; sw = models.BY_NAME['swift-1.5']
    check(manage.start_kv(q3, 1, 'q8', []) == 'q8' and manage.start_kv(q3, 2, None, ['--kv', 'q8']) == 'q8'
          and manage.start_kv(q3, 1, None, ['--kv=f16']) == 'f16' and manage.start_kv(q3, 1, None, []) is None,
          'start --kv: qwen3-30b-a3b takes q8 on one card and two (--kv or the serve flag); none given: the server\'s default')
    refused = []
    for mm, nc, kv, extra in ((sw, 1, 'q8', []), (sw, 2, None, ['--kv', 'q8']), (models.BY_NAME['coder'], 2, 'q8', []),
                              (q3, 1, 'q8', ['--kv', 'f16']), (q3, 1, None, ['--kv', 'q4'])):
        try:
            manage.start_kv(mm, nc, kv, extra); refused.append((mm.name, kv, extra, 'accepted'))
        except manage.ManageError as err:
            if not ("kv 'q8'" in str(err) and 'qwen3-30b-a3b' in str(err) or 'disagree' in str(err) or "'f16' or 'q8'" in str(err)):
                refused.append((mm.name, kv, extra, str(err)))
    check(not refused, f'start --kv: q8 refused for swift-1.5 (one card or two) and the Coder, naming the model that takes it; '
                       f'--kv and the serve flag disagreeing, or another format, refused ({refused or "all refused"})')
    real_start_one, team_starts = manage.start_one, []
    manage.start_one = lambda *x, **k: team_starts.append((x, k)) or {}
    team_out = []
    try:
        for argv in (['--team', 'brain-hands', '--kv', 'q8'], ['--team', 'brain-hands', '--kv', 'f16'],
                     ['--team', 'brain-hands', '--', '--kv', 'q8']):
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    rc = manage.cmd_start(argv)
            except SystemExit as e:
                rc = e.code
            team_out.append((rc, err.getvalue()))
    finally:
        manage.start_one = real_start_one
    check(not team_starts and all(rc == 2 and '--kv applies to one model, not to --team brain-hands' in e for rc, e in team_out),
          f'start --team with --kv (or `-- --kv`) is refused before any server starts, not dropped '
          f'({[(rc, e.strip().splitlines()[-1:]) for rc, e in team_out]}, starts {len(team_starts)})')
    from caminus import arch as _ar
    seen = []

    class Rec(_ar.Arch):
        cards = 1
        def auto_need(self, opt, slots, cards=None): seen.append(opt.get('kv')); return 40960, [0], 1.0, lambda c: [1.0]
        def fits(self, opt, cards): seen.append(opt.get('kv')); return True
        def variant(self, cards): return self
    real_of = _ar.of
    _ar.of = lambda path: Rec('rec', 'rec')
    try:
        ent = dict(path='/m/q3.gguf', size_gb=18.6, need_gb=18.6, paths={})
        manage.fits_fn(ent, kv='q8')(dict(line_mib=24000, display=False)); manage.pair_fits(q3, ent, cards=2, kv='q8')([{}, {}])
        manage.fits_fn(ent)(dict(line_mib=24000, display=False))
    finally:
        _ar.of = real_of
    check(seen == ['q8', 'q8', 'q8', 'f16', 'f16'], f'start --kv: the one-card and two-card fits project the asked cache (opt kv {seen})')
    sv = dict(name='kvq8', model='qwen3-30b-a3b', path='/m/q3.gguf', port=18099, devices=[0], card_names=['XTX'], argv=[],
              guard_pid=1, log=os.path.join(S, 'none.log'), started=time.time(), alias='qwen3-30b-a3b', kv='q8')
    props = {'n_ctx': 40960, 'model_alias': 'qwen3-30b-a3b', 'caminus': {'kv_cache': {'type': 'q8', 'format': 'q8_0-v1:x'}}}
    real_alive, real_http = manage.alive, manage.http_json
    manage.alive = lambda pid: True
    manage.http_json = lambda url, timeout=1.0: (200, props if url.endswith('/props') else {'status': 'ok'})
    try:
        v = manage.server_view(sv)
        props_f16 = dict(props, caminus={})                     # (an older server: no kv_cache key -> nothing shown)
        manage.http_json = lambda url, timeout=1.0: (200, props_f16 if url.endswith('/props') else {'status': 'ok'})
        v_old = manage.server_view(sv)
        buf = io.StringIO()
        real_all = manage.all_servers
        manage.all_servers = lambda: [sv]
        manage.http_json = lambda url, timeout=1.0: (200, props if url.endswith('/props') else {'status': 'ok'})
        try:
            with contextlib.redirect_stdout(buf):
                manage.ROOT = TH.tree(os.path.join(S, 'sysfs-kv'), [TH.card('card1', '0000:03:00.0', TH.XTX, 128, display=True)])
                manage.cmd_ps([])
        finally:
            manage.all_servers = real_all; manage.ROOT = keep
    finally:
        manage.alive, manage.http_json = real_alive, real_http
    check(v['state'] == 'ready' and v['kv'] == 'q8' and v_old['kv'] is None and 'kv q8' in buf.getvalue() and '40,960 ctx' in buf.getvalue(),
          f'ps: the KV cache from /props (q8; an older server without the key: none shown): {buf.getvalue().splitlines()[:1]}')

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
    unmanaged_checks(S)
    reload_checks(S)
    restart_checks(S)
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


def restart_checks(S):
    """caminus restart of a server recorded before its catalog file moved (Qwen3-30B-A3B: the official file's path, before the
    switch to unsloth's file in its own folder): refused before the stop while the catalog file is missing; with it, the
    restart serves the catalog file, not the recorded path. A user's own --model (`caminus start NAME -- --model X`, serve
    takes the last one) survives a restart, and needs no catalog file"""
    from caminus import models
    from caminus.cmd import manage
    m = models.BY_NAME['qwen3-30b-a3b']
    it = models.BY_KEY['qwen3-30b-a3b']
    old = os.path.join(S, 'old-models', 'Qwen3-30B-A3B-GGUF', os.path.basename(it.path))
    own = os.path.join(S, 'own-models', 'Qwen3-30B-A3B-Q5_K_M.gguf')
    os.makedirs(os.path.dirname(own), exist_ok=True)
    with open(own, 'wb') as f:
        f.write(b'GGUF')
    kv = ['--cache-dir', os.path.join(S, 'kv', 'q3-restart')]
    recs = dict(plain=dict(name='q3-restart', argv=['--model', old, '--port', '8100'] + kv),
                own=dict(name='q3-own', argv=['--model', old, '--port', '8100'] + kv + ['--model', own]))
    for r in recs.values():
        r.update(model='qwen3-30b-a3b', path=old, port=8100, devices=[0], guard_pid=999999999)
        d = manage.server_dir(r['name']); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'server.json'), 'w') as f:
            json.dump(r, f)
    calls = []
    real_stop, real_start = manage.stop_one, manage.start_one
    manage.stop_one = lambda n, log=print: calls.append(('stop', n))
    manage.start_one = lambda *x, **k: calls.append(('start', x, k)) or dict(name=k.get('name'), port=8101)

    def restart(name):
        calls.clear()
        with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
            return manage.cmd_restart([name])
    new = os.path.join(os.environ['CAMINUS_MODEL_DIR'], models.local_rel(it))
    try:
        if os.path.exists(new): os.remove(new)
        rc = restart('q3-restart')
        check(rc != 0 and calls == [], f'restart: the catalog file missing: refused before the stop, the server keeps running ({rc}, {calls})')
        rc = restart('q3-own')
        check(rc == 0 and [c[0] for c in calls] == ['stop', 'start'] and calls[1][2]['recorded_path'] == old,
              f'restart: the catalog file missing, a user\'s own --model: restarted, no catalog file needed ({rc}, {[c[0] for c in calls]})')
        e0 = manage.model_entry(m)
        check(not e0['present'] and manage.restart_argv(recs['own']['argv'], e0, 8102, m, old) ==
              ['--model', old, '--port', '8102'] + kv + ['--model', own] and not manage.needs_catalog(recs['own']['argv'], m, old),
              'restart: without the catalog file the argv is kept (only the port changes); it does not depend on the catalog file')
        os.makedirs(os.path.dirname(new), exist_ok=True)
        with open(new, 'wb') as f:
            f.truncate(it.size)
        rc = restart('q3-restart')
        check(rc == 0 and [c[0] for c in calls] == ['stop', 'start'] and calls[1][2]['argv_override'] == recs['plain']['argv'],
              f'restart: stop, then start with the recorded argv ({[c[0] for c in calls]})')
    finally:
        manage.stop_one, manage.start_one = real_stop, real_start
    e = manage.model_entry(m)
    got = manage.restart_argv(recs['plain']['argv'], e, 8102, m, old)
    check(e['path'] == new and got == ['--model', new, '--port', '8102'] + kv,
          f'restart: --model at the catalog file found now ({models.local_rel(it)}), not the recorded old path; the new port')
    got = manage.restart_argv(recs['own']['argv'], e, 8102, m, old)
    check(got == ['--model', new, '--port', '8102'] + kv + ['--model', own],
          'restart: a user\'s own --model survives (the catalog one re-pointed, the user\'s last one kept: serve takes the last)')
    # a recorded --model the catalog identifies by its file (the official file kept, a PREVIOUS_FILES entry), whatever the
    # record's path says, is re-pointed too
    prev = os.path.join(S, 'prev-models', models.PREVIOUS_FILES['qwen3-30b-a3b'][0][0])
    os.makedirs(os.path.dirname(prev), exist_ok=True)
    with open(prev, 'wb') as f:
        f.truncate(models.PREVIOUS_FILES['qwen3-30b-a3b'][0][1])
    check(manage.restart_argv(['--model', prev], e, 1, m, None) == ['--model', new] and manage.catalog_arg(new, m)
          and not manage.catalog_arg(own, m),
          'restart: a --model model_of takes for this model (its catalog file, the official file kept) is re-pointed; a user file is not')


def reload_checks(S):
    """caminus start's launcher reloads a server that exited 75 (a broken engine) once, and passes any other exit through:
    the real loop of guarded.command_reloading around a stub in the guard's place"""
    from caminus.cmd import guarded
    d = os.path.join(S, 'reload'); os.makedirs(d, exist_ok=True)
    stub, cnt, log = os.path.join(d, 'stub.sh'), os.path.join(d, 'n'), os.path.join(d, 'serve.log')
    with open(stub, 'w') as f:   # (exits with the codes in $CODES, one per run)
        f.write('n=$(cat "$CNT" 2>/dev/null || echo 0); echo $((n + 1)) > "$CNT"\n'
                'set -- $CODES; shift "$n"; exit "${1:-0}"\n')
    cmd = guarded.command_reloading('serve', ['--port', '1'], log, 0)
    i = cmd.index('caminus-reload')
    for codes, want_rc, want_runs in (('75 0', 0, 2), ('75 75', 75, 2), ('99', 99, 1), ('0', 0, 1)):
        if os.path.exists(cnt): os.remove(cnt)
        env = dict(os.environ, CNT=cnt, CODES=codes, CAMINUS_RELOAD_SLEEP='0')
        r = subprocess.run(cmd[:i + 1] + [stub, log], env=env, capture_output=True, text=True, timeout=30)
        runs = int(open(cnt).read().strip())
        check(r.returncode == want_rc and runs == want_runs,
              f'reload: exits {codes} -> launcher rc {r.returncode} after {runs} run(s) (want {want_rc} after {want_runs}: '
              f'75 reloads once, a fault or a stop never)')


def paused_checks(S):
    """without CAMINUS_ALLOW_PAUSED: the paused models are left out of the catalog and refused by pull, download, start and
    caminus serve's architecture check, each with the reason; the shipped models stay"""
    env = dict(PAUSED_ENV, CAMINUS_CONFIG=os.path.join(S, 'paused-none.toml'), CAMINUS_MODEL_DIR=os.path.join(S, 'paused-models'))
    r = subprocess.run([sys.executable, '-I', '-c', PAUSED_CHILD, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python')],
                       capture_output=True, text=True, env=env)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        check(False, f'paused: the child printed no JSON (rc {r.returncode}): {r.stderr[-400:]}'); return
    check(o['catalog'] == ['coder', 'swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b'] and len(o['all']) == 9,
          f'paused: the catalog offers the five shipped models ({o["catalog"]}), the tree keeps all nine')
    check(o['pull'] == 2 and o['download'] == 2 and 'paused' in r.stderr and 'Qwen3-Coder-Next is paused' in r.stderr,
          f'paused: caminus pull / caminus download refuse a paused model with the reason (pull {o["pull"]}, download {o["download"]})')
    check('paused' in o['start'], f'paused: caminus start refuses a paused model ({o["start"]})')
    check(o['team'] not in (0, None) and 'brain-hands team is paused' in r.stdout + r.stderr,
          f'paused: caminus start --team brain-hands is refused before any server starts (rc {o["team"]})')
    sv = o['serve']
    check(all(sv[a] and 'CAMINUS_ALLOW_PAUSED=1' in sv[a] for a in ('qwen35moe', 'qwen3next', 'qwen35'))
          and not any(sv[a] for a in ('qwen35-own', 'qwen3moe', 'qwen4exp')),
          'paused: caminus serve refuses the qwen35moe, qwen3next and qwen35 (AREX-2) architectures only')


def unmanaged_checks(S):
    """unmanaged_servers() scans /proc for caminus_serve.py processes outside the registry and never raises, even on a
    machine with none running (CI): every entry is a dict with the keys cmd_ps and cmd_top read"""
    from caminus.cmd import manage
    try:
        found = manage.unmanaged_servers()
    except Exception as e:
        check(False, f'unmanaged: unmanaged_servers() raised {type(e).__name__}: {e}')
        return
    check(isinstance(found, list)
          and all(isinstance(s, dict) and isinstance(s.get('port'), int) and s.get('name') and 'model' in s
                  and 'url' in s and 'state' in s for s in found),
          f'unmanaged: read-only scan of /proc for live servers outside the registry -> {len(found)} entry(ies), each '
          f'with port, name, model, url and state ({found})')
    ports = [s['port'] for s in found]
    check(ports == sorted(ports) and len(ports) == len(set(ports)),
          f'unmanaged: one entry per server, sorted by port ({ports})')
    check(manage.serve_argv_of(['python', '-I', 'tools/caminus_serve.py', '--model', 'x.gguf'])
          and manage.serve_argv_of(['/usr/bin/caminus', 'serve', '--port', '8100'])
          and manage.serve_argv_of(['python', '-I', '-c', 'from caminus.__main__ import main', 'serve'])
          and not manage.serve_argv_of(['python', '-I', '-c', 'from caminus.__main__ import main', 'ps', '--json'])
          and not manage.serve_argv_of(['nginx', 'serve']),
          'unmanaged: caminus_serve.py and the CLI\'s serve command are servers; other caminus commands and programs are not')
    # a fake /proc: a server started by a script without --port (it listens on 8092), its wrapper, and a guard's command
    # line; the port comes from the socket table, the catalog name from /props' model path (none here: the file label)
    proc = os.path.join(S, 'proc')
    def proc_pid(pid, argv, sockets=()):
        d = os.path.join(proc, str(pid))
        os.makedirs(os.path.join(d, 'fd'), exist_ok=True)
        with open(os.path.join(d, 'cmdline'), 'wb') as f:
            f.write(b'\0'.join(a.encode() for a in argv) + b'\0')
        with open(os.path.join(d, 'stat'), 'w') as f:
            f.write(f'{pid} (x) S ' + ' '.join(['0'] * 18) + ' 100 0\n')
        for i, ino in enumerate(sockets):
            with contextlib.suppress(FileExistsError):
                os.symlink(f'socket:[{ino}]', os.path.join(d, 'fd', str(i + 3)))
    proc_pid(4101, ['timeout', '9h', 'python', '-I', 'tools/caminus_serve.py', '--model', '/m/Coder-00001-of-00002.gguf'])
    proc_pid(4102, ['python', '-I', 'tools/caminus_serve.py', '--max-ctx', '1048576', '--model', '/m/Coder/IQ1_M/Coder-00001-of-00002.gguf'],
             sockets=('7777',))
    proc_pid(4103, ['bash', '/x/guard.sh', 'log', '0', 'python', '-I', 'tools/caminus_serve.py', '--port', '18200'])
    os.makedirs(os.path.join(proc, 'net'), exist_ok=True)
    with open(os.path.join(proc, 'net', 'tcp'), 'w') as f:
        f.write('  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n'
                '   0: 0100007F:1F9C 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 7777 1\n')
    with open(os.path.join(proc, 'uptime'), 'w') as f:
        f.write('5000.0 1.0\n')
    check(manage.listen_ports(4102, proc) == [8092] and manage.listen_ports(4101, proc) == [],
          'unmanaged: the listening port is read from the process\'s sockets')
    real_http, real_all = manage.http_json, manage.all_servers
    asked = []
    def fake_http(url, timeout=1.0):
        asked.append(url)
        if url.endswith('/health'):
            return (200, {'status': 'ok'}) if ':8092/' in url else (None, None)
        if url.endswith('/props'):
            return 200, {'engine': 'caminus', 'model_alias': 'k3-coder', 'model_path': '/m/none.gguf',
                         'default_generation_settings': {'n_ctx': 1048576}}
        return 200, {'version': 1, 'engine': 'caminus', 'decode': {'ms_per_token': 10.0}}
    manage.http_json, manage.all_servers = fake_http, lambda: []
    try:
        seen = manage.unmanaged_servers(proc)
    finally:
        manage.http_json, manage.all_servers = real_http, real_all
    check(len(seen) == 1 and seen[0]['port'] == 8092 and seen[0]['pid'] == 4102 and seen[0]['external'] is True
          and seen[0]['state'] == 'ready' and seen[0]['max_ctx'] == 1048576 and seen[0]['decode_tps'] == 100.0
          and not any(':18200/' in u for u in asked),
          f'unmanaged: a server started without --port is found on its socket\'s port, once, marked external; a guard\'s '
          f'command line is not a server ({seen})')
    manage.http_json, manage.all_servers = (lambda url, timeout=1.0: (200, {'engine': 'llama.cpp'} if url.endswith('/props') else {})), (lambda: [])
    try:
        other = manage.unmanaged_servers(proc)
    finally:
        manage.http_json, manage.all_servers = real_http, real_all
    check(other == [], f'unmanaged: a port answered by another engine is not this process\'s server ({other})')
    # caminus top NAME picks the named outside server, not the first one
    configs = []
    real = (manage.all_servers, manage.unmanaged_servers, manage.top_config, manage.exec_panel)
    manage.all_servers = lambda: []
    manage.unmanaged_servers = lambda: [dict(name='observed-8092', port=8092), dict(name='observed-8093', port=8093)]
    manage.top_config = lambda s: configs.append(s['name']) or '/dev/null'
    manage.exec_panel = lambda argv: 0
    try:
        rc = manage.cmd_top(['observed-8093'])
    finally:
        manage.all_servers, manage.unmanaged_servers, manage.top_config, manage.exec_panel = real
    check(rc == 0 and configs == ['observed-8093'], f'top: the named outside server is watched, not the first ({rc} {configs})')


if __name__ == '__main__':
    sys.exit(main())
