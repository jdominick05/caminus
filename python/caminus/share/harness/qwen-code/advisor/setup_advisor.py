"""qwen-advisor's interactive setup (run by `caminus advisor setup`, which makes the Python environment for the hosted
advisors' SDKs first; --local-only needs none):
  1. installs a copy of the hook into ~/.local/share/qwen-advisor/app (the hook never runs from this source tree)
  2. asks for the advisors (Anthropic, Mistral, or both with one as the fallback), their models, effort and API keys
     (an existing ~/.config/anthropic/api_key or MISTRAL_API_KEY can be used as is; a pasted key is stored 0600 and never
     printed), and writes ~/.config/qwen-advisor/config.toml (0600)
  3. offers one live test request per advisor
  4. shows the hook entries and, when you agree, merges them into the Qwen Code settings file you choose (a timestamped
     backup first; earlier qwen-advisor entries are replaced, nothing else changes), and adds the /advise command

  setup_advisor.py [--uninstall | --update] [--settings FILE ...] [--local-only] [--non-interactive] [--no-live-test] [--dry-run]"""
import argparse
import datetime
import getpass
import json
import os
import pathlib
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen_advisor import advisor, config  # noqa: E402

NAME = 'qwen-advisor'
APP = os.path.join(config.SHARE_DIR, 'app')
VENV_PY = os.path.join(config.SHARE_DIR, 'venv', 'bin', 'python')
ANTHROPIC_MODELS = ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1')
A = None   # argparse namespace


# ---- small terminal helpers ---------------------------------------------------------------------------------------
def ask(prompt, default=''):
    if A.non_interactive:
        return default
    v = input(f'{prompt}' + (f' [{default}]' if default != '' else '') + ': ').strip()
    return v or default


def yes(prompt, default=True):
    if A.non_interactive:
        return default
    v = input(f'{prompt} [{"Y/n" if default else "y/N"}]: ').strip().lower()
    return default if not v else v.startswith('y')


def choose(prompt, options, default=0):
    if A.non_interactive:
        return options[default]
    for i, o in enumerate(options, 1):
        print(f'  {i}. {o}')
    while True:
        v = input(f'{prompt} [{default + 1}]: ').strip()
        if not v:
            return options[default]
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1]
        if v in options:
            return v
        print('  (a number from the list)')


def write_private(path, text):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp-')
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def toml_dump(d):
    """the config dict as TOML (strings, ints, bools; one level of tables)"""
    def val(v):
        if isinstance(v, list):
            return '[' + ', '.join(json.dumps(str(x)) for x in v) + ']'
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return str(v)
        return json.dumps(str(v))
    top = [f'{k} = {val(v)}' for k, v in d.items() if not isinstance(v, dict)]
    tables = [f'\n[{k}]\n' + '\n'.join(f'{kk} = {val(vv)}' for kk, vv in v.items()) for k, v in d.items() if isinstance(v, dict)]
    return '# qwen-advisor settings (caminus advisor setup writes this; edit freely). Keys are not stored here: see the key = lines.\n' + \
        '\n'.join(top) + '\n' + '\n'.join(tables) + '\n'


# ---- 1. the installed copy ------------------------------------------------------------------------------------------
def install_app():
    src = os.path.join(HERE, 'qwen_advisor')
    new = APP + '.new'
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new, exist_ok=True)
    shutil.copytree(src, os.path.join(new, 'qwen_advisor'), ignore=shutil.ignore_patterns('__pycache__'))
    if os.path.isdir(APP):
        old = APP + '.old'
        shutil.rmtree(old, ignore_errors=True)
        os.replace(APP, old)
        os.replace(new, APP)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(new, APP)
    print(f'installed the hook into {APP}')


def hook_command():
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    return f'{py} -I {os.path.join(APP, "qwen_advisor", "hook.py")}'


# ---- 2. providers, keys, config -------------------------------------------------------------------------------------
def key_for(provider, cur):
    """where this provider's key comes from: an existing file / env var, or a pasted key stored 0600"""
    std = os.path.join(config.CONFIG_DIR, 'keys', provider)
    cands = []
    if provider == 'anthropic' and os.path.exists(os.path.expanduser('~/.config/anthropic/api_key')):
        cands.append('~/.config/anthropic/api_key')
    env = 'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'MISTRAL_API_KEY'
    if os.environ.get(env):
        cands.append(f'env:{env}')
    if cur and config.read_key(cur) and cur not in cands:
        cands.insert(0, cur)
    for c in cands:
        if yes(f'  use the {provider} key from {c}?', True):
            return c
    if A.non_interactive:
        return cur or std
    k = getpass.getpass(f'  paste your {provider} API key (input hidden; stored in {std}, mode 600): ').strip()
    if not k:
        print('  no key given: this advisor stays unconfigured')
        return cur or std
    write_private(std, k + '\n')
    return std


def anthropic_models(keyspec):
    """every model the key can use, newest first, as (id, label) from the Models API; the built-in list when it fails"""
    key = config.read_key(keyspec)
    if key:
        try:
            import anthropic
            ms = list(anthropic.Anthropic(api_key=key, timeout=20.0, max_retries=1).models.list())
            if ms:
                return [(m.id, getattr(m, 'display_name', '') or '') for m in ms]
        except Exception as e:
            print(f'  (could not list Anthropic models: {type(e).__name__}: {str(e)[:120]}; showing the built-in list)')
    return [(m, '') for m in ANTHROPIC_MODELS]


def pick_model(ms, current, prefer=()):
    """the model from a full list: a number, an id from the list, or any other id typed in"""
    ids = [m for m, _ in ms]
    default = current if current in ids else next((m for w in prefer for m in ids if w in m), ids[0] if ids else '')
    if A.non_interactive or not ids:
        return default if ids else ask('  model id', current or '')
    print(f'  {len(ids)} models available on this key:')
    for i, (m, label) in enumerate(ms, 1):
        print(f'  {i:3d}. {m}' + (f'  ({label})' if label and label != m else ''))
    while True:
        v = input(f'  model (number or id) [{default}]: ').strip()
        if not v:
            return default
        if v.isdigit() and 1 <= int(v) <= len(ids):
            return ids[int(v) - 1]
        if v in ids or yes(f'  {v!r} is not in the list; use it anyway?', False):
            return v


def mistral_models(keyspec):
    """chat models the key can use, as (id, 'reasoning' when it takes reasoning_effort, else ''); [] when listing fails"""
    key = config.read_key(keyspec)
    if not key:
        return []
    try:
        from mistralai.client import Mistral
        res = Mistral(api_key=key, timeout_ms=20000).models.list()
        out = []
        for m in getattr(res, 'data', None) or []:
            caps = getattr(m, 'capabilities', None)
            if caps is not None and getattr(caps, 'completion_chat', True) is False:
                continue
            out.append((m.id, 'reasoning' if caps is not None and getattr(caps, 'reasoning', False) else ''))
        return sorted(set(out))
    except Exception as e:
        print(f'  (could not list Mistral models: {type(e).__name__}: {str(e)[:120]})')
        return []


def local_models(url, keyspec):
    """the ids a local OpenAI-compatible server lists at /models ([] when it is down or lists none)"""
    import json, urllib.request
    req = urllib.request.Request(url.rstrip('/') + '/models')
    key = config.read_key(keyspec)
    if key:
        req.add_header('Authorization', f'Bearer {key}')
    try:
        with urllib.request.urlopen(req, timeout=5) as f:
            return [(m.get('id'), '') for m in json.loads(f.read()).get('data') or [] if m.get('id')]
    except Exception as e:
        print(f'  (could not list models at {url}: {type(e).__name__}: {str(e)[:120]})')
        return []


def configure_local(pc, cfg_budget):
    print('  Any OpenAI-compatible server works: Caminus or llama.cpp serving a model on another card, vLLM, Ollama ...')
    pc['url'] = ask('  server URL (up to /v1)', pc.get('url') or 'http://127.0.0.1:8093/v1').strip()
    ks = ask('  API key: a file path or env:NAME (Enter for none; most local servers need none)', pc.get('key') or '').strip()
    pc['key'] = '' if ks.lower() == 'none' else ks
    ms = local_models(pc['url'], pc['key'])
    if ms:
        print(f'  the server lists: {", ".join(m for m, _ in ms[:8])}')
        pc['model'] = pick_model(ms, pc.get('model') or ms[0][0], ())
    else:
        pc['model'] = ask('  model id (Enter: whatever the server serves)', pc.get('model') or '').strip()
    pc['max_tokens'] = int(ask('  max tokens per reply (a thinking model needs room to think)', pc.get('max_tokens') or 8000))
    print('  A late reply adds nothing. A thinking model on one card can take a minute; a model id without thinking\n'
          '  (on Caminus, the -nothink ids) skips that and answers much sooner.')
    pc['timeout_s'] = int(float(ask('  seconds to wait for its reply', pc.get('timeout_s') or 120)))
    if float(cfg_budget[0]) < pc['timeout_s'] + 10:
        cfg_budget[0] = pc['timeout_s'] + 30
        print(f'  (the whole advisor chain may now take up to {cfg_budget[0]} s: chain_budget_s raised to match)')
    pc['effort'] = ''


def configure():
    cfg = config.load() if os.path.exists(config.CONFIG_PATH) else config.load('/nonexistent')
    print('\nWhich outside advisors should steer the local model?')
    plans = [('Anthropic', ('anthropic', '')), ('Mistral', ('mistral', '')),
             ('Anthropic first, Mistral as the fallback', ('anthropic', 'mistral')),
             ('Mistral first, Anthropic as the fallback', ('mistral', 'anthropic')),
             ('Local: a model on your own server (e.g. your second card), no cloud', ('local', '')),
             ('Local first, Anthropic as the fallback', ('local', 'anthropic')),
             ('Local first, Mistral as the fallback', ('local', 'mistral'))]
    if A.local_only:
        plans = [p for p in plans if p[1] == ('local', '')]
        cfg['chain'] = [c for c in (cfg.get('chain') or []) if str(c).startswith('local:')]
    now = (cfg['provider'], cfg.get('fallback_provider') or '')
    cur = next((i for i, (_, o) in enumerate(plans) if o == now), 0)
    plan = choose('advisors', [n for n, _ in plans], cur)
    order = dict(plans)[plan]
    cfg['provider'], cfg['fallback_provider'] = order
    for p in [x for x in order if x]:
        pc = cfg[p]
        print(f'\n{p.capitalize()}:')
        if p == 'local':
            budget = [cfg.get('chain_budget_s') or 150]
            configure_local(pc, budget)
            cfg['chain_budget_s'] = int(float(budget[0]))
            continue
        pc['key'] = key_for(p, pc.get('key'))
        if p == 'anthropic':
            pc['model'] = pick_model(anthropic_models(pc['key']), pc.get('model') or 'claude-opus-5-5', ('claude-opus-5-5',))
        else:
            ms = mistral_models(pc['key'])
            if not ms:
                print('  (no model list: type an id, e.g. from https://docs.mistral.ai/getting-started/models/)')
            # (mistral-large-latest is a model without reasoning; Large 4 has it)
            pc['model'] = pick_model(ms, pc.get('model') or '', ('mistral-large-4', 'mistral-large-latest', 'medium'))
            if ms and dict(ms).get(pc['model'], 'reasoning') != 'reasoning':
                pc['effort'] = ''
                print(f'  {pc["model"]} has no reasoning setting: effort left at the model default')
                continue
        efs = list(config.EFFORTS[p]) + ['(model default)']
        de = efs.index(pc['effort']) if pc.get('effort') in efs else (efs.index('medium') if 'medium' in efs else 0)
        e = choose('  effort (how hard it thinks: higher is slower and costs more)', efs, de)
        pc['effort'] = '' if e == '(model default)' else e
    if A.local_only:
        return finish(cfg)
    print('\nWhen the advisors above refuse (hard or soft) or fail, more can be tried in order until one helps.')
    print('  Entries: provider:model[@effort], comma-separated (e.g. mistral:mistral-medium-latest, anthropic:claude-opus-4-8@high,')
    print('  local:<model> for the local server).')
    chain = ask('  extra advisors (Enter keeps the current list; "none" clears it)', ', '.join(cfg.get('chain') or []))
    cfg['chain'] = [] if chain.strip().lower() == 'none' else [x.strip() for x in chain.split(',') if ':' in x]
    return finish(cfg)


def finish(cfg):
    print('\nLimits (Enter keeps the value):')
    cfg['max_calls_per_session'] = int(ask('  advisor calls per Qwen Code session at most', cfg['max_calls_per_session']))
    mins = ask('  minutes between automatic hints (stuck detection, the done-check; /advise is never limited)',
               round(float(cfg.get('min_interval_s') or 0) / 60, 1))
    cfg['min_interval_s'] = int(float(mins) * 60)
    cfg['review_on_stop'] = yes('  also check the work once when the agent says it is done (one call per message)?', bool(cfg['review_on_stop']))
    write_private(config.CONFIG_PATH, toml_dump(cfg))
    print(f'wrote {config.CONFIG_PATH}')
    return cfg


# ---- 3. live test ---------------------------------------------------------------------------------------------------
TEST_BRIEF = """Why you were called: a setup check.

THE USER'S LATEST REQUEST:
Run the unit tests and fix the failing one.

THE AGENT'S RECENT STEPS (oldest first):
CALL run_shell_command: {"command": "python -m pytest -q"}
RESULT run_shell_command: 1 failed, 41 passed. FAILED tests/test_dates.py::test_parse_iso - ValueError: Invalid isoformat string: '2026-11-09T10:00:00Z'
CALL run_shell_command: {"command": "python -m pytest -q"}
RESULT run_shell_command: 1 failed, 41 passed. FAILED tests/test_dates.py::test_parse_iso - ValueError: Invalid isoformat string: '2026-11-09T10:00:00Z'"""


def live_test(cfg):
    for p in [x for x in (cfg['provider'], cfg.get('fallback_provider')) if x]:
        one = dict(cfg, provider=p, fallback_provider='')
        r = advisor.ask(one, TEST_BRIEF)
        if r['ok']:
            print(f'  {p}: OK, {r["model"]} answered in {r["latency_s"]} s ({r["input_tokens"]} in / {r["output_tokens"]} out tokens):')
            lim = float((cfg.get(p) or {}).get('timeout_s') or cfg['timeout_s'])
            if r['latency_s'] > 0.7 * lim:
                print(f'  WARNING: that is close to its {lim:.0f} s limit; a slower reply would be dropped. Run setup again'
                      ' and give it more seconds (setup also updates the hooks\' own time limits), or pick a model without thinking.')
            print('    ' + r['text'].replace('\n', '\n    ')[:600])
        else:
            print(f'  {p}: FAILED: {r["error"]}')


# ---- 4. Qwen Code settings ------------------------------------------------------------------------------------------
def candidate_settings():
    """Caminus's own Qwen homes (caminus qwen-code setup); ~/.qwen is used only when named with --settings"""
    base = os.environ.get('CAMINUS_QWEN_HOME') or os.path.join(os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
                                                               'caminus', 'qwen-home')
    base = os.path.expanduser(base)
    return [p for p in (os.path.join(base, 'settings.json'), os.path.join(base + '-bh', 'settings.json')) if os.path.exists(p)]


def ours(h):
    return isinstance(h, dict) and (h.get('name') == NAME or NAME in str(h.get('command', '')))


def strip_ours(settings):
    hooks = settings.get('hooks')
    if not isinstance(hooks, dict):
        return settings
    for ev in list(hooks):
        groups = hooks[ev] if isinstance(hooks[ev], list) else []
        kept = []
        for g in groups:
            if isinstance(g, dict) and isinstance(g.get('hooks'), list):
                g = dict(g, hooks=[h for h in g['hooks'] if not ours(h)])
                if g['hooks']:
                    kept.append(g)
            else:
                kept.append(g)
        if kept:
            hooks[ev] = kept
        else:
            del hooks[ev]
    if not hooks:
        del settings['hooks']
    return settings


def our_hooks(cfg):
    cmd = hook_command()
    t = int(float(cfg.get('chain_budget_s') or cfg['timeout_s'])) + 15   # (the whole advisor chain, plus margin)
    h = lambda timeout: {'hooks': [{'type': 'command', 'name': NAME, 'command': cmd, 'timeout': timeout}]}
    out = {'PostToolBatch': h(10), 'UserPromptSubmit': h(t)}
    if cfg.get('review_on_stop'):
        out['Stop'] = h(t)
    return out


def write_settings(path, settings):
    bak = f'{path}.bak-{NAME}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    shutil.copy2(path, bak)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.settings-')
    with os.fdopen(fd, 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')
    os.chmod(tmp, os.stat(path).st_mode & 0o777)
    os.replace(tmp, path)
    return bak


def install_hooks(cfg, paths):
    add = our_hooks(cfg)
    for path in paths:
        with open(path) as f:
            s = json.load(f)
        s = strip_ours(s)
        hooks = s.setdefault('hooks', {})
        for ev, group in add.items():
            hooks.setdefault(ev, []).append(group)
        print(f'\n{path}: these hook entries will be added (earlier qwen-advisor entries replaced):')
        print('  ' + json.dumps(add, indent=2).replace('\n', '\n  '))
        if s.get('disableAllHooks'):
            print('  NOTE: this file has "disableAllHooks": true, so no hook runs until you set it to false.')
        if A.dry_run:
            print('  (dry run: nothing written)')
            continue
        if not yes('  write it?', True):
            continue
        bak = write_settings(path, s)
        print(f'  written (backup: {bak})')
        cmd_dir = os.path.join(os.path.dirname(path), 'commands')
        dst = os.path.join(cmd_dir, 'advise.md')
        if not os.path.exists(dst) or '[qwen-advisor: requested]' in pathlib.Path(dst).read_text():
            if yes(f'  add the /advise command ({dst})?', True):
                os.makedirs(cmd_dir, exist_ok=True)
                shutil.copy(os.path.join(HERE, 'commands', 'advise.md'), dst)
                print('  added /advise')


def uninstall(paths):
    for path in paths:
        with open(path) as f:
            s = json.load(f)
        before = json.dumps(s, sort_keys=True)
        s = strip_ours(s)
        if json.dumps(s, sort_keys=True) == before:
            print(f'{path}: no qwen-advisor hooks')
        elif A.dry_run:
            print(f'{path}: would remove the qwen-advisor hooks (dry run)')
        else:
            print(f'{path}: removed the qwen-advisor hooks (backup: {write_settings(path, s)})')
        cmd = os.path.join(os.path.dirname(path), 'commands', 'advise.md')
        if os.path.exists(cmd) and '[qwen-advisor: requested]' in pathlib.Path(cmd).read_text() and not A.dry_run:
            os.remove(cmd)
            print(f'  removed {cmd}')
    print(f'The config ({config.CONFIG_PATH}), stored keys ({os.path.join(config.CONFIG_DIR, "keys")}), the log '
          f'({os.path.join(config.STATE_DIR, "advisor.log")}) and {config.SHARE_DIR} are left in place; delete them by hand if you like.')


def main():
    global A
    ap = argparse.ArgumentParser(description='qwen-advisor setup')
    ap.add_argument('--uninstall', action='store_true')
    ap.add_argument('--settings', action='append', help='a Qwen Code settings.json to edit (default: ask)')
    ap.add_argument('--non-interactive', action='store_true', help='take the current / default answers')
    ap.add_argument('--no-live-test', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='show the settings change, write nothing there')
    ap.add_argument('--skip-install', action='store_true', help='(accepted for compatibility; does nothing)')
    ap.add_argument('--local-only', action='store_true', help='offer only the local advisor (no hosted SDKs installed)')
    ap.add_argument('--update', action='store_true', help='only refresh the installed hook copy (no questions, config and settings untouched)')
    A = ap.parse_args()
    if A.update:
        install_app()
        print('Updated. Config, keys and Qwen Code settings were not touched; running Qwen Code sessions pick it up on the next hook call.')
        return
    found = A.settings or candidate_settings()
    if not A.settings and found and not A.non_interactive:
        print('Qwen Code settings files found:')
        pick = choose('which one gets the hooks', found + ['all of them'], 0)
        found = found if pick == 'all of them' else [pick]
    if A.uninstall:
        return uninstall(found)
    install_app()
    cfg = configure()
    if not A.no_live_test and yes('\nSend one live test request per advisor (a fraction of a cent each)?', True):
        live_test(cfg)
    if not found:
        print('\nNo Qwen Code settings.json found. Add this "hooks" block to yours:')
        print(json.dumps({'hooks': our_hooks(cfg)}, indent=2))
    else:
        install_hooks(cfg, found)
    print(f'\nDone. Restart Qwen Code to load the hooks. Log: {os.path.join(config.STATE_DIR, "advisor.log")}. '
          'Turn it off: enabled = false in the config, QWEN_ADVISOR_OFF=1, or a .qwen-advisor-off file in a project. '
          'Remove it: caminus advisor uninstall')


if __name__ == '__main__':
    main()
