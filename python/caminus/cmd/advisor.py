"""caminus advisor setup | status | update | uninstall: the Qwen Code advisor add-on (docs/advisor.md).

  caminus advisor setup [--local-only] [--settings FILE ...] [--no-live-test] [--dry-run]
  caminus advisor status [--json]
  caminus advisor update
  caminus advisor uninstall [--settings FILE ...]

When the local model looks lost in Qwen Code (failing commands in a row, the same call repeated, long runs), another model
reads the task and the recent steps and sends back a short hint, added to the newest message so the server's prompt cache
keeps its prefix. The advisor is hosted (Anthropic, Mistral) or local: any OpenAI-compatible server, such as a second
model on your other card. `/advise` in Qwen Code asks on demand.

The add-on lives in the package (share/harness/qwen-code/advisor) and installs a copy of its hook into
$XDG_DATA_HOME/qwen-advisor/app; its settings are $XDG_CONFIG_HOME/qwen-advisor/config.toml. setup puts the hooks and the
/advise command into Caminus's own Qwen homes (caminus qwen-code setup makes them); ~/.qwen only when named with
--settings. Hosted advisors need their SDKs: setup installs anthropic and mistralai into $XDG_DATA_HOME/qwen-advisor/venv
(network); --local-only skips that and offers the local advisor alone (standard library only)."""
import argparse
import json
import os
import subprocess
import sys

from .. import config
from . import qwen_code

ADDON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share', 'harness', 'qwen-code', 'advisor')
SETUP = os.path.join(ADDON, 'setup_advisor.py')
SDKS = ('anthropic>=2,<3', 'mistralai>=3,<4')


def share_dir():
    return os.path.join(config.xdg('XDG_DATA_HOME', os.path.join('.local', 'share')), 'qwen-advisor')


def venv_py():
    return os.path.join(share_dir(), 'venv', 'bin', 'python')


def homes():
    """Caminus's Qwen homes that exist (the single-server home, the brain / hands team's)"""
    b = qwen_code.base_home()
    return [h for h in (b, b + '-bh') if os.path.isfile(os.path.join(h, 'settings.json'))]


def addon():
    """the add-on's own modules (config, for the status), imported from the package's copy"""
    if ADDON not in sys.path:
        sys.path.insert(0, ADDON)
    from qwen_advisor import config as acfg
    return acfg


def ensure_venv(say):
    """the hosted advisors' SDKs in the add-on's own environment (created once; pip needs the network)"""
    py = venv_py()
    if not os.path.exists(py):
        say(f'advisor: creating the Python environment for the hosted advisors in {os.path.dirname(os.path.dirname(py))}')
        r = subprocess.run([sys.executable, '-m', 'venv', os.path.dirname(os.path.dirname(py))])
        if r.returncode:
            return None
    say('advisor: installing the anthropic and mistralai SDKs into it')
    r = subprocess.run([py, '-m', 'pip', 'install', '--quiet', '--upgrade', *SDKS])
    return py if r.returncode == 0 else None


def run_setup(py, args):
    return subprocess.run([py, '-I', SETUP, *args]).returncode


def settings_args(a):
    files = a.settings or [os.path.join(h, 'settings.json') for h in homes()]
    return [x for f in files for x in ('--settings', os.path.expanduser(f))]


def cmd_setup(a, say):
    s = settings_args(a)
    if not s:
        say('advisor: no Caminus Qwen home yet: run `caminus qwen-code setup` first (or name a settings file: --settings FILE)')
        return 2
    extra = s + (['--no-live-test'] if a.no_live_test else []) + (['--dry-run'] if a.dry_run else [])
    if a.local_only:
        return run_setup(sys.executable, ['--local-only', *extra])
    py = ensure_venv(say)
    if not py:
        say('advisor: the SDKs could not be installed (no network?). `caminus advisor setup --local-only` needs none.')
        return 1
    return run_setup(py, extra)


def hooked(home):
    """(hooks registered, /advise command present) in one Qwen home"""
    try:
        with open(os.path.join(home, 'settings.json')) as f:
            s = json.load(f)
    except (OSError, ValueError):
        return False, False
    blob = json.dumps(s.get('hooks') or {})
    return 'qwen_advisor' in blob, os.path.isfile(os.path.join(home, 'commands', 'advise.md'))


def cmd_status(a, say):
    acfg = addon()
    cfg = acfg.load() if os.path.exists(acfg.CONFIG_PATH) else None
    app = os.path.join(acfg.SHARE_DIR, 'app', 'qwen_advisor', 'hook.py')
    st = dict(installed=os.path.isfile(app), app=os.path.dirname(app), config=acfg.CONFIG_PATH if cfg else None,
              sdk_env=os.path.exists(venv_py()), homes={})
    if cfg:
        chain = [cfg['provider']] + ([cfg['fallback_provider']] if cfg.get('fallback_provider') else []) + list(cfg.get('chain') or [])
        st.update(enabled=bool(cfg.get('enabled', True)), advisors=chain, min_interval_s=cfg.get('min_interval_s'),
                  review_on_stop=bool(cfg.get('review_on_stop')))
        if 'local' in chain or any(str(c).startswith('local:') for c in chain):
            st['local'] = {k: cfg['local'].get(k) for k in ('url', 'model', 'timeout_s')}
    for h in [qwen_code.base_home(), qwen_code.base_home() + '-bh']:
        if os.path.isdir(h):
            hk, cmd = hooked(h)
            st['homes'][h] = dict(hooks=hk, advise_command=cmd)
    if a.json:
        print(json.dumps(st, indent=2))
        return 0
    say(f'advisor: hook {"installed in " + st["app"] if st["installed"] else "not installed (caminus advisor setup)"}')
    if not cfg:
        say('advisor: not configured (caminus advisor setup)')
    else:
        say(f'advisor: {"on" if st["enabled"] else "OFF (enabled = false)"}; advisors in order: {", ".join(st["advisors"])}; '
            f'automatic hints at most every {round((st["min_interval_s"] or 0) / 60, 1)} min; '
            f'done-check {"on" if st["review_on_stop"] else "off"}')
        if st.get('local'):
            say(f'advisor: local server {st["local"]["url"]} model {st["local"]["model"] or "(the server\'s)"}, '
                f'{st["local"]["timeout_s"]} s per reply')
    for h, v in st['homes'].items():
        say(f'advisor: {h}: hooks {"yes" if v["hooks"] else "no"}, /advise {"yes" if v["advise_command"] else "no"}')
    if not st['homes']:
        say('advisor: no Caminus Qwen home yet (caminus qwen-code setup)')
    return 0


def main(argv, say=print):
    ap = argparse.ArgumentParser(prog='caminus advisor', description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('setup', help='configure the advisors and put the hooks into Caminus\'s Qwen homes')
    p.add_argument('--local-only', action='store_true', help='the local advisor only: no hosted SDKs, no network')
    p.add_argument('--settings', action='append', help='a Qwen Code settings.json to edit (default: Caminus\'s Qwen homes)')
    p.add_argument('--no-live-test', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='show the settings change, write nothing there')
    p = sub.add_parser('status', help='installed? which advisors? hooked into which homes?')
    p.add_argument('--json', action='store_true')
    sub.add_parser('update', help='refresh the installed hook copy from this package (asks nothing)')
    p = sub.add_parser('uninstall', help='remove the hooks and /advise from the homes (the config stays)')
    p.add_argument('--settings', action='append')
    a = ap.parse_args(argv)
    if a.cmd == 'setup':
        return cmd_setup(a, say)
    if a.cmd == 'status':
        return cmd_status(a, say)
    py = venv_py() if os.path.exists(venv_py()) else sys.executable
    if a.cmd == 'update':
        return run_setup(py, ['--update'])
    if a.cmd == 'uninstall':
        s = settings_args(a)
        if not s:
            say('advisor: no Caminus Qwen home to remove it from (name one: --settings FILE)')
            return 2
        return run_setup(py, ['--uninstall', *s])
    ap.print_help()
    return 2
