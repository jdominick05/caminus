"""caminus wizard: the first run, from an installed package to a served model and Qwen Code (docs/quickstart.md).

  caminus wizard [--yes] [--dry-run]

Six steps. The wizard only orchestrates: each step is a caminus command, printed (`$ caminus ...`) before it runs, so
nothing here is something the CLI can't do:
  1. check the machine: caminus doctor, then caminus doctor --gpu unless a process holds a GPU context already
  2. choose a model: the catalog, with what runs on these cards (caminus.models.SETUPS) and the model to start with on
     this setup preselected (while a server runs: its model); the cards, drafter and context caminus start will use, on
     one line. A running server is kept unless you switch to another model (asked here, before any download: the
     question names every server the switch stops); a server started outside caminus start is kept, steps 3-5 skipped
  3. caminus pull MODEL: the size, the folder and its free space first; refused when it won't fit
  4. caminus stop NAME for a switch, caminus fit (the context that fits, when the files are on disk), caminus start MODEL
  5. caminus qwen-code setup --server NAME (--install when Qwen Code is not installed yet)
  6. Open Qwen Code now? caminus qwen --server NAME
Re-runnable: what is done already (the files on disk, the server running, Qwen Code set up for it) is skipped. It never
runs sudo: where a step needs root (the render group, Node.js), it prints the command and stops or skips that step. No
undo command: caminus stop NAME stops the server, and the files stay in the model directory (caminus doctor prints it).
Supported cards: the RX 7900 XTX and XT; another gfx1100 card is allowed as untested (a hardware report is asked for);
anything else stops at step 1.

--yes      every question takes its default (a running server is kept); Qwen Code is not opened: its command is printed
--dry-run  runs only the read-only steps (caminus doctor without --gpu, caminus pull --dry-run, caminus fit) and prints
           the others; the one thing it writes is its log
The log: wizard.log in the state directory (~/.local/state/caminus): the lines shown and each command's exit code (no
keys; nothing is sent anywhere)."""
import argparse
import os
import platform
import shlex
import sys
import time

from .. import config, models
from ..device import host
from . import doctor, download, manage, qwen_code

STEPS = 6
REPORT = 'https://github.com/jdominick05/caminus/issues/new?template=hardware_report.yml'
GROUPS = 'sudo usermod -aG render,video $USER'     # (docs/quickstart.md "What you need": then log out and back in)


class Stop(Exception):
    """the wizard ends here: rc 1 (a step failed or was refused) or 130 (interrupted)"""

    def __init__(self, rc, text):
        super().__init__(text)
        self.rc = rc


def run_cmd(argv):
    """one caminus command in this process, as `caminus ARGV` runs it (caminus.__main__.main: its configuration-error
    handling included): its exit code"""
    from ..__main__ import main
    try:
        return main(list(argv)) or 0
    except SystemExit as e:                     # (argparse's usage errors)
        return e.code if isinstance(e.code, int) else 1


def kfd_closed(root):
    """the path of /dev/kfd under root when it exists but this user cannot open it (not in the render group), else None"""
    kfd = os.path.join(root, 'dev', 'kfd')
    return kfd if os.path.exists(kfd) and not os.access(kfd, os.R_OK | os.W_OK) else None


class Wizard:
    def __init__(self, yes=False, dry=False, out=None, ask=input):
        self.yes, self.dry, self.out, self.input = yes, dry, out or sys.stdout, ask
        self.color = bool(getattr(self.out, 'isatty', lambda: False)()) and not os.environ.get('NO_COLOR') \
            and os.environ.get('TERM') != 'dumb'
        self.skipped = []
        self.log = None
        p = os.path.join(config.state_dir(), 'wizard.log')
        try:
            os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
            self.log = open(p, 'a')
            self.log.write(f"\n== caminus wizard {time.strftime('%F %T')}" + (' --yes' if yes else '')
                           + (' --dry-run' if dry else '') + '\n')
        except OSError as e:
            self.say(f'(no log: {p}: {e})')
        self.log_path = p

    # ---- output
    def say(self, text='', style=None):
        codes = {'step': '1', 'cmd': '36', 'warn': '33', 'bad': '31', 'ok': '32', 'dim': '2'}
        shown = f'\033[{codes[style]}m{text}\033[0m' if self.color and style else text
        print(shown, file=self.out, flush=True)
        if self.log:
            self.log.write(text + '\n')
            self.log.flush()

    def step(self, n, title):
        self.say()
        self.say(f'Step {n}/{STEPS}: {title}', 'step')

    def skip(self, n, why):
        self.skipped.append(f'step {n}: {why}')
        self.say(f'skipped: {why}', 'warn')

    def run(self, argv, read_only=False):
        """print the command, then run it (a dry run runs only a read-only one); its exit code"""
        self.say('$ caminus ' + shlex.join(argv), 'cmd')
        if self.dry and not read_only:
            self.say('  (dry run: not run)', 'dim')
            return 0
        if self.log:
            self.log.flush()
        rc = run_cmd(argv)
        if self.log:
            self.log.write(f'-> exit {rc}\n')
        return rc

    def ask(self, question, default, valid=None):
        """the answer to a question (the default under --yes or for an empty line); EOF stops the wizard"""
        if self.yes:
            self.say(f'{question} {default} (--yes)')
            return default
        while True:
            try:
                got = self.input(f'{question} ').strip()
            except EOFError:
                raise Stop(1, 'no answer (end of input): stopped; run caminus wizard again to continue') from None
            if self.log:
                self.log.write(f'{question} {got}\n')
            got = got or default
            if valid is None or valid(got):
                return got
            self.say(f'  {got!r}: not one of the choices', 'warn')

    # ---- the steps
    def check(self):
        self.step(1, 'check the machine')
        gs = host.gpus(manage.ROOT)
        rc = self.run(['doctor'], read_only=True)
        u = host.usable(gs)
        if not u:
            for _, _, text in host.problems(gs):
                self.say(text, 'bad')
            raise Stop(1, 'no supported GPU: Caminus runs on AMD RX 7900 XTX / XT (gfx1100) cards. Other AMD hardware: '
                          f'please send a hardware report with the caminus doctor lines above ({REPORT})')
        if rc != 0:
            raise Stop(1, 'caminus doctor found a FAIL: fix each FAIL line above (docs/troubleshooting.md, "Reading caminus '
                          'doctor"), then run caminus wizard again')
        kfd = kfd_closed(manage.ROOT)
        if kfd:
            self.say(f'{kfd} is not open to this user: the GPUs need the render and video groups. Run this yourself (the '
                     'wizard never runs sudo), then log out and back in:', 'bad')
            self.say(f'  {GROUPS}', 'cmd')
            raise Stop(1, 'run caminus wizard again after logging back in')
        untested = [g for g in u if g['status'] == 'untested']
        if untested:
            self.say(', '.join(host.describe(g) for g in untested) + ': not tested with Caminus yet; it is used. Please send '
                     'a hardware report, even when everything works:', 'warn')
            self.say(f'  {REPORT}')
            for line in self.report(gs):
                self.say('  ' + line)
        held = self.gpu_holders()
        if held:
            self.skip(1, f'caminus doctor --gpu (a test kernel on each card) waits for idle cards: {held}')
        elif self.run(['doctor', '--gpu']) != 0:
            raise Stop(1, 'caminus doctor --gpu failed: the cards above did not run the test kernel (docs/troubleshooting.md)')
        return gs

    def report(self, gs):
        """the hardware report's first field, ready to paste (the form also asks for caminus doctor's lines: above)"""
        from .. import __version__
        cards = [g for g in gs if g['status'] != 'unsupported']
        out = ['Your cards: ' + ' + '.join(host.SUPPORTED.get(g['label'], g['label']) for g in cards)]
        for g in cards:
            out.append(f"  {host.describe(g)}: {host.gfx_name(g['gfx'])}, {g['total_mib']:,} MiB, {host.link_text(g['link'])}"
                       + (', drives a display' if g['display'] else '') + f", {g['status']}")
        out.append(f'  caminus {__version__}, {platform.system()} {platform.release()}')
        return out

    def gpu_holders(self):
        """what holds a GPU context now (other processes, running Caminus servers), as text; '' when nothing"""
        ps = doctor.kfd_procs(manage.ROOT) or []
        servers = [s['name'] for s in manage.all_servers() if manage.alive(s.get('guard_pid'))]
        servers += [o['name'] for o in manage.unmanaged_servers()]
        parts = ([f'{len(ps)} process(es) hold a GPU context'] if ps else []) + ([f"server(s) {', '.join(servers)}"] if servers else [])
        return '; '.join(parts)

    def running(self):
        """(the servers caminus start recorded that run, with their state; the servers started outside it)"""
        rs = [dict(s, state=manage.server_view(s)['state']) for s in manage.all_servers() if manage.alive(s.get('guard_pid'))]
        return rs, ([] if rs else manage.unmanaged_servers())

    def choose(self, gs):
        """step 2: (model, cards, plan): plan ('keep', server) or ('start', [servers to stop first]); None when a server
        started outside caminus start holds the cards (steps 3-5 skipped)"""
        self.step(2, 'choose a model')
        running, outside = self.running()
        if outside:
            for o in outside:
                self.say(f"{o['url']} runs a Caminus server started outside caminus start ({o['model'] or 'its model unknown'}): "
                         'kept. The wizard can neither stop it nor set Qwen Code up for it (docs/harnesses.md has the '
                         'addresses for any client)', 'warn')
            self.skip(2, 'a server started outside caminus start holds the cards')
            for n, what in ((3, 'download'), (4, 'start'), (5, 'Qwen Code setup')):
                self.skip(n, f'no {what}: the server started outside caminus start is kept')
            return None
        cards = host.ranked(gs)
        row, why = models.setup_here(gs)
        if why:
            self.say(f'two-card models: {why}; the setup of the first card applies', 'warn')
        self.say('Your cards: ' + ' + '.join(host.describe(g) for g in cards)
                 + (f" (the setup \"{row['row']}\")" if row else ' (no measured setup: every model is offered, untested here)'))
        for s in running:
            self.say(f"running: {s['name']} ({s['model']}) on {'+'.join(s.get('card_names') or [])} at "
                     f"http://127.0.0.1:{s['port']}" + (' (still loading)' if s['state'] == 'loading' else ''))
        names, mark, lines = [], {}, []
        for m in models.CATALOG:
            e = manage.model_entry(m)
            if row is None:
                here, nc = 'untested on these cards', m.cards
            elif m.name in row['runs']:
                here, nc = 'runs here' + (' (or on two cards: caminus start --cards 2)' if f'{m.name}:2' in row['runs'] else ''), m.cards
            elif f'{m.name}:2' in row['runs']:
                here, nc = 'runs here on two cards', 2
            else:
                here, nc = 'does not run on these cards', None
            if any(s['model'] == m.name for s in running):
                here, nc = 'running', nc or m.cards
            names.append(m.name)
            mark[m.name] = nc
            lines.append((m, e, here))
        default = next((s['model'] for s in running if mark.get(s['model'])), None) or (row['start'] if row else None)
        for i, (m, e, here) in enumerate(lines, 1):
            self.say(f"{i:>2}. {'*' if m.name == default else ' '} {m.name:<14} {e['size_gb']:>5.1f} GB  {m.cards} "
                     f"card{'s' if m.cards > 1 else ' '}  {'on disk' if e['present'] else 'not downloaded':<14}  {here}",
                     None if mark[m.name] else 'dim')
        if default is None and self.yes:
            raise Stop(1, 'no model to start with on these cards: run caminus wizard without --yes and choose one')

        def pick(s):
            name = names[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(names) else s
            return name if name in mark and mark[name] else None
        got = self.ask(f"Model (a number or a name){f' [{default}]' if default else ''}:", default or '',
                       valid=lambda s: pick(s) is not None)
        m = models.BY_NAME[pick(got)]
        nc = mark[m.name]
        plan = self.decide(m, running)
        if plan[0] == 'keep':
            return m, nc, plan
        if nc >= 2:
            try:
                where = ' + '.join(host.describe(g) for g in host.pair(gs))
            except host.PlacementError as e:
                raise Stop(1, f'{m.name}: {e}') from None
        else:
            where = 'the smallest free card it fits'
        draft = 'DFlash2 (used where its file is on disk and the context fits)' if models.drafter_item(m) else 'none'
        self.say(f"{m.name}: {nc} card{'s' if nc > 1 else ''} ({where}), drafter {draft}, context auto (the largest that "
                 f"fits): caminus start's defaults (options: docs/cli.md)")
        return m, nc, plan

    def decide(self, m, running):
        """keep a running server or switch to m, asked before any download: ('keep', server name) or ('start', [the
        servers stopped first])"""
        mine = next((s for s in running if s['model'] == m.name), None)
        if mine:
            self.say(f"{mine['name']} runs {m.name} already at http://127.0.0.1:{mine['port']}: kept"
                     + (' (still loading: Qwen Code\'s setup waits for it)' if mine['state'] == 'loading' else ''), 'ok')
            return 'keep', mine['name']
        if not running:
            return 'start', []
        names = [s['name'] for s in running]
        stops = ', '.join(f"{s['name']} (:{s['port']})" for s in running)
        got = self.ask(f'Switch to {m.name}? This stops {stops}. [y/N]', 'n',
                       valid=lambda x: x.lower() in ('y', 'n', 'yes', 'no'))
        if got.lower() in ('y', 'yes'):
            return 'start', names
        if len(running) == 1:
            self.say(f'kept {names[0]}: no download, no start; Qwen Code is set up for it')
            return 'keep', names[0]
        for i, s in enumerate(running, 1):
            self.say(f"  {i}. {s['name']} ({s['model']})")
        got = self.ask(f'Set Qwen Code up for which server (1-{len(running)})? [1]', '1',
                       valid=lambda x: x.isdigit() and 1 <= int(x) <= len(running))
        return 'keep', names[int(got) - 1]

    def pull(self, m):
        self.step(3, f'download {m.name}')
        items = [models.BY_KEY[k] for k in m.items]
        if all(manage.locate_item(it) for it in items):
            self.say(f'{m.name} is on disk already: nothing to download', 'ok')
            return
        dest = os.path.abspath(os.path.expanduser(config.model_dir()))
        finals = [(it, os.path.join(dest, models.local_rel(it))) for it in items]
        wrong = [(f, os.path.getsize(f), it.size) for it, f in finals if os.path.exists(f) and os.path.getsize(f) != it.size]
        if wrong:                               # (caminus pull leaves such a file alone: a re-run would never get past it)
            raise Stop(1, '; '.join(f'{f} is {got:,} bytes, the release\'s file {want:,}' for f, got, want in wrong)
                       + ': caminus pull leaves it alone. Move it away (or delete it), then run caminus wizard again')
        todo = [(it, f) for it, f in finals if not os.path.exists(f)]
        need = download.need_bytes(todo)
        ok, free = download.room(dest, need)
        self.say(f'{need / 1e9:.1f} GB to download into {dest} ({free / 1e9:.0f} GB free there)')
        if not ok:
            raise Stop(1, f'not enough disk: {need / 1e9:.1f} GB + 1 GiB needed, {free / 1e9:.1f} GB free. Free some space, or '
                          'set model_dir to a bigger disk (docs/configuration.md), then run caminus wizard again')
        rc = self.run(['pull', m.name, '--dry-run'], read_only=True) if self.dry else self.run(['pull', m.name])
        if rc == 130:                           # (caminus pull catches Ctrl+C itself and returns 130: an interruption)
            raise Stop(130, 'interrupted: run caminus wizard again to resume the download (it skips what is done)')
        if rc != 0:
            raise Stop(1, 'the download did not finish: the lines above say why (after a dropped connection, run caminus '
                          'wizard again: the download resumes)')

    def start(self, m, nc, stops):
        self.step(4, f'start {m.name}')
        done = []
        for name in stops:
            if self.run(['stop', name]) != 0:
                left = [n for n in stops if n not in done]
                raise Stop(1, (f"stopped {', '.join(done)}; " if done else '') + f'{name} did not stop (caminus logs {name}); '
                              f"still running: {', '.join(left)}. Nothing was started")
            done.append(name)
        self.fit(m, nc)
        argv = ['start', m.name] + (['--cards', str(nc)] if nc != m.cards else [])
        if self.run(argv) != 0:
            raise Stop(1, f'{m.name} did not start: the lines above say why (caminus logs {m.name})'
                          + (f"; stopped for it: {', '.join(done)}" if done else ''))
        return m.name

    def fit(self, m, nc):
        """caminus fit on the files start will serve (read-only, the projection start places with): the context that fits
        now. Informational: start checks again, and nothing is projected before the files are on disk"""
        e = manage.model_entry(m)
        if not e['present']:
            return
        dk = models.drafter_item(m)
        dp = e['paths'].get(dk) if dk else None
        self.say('caminus fit reads the model to project the context that fits: this may take a while...', 'dim')
        if self.run(['fit', '--model', e['path'], '--cards', str(nc)] + (['--drafter', dp] if dp else []), read_only=True) != 0:
            self.say('caminus fit could not project it here (what the cards hold now counts); caminus start checks again', 'warn')

    def qwen(self, server):
        self.step(5, 'set up Qwen Code')
        installed = qwen_code.binary()
        if installed and qwen_code.status().get('target') == server:
            self.say(f'Qwen Code is set up for {server} already', 'ok')
            return True
        argv = ['qwen-code', 'setup', '--server', server]
        if not installed:
            lacks = qwen_code.node_missing()
            if lacks:
                self.say(f'Qwen Code needs Node.js {qwen_code.NODE_MIN} or newer with npm (found: {lacks}). Install it '
                         f'({qwen_code.NODE_URL}, or your distribution\'s package when it is {qwen_code.NODE_MIN} or newer), '
                         'then run:', 'warn')
                self.say('  caminus ' + shlex.join(argv + ['--install']), 'cmd')
                self.skip(5, f'Node.js {qwen_code.NODE_MIN} or newer with npm is not installed')
                return False
            argv.append('--install')
        if self.run(argv) != 0:
            raise Stop(1, 'caminus qwen-code setup failed: the lines above say why')
        return True

    def open_qwen(self, server):
        self.step(6, 'open Qwen Code')
        argv = ['qwen', '--server', server]
        if self.dry:
            self.run(argv)
            return
        if self.yes:
            self.say('When you want it: caminus ' + shlex.join(argv) + ' (--yes opens nothing)')
            return
        got = self.ask('Open Qwen Code now? [Y/n]', 'y', valid=lambda s: s.lower() in ('y', 'n', 'yes', 'no'))
        if got.lower() in ('n', 'no'):
            self.say('Later: caminus ' + shlex.join(argv))
            return
        self.say(f'(this log: {self.log_path})', 'dim')
        rc = self.run(argv)                     # (on success Qwen Code takes this process over: no return)
        if rc != 0:
            raise Stop(1, f'caminus qwen did not open Qwen Code (exit {rc}): the lines above say why; later: caminus '
                          + shlex.join(argv))

    def main(self):
        if self.dry:
            self.say('Dry run: only caminus doctor, caminus pull --dry-run and caminus fit run; every other command is printed, '
                     f'not run. The one thing it writes is its log ({self.log_path}).')
        try:
            gs = self.check()
            got = self.choose(gs)
            if got is None:
                server = None
            else:
                m, nc, (how, what) = got
                if how == 'keep':
                    self.step(3, 'download')
                    self.skip(3, f'{what} is kept: no download')
                    self.step(4, 'start')
                    self.skip(4, f'{what} is kept: no start')
                    server = what
                else:
                    self.pull(m)
                    server = self.start(m, nc, what)
            if server is not None and self.qwen(server):
                self.open_qwen(server)
            else:
                self.skip(6, 'Qwen Code is not set up')
        except Stop as e:
            self.say()
            self.say(str(e), 'warn' if e.rc == 130 else 'bad' if e.rc else None)
            return e.rc
        except KeyboardInterrupt:
            self.say()
            self.say('interrupted: run caminus wizard again to continue (it skips what is done)', 'warn')
            return 130
        finally:
            if self.log:
                self.log.flush()
        self.say()
        self.say('Done' + (f" ({len(self.skipped)} skipped: {'; '.join(self.skipped)})" if self.skipped else '') + '.', 'ok')
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus wizard', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yes', action='store_true', help='take every default (keep a running server; open nothing at the end)')
    ap.add_argument('--dry-run', action='store_true', help='run only caminus doctor, pull --dry-run and fit; print the rest '
                    '(it writes only its log)')
    a = ap.parse_args(argv)
    return Wizard(yes=a.yes, dry=a.dry_run).main()
