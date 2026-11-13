"""Running a command of the installed package under its own guard (caminus/share/guard.sh, docs/dev/packaging.md "Guard").

The guard exports CAMINUS_GUARD to what it runs, so a command started by the guard (or under any outer guard, as the tests run
everything) does not wrap itself a second time. caminus serve execs the guard (the guard becomes the process the terminal or a
service manager sees; the server's output goes to its log); caminus replay and caminus doctor --gpu run it as a child and print
its log when it exits."""
import os
import subprocess
import sys

from .. import config

GUARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share', 'guard.sh')
# the directory holding this caminus package (site-packages, or a checkout's python/): the guarded child imports the same copy
PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def under_guard():
    return bool(os.environ.get('CAMINUS_GUARD'))


def command(name, argv, log, secs):
    boot = f'import sys; sys.path.insert(0, {PARENT!r}); from caminus.__main__ import main; sys.exit(main())'
    # -I ignores PYTHONFAULTHANDLER; enable it explicitly so native failures leave thread tracebacks in the guard's log.
    return ['bash', GUARD, log, str(int(secs)), sys.executable, '-I', '-X', 'faulthandler', '-c', boot, name, *argv]


# caminus serve's exit code when its engine produced non-finite logits (serve/cli.py EXIT_BROKEN): the one exit a launcher
# reloads on, once. GPU faults (the guard's 99 / 98) are never retried
RELOAD_RC, RELOADS = 75, 1
_RELOAD_SH = (
    'n=0; while :; do bash "$@"; rc=$?; '
    'if [ "$rc" = {rc} ] && [ "$n" -lt {k} ]; then n=$((n + 1)); '
    'echo "caminus: the server exited {rc} (engine broken: non-finite logits): reloading ($n of {k}) in 10 s" | tee -a "$2"; '
    'sleep "${{CAMINUS_RELOAD_SLEEP:-10}}"; continue; fi; exit "$rc"; done')


def command_reloading(name, argv, log, secs):
    """command() inside a loop that runs it again once when the server exits RELOAD_RC (a fresh load re-uploads the
    weights); any other exit, a stop's included, ends the loop with that code"""
    return ['bash', '-c', _RELOAD_SH.format(rc=RELOAD_RC, k=RELOADS), 'caminus-reload'] + command(name, argv, log, secs)[1:]


def log_path(name):
    return os.path.join(config.state_dir(), f'{name}.log')


def exec_guarded(name, argv, secs=0):
    """replace this process by the guard running `caminus NAME ARGV` (never returns)"""
    log = log_path(name)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f'caminus {name}: under the guard ({GUARD}: the VRAM / host-RAM kill lines, the amdgpu fault watch, gpu-high); '
          f'the output goes to {log} (tail -f it); stop the server with kill -TERM $(pgrep -x caminus)', file=sys.stderr, flush=True)
    os.execvp('bash', command(name, argv, log, secs))


def run_guarded(name, argv, secs):
    """run `caminus NAME ARGV` under the guard, then print its log; returns the guard's exit code (99: a GPU fault)"""
    log = log_path(name)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f'caminus {name}: under the guard ({GUARD}); log {log}', file=sys.stderr, flush=True)
    rc = subprocess.run(command(name, argv, log, secs)).returncode
    try:
        with open(log, errors='replace') as f:
            sys.stdout.write(f.read())
    except OSError:
        pass
    if rc == 99:
        print(f'caminus {name}: the guard saw a GPU fault (exit 99): stop all GPU work; the kernel-log lines are in the STOP file',
              file=sys.stderr)
    return rc
