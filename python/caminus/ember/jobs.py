"""Detached processes in the agent environment (docs/specs/spec-ember.md section 9): `caminus ember spawn` starts
CMD detached in the agent user (nohup + setsid, log and pid under ~/env-logs and ~/env-jobs), `jobs` lists them,
`stop NAME` kills one. Pure script builders; the runner executes them."""
import re
import shlex

from . import shell

JOBS_DIR = '~/env-jobs'
LOGS_DIR = '~/env-logs'
_NAME = re.compile(r'^[A-Za-z0-9_.-]+$')


def check_name(name):
    if not _NAME.match(name or ''):
        raise ValueError(f'bad job name {name!r}: use letters, digits, _ . -')
    return name


def spawn_script(name, cmd_argv, jobs_dir=JOBS_DIR, logs_dir=LOGS_DIR):
    """the shell script that starts cmd_argv detached and prints its pid"""
    check_name(name)
    cmd = shell.quote_argv(cmd_argv)
    return (f'mkdir -p {jobs_dir} {logs_dir}; '
            f'nohup setsid bash -c {shlex.quote(cmd)} > {logs_dir}/{name}.log 2>&1 & '
            f'echo $! > {jobs_dir}/{name}.pid; cat {jobs_dir}/{name}.pid')


def jobs_script(jobs_dir=JOBS_DIR, logs_dir=LOGS_DIR):
    """lists the spawned jobs: name, pid, alive?, log tail"""
    return (f'for p in {jobs_dir}/*.pid; do [ -e "$p" ] || continue; '
            f'n=$(basename "$p" .pid); pid=$(cat "$p"); '
            f'if kill -0 "$pid" 2>/dev/null; then s=alive; else s=dead; fi; '
            f'echo "$n pid=$pid $s log={logs_dir}/$n.log"; done')


def stop_script(name, jobs_dir=JOBS_DIR):
    check_name(name)
    return f'pid=$(cat {jobs_dir}/{name}.pid 2>/dev/null) && kill "$pid" 2>/dev/null; rm -f {jobs_dir}/{name}.pid; echo stopped {name}'
