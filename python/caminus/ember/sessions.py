"""Named tmux sessions in the agent user (docs/specs/spec-ember.md section 11): the human's way to run an
agent interactively, watch it, and type into it.

  new NAME CMD...     tmux new-session -d -s NAME running CMD (an agent CLI, a shell, anything)
  sessions            tmux ls
  send NAME TEXT      tmux send-keys: type TEXT + Enter into the session (a new instruction, live)
  attach NAME         ssh -t ... tmux attach -t NAME: drop into the same terminal
"""
import shlex

from . import jobs, shell


def check_name(name):
    return jobs.check_name(name)


def new_script(name, cmd_argv):
    check_name(name)
    cmd = shell.quote_argv(cmd_argv)
    return f'tmux new-session -d -s {shlex.quote(name)} bash -lc {shlex.quote(cmd)}'


def sessions_script():
    return 'tmux ls 2>/dev/null || echo "(no sessions)"'


def send_script(name, text):
    """type text + Enter into the session; -l sends the text literally, Enter is the key"""
    check_name(name)
    return f'tmux send-keys -t {shlex.quote(name)} -l {shlex.quote(text)} && tmux send-keys -t {shlex.quote(name)} Enter'


def attach_argv(name, base_ssh_argv):
    check_name(name)
    return base_ssh_argv + ['-t', 'tmux', 'attach', '-t', name]
