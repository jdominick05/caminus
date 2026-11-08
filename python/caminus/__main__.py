"""caminus COMMAND [ARGS...]   (`caminus COMMAND --help` for a command's options; python -I -m caminus works the same)

  models    the named models: on disk or not, the cards each needs, measured speeds    (pull NAME downloads one)
  start     start a server in the background: caminus start NAME [--card N] [--ctx auto|N] (--team brain-hands)
  ps        the running servers and the cards; stop NAME | --all, restart NAME, logs NAME [-f], top [NAME]
  panel     the control panel (a terminal UI: basic and advanced layouts)
  qwen      Qwen Code on the running server, in Caminus's own Qwen home (qwen-code status | setup [--install])
  serve     the OpenAI / Anthropic server in the foreground (runs itself under the guard; its log in
            ~/.local/state/caminus/serve.log)
  download  the model files from their Hugging Face repositories (resume, size and sha256 checks)
  doctor    is this machine ready? (CPU only; --gpu: one launch per card under the guard)
  purge     empty the prompt cache's disk tier and the flight recorder (lists only without --yes)
  replay    rerun a flight-recorder bundle, bitwise (GPU, under the guard)
  flight    the flight recorder: list / purge / bundle
  kv-cache  the prompt cache's disk tier: list / purge
  guard     run any command under the guard: caminus guard LOG SECS CMD... (--lines: the cards and their kill lines)
  version   the package version and its code objects' provenance

Paths come from the environment, ~/.config/caminus/config.toml, then the XDG defaults (caminus.config; caminus doctor prints
them). PyTorch is not a declared dependency: install the ROCm build (caminus.device.torchcheck)."""
import sys

COMMANDS = ('serve', 'download', 'doctor', 'purge', 'replay', 'flight', 'kv-cache', 'guard', 'version',
            'models', 'pull', 'start', 'stop', 'restart', 'ps', 'logs', 'top', 'panel', 'qwen-code', 'qwen')
MANAGE = {'models': 'cmd_models', 'pull': 'cmd_pull', 'start': 'cmd_start', 'stop': 'cmd_stop', 'restart': 'cmd_restart',
          'ps': 'cmd_ps', 'logs': 'cmd_logs', 'top': 'cmd_top', 'panel': 'cmd_panel'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__.strip())
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ('--version', 'version'):
        from . import __version__
        from .device import kernels
        print(f'caminus {__version__}\n{kernels.describe()}')
        return 0
    if cmd == 'kv_cache':
        cmd = 'kv-cache'
    if cmd not in COMMANDS:
        print(f'caminus: unknown command {cmd!r} ({", ".join(COMMANDS)}; caminus --help)', file=sys.stderr)
        return 2
    from .config import ConfigError
    try:
        return dispatch(cmd, rest)
    except ConfigError as e:
        print(f'caminus {cmd}: the configuration: {e}', file=sys.stderr)
        return 2


def dispatch(cmd, rest):
    from .cmd import guarded
    if cmd in MANAGE:
        from .cmd import manage
        return getattr(manage, MANAGE[cmd])(rest)
    if cmd == 'qwen-code':
        from .cmd import qwen_code
        return qwen_code.main(rest)
    if cmd == 'qwen':
        from .cmd import qwen_code
        return qwen_code.launch(rest)
    help_only = any(a in ('-h', '--help', '--options-json') for a in rest)   # (no torch, no guard: nothing is loaded)
    if cmd in ('serve', 'replay') and not help_only and not guarded.under_guard():
        # the engine needs the ROCm PyTorch: say so here, in the terminal, before the guard takes over (its log would hide it);
        # under a guard the command checks itself (serve without importing torch before it names its process)
        from .device.torchcheck import refusal
        msg = refusal(f'caminus {cmd}')
        if msg:
            print(msg, file=sys.stderr)
            return 2
    if cmd == 'serve':
        if not guarded.under_guard() and not help_only:
            guarded.exec_guarded('serve', rest)      # (never returns)
        from .serve.cli import main as serve
        return serve(rest)
    if cmd == 'replay':
        if not guarded.under_guard() and not help_only:
            return guarded.run_guarded('replay', rest, 1800)
        from .cmd.replay import main as f
        return f(rest)
    if cmd == 'guard':
        import os
        if rest[:1] != ['--lines'] and len(rest) < 3:
            print('usage: caminus guard LOG SECS CMD...  (SECS 0: no time limit) | caminus guard --lines', file=sys.stderr)
            return 2
        os.execvp('bash', ['bash', guarded.GUARD, *rest])
    mod = {'download': 'download', 'doctor': 'doctor', 'purge': 'purge', 'flight': 'flight', 'kv-cache': 'kv_cache'}[cmd]
    import importlib
    return importlib.import_module(f'.cmd.{mod}', __package__).main(rest)


if __name__ == '__main__':
    sys.exit(main())
