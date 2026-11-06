"""caminus COMMAND [ARGS...]   (`caminus COMMAND --help` for a command's options; python -I -m caminus works the same)

  serve     the OpenAI / Anthropic server on the K3 engine, 127.0.0.1:8092 (runs itself under the guard; its log in
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
them). PyTorch is not a declared dependency: install the ROCm build (caminus.torchcheck)."""
import sys

COMMANDS = ('serve', 'download', 'doctor', 'purge', 'replay', 'flight', 'kv-cache', 'guard', 'version')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__.strip())
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd in ('--version', 'version'):
        from . import __version__, kernels
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
    help_only = any(a in ('-h', '--help') for a in rest)
    if cmd in ('serve', 'replay') and not help_only and not guarded.under_guard():
        # the engine needs the ROCm PyTorch: say so here, in the terminal, before the guard takes over (its log would hide it);
        # under a guard the command checks itself (serve without importing torch before it names its process)
        from .torchcheck import refusal
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
