#!/usr/bin/env python3
"""`caminus serve` from a checkout: the repo's python/ on sys.path (python -I ignores PYTHONPATH), then caminus.serve.cli.

    python -I tools/caminus_serve.py [--config FILE] [--port 8092] [--max-ctx 262144] ...

(`python -I -m caminus serve` needs the package importable without PYTHONPATH, e.g. installed; caminus serve --help lists the
options.) Every GPU process runs under tools/guard.sh (CLAUDE.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from caminus.serve.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
