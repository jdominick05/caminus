#!/usr/bin/env python3
"""`caminus` from a checkout (every command: models, pull, start, ps, stop, logs, top, panel, qwen-code, qwen, serve, doctor...):
the repo's python/ on sys.path (python -I ignores PYTHONPATH), then caminus.__main__.

    python -I tools/caminus.py panel            # the control panel
    python -I tools/caminus.py start qwen3.6    # a server in the background (under the guard)

Run it with the Python that has the ROCm PyTorch: caminus start launches caminus serve with this same interpreter."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from caminus.__main__ import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
