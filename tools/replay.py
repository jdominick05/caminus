"""From a checkout: rerun a flight-recorder bundle bitwise (GPU: run it under tools/guard.sh). The code is caminus.cmd.replay, the same as `caminus replay` of the installed package; --help lists its
options. This wrapper puts the repo's python/ on sys.path (python -I ignores PYTHONPATH)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from caminus.cmd.replay import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
