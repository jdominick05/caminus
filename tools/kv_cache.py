"""From a checkout: the prompt cache's disk tier: list / purge (CPU). The code is caminus.cmd.kv_cache, the same as `caminus kv-cache` of the installed package; --help lists its
options. This wrapper puts the repo's python/ on sys.path (python -I ignores PYTHONPATH)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from caminus.cmd.kv_cache import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
