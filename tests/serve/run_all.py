"""Runs every server test (CPU only, no GPU):  python -I tests/serve/run_all.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(HERE, pattern='test_*.py', top_level_dir=HERE)
    r = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if r.wasSuccessful() else 1)
