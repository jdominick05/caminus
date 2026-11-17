"""ROBOTS.md, the agent environment's own instructions (docs/specs/spec-ember.md section 7): shipped as package
data (share/harness/ember/ROBOTS.md), printed by `caminus ember prompt` and installed into the agent user's home as
~/ROBOTS.md by `caminus ember setup --agent-side`."""
import os

ROBOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share', 'harness', 'ember', 'ROBOTS.md')


def text():
    with open(ROBOTS, encoding='utf-8') as f:
        return f.read()

MEMORY_TEMPLATE = """# MEMORY.md — your persistent memory

Read this at the start of every task. Append what you learn at the end — decisions, gotchas, paths, what your
human prefers — and keep it short and factual; link to files for the long stuff. It survives restarts and new
sessions. Questions your human answers through `env-ask` are appended here automatically (Q/A blocks).
"""
