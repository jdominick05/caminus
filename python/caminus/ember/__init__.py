"""The agent environment (docs/specs/spec-ember.md): run agent work as a second user on this machine over SSH,
with root available only through a human-approved bridge. Experimental.

  runner    the SSH transport: every command runs as the agent user on 127.0.0.1
  bridge    the root bridge: the agent user files requests, only the invoking user executes them under sudo
  audit     the append-only log of runs, requests, decisions and results
  settings  names and paths (environment first)
  agent_side  the ask-root script installed into the agent user's ~/bin
"""
