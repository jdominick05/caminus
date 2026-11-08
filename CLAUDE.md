# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Claude Code specifics

- Read AGENTS.md's Invariants before any GPU command; every one runs under `tools/guard.sh`.
- Use the job's own scratch directory for temporary files, never `/tmp`.
- Commit trailer: `Assisted-by: Claude <model>`; no `Co-Authored-By` and no session links.
