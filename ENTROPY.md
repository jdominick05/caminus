# ENTROPY.md

Why this tree has rules against its own growth. The rules are a few lines in AGENTS.md ("Invariants", Growth), where
every agent session reads them. This file is the reasoning, read once.

## The decay

Most changes here are written by AI coding agents, and agents fail in two ways that compound:

- **They add.** Asked for a behaviour, an agent writes new code beside the old: a new file, a new helper, a new
  paragraph. Merging or deleting needs to know what else depends on the old thing; adding needs nothing. Each addition
  is reasonable alone. Together the tree only grows, and every later change has more to read, more to keep true and
  more to miss.
- **They see a slice.** An agent works from the files it opened in one session, never from the whole system. It cannot
  know that the helper it is about to write exists two directories away, that a doc it trusts names files that were
  renamed, or that the file it left behind is run by nothing.

Asking for care fixes neither: the next session starts without the memory. So the rules sit in the one file every
session loads, and each has a place where it is checked.

## The rules and what checks them

| Rule (AGENTS.md) | Check |
|---|---|
| Search for what exists and extend it | The review flags a duplicate (AGENTS.md, Code Review Rules, Growth) |
| What a change replaces, it deletes | `orphans`: a test, tool or benchmark that no other file names was left behind. The review flags a replaced path left in place |
| Change what names a thing with the thing | `refs`: a path a doc names is in the tree. For code, the CPU suites |
| One fact in one file | `copies`: no two files hold the same bytes |
| AGENTS.md is the map | `refs` keeps it true; nothing keeps it small yet |

`python -I tools/entropy.py` runs `refs`, `orphans` and `copies` (the CPU suite `entropy`). Each failure says what to
delete, link or correct: the cheapest way to pass is never to add.

## The baseline

`tools/entropy.json` lists by name what was already wrong when a check arrived: dead names in docs, files nothing
names, pairs of files that are still copies (kept equal until one is a link). A check fails on anything not listed, and on a listed entry that no
longer applies, so the file is exactly what is wrong today.

- It only shrinks. Fix an entry, and `python -I tools/entropy.py --prune` drops it. The tool never adds one.
- Adding an entry is an edit by hand, and the pull request body says why.
- It is the consolidation backlog: an orphan is deleted or wired in, a dead name corrected or its sentence removed.
  An orphan is named in no file, which does not prove that nobody runs it by hand: ask before deleting one.

## Adding a rule

When something rots that no rule caught, extend a check that exists before writing a new one. Its rule goes in
AGENTS.md, its reason here and its case in tests/tools/test_entropy.py, in the same commit.

## What consolidating does not mean here

- **Not merging code objects.** A decode module and its verify module are separate objects on purpose: an edit to one
  must leave the other's machine code unchanged (AGENTS.md, Layout). They share one source through defines, and that
  is their consolidated form. Folding kernel or tape code together is a kernel change and runs every gate.
- **Not generalizing ahead.** "Never generalize speculatively or at a speed cost before a second target exists"
  (AGENTS.md, Conventions) stands. Merge two things that are the same today.
- **Not a shared test library.** Tests are plain scripts with their own small helpers, so each reads and runs alone.
