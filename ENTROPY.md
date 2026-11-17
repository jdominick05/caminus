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

## What checks the rules

The review: AGENTS.md, Code Review Rules, Growth.

## What consolidating does not mean here

- **Not merging code objects.** A decode module and its verify module are separate objects on purpose: an edit to one
  must leave the other's machine code unchanged (AGENTS.md, Layout). They share one source through defines, and that
  is their consolidated form. Folding kernel or tape code together is a kernel change and runs every gate.
- **Not generalizing ahead.** "Never generalize speculatively or at a speed cost before a second target exists"
  (AGENTS.md, Conventions) stands. Merge two things that are the same today.
- **Not a shared test library.** Tests are plain scripts with their own small helpers, so each reads and runs alone.
