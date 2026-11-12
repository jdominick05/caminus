---
name: kernel-change-evidence
description: Use after editing a file under kernels/, or when comparing two kernel builds (an A/B, a rebase, a merge of testing), to prove which built code objects changed and that the persistent kernel stays inside its register budget, before any GPU run or pull request.
---

# Kernel change evidence

A kernel edit is done when `tools/gates/objcheck.py check` prints `RESULT PASS` for the build directory the GPU gates
then run from. The rules it checks are AGENTS.md's ("Layout" says which modules an edit must leave unchanged;
"Commands" gives the register budget). This skill is the order of work around the script.

## Steps

1. **Snapshot the objects before the edit.** With the branch point built (`./build.sh`):

   ```
   python -I tools/gates/objcheck.py snap build -o SCRATCH/before.txt
   ```

   Already edited? Build the merge base with `testing` in a second checkout and snapshot its `build/`.
   Done when `before.txt` exists and its last line counts every object of the base build.

2. **Build the whole tree with `./build.sh`.** It records each object's compile command and the state of the sources
   (docs/dev/packaging.md, "Provenance"); an object compiled by hand is missing from that record.
   Done when build.sh exits 0.

3. **Check.** Name each module the change is meant to alter, and nothing else:

   ```
   python -I tools/gates/objcheck.py check SCRATCH/before.txt build --expect 'MODULE*'
   ```

   Done when it prints `RESULT PASS`. Each `FAIL` line names its case:

   | Line | Meaning | Next |
   |---|---|---|
   | `changed ... (no --expect pattern names it)` | the edit reached a module it must leave alone | keep the edit inside its own module (AGENTS.md, "Layout"), rebuild |
   | `--expect ...: none changed` | the build is stale, or this is the wrong directory | rebuild; run nothing on these objects |
   | `gone` | an object the base had is missing | restore it, or pass `--allow-missing` when the change removes it |
   | `N VGPRs > budget`, `spilled VGPRs` | the persistent kernel left its register budget | restructure the op until it fits (AGENTS.md, "Layout" and "Commands") |
   | `spilled SGPRs A -> B` | more scalars reload at every segment hop (AGENTS.md, "Commands": the count does not go up) | bring it back down; `--allow-sgpr-spill-up` is for a rise the maintainer has accepted, with the measured cost in the pull request |

4. **Run the GPU gates** for the change, on the objects just checked (AGENTS.md: "Invariants", then "Commands").
   When a gate reads a kernel directory other than `build/`, snapshot that directory too: two directories with the
   same `set digest` hold the same objects, whatever their names say.

5. **Put the evidence in the pull request body**: the `objects:` line, each `changed` / `new` line and the
   `registers` lines, copied from the script's output.

## Comparing two builds (A/B, after a rebase or a merge)

Build each arm into its own new directory, then `objcheck.py check ARM_A ARM_B --expect 'MODULE*'` (a directory works
as BEFORE). The comparison is by section: instructions and loaded data. Whole-file hashes are the wrong tool here:
they can differ between two directories for the same source, so a whole-file difference alone is not a finding.
