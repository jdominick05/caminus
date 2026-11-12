---
name: pr-ready
description: Use before calling a pull request into testing ready or merging it, after every push to an open pull request, and instead of waiting on CI or the reviewer by hand, to check CI and the review bot's verdict on the head commit, unresolved review threads and the GPU gates named in the body.
---

# Is this pull request ready?

A pull request is ready when `python -I tools/pr_ready.py N` prints `RESULT PASS` for its current head commit. The
script only reads GitHub; it changes and posts nothing. What a pull request must contain is written in AGENTS.md
("Conventions", "Code Review Rules") and CONTRIBUTING.md; this skill is the loop that gets a branch to PASS.

## Steps

1. **Before the first push**, run AGENTS.md's CPU suites and privacy scan ("Commands") on the branch tip.
   Done when both end PASS / CLEAN on the commit you push.

2. **Run the check** after opening the pull request and after every later push:

   ```
   python -I tools/pr_ready.py N
   ```

3. **Clear each FAIL line**, then run the check again.

   | Line | Meaning | Next |
   |---|---|---|
   | `ci` failed | a check failed on the head commit | read the failing suite's report in the run's log, fix, push |
   | `review: last reviewed commit X, head is Y` | the reviewer saw an older commit: it looks when a pull request opens (or leaves draft) and when asked, not after a push | comment `@codex review` once, then wait (step 4) |
   | `review: findings open` | a finding stands on the head commit | AGENTS.md, "Conventions": fix it, or answer in its thread; then ask for a review again |
   | `threads: N unresolved` | the branch rules refuse the merge | reply in each thread with the commit that fixes it or the reason; the thread is then resolved by the pull request's author or the maintainer |
   | `gates` | a kernel or tape change whose body names no GPU gate or no card | add the gates you ran, on which cards, with their numbers (the `kernel-change-evidence` skill produces the object and register lines) |
   | `text` | a decision id, a machine path or a session link in the title, body or a commit message | reword the title or body; for a commit message, say so in the pull request (`testing` gets one commit per pull request, with its own message) |
   | `base` | wrong target, a draft, or a conflict | CONTRIBUTING.md, step 6, names the target; for a conflict see "When testing moved" |

4. **Wait with a bound, in the background**, for CI and the reviewer:

   ```
   python -I tools/pr_ready.py N --wait 1800
   ```

   It returns as soon as nothing is pending, on the first failure, or at the bound. Done when it prints `RESULT PASS`
   or a FAIL line to act on.

5. **Stop at `RESULT PASS` on the head commit** and report it. The merge is the maintainer's.

## When testing moved

Bring `testing` into a reviewed branch with a merge commit, then ask for a review of the new head: the check ties
the reviewer's verdict to the head commit, and a merge is a new head. A merge keeps the earlier review threads
attached to their commits; a rebase replaces every commit, so keep it for a branch nobody has reviewed yet. Rerun the
gates the merged-in changes can affect, then step 2. `--allow-base-merges` accepts a merge of `testing` without a new
review; that choice is the maintainer's.
