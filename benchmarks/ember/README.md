# Ember suites

Tasks for the Ember lab (`tools/emberlab/lab.py`; how to run it: [docs/dev/ember.md](../../docs/dev/ember.md#the-lab)).

A suite is a folder:

- `suite.json`: `name`, `about`, `timeout` (seconds per goal), `pristine` (paths copied to `DIR/.orig` for checks that
  compare against the original), `dirs` (empty folders to make), and `tasks`. Each task has a `key`, a `kind`, the
  `prompt` (`DIR` stands for the run's work folder) and a `check`: a shell command run in `DIR` that passes on exit 0.
  `scripts` holds optional tool-call sequences for the scripted brain (`scripted:NAME`), with `DIR` filled in the same way.
- `fixture/`: the files each run starts from. Every run gets a fresh copy, so tasks never see each other's changes.

The check is the score. The agent's own verdict is recorded next to it: a goal that finished while its check failed is
a false completion, counted on its own.

| Suite | Tasks |
|---|---|
| `basic` | five everyday tasks on a small invented Python project: read code, edit, run tests, use a tool, report back. Its `solve` script passes every check through the resident's gates; its `claim` script declares done without doing anything. |

Keep a suite's prompts and checks unchanged once results exist for it. Earlier scores stay comparable only while the
tasks are the same. Add a new suite or task rather than editing one.
