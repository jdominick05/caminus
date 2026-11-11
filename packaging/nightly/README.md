# The nightly build as a systemd user timer

`caminus-nightly.timer` starts `caminus-nightly.service` every night at midnight (local time). The service runs
`tools/release/nightly.sh`: when the head of `testing` has a green CI run and no nightly yet, it builds that commit the
way a release is built and publishes it as a GitHub pre-release; `.github/workflows/nightly.yml` then verifies the files
and uploads the wheel to PyPI. What the script does, step by step, and why it runs on the maintainer's machine:
[docs/dev/releasing.md](../../docs/dev/releasing.md), "Nightly builds". Nothing in the repository installs or enables the
timer; these are the steps to take by hand.

It needs what a release build needs (docs/dev/releasing.md, "Making a release": the TheRock toolchain, the ignite-llama
fork's headers, Go, a Python with ROCm PyTorch) plus `gh`, logged in to an account that may push tags and create
releases. It is CPU only and never opens a GPU.

## Set it up

```
git clone -b testing https://github.com/jdominick05/caminus ~/.local/share/caminus-nightly/src   # the job's own clone
mkdir -p ~/.config/caminus-nightly
install -m 600 ~/.local/share/caminus-nightly/src/packaging/nightly/nightly.env.example ~/.config/caminus-nightly/env
$EDITOR ~/.config/caminus-nightly/env                         # the commit identity, PY, ROCM_PATH, IGNITE, PATH
systemctl --user link ~/.local/share/caminus-nightly/src/packaging/nightly/caminus-nightly.service
systemctl --user link ~/.local/share/caminus-nightly/src/packaging/nightly/caminus-nightly.timer
```

Try it before the timer does, in this order:

```
set -a; . ~/.config/caminus-nightly/env; set +a
~/.local/share/caminus-nightly/src/tools/release/nightly.sh --dry-run    # builds and verifies; pushes and publishes nothing
systemctl --user start caminus-nightly          # the real thing, once, as the timer runs it: a public tag and pre-release
journalctl --user -u caminus-nightly -e         # what it did
systemctl --user enable --now caminus-nightly.timer
systemctl --user list-timers caminus-nightly.timer
```

The run under systemd is its own test: a unit sees the env file's `PATH` and no login shell, and `gh` cannot read a
token kept in a desktop keyring when nobody is logged in (set `GH_TOKEN` in the env file for that case).
`loginctl enable-linger` lets the timer run without a login at all.

## Living with it

- **Skip nights:** `touch ~/.local/state/caminus-nightly/HOLD` (remove the file to resume). A night is also skipped while
  a guarded GPU job other than the server is running, so the two kernel builds never land on a benchmark's timings; the
  skipped commit is built by the next run.
- **What happened:** `cat ~/.local/state/caminus-nightly/last` (one line), `journalctl --user -u caminus-nightly`, and the
  two build logs beside the clones in `~/.local/state/caminus-nightly/builds/<date>-<commit>-<time>/`. The script keeps the
  last two build directories and deletes older ones.
- **A failed night** changes nothing public unless it died between pushing the tag and creating the pre-release; the next
  run finishes that from the tag. A pre-release whose PyPI upload failed: run the `nightly` workflow by hand on the tag
  (`gh workflow run nightly.yml --ref <tag> -f tag=<tag>`; GitHub allows that once `nightly.yml` is on `main`).
- **One nightly a day at most**, and none when `testing` did not move: a PyPI version is published once.
- **The service runs the head of `testing`** (it updates its clone first), with the account's GitHub rights. It builds
  and runs only what was merged there.
- **Stop:** `systemctl --user disable --now caminus-nightly.timer`; `systemctl --user unlink caminus-nightly.timer
  caminus-nightly.service`.
