# Without the panel

By the end of this page you'll be able to do everything the panel does from a terminal or a script. Every command takes
`--help`; most take `--json` for scripts.

## The usual flow

```bash
caminus models                 # what's available, what's on disk, how many cards each model needs
caminus pull thinkingcap       # download a model (resumes; checks every file's size and hash)
caminus start thinkingcap      # start it in the background on the best card, with the largest context that fits
caminus ps                     # running servers, their addresses, and your cards
caminus logs thinkingcap -f    # follow a server's log
caminus top                    # the live monitor
caminus stop thinkingcap       # stop it (--all stops every server)
```

`caminus start` options:

| Option | What it does |
|---|---|
| `--card N` | Use card N (as `caminus ps` numbers them) instead of choosing. |
| `--ctx N` | A context of N tokens instead of the largest that fits. |
| `--name NAME` | Call the server NAME (default: the model's name). |
| `--port N` | Listen on port N (default: the first free port from 8092). |
| `--team brain-hands` | Start AREX-2 and Qwen3.6 together on two cards ([the team](panel.md#the-brain--hands-team)); paused with Qwen3.6 for this release. |

`caminus restart NAME` restarts a server with the same settings.

## Check context capacity

`caminus fit` uses the same model options, drafter selection, card placement and memory projection as `caminus serve`.
It reads model metadata and current card usage without starting a server or initializing a GPU:

```bash
caminus fit --model MODEL.gguf --draft off --json
caminus fit --model MODEL.gguf --drafter DRAFTER.gguf --draft dflash --json
```

The report gives the largest native context that fits, the chosen cards, and projected and available memory per card.
It includes checkpoint slots and drafter buffers. An unavailable projection returns an error and no maximum.
Card usage can change between the query and startup, so serving checks capacity again. The query currently covers
the configured serving recipe's native device-memory context; host KV and context extension need separate projections.
F16/Q8 comparison and context choices in the panel remain separate integration work.

## Qwen Code

```bash
caminus qwen-code status            # installed? set up for which server?
caminus qwen-code setup --install   # install it and point it at the running server
caminus qwen                        # run it (add Qwen Code's own arguments after it)
caminus advisor setup               # optional: an API advisor steers Qwen Code when it gets stuck (docs/advisor.md)
caminus advisor status              # which advisor, hooked into which Qwen home
```

## Running the server yourself

`caminus serve` runs a server in the foreground, for scripts or a systemd unit. It starts itself under the
[guard](hardware.md#the-guard) and writes its log to `~/.local/state/caminus/serve.log`.

```bash
caminus serve --model ~/.local/share/caminus/models/<file>.gguf
caminus serve --help                # every option
```

Stop it with `kill -TERM` on its process (it's named `caminus`). It finishes the current request, saves the
conversation to disk, then exits. Ctrl-C ends only the guard and leaves the server running unwatched.

## Maintenance

| Command | What it does |
|---|---|
| `caminus doctor` | Checks your machine; `--gpu` also runs a tiny test on each card under the guard. |
| `caminus purge` | Lists the prompt cache and the request log on disk; `--yes` deletes them. |
| `caminus kv-cache list` / `purge` | The prompt cache on disk only. |
| `caminus flight list` / `purge` / `bundle` | The request log: one line per request (never the text) and replay files. |
| `caminus replay BUNDLE` | Reruns a saved request exactly, for a bug report. |
| `caminus guard LOG SECS CMD...` | Runs any command under the guard. `caminus guard --lines` prints each card's memory limit. |
| `caminus version` | The version and where its GPU code came from. |
