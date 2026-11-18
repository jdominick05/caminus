# Without the panel

By the end of this page you'll be able to do everything the panel does from a terminal or a script. Most commands take
`--help`; most take `--json` for scripts.

## The usual flow

```bash
caminus wizard                 # the first run: these steps one by one, each command printed first (the quickstart)
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
| `--long-context` | `qwen3-30b-a3b`'s long-context mode ([models](models.md#long-context-for-qwen3-30b-a3b-opt-in)). Other models refuse it; the `coder` has its own ([configuration](configuration.md#long-context-coder-up-to-1m-tokens)). |
| `--kv q8` | `qwen3-30b-a3b` only: the 8-bit KV cache instead of F16, the default ([models](models.md#the-8-bit-kv-cache-qwen3-30b-a3b)). Other models refuse it before a card is picked, and `--team` refuses it (each team model keeps its own cache). |
| `--name NAME` | Call the server NAME (default: the model's name). |
| `--port N` | Listen on port N (default: the first free port from 8092). |
| `--cards 2` | Run `swift-1.5`, `thinkingcap` or `qwen3-30b-a3b` on both cards ([models](models.md)). |
| `--no-wait` | Return once the server is launched (`caminus ps` shows it loading). |
| `-- FLAGS` | Everything after `--` goes to `caminus serve`, for example `caminus start thinkingcap -- --draft mtp`. |

`caminus restart NAME` restarts a server with the same settings.

## Check context capacity

`caminus fit` uses the same model options, drafter selection, card placement and memory projection as `caminus serve`.
It reads model metadata and current card usage without starting a server or initializing a GPU:

```bash
caminus fit --model MODEL.gguf --draft off --json
caminus fit --model MODEL.gguf --drafter DRAFTER.gguf --draft dflash --json
```

The report gives the largest native context that fits, the chosen cards, and projected and available memory per card.
It includes checkpoint slots and drafter buffers. For a model with more than one KV cache format (Qwen3-30B-A3B), the
JSON's `kv` names the format the maximum assumes (`f16` or `q8`): in the long-context mode the mode picks it. An
unavailable projection returns an error and no maximum.
Card usage can change between the query and startup, so serving checks capacity again. The query covers
the native context (and Qwen3-30B-A3B's `--long-context`); the Coder's mode past 262,144 and host KV are not projected.
`max_ctx` has one entry per KV cache format, `{"f16": N, "q8": N}`, each the largest context that fits with that cache.
`q8` is `null` for a model without the 8-bit cache (`kv_formats` lists the formats a model has: only `qwen3-30b-a3b`,
on one card or two, has `q8`) and when no cards hold it. For a model that always keeps its own cache (the Coder's
8-bit rows), `f16` is that cache's figure and `kv_fixed` names the cache (`"q8_0"`; `null` for every other model).
`--kv` never changes these figures: it picks the format a `caminus serve` with the same options would start with, and
the cards, `need_mib` and `room_mib` describe that one (`kv` names it). When no cards hold that format the query fails
(exit 1, `ok: false`) but still gives the other format's figure. A model without the 8-bit cache refuses `--kv q8`
here as `caminus serve` does. The panel's model picker shows both figures, or the one figure of a model that keeps
its own cache.

## Qwen Code

[Harnesses](harnesses.md#qwen-code-recommended) has `caminus qwen-code status|setup --install` and `caminus qwen` (`--model MODEL`
starts a server first when none runs; `--server NAME` picks one); [the advisor](advisor.md) has `caminus advisor setup|status`.

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
| `caminus kv-cache list` / `prune` / `purge` | The prompt cache on disk: every model's store with each entry's size, age and last use (beside a running server too); prune by age (`--older-than-days`), parks per conversation (`--keep-per-root N`) or size (`--max-gb G`), skipping a running server's store (`--dry-run` first). |
| `caminus flight list` / `purge` / `bundle` | The request log: one line per request (never the text) and replay files. |
| `caminus replay BUNDLE` | Reruns a saved request exactly, for a bug report. |
| `caminus guard LOG SECS CMD...` | Runs any command under the guard. `caminus guard --lines` prints each card's memory limit. |
| `caminus version` | The version and where its GPU code came from. |
| `caminus ember ...` | The agent environment (experimental): work as a second user over SSH, root only with your approval. See [harnesses.md](harnesses.md#an-agent-with-its-own-user-experimental). |
