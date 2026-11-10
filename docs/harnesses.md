# Harnesses

By the end of this page your coding tool will be talking to a model running in Caminus. A harness is the tool you code
with: Qwen Code, Hermes Agent, Claude Code, or anything that speaks the OpenAI or Anthropic API.

## Qwen Code (the easiest)

The panel sets it up for you: in **QWEN CODE** press `i` to install it, `u` to set it up, then `enter` on a server in
**SERVERS**. From a terminal:

```bash
caminus qwen-code setup --install   # installs Qwen Code (needs Node.js) and points it at your running server
caminus qwen                        # starts Qwen Code on that server
```

Caminus gives Qwen Code a home of its own (`~/.config/caminus/qwen-home`). Your own Qwen Code settings, sessions and keys
in `~/.qwen` stay untouched, and a Qwen Code you installed yourself isn't changed. `caminus qwen-code status` shows what's
set up.

Caminus sizes each model's context to your cards, so it isn't always 262K. Every time you start Qwen Code from the panel
or with `caminus qwen`, Caminus sets Qwen Code's context window to what the running server actually holds, so Qwen Code
compacts the conversation before it runs out of room.

### The context note

The model can't see how full its context is. So after each batch of tool calls, Caminus adds one line to the last tool
result, like this:

```text
[context: ~498K of 1,048K tokens used; ~392K left before automatic compaction]
```

The model can then plan the rest of the task, finishing or writing notes before Qwen Code compacts the conversation. The
line stays in the conversation, so the prompt cache keeps working. The numbers come from the model's last call plus a
high estimate of the new tool output, so the room left is never overstated.

It's on in Caminus's Qwen homes. To turn it off, run `caminus qwen-code setup --context-note off` (it stays off), or set
`CAMINUS_CONTEXT_NOTE=0` for one session. Subagents don't get it. After updating Caminus, run `caminus qwen-code setup`
once and restart Qwen Code.

### The advisor (optional)

When the model gets stuck, a second model (local or hosted) can send it a hint. `caminus advisor setup --local-only`
sets it up with a model on your other card; see [the advisor](advisor.md).

## Any OpenAI- or Anthropic-compatible tool

A running server answers on your own machine only:

| | |
|---|---|
| OpenAI base URL | `http://127.0.0.1:8092/v1` |
| Anthropic base URL | `http://127.0.0.1:8092` |
| API key | anything: the server doesn't check it |
| Model name | the server's id, for example `qwen3.6` or `qwen3.6-nothink` ([the names](models.md#the-names-your-tools-use)) |

The first server takes port 8092, the next 8093, and so on. `caminus ps` shows each server's address. Both APIs support
streaming and tool calls. Don't expose the port to your network: there's no authentication.

## Hermes Agent and Claude Code

The repository has wrappers for both in `packaging/harness/` (clone it with
`git clone https://github.com/jdominick05/caminus`). Like the Qwen Code setup, each wrapper gives the tool a home of its
own under `~/.config/caminus/`, so your own settings stay where they are. Install the tool first.

| Tool | Run | Notes |
|---|---|---|
| Hermes Agent | `packaging/harness/hermes/hermes-k3` | Set up for `coder`. For another model, edit the model names in `~/.config/caminus/hermes-home/config.yaml` after the first run. |
| Claude Code | `packaging/harness/claude-code/claude-k3` | Set up for `coder`. For another model: `CAMINUS_MODEL=qwen3.6 CAMINUS_SMALL_MODEL=qwen3.6-nothink packaging/harness/claude-code/claude-k3`. It never sends a real Anthropic key. |

Each time it starts, the Hermes wrapper sets its context length to what the running server holds. Claude Code doesn't
read a context size from the server: with a model whose context is below about 200K tokens it can send a prompt the server
refuses as too long. Run `/compact` in Claude Code when that happens.

These two wrappers haven't been tested against a real session yet. If one misbehaves, please
[open an issue](https://github.com/jdominick05/caminus/issues/new/choose).

## Long conversations

Caminus keeps your conversation in a prompt cache, so each new turn only reads what's new: about 0.2 s to the first token
on an 11K-token conversation with `coder`. Short side requests from your tool (titles, summaries) run beside the
conversation without pushing it out. A conversation you leave for 8 hours, or one that was live when the server stopped,
is saved to disk and comes back in about a second.
