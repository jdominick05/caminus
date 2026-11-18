# Harnesses

By the end of this page your coding tool will be talking to a model running in Caminus. A harness is the tool you code
with: Qwen Code, Hermes Agent, Claude Code, or anything that speaks the OpenAI or Anthropic API.

## Qwen Code (the easiest)

The panel sets it up for you: press `i` to install it, `u` to set it up for the running server, then `c` to open it.
From a terminal:

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

Setup waits for the server to report its context. If the model is still loading after that wait, setup asks you to try
again when it is ready; it keeps your existing settings. It does not guess a context size.

Qwen Code's thinking switch works on the main model entry. The `-low` and `-xhigh` entries keep their named effort
while allowing thinking to be turned off. A `-nothink` entry always stays off; select the main entry to turn thinking
on again. When you switch servers, Caminus updates its own selected model and removed fast/compaction model routes,
while keeping your external provider choices, MCP servers and hooks.

### Side questions (`/btw`)

`/btw <question>` asks about the conversation without adding to it. Caminus recognizes the request and answers it in its
side slot, so your conversation's next turn picks up where it was instead of being read again. Two things to know:

- It waits for the current answer to finish: the server answers one request at a time.
- In a short conversation it's quick: only the question is new.
- In a long one Qwen Code sends just the opening and its last 40 messages. When those match the end of the conversation
  the server holds, token for token, Caminus answers from the whole stored conversation instead: only the question is
  read, and the model also sees the older turns. That's the case on Swift 1.5 and ThinkingCap. On Qwen3-30B-A3B its
  template writes past replies differently from how they were generated, so they never match: the 40 messages are read
  fresh, which takes longer. The conversation itself is untouched either way.

A side question's answer is capped at 16,384 tokens. One whose prompt and answer won't fit the side slot (98,304
positions) runs in the main slot as before.

### The context note

The model can't see how full its context is, or what time it is. So after each batch of tool calls, Caminus adds one
line to the last tool result, like this:

```text
[time: Wed 2026-11-18 00:41 EST; context: ~498K of 1,048K tokens used; ~392K left before automatic compaction]
```

The model can then plan the rest of the task, finishing or writing notes before Qwen Code compacts the conversation, and
knows how long a step took. The line stays in the conversation, so the prompt cache keeps working. The numbers come from the model's last call plus a
high estimate of the new tool output, so the room left is never overstated.

It's on in Caminus's Qwen homes. To turn it off, run `caminus qwen-code setup --context-note off` (it stays off), or set
`CAMINUS_CONTEXT_NOTE=0` for one session. Subagents don't get it. After updating Caminus, run `caminus qwen-code setup`
once and restart Qwen Code.

### The advisor (optional)

When the model gets stuck, an API advisor can send it a hint. `caminus advisor setup` sets up Anthropic or Mistral
with your own API key; see [the advisor](advisor.md).

## Any OpenAI- or Anthropic-compatible tool

A running server answers on your own machine only:

| | |
|---|---|
| OpenAI base URL | `http://127.0.0.1:8092/v1` |
| Anthropic base URL | `http://127.0.0.1:8092` |
| API key | anything: the server doesn't check it |
| Model name | the server's id, for example `thinkingcap` or `thinkingcap-nothink` ([the names](models.md#the-names-your-tools-use)) |

The first server takes port 8092, the next 8093, and so on. `caminus ps` shows each server's address. Both APIs support
streaming and tool calls. Don't expose the port to your network: there's no authentication.

## Hermes Agent and Claude Code

The repository has wrappers for both in `packaging/harness/` (clone it with
`git clone https://github.com/jdominick05/caminus`). Like the Qwen Code setup, each wrapper gives the tool a home of its
own under `~/.config/caminus/`, so your own settings stay where they are. Install the tool first.

| Tool | Run | Notes |
|---|---|---|
| Hermes Agent | `packaging/harness/hermes/hermes-k3` | Set up for `coder`. For another model, edit the model names in `~/.config/caminus/hermes-home/config.yaml` after the first run. |
| Claude Code | `packaging/harness/claude-code/claude-k3` | Set up for `coder`. For another model: `CAMINUS_MODEL=thinkingcap CAMINUS_SMALL_MODEL=thinkingcap-nothink packaging/harness/claude-code/claude-k3`. It never sends a real Anthropic key. |

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

## An agent with its own user (experimental)

Instead of a fixed tool list, an agent can get an environment: a second user on this machine (`caminus-agent` by
default) that it reaches over SSH and can do anything in. Your own files, keys and processes stay out of its reach by
Unix permissions, and root is never in its hands: a root command waits in a queue until you approve it in
`caminus ember watch`, and only your side runs it under sudo.

```bash
caminus ember setup --agent-side   # one-time provisioning (prints the sudo commands to run yourself)
caminus ember status               # user, key, ssh login, bridge, ask-root all OK?
caminus ember shell                # interactive session as the agent user: run any agent here
caminus ember run -- git status    # one command as the agent user
caminus ember ui                   # the TUI: queued root requests + the live audit log
caminus ember watch                # same, in plain text (approve / edit / deny queued root commands)
caminus ember spawn -n build -- make -j4   # long work, detached (log ~/env-logs/build.log)
caminus ember jobs                 # the detached jobs
caminus ember stop build           # kill one
caminus ember reqs --json          # pending root/question requests, machine-readable
caminus ember answer ID answered "yes"   # answer one from a script (approved/edited/denied/answered)
caminus ember memory               # the agent's ~/MEMORY.md (its persistent memory)
caminus ember remember "TP wins at these shapes"   # append to it
caminus ember serve coder coder --placement 2   # a model server as a job: own home/port/session, split on both cards
caminus ember serve swift swift-1.5 --placement 0  # one small model on the 24 GB card (S in `caminus ember ui` does this)
caminus ember new coder -- qwen    # a named session running your agent CLI (tmux, in the agent user)
caminus ember sessions             # the sessions
caminus ember send coder "run the TP benchmarks"   # type an instruction into it, live
caminus ember attach coder         # sit in the same terminal
caminus ember ctx                  # running servers and the context window each reports
caminus ember audit --tail 20      # everything the environment did
```

Inside the agent user, `ask-root CMD...` asks you to run CMD as root and waits for the answer, and `~/ROBOTS.md` is
its standing instruction — what it is, what it can do, and how root works (`caminus ember prompt` prints it from your
side). The full design is in
[spec-ember.md](specs/spec-ember.md). This is an environment, not a sandbox: the agent user still has the
network and the GPU, so give it only scoped credentials in its own environment.
