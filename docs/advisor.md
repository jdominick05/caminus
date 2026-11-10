# The advisor

A local model sometimes gets stuck: the same failing command again and again, or a long run going nowhere. The advisor
is an optional add-on for Qwen Code. When the model looks stuck, a second model reads the task and the recent steps and
sends back a short hint, and Qwen Code adds it to the conversation. You can also ask it yourself with `/advise`.

The second model can be **local** (another model on your other card, or any OpenAI-compatible server) or **hosted**
(Anthropic or Mistral, with your own API key).

## Set it up

Set up Qwen Code first (`caminus qwen-code setup`, see [harnesses](harnesses.md)). Then:

```bash
caminus advisor setup --local-only   # a local advisor: nothing to install, no API key
caminus advisor setup                # local, Anthropic or Mistral (installs their Python SDKs once)
caminus advisor status               # what's set up, and where
```

Setup asks which advisor to use, writes `~/.config/qwen-advisor/config.toml` and adds the hooks and the `/advise` command
to Caminus's own Qwen home. Your own `~/.qwen` isn't touched unless you name it with `--settings`. Restart Qwen Code
afterwards.

## Choosing an advisor

**Local.** Give setup the server's URL (up to `/v1`); it lists the server's models to pick from. Point it at a different
server from the one Qwen Code is using: the model on your other card, for example. A thinking model can take a minute to
answer on one card, and a hint that arrives after the limit (120 seconds by default) is dropped. A model id without
thinking (on Caminus, the `-nothink` ids) skips that and answers much sooner. Setup's test request shows how long a
reply takes; to change the limit, run setup again (it also updates the hooks' own time limits).

**Hosted.** Setup asks for your API key (or uses `ANTHROPIC_API_KEY` / `MISTRAL_API_KEY`) and lists the models your key
can use. Measured: Claude Opus 5.5 at medium effort about 5 seconds and $0.01 a hint, Claude Haiku 5.5 about 2 seconds and
$0.0003, Mistral Large 4 2 to 4 seconds. You can chain several: if one refuses or fails, the next one is asked, and a
local server can be the last in line (`local:<model>`).

## When it speaks

- Three failed commands in a row, the same call three times, the same file edited four times with a failure after it,
  or 40 rounds of tool calls since your last message.
- Automatic hints come at most once every 10 minutes (`min_interval_s`), so the model mostly works on its own. A local
  advisor costs nothing per hint, so you can make that much shorter.
- Optionally, once per message, it checks whether the task is really done before the model stops.
- `/advise` asks right away, any time.

If the advisor thinks the model is on track, nothing is added.

## Why it can't break your session

Hints are added to the newest message only. Nothing earlier in the conversation changes, so Caminus keeps reusing its
prompt cache. Any error, timeout or refusal adds nothing. While a hint is being fetched, Qwen Code waits for it (up to
the reply limit), and the done-check can send the model back to work once per message; otherwise the hooks never get
in its way.

## What it sends

The task, your latest message, the first and the most recent steps (commands and their output, shortened), and why it
was called. API keys and tokens that look like secrets are removed first. With a hosted advisor this leaves your machine.

## Turning it off

- For now: `enabled = false` in `~/.config/qwen-advisor/config.toml`, or `QWEN_ADVISOR_OFF=1`.
- For one project: an empty `.qwen-advisor-off` file in it.
- For good: `caminus advisor uninstall` removes the hooks and `/advise`.

After a Caminus update, `caminus advisor update` refreshes the installed hook. Its log is
`~/.local/state/qwen-advisor/advisor.log`.
