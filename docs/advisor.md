# The advisor

A local model sometimes gets stuck: the same failing command again and again, or a long run going nowhere. The advisor
is an optional add-on for Qwen Code. When the model looks stuck, a second model reads the task and the recent steps and
sends back a short hint, and Qwen Code adds it to the conversation. You can also ask it yourself with `/advise`.

The default setup uses an API advisor from Anthropic or Mistral, with your own API key. You can use one provider or
try several in order when an advisor refuses or fails.

## Set it up

Set up Qwen Code first (`caminus qwen-code setup`, see [harnesses](harnesses.md)). Then:

```bash
caminus advisor setup                # Anthropic or Mistral (installs their Python SDKs once)
caminus advisor status               # what's set up, and where
```

Setup asks which advisor to use, writes `~/.config/qwen-advisor/config.toml` and adds the hooks and the `/advise` command
to Caminus's own Qwen home. Your own `~/.qwen` isn't touched unless you name it with `--settings`. Restart Qwen Code
afterwards.

## Choosing an advisor

Setup asks for your API key (or uses `ANTHROPIC_API_KEY` / `MISTRAL_API_KEY`) and lists the models your key can use.
Choose its model, effort and fallback providers. Extra chain entries use `anthropic:<model>` or `mistral:<model>`;
append `@<effort>` when needed. Setup offers a live test request so you can check its response time before using it.

Existing local-advisor configurations continue to run. The explicit `caminus advisor setup --local-only` option keeps
the earlier local setup for compatibility.
If an existing extra chain contains local entries, API setup asks you to replace that list or enter `none` before
writing the new configuration. Updating or uninstalling the hook leaves provider configuration intact.

## When it speaks

- Three failed commands in a row, the same call three times, the same file edited four times with a failure after it,
  or 40 rounds of tool calls since your last message.
- Automatic hints come at most once every 10 minutes (`min_interval_s`), so the model mostly works on its own.
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
