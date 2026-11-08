# Brain / hands: two models, one card each

A dense model plans and a fast MoE does the work. Each runs its own `caminus serve` process on its own card:

| role | model | card | port | model ids |
|---|---|---|---|---|
| brain | AREX-2 (BAAI, dense 27B on Qwen3.8-27B, Q4_K_M) | XTX (HIP_VISIBLE_DEVICES=0) | 8092 | `arex-2`, `arex-2-nothink` |
| hands | Qwen3.6-35B-A3B 2-bit GSQ (MoE, 3B active) | XT (HIP_VISIBLE_DEVICES=1) | 8093 | `qwen3.6`, `qwen3.6-nothink` |

The hands model decodes about four times faster than the brain. The Qwen Code setup is built around that:

- **The brain:** AREX-2 is the main model. It plans, decides and reviews, and keeps its own output short (`BRAIN.md`).
- **The hands:** they run the `hands` subagent, which takes whole subtasks and reports back in about 200 words (`agents/hands.md`). They also run the built-in Explore agent and the side calls such as titles and summaries (`fastModel`).
- **One hands task at a time.** The hands engine serves one request at a time, so `maxParallelAgentsByModel` is 1.
- **The two cards work in parallel.** The brain can keep planning while a hands task runs in the background.

```
packaging/brain-hands/caminus-bh start      # hands on the XT, then the brain on the XTX; each under tools/guard.sh
packaging/harness/qwen-code/brain-hands/qwen-bh       # Qwen Code in ~/.config/caminus/qwen-home-bh
packaging/brain-hands/caminus-bh status
packaging/brain-hands/caminus-bh stop       # SIGTERM to each: drain, park the conversation, exit
llm-top -config packaging/brain-hands/llm-top-brain.json      # one llm-top per server (or llm-top-hands.json)
```

- **State and logs:** `~/.local/state/caminus/{brain,hands}/serve.log`.
- **Prompt-cache disk tiers:** `~/.cache/caminus/kv-{brain,hands}`. Each takes an owner lock, so the two servers never share one.
- **Flight recorders:** `~/.cache/caminus/flight-{brain,hands}`.
- **Overrides:** `BRAIN_GGUF`, `HANDS_GGUF`, `BRAIN_CTX` (65,536) and `HANDS_CTX` (131,072). Both servers refuse at startup, before reading weights, when the model and its context do not fit their card's guard line.
