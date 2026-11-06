# Metrics and the monitoring surface (llm-top support)

The server's monitoring surface, as asked in docs/AGENTIC-PLAN.md (2026-11-04 entries "llm-top support" and "the full
Caminus metric set"). It has two parts:
- **llama-server's surface**, with llama-server's own names and shapes: `/health`, `/props`, `/slots`, `/metrics`.
  llm-top and other llama.cpp tools read Caminus with no change.
- **The Caminus metric set**: `caminus:` series beside the `llamacpp:` ones, plus a versioned JSON status
  (`GET /caminus/status`). A client gets the histograms and per-card data from it with no label parsing.

Status (11-05): built and tested on `MockSession`; M7 wired the real engine (`serve/k3_session.py`: the pass split `pass_k3_ms` / `pass_sampler_ms`, bails and retries from `FlashNextTP.bails`, `cache_update` with the prompt cache's own counters after each request). Server overhead with the metrics on, tests/test_serve_k3.py overhead: +0.68 % (short) / -0.42 % (32K) of decode ms/token.

M7 ops (impl:m7-ops) fills the rest from the real engine (section 10): the scrape off the decode path (a snapshot under the stats lock, the text formatted on the HTTP thread), per-card pass spans from device events, the exchange payload bytes, live prompt progress per chunk, filler drafts no longer counted as drafts, the plain pass time (measured, else the policy's estimate), refusals by kind, KFD evictions and page migrations from the hardware sampler thread, the hardware reader on the engine's own cards by PCI slot, the floor's per-card bytes and the KV bytes per position from the engine, and the startup clock check. The exchange wait is the one series the engine cannot fill (section 10).

**Code:**
- `python/caminus/serve/stats.py`: the hook and the recorder.
- `metrics.py`: the renderers.
- `agentmon.py`: the content class, harness classes, the tool-call report and the loop detector.
- `hw.py`: the hardware reader.
- `server.py`: the routes and the process name.

**Tests:** tests/serve/test_stats.py, test_metrics.py, test_hw.py.

## 1. Endpoints

All are GET only. While the server loads, every route answers 503
`{"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}`, as llama-server's middleware does
(`K3Server(loading=True)` / `set_loading()`; `tools/k3_serve.py --load-s S`).

| Route | Shape | Notes |
|---|---|---|
| `/health` | `{"status": "ok"}`, or 503 while loading | llm-top's liveness probe |
| `/props` | llama-server's `get_res_props` keys and JSON types | `default_generation_settings` = `{params, n_ctx}`. `params` is llama-server's `task_params::to_json(only_metrics=true)` set, with `n_predict` -1. Caminus adds `engine: "caminus"` (the engine marker), `caminus: {build_info, status, status_version, metrics_prefix, speculative_types, stats_enabled}`, and its older keys `n_ctx`, `model`, `models`. `model_alias` is the default model id (`k3-coder`). `build_info` is `caminus-<short sha>`, read from .git. |
| `/slots` | a list with one slot, llama-server's `server_slot::to_json(only_metrics=true)` | Before the first request: only `id`, `n_ctx`, `speculative`, `is_processing`. After that, the last request's numbers stay, like llama-server's `task_prev`. `?fail_on_no_slot=1` answers 503 while busy. |
| `/metrics` | Prometheus text, `Content-Type: text/plain; version=0.0.4`, header `Process-Start-Time-Unix` | First llama-server's block, in its order with its HELP texts. Then the `caminus:` series. |
| `/caminus/status` | JSON, `version` 1 | Section 5 |

**Where Caminus differs from llama-server on purpose:**
- **Speculative counters move every pass.** llama-server adds them when a request ends.
- **Values print exactly,** rounded to 6 decimals. llama-server prints 6 significant digits.
- **No request text.** `/slots` never carries prompt or generated text (llama-server omits it unless slots debugging is on). The status has none either. The loop detector keeps line hashes only. Harnesses are a fixed set of classes; the raw User-Agent is never a label.
- **`n_decode_total` counts engine passes:** prompt passes, batched-prefill segments and decode passes.

**The label in llama-server's source is `position`, not `pos`.** llama-server writes the per-position acceptance as
`llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}` (server-task.cpp `to_metrics`), and llm-top's
`parseMetrics` matches `{position="` exactly. The step text's `{pos=...}` is not what either side uses.

**Prefix.** The plan says `caminus_*`; the step says `caminus:`, "the same prefix convention" as `llamacpp:`. The
surface follows the step. Colons are legal in Prometheus metric names. llm-top's parser skips labelled lines other
than the per-position series, and stores unlabelled `caminus:` lines under their full names, so they never collide
with `llamacpp:` keys.

**Process name.** `start()` names the process `caminus` with `prctl(PR_SET_NAME)` through ctypes, so llm-top finds it
by `/proc/<pid>/comm` and reads per-card VRAM from its DRM fdinfo.
- prctl names the calling thread, so the launcher calls `set_process_name` on the main thread before any thread starts.
- From another thread, `/proc/self/comm` (the thread group leader's) is written instead.
- Tested in child processes (test_metrics `TestProcessName`), never in the test runner.

## 2. The stats hook (the M7 contract) and the cost rule

`EngineStats` is one object per server (`K3Server.stats`, also `session.stats`).

**Who writes:**
- The engine thread is the only writer of its engine-side fields: the worker, `run_generation` and the
  `EngineSession`.
- HTTP threads write only the agent fields (User-Agent classes, tool calls, the loop detector), under `stats.lock`.
- Readers copy and compute quantiles on their own thread.

**What a pass may cost.** Every value comes from host timestamps (`stats.clock`, default `time.perf_counter`) and
counters the loop already has. No device sync is added, and per-op device timing stays in the `*_trace` builds.
`EngineStats(enabled=False)` turns every hook into an early return, for M7's "server adds <= 3 % to decode ms/token"
gate, which is measured with the metrics on.

| Hook | Caller | When |
|---|---|---|
| `begin(job, max_ctx)` | worker | the job reached the engine (id_task, n_predict, params for /slots) |
| `cached_known(n)` | session | the reused prefix is known, before processing |
| `prefill_progress(n)` | session | after each prompt pass / segment: prompt tokens processed so far |
| `prefill_done(pf)` | run_generation | `pf` = `PrefillStats`: processed / cached, `ms`, `source` (the prompt cache's names), `exact`, `lcp`, the TTFT split `restore_ms` / `batched_ms` / `handoff_ms` / `passes_ms` |
| `first_token(n)` | run_generation | the first sampled token(s) emitted: ends TTFT |
| `pass_k3_ms`, `pass_sampler_ms` | `EngineSession.sample_pass` | set around each verify and its sampling (the base class times its host sampler; an engine with a device sampler sets them itself) |
| `pass_cards` | engine | `(card ms tuple, exchange-wait ms tuple, link bytes)` of the pass, when known |
| `pass_done(t0, draft_ms, fed, drafted, accepted, emitted, cls)` | run_generation | one decode pass, from its drafter call to its emitted tokens' bookkeeping. `accepted` drafts are a prefix (positions 0 .. accepted - 1). `cls` = the content class when the pass started |
| `other_pass(n)` | run_generation | forced tool-call tokens beyond one pass |
| `forced(n)` | run_generation | n forced tool-call ids emitted: /slots `n_decoded` / `n_remain` and the position move as for sampled ids (M7) |
| `tokens(reasoning, answer)` | worker | generated ids by kind (reasoning up to and including `</think>` when the prompt opened a think block) |
| `cls` | worker | `agentmon.ContentClass` over each pass's released text |
| `end(job, res)` / `error(job, kind)` | worker | the request is over (`kind` engine: a failed bail; internal: a bug) |
| `bail(recovered)`, `retry(n)`, `refusal(kind)`, `evicted(ms)`, `pgmigrate_add(n)` | engine (M7) | `FlashNextTP.verify`'s bitwise retry; host refusals (routing flag, bounds, context, prefill, cache, capacity); `passpolice`'s KFD `evicted_ms` and page migrations |
| `cache_update(counters, gauges)` | session / `PromptCache` owner | `counters` = `PromptCache.stats` as is. Its keys `checkpoints evicted spilled parked unparked park_failed refused side_requests side_ends main_lost external_reset` are copied as totals. `gauges` = `n_slots`, `slots_used` (`len([c for c in pc.ckpts if c.slot is not None])`), `tier_mib`, `headroom_mib`, `disk_bytes` (`pc.store.total_bytes()`), `disk_cap_bytes` (`pc.store.cap`), `side_active` (`pc.side is not None`), `side_rows` |
| `route(decision)` | session (M7 finish) | the request's side-slot routing decision, `PromptCache.route`'s dict (`route`, `reason`, `shared`, `root`, `need`, `main_n`, `side_rows`): counted by route and reason, the last one kept |

**Measured cost** (test_stats `TestOverhead`; this box, 11-05, with a GPU job running on the cards; best of 7 repeats,
two runs):
- **3.98-4.19 us per decode pass recorded.** That covers what the loop, the session and the worker add with the hook
  on: clock reads, `pass_done`, `tokens`, `ContentClass.feed`.
- **0.08-0.09 us per pass with the hook disabled.**
- **7.81-8.18 us per request:** begin, 4 progress calls, prefill_done, first_token, cache_update, end.
- **End to end on the CPU mock** (4-token passes, 3,000 tokens): 101.7 / 101.5 / 106.6 and 102.9 / 102.7 / 107.9 us
  per pass with no stats / disabled / on.
- Against a 16.5 ms T=4 pass, 4.2 us is 0.03 %. M7 measures the real gate on the engine.

## 3. llama-server's series (`llamacpp:`)

Names, HELP texts, types and order are llama-server's (`server_task_result_metrics::to_metrics`).

| Series | Type | Unit | Caminus source |
|---|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | tokens | prompt tokens processed, cached excluded (`PrefillStats.processed_n`) |
| `llamacpp:prompt_tokens_cached_total` | counter | tokens | prompt tokens reused (`PrefillStats.cached_n`) |
| `llamacpp:prompt_seconds_total` | counter | s | `PrefillStats.ms` |
| `llamacpp:tokens_predicted_total` | counter | tokens | generated tokens (EOS included, as llama.cpp's `n_decoded`) |
| `llamacpp:tokens_predicted_seconds_total` | counter | s | prompt processed -> request end (llama.cpp's `predicted_ms`) |
| `llamacpp:n_decode_total` | counter | passes | engine passes: prompt passes + decode passes |
| `llamacpp:n_tokens_max` | counter (llama-server's type) | tokens | the largest prompt + generated |
| `llamacpp:spec_decode_num_draft_tokens_total` | counter | tokens | drafts verified |
| `llamacpp:spec_decode_num_accepted_tokens_total` | counter | tokens | drafts accepted |
| `llamacpp:spec_decode_num_drafts_total` | counter | passes | verification passes with at least one draft |
| `llamacpp:prompt_tokens_seconds` | gauge | tok/s | over the window since the last scrape, then reset (llama-server's bucket; two scrapers share it, as with llama-server) |
| `llamacpp:predicted_tokens_seconds` | gauge | steps/s | decode passes per second over the same window (llama-server counts decode steps, not tokens) |
| `llamacpp:requests_processing` | gauge | | 1 while the engine worker runs a job |
| `llamacpp:requests_deferred` | gauge | | jobs waiting in the FIFO |
| `llamacpp:n_busy_slots_per_decode` | gauge | | 1 once any pass ran (one slot) |
| `llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position}` | counter | tokens | accepted drafts per draft position 0 .. max drafted - 1 |

**What llm-top uses:**
- `requests_processing + requests_deferred > 0`, or a change of `prompt_tokens_total + prompt_tokens_cached_total +
  tokens_predicted_total`, makes it poll `/slots`.
- `/slots`' `n_prompt_tokens_processed` and `next_token[0].n_decoded` deltas give the read / write rates. Both move
  per pass (test_metrics `test_live_progress`).
- `params["speculative.types"]` (`draft-dflash` with a drafter, `none` without) and the `spec_decode_*` counters give
  the speculative rows.
- `params.n_predict` is the client's requested `max_tokens`, never the context clamp, -1 when none (16 for
  `/v1/completions`' default). A Qwen Code compaction is recognised by exactly 20000.

## 4. The Caminus series (`caminus:`)

Quantile series are Prometheus summaries: `quantile` 0.5 / 0.95 over a window of the last 1,024 values, `_sum` /
`_count` since start. A series with no value yet (unknown) is left out.

### Speed vs physics

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:physics_floor_ms` | gauge | ms | the bandwidth floor of one decode token (below) |
| `caminus:physics_pass_pct` | gauge | % | floor / median K3 pass time. A pass streams at least one token's bytes, so on the engine this is at most 100 (the CPU mock's microsecond passes read far above it). |
| `caminus:physics_token_pct` | gauge | % | floor / decode ms per emitted token over the window; above 100 when drafts are accepted |
| `caminus:decode_ms_per_token` | gauge | ms | decode pass wall time / emitted tokens over the window |
| `caminus:decode_pass_ms{part}` | summary | ms | `part` = `k3` (the verify: `pass_k3_ms`), `drafter`, `sampler`, `host` (the rest of the pass), `total` |
| `caminus:ttft_ms{part}` | summary | ms | `part` = `queue` (FIFO wait), `restore`, `batched` (batched prefill), `handoff`, `passes` (K3 prompt passes incl. the first-logits pass), `host` (the rest), `total` (queue included) |
| `caminus:prefill_tokens_per_second` | summary | tok/s | processed prompt tokens / prompt seconds, per request with new tokens |

**The floor.** `PhysicsConfig` gives each card the bytes one token streams and the card's measured streaming bandwidth.
- **Default:** the Coder IQ1_M at TP=2, 3.60 / 2.87 GB at 881 / 774 GB/s (spec-gsq-port.md section 2).
- **Mode `max` (default):** the slowest card's bytes / bandwidth. The cards stream in parallel and a pass waits for
  both, so this is 3.60 / 881 = **4.086 ms**. That reproduces the plan's 4.09 ms.
- **Mode `sum`:** all bytes / summed bandwidth, 6.47 / 1655 = 3.91 ms. Only a perfectly balanced split reaches it.
  The plan's wording "bytes / summed bandwidth" gives this number, not 4.09, so the default follows the number.
- **Configurable** per server: `K3Server(physics=PhysicsConfig(...))`. M7 passes the engine's split.

### DFlash

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:spec_draft_tokens_total{class}` | counter | tokens | drafts verified, per content class |
| `caminus:spec_accepted_tokens_total{class}` | counter | tokens | drafts accepted, per class |
| `caminus:spec_drafts_total{class}` | counter | passes | passes with drafts, per class |
| `caminus:spec_accepted_per_pos_total{class,position}` | counter | tokens | accepted drafts per class and draft position |
| `caminus:spec_acceptance_ratio{class}` | gauge | | accepted / drafted since start; `class="all"` too |
| `caminus:spec_tokens_per_pass` | gauge | tokens | emitted tokens per decode pass over the window |
| `caminus:plain_ms_per_token` | gauge | ms | a plain pass (no drafts, one token fed): an EMA of measured ones, else `PhysicsConfig.plain_ms_per_token` (unset: absent) |
| `caminus:spec_speedup_vs_plain` | gauge | x | plain ms per token / decode ms per emitted token over the window (absent until a plain pass was measured or configured) |

**Content classes** (`agentmon.ContentClass`, fed each pass's released text on the engine thread):
- `tool`: inside `<tool_call>` .. `</tool_call>`;
- `shell`: inside a call of a shell tool (`run_shell_command`, `bash`, ...), or in a fenced block tagged bash / sh /
  console / ...;
- `code`: in any other fenced block;
- `prose`: everything else, reasoning and answer text alike.

A pass counts in the class it started in.

### Prompt cache

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:cache_tokens_total{tier}` | counter | tokens | reused prompt tokens by tier: `vram` (sources `live`, `vram`), `disk` (`spill`, `disk`), `exact` (exact hits) |
| `caminus:cache_requests_total{source}` | counter | requests | `PromptCache.plan` / `prepare` source: `live`, `vram`, `spill`, `disk`, `cold` |
| `caminus:cache_restore_ms` | summary | ms | restore time of a cached state that is not the live one (`PrefillStats.restore_ms`) |
| `caminus:cache_lcp_tokens` | summary | tokens | longest common prefix of each request with the live token log |
| `caminus:cache_vram_slots{state}` | gauge | slots | `used` / `total` VRAM checkpoint slots |
| `caminus:cache_vram_tier_mib` | gauge | MiB | the tier's VRAM (`pc.tier_mib`) |
| `caminus:cache_vram_headroom_mib` | gauge | MiB | the headroom the tier was sized from (`pc.headroom`) |
| `caminus:cache_disk_bytes` | gauge | bytes | `pc.store.total_bytes()` |
| `caminus:cache_disk_cap_bytes` | gauge | bytes | `pc.store.cap` (30 GB default) |
| `caminus:cache_events_total{event}` | counter | | the prompt cache's own counters: `checkpoints`, `evicted`, `spilled`, `parked`, `unparked`, `park_failed`, `refused`, `side_requests`, `side_ends`, `main_lost`, `external_reset` |
| `caminus:cache_route_total{route,reason}` | counter | requests | side-slot routing (section 13): `route="side"` `reason="unrelated"`; `route="main"` with `reason` `match`, `no_main`, `no_fit`, `main_shorter`, `parked`, `off` |
| `caminus:cache_side_active` | gauge | | 1 while a side call holds the engine (the main line waits in the side slot), else 0; absent until the engine reports it |
| `caminus:cache_prefill_seconds_saved_total` | counter | s | estimate: each reused token / the prefill rate EMA at the time |
| `caminus:cache_prefill_joules_saved_total` | counter | J | estimate: reused tokens x measured prefill joules per processed token (needs the hardware reader) |

### Context

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:context_position` | gauge | tokens | tokens the engine holds for the current (or last) sequence |
| `caminus:context_max` | gauge | tokens | the session's max_ctx |
| `caminus:context_remaining` | gauge | tokens | max_ctx - position |
| `caminus:context_qsa` | gauge | 0/1 | 1 past the dense range (position >= `PhysicsConfig.qsa_from`, capacity.QSA_EXACT 2051) |
| `caminus:kv_bytes{card}` | gauge | bytes | position x KV + indexer bytes per token per card: 6,528 B of KV each (13,056 across both) + 2,040 B indexer, replicated (AGENTIC-PLAN 0.5) |

### Reliability

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:bails_total{outcome}` | counter | | `recovered` (the bitwise retry succeeded, M7 hook) / `failed` (retries exhausted: an `EngineError`) |
| `caminus:pass_retries_total` | counter | | pass retries after a bail |
| `caminus:kfd_evicted_ms_total` | counter | ms | this process's KFD queue eviction time (`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms`, read by the hardware sampler thread with passpolice's reader): per interval the largest per-node rise (one eviction stops every node at once: never the sum) |
| `caminus:pgmigrate_total` | counter | | host page migrations (`/proc/vmstat` `pgmigrate_success`: system-wide, not per process) in hardware-sampler intervals (1 s) while the engine was busy (phase prefill / decode at either end) |
| `caminus:pgmigrate_all_total` | counter | | the same counter in every interval, idle included |
| `caminus:refusals_total{kind}` | counter | | host refusals: `routing_flag`, `bounds`, `context`, `prefill`, `cache`, `capacity` |
| `caminus:errors_total{kind}` | counter | requests | `engine` (EngineError: the session reset, 503 / 529), `internal` (a bug: 500) |
| `caminus:vram_guard_margin_mib{card,name}` | gauge | MiB | guard line - reserve - VRAM used. Guard lines: tools/guard.sh, `capacity.GUARD_LINE_MIB`, XTX 23,800 / XT 19,850. Reserve: the XTX keeps 512 MiB. This is the XTX reserve margin. |

### TP / hardware

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:card_pass_ms{card}` | summary | ms | per-card busy time of a decode pass (`pass_cards`; the real engine: device events around the target's pass on each card's stream, `FlashNextTP.enable_card_timing`, read after the pass's own wait) |
| `caminus:exchange_wait_ms{card}` | summary | ms | per-card wait on the TP exchange in a pass: the x4 link's drag (synthesized by MockSession; the real engine cannot measure it: absent, the reason in `cards[].exchange_wait_note`, section 10) |
| `caminus:link_bytes_total` | counter | bytes | bytes over the TP link by decode passes (the real engine: the exchanges' payload, 2 cards x 2 exchanges per layer x T x 2560 x 4 B = 7.86 MB per T = 4 pass; the epoch flags not counted) |
| `caminus:link_bytes_per_second` | gauge | B/s | link bytes / decode pass time over the window |
| `caminus:vram_used_mib{card,name}` | gauge | MiB | sysfs `mem_info_vram_used` (every process, as the guard reads it) |
| `caminus:vram_total_mib{card,name}` | gauge | MiB | sysfs `mem_info_vram_total` |
| `caminus:power_watts{card,name}` | gauge | W | hwmon `power1_average` (uW) |
| `caminus:sclk_mhz{card,name}` | gauge | MHz | hwmon `freq1_input`, else the `*` line of `pp_dpm_sclk` |
| `caminus:mclk_mhz{card,name}` | gauge | MHz | hwmon `freq2_input`, else `pp_dpm_mclk` |
| `caminus:gpu_busy_percent{card,name}` | gauge | % | sysfs `gpu_busy_percent` |
| `caminus:temp_junction_celsius{card,name}` | gauge | C | hwmon `temp2_input` |
| `caminus:clock_forced_high{card,name}` | gauge | 0/1 | `power_dpm_force_performance_level` == `high` (what gpu-high sets; `auto` is ~13 % slower) |
| `caminus:runtime_suspend_allowed{card,name}` | gauge | 0/1 | the card's runtime PM control (`device/power/control`) is `auto`: it may runtime-suspend, and after a resume the SMU runs at auto clocks while the level still reads `high` (the gpu-high note); 0 when `on` |
| `caminus:clock_check_ok` | gauge | 0/1 | the startup clock check (section 10): every engine card at DPM `high` with runtime PM `on`; reported, never changed |
| `caminus:energy_joules_total{card,phase}` | counter | J | power integrated over time (trapezoid between readings), by engine phase `idle` / `prefill` / `decode` |
| `caminus:tokens_per_joule{phase}` | gauge | tok/J | `decode`: generated tokens / decode-phase joules of both cards; `prefill`: processed prompt tokens / prefill-phase joules |

### Agent behaviour

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:requests_total{harness}` | counter | requests | generation POSTs (chat, messages, completions) by User-Agent class: `qwen-code`, `claude-code`, `hermes`, `opencode`, `codex`, `aider`, `llm-top`, `anthropic-sdk`, `openai-sdk`, `curl`, `python`, `other`, `none`. GET polls are not counted. |
| `caminus:tool_calls_total` | counter | calls | tool calls parsed and reported (not cut) |
| `caminus:tool_call_failures_total{kind}` | counter | | `cut` (the end fell inside a value: never reported), `unclosed` (closing tags lost), `undeclared` (a tool the request did not declare), `malformed` (tag text that was not a call) |
| `caminus:tool_call_lenient_total` | counter | | leniency the parser applied (name aliases, coerced values, missing newlines) |
| `caminus:tokens_generated_total{kind}` | counter | tokens | `reasoning` (up to and including `</think>`) / `answer` |
| `caminus:finish_total{reason}` | counter | requests | `eos`, `stop`, `length`, `cancel`, `error` |
| `caminus:max_tokens_hits_total` | counter | requests | requests that ended at max_tokens (finish `length`) |
| `caminus:loop_requests_total` | counter | requests | requests whose reasoning repeated one line `LoopDetector.REPEAT` (3) times or more |
| `caminus:reasoning_repeated_lines_total` | counter | lines | reasoning line occurrences beyond the first, summed over requests |
| `caminus:loop_live_repeats` | gauge | | the current request's highest repeat count of one reasoning line (0 between requests) |

**The loop detector** (`agentmon.LoopDetector`) runs on the HTTP thread over the raw text as it streams:
- **Lines:** complete reasoning lines, stripped, of at least 12 characters, counted by hash.
- **Reasoning:** the text before `</think>` when the request thinks.
- **Bound:** at most 4,096 distinct lines per request.

Together with `caminus:max_tokens_hits_total`, it is the "live loop detector" the bake-off plan asks for. Neither the
threshold nor the minimum line length has been tuned on real traces yet.

### Server

| Series | Type | Unit | Source |
|---|---|---|---|
| `caminus:info{engine,build,model,status_version}` | gauge | | always 1 |
| `caminus:stats_enabled` | gauge | 0/1 | the hook records (0: the overhead A/B) |
| `caminus:queue_depth` | gauge | requests | on the engine plus waiting |
| `caminus:http_responses_total{route,status}` | counter | | HTTP responses (replaces the old `k3_requests_total`) |
| `caminus:worker_total{counter}` | counter | | the worker's own counters: `jobs`, `engine_errors`, `internal_errors`, `cancelled`, `prompt_tokens`, `cached_tokens`, `generated_tokens`, `passes`, `drafted`, `accepted`, `aborts`, `resets` (replaces the old `k3_*_total`) |

## 5. GET /caminus/status (version 1)

One JSON object. Windows (`decode.split_ms.*`, `ttft.split_ms.*`, `prefill.tokens_per_s`, `cache.restore_ms`,
`cache.lcp`, `decode.tokens_per_pass`, `cards[].pass_ms`, `cards[].exchange_wait_ms`) are
`{p50, p95, mean, window, max, count, sum, unit, hist}`:
- p50 / p95 / mean over the last `window` values (at most 1,024);
- max / count / sum since start;
- `hist`: `{le: [edges..., "+Inf"], counts: [...]}` per bucket since start. Not cumulative, unlike Prometheus.
  Edges ~x2 from 0.05 ms to 65 s for ms series.

Absent values are `null`. Bump `version` on any incompatible change.

| Field | Unit | Meaning / source |
|---|---|---|
| `version` | | `STATUS_VERSION` (1) |
| `engine` | | `"caminus"` |
| `build_info` | | `caminus-<short sha>` |
| `time` | s | Unix time of the reading |
| `uptime_s` | s | since the stats object was made (its clock) |
| `stats_enabled` | | the hook records |
| `server.model_alias`, `server.model_path`, `server.n_ctx` | | as in /props |
| `server.requests_processing`, `server.requests_deferred`, `server.queue_depth` | requests | as the `llamacpp:` gauges |
| `server.loading` | | answering 503 |
| `server.speculative_types` | | `draft-dflash` or `none` |
| `server.requests` | requests | requests that reached the engine and ended |
| `llamacpp.prompt_tokens_total`, `llamacpp.prompt_tokens_cached_total`, `llamacpp.prompt_seconds_total`, `llamacpp.tokens_predicted_total`, `llamacpp.tokens_predicted_seconds_total`, `llamacpp.n_decode_total`, `llamacpp.n_tokens_max` | | the counters of section 3 |
| `physics.floor_ms`, `physics.mode`, `physics.pass_pct`, `physics.token_pct`, `physics.k3_pass_p50_ms`, `physics.ms_per_token` | ms, % | section 4, speed vs physics |
| `physics.source` | | `spec` (the defaults) or `engine` (the per-card bytes from the engine's weights, section 10) |
| `physics.pass_source`, `physics.target_pass_p50_ms` | , ms | the denominator of `pass_pct`: `card_events` (the median target pass span: the slower card's `card_pass_ms`) or `k3_host` (`pass_k3_ms`, which on the device-accept path also holds the drafter step) |
| `physics.cards` | B, B/s | per card `{bytes_per_token, bandwidth_bytes_s}` (the configuration) |
| `decode.split_ms` | ms | windows by part: `k3`, `drafter`, `sampler`, `host`, `total` |
| `decode.tokens_per_pass` | tokens | window |
| `decode.ms_per_token` | ms | over the window |
| `decode.passes` | passes | decode passes since start |
| `ttft.split_ms` | ms | windows by part: `queue`, `restore`, `batched`, `handoff`, `passes`, `host`, `total` |
| `prefill.tokens_per_s` | tok/s | window, per request |
| `prefill.rate_ema` | tok/s | the EMA behind the seconds-saved estimate |
| `dflash.drafts`, `dflash.draft_tokens`, `dflash.accepted`, `dflash.acceptance` | | totals, accepted / drafted |
| `dflash.per_pos`, `dflash.per_pos_rate` | | accepted per draft position; / drafts (llm-top's per-position share) |
| `dflash.classes` | | per class (`prose`, `code`, `shell`, `tool`): `{drafts, draft_tokens, accepted, acceptance, per_pos, per_pos_rate}` |
| `dflash.tokens_per_pass` | tokens | window mean |
| `dflash.plain_ms_per_token`, `dflash.plain_measured` | ms | the plain reference and whether it was measured (else configured) |
| `dflash.plain_source` | | `measured` (an EMA of plain passes' ms per emitted token), `estimate` (K3Session: the draft policy's plain cost, a DFlash pass less the drafter step's device time, while no plain pass ran: DFlash-always on this engine), `configured`, or null |
| `dflash.speedup_vs_plain` | x | live speedup over plain |
| `cache.tokens` | tokens | reused by tier: `vram`, `disk`, `exact` |
| `cache.sources` | requests | by source: `live`, `vram`, `spill`, `disk`, `cold` |
| `cache.restore_ms`, `cache.lcp` | ms, tokens | windows |
| `cache.vram` | | `{slots_used, slots, tier_mib, headroom_mib}` |
| `cache.disk` | | `{bytes, cap_bytes, gb, cap_gb}` (GB = 1e9 bytes, the cap's unit) |
| `cache.events` | | the prompt cache's counters (section 4) |
| `cache.route` | | side-slot routing (section 13): `{main: {match, no_main, no_fit, main_shorter, parked, off}, side: {unrelated}, last: {route, reason, shared, root, need, main_n, side_rows}, side_active, side_rows}` |
| `cache.saved` | | `{prefill_s, tokens, joules}`: the estimates of section 4 |
| `context.position`, `context.max_ctx`, `context.remaining` | tokens | section 4, context |
| `context.attention` | | `dense` or `qsa` |
| `context.qsa_from` | position | the first QSA position |
| `context.kv_bytes` | bytes | per card |
| `reliability.bails` | | `{recovered, failed}` |
| `reliability.retries`, `reliability.kfd_evicted_ms`, `reliability.pgmigrate` | | totals |
| `reliability.pgmigrate_all`, `reliability.host_samples` | | page migrations in every sampler interval; the sampler's host samples (0: no sampler feeds the host counters) |
| `reliability.refusals` | | by kind |
| `reliability.errors` | requests | `{engine, internal}` |
| `reliability.xtx_margin_mib` | MiB | card 0's guard margin (null without the hardware reader) |
| `cards[].index`, `cards[].name`, `cards[].pci` | | engine device index, XTX / XT, PCI slot |
| `cards[].vram_used_mib`, `cards[].vram_total_mib` | MiB | sysfs |
| `cards[].power_w` | W | hwmon |
| `cards[].sclk_mhz`, `cards[].mclk_mhz` | MHz | hwmon / pp_dpm |
| `cards[].perf_level`, `cards[].forced_high` | | `power_dpm_force_performance_level`; == `high` |
| `cards[].busy_pct` | % | `gpu_busy_percent` |
| `cards[].temp_edge_c`, `cards[].temp_junction_c` | C | hwmon `temp1_input` / `temp2_input` |
| `cards[].link_speed`, `cards[].link_width` | | sysfs `current_link_speed` / `current_link_width` |
| `cards[].runtime_pm`, `cards[].runtime_status` | | the PCI device's `power/control` (`on` / `auto`) and `power/runtime_status` (`active` / `suspended`) |
| `cards[].exchange_wait_note` | | null when the card has exchange-wait values; else the engine's reason (section 10) |
| `cards[].guard_mib`, `cards[].margin_mib` | MiB | the guard line, the margin below it after the reserve |
| `cards[].energy_j` | J | by phase `{idle, prefill, decode}` |
| `cards[].kv_bytes` | bytes | this card's KV + indexer bytes |
| `cards[].pass_ms`, `cards[].exchange_wait_ms` | ms | windows (engine per-card timings) |
| `link.bytes_total`, `link.bytes_per_s` | B, B/s | the TP link |
| `energy.tokens_per_joule` | tok/J | `{decode, prefill}` |
| `energy.joules` | J | `{decode, prefill}` summed over the cards |
| `agent.harness` | requests | by User-Agent class |
| `agent.tool_calls`, `agent.tool_call_failures`, `agent.tool_call_lenient` | | section 4, agent behaviour |
| `agent.loop` | | `{requests_flagged, repeated_lines, live_max_repeat, repeat_threshold}` |
| `agent.tokens` | tokens | `{reasoning, answer}` |
| `agent.finish` | requests | by finish reason |
| `agent.max_tokens_hits` | requests | finish `length` |
| `slot.id`, `slot.n_ctx`, `slot.speculative`, `slot.is_processing`, `slot.id_task`, `slot.n_prompt_tokens`, `slot.n_prompt_tokens_processed`, `slot.n_prompt_tokens_cache`, `slot.params`, `slot.next_token` | | the /slots entry |
| `clock_check` | | the startup clock check (section 10): `{ok, cards: [{index, name, pci, perf_level, forced_high, runtime_pm, runtime_status, sclk_mhz, ok, notes}], summary}`; null when the server ran none (the mock) |

Without a hardware reader, `cards` still lists the engine's cards with `index`, `name`, `pass_ms`, `exchange_wait_ms`,
`kv_bytes`.

## 6. Hardware reader (`hw.HwReader`)

**What it reads.** Read-only amdgpu sysfs under a root directory: `/` on the real machine, a fixture tree in tests
(`hw.write_fixture`). Per card:
- `/sys/class/drm/cardN/device/{mem_info_vram_used, mem_info_vram_total, gpu_busy_percent,
  power_dpm_force_performance_level, pp_dpm_sclk, pp_dpm_mclk, current_link_speed, current_link_width, uevent}`;
- `hwmon/hwmonM/{power1_average, freq1_input, freq2_input, temp1_input, temp2_input}`.

A missing or unreadable file reads as None.

**Which cards.** Both cards are PCI id 1002:744c, so the id cannot tell them apart.
- Default: every card with at least 4 GiB of VRAM and a `power1_average`, the larger VRAM first. That gives the XTX
  (device 0, 24 GiB) then the XT (20 GiB). The iGPU has neither and is left out.
- `cards=[PCI slots]` sets the engine's order explicitly; M7 passes the engine devices' slots.

**When it reads.** A reading is cached for `min_interval` (1 s). `start(interval)` runs a daemon sampler thread (the
server's; never the engine thread). Each reading integrates power into joules per card, attributed to the engine phase
(`stats.phase`: idle / prefill / decode); a gap over 30 s is not integrated. GET handlers only read the latest
reading.

**Caveat.** Power and clocks come from the SMU. The mock server reads hardware only when asked: `--hw-root /` for the
real cards, or `--hw-fixture DIR` for a synthetic tree.

**What read the real cards in this step:**
- Caminus' own reader read fixture trees only.
- The llm-top `-once` run (section 8) is llm-top's own behaviour. During its ~7 s window it read the real cards'
  `mem_info_vram_*` and `gpu_busy_percent`, while the GPU chain was running: its MACHINE panel shows their live VRAM.
  It does this on every run.
- No HIP context was created and no compute ran.

## 7. MockSession: synthetic engine values and the deterministic mode

`MockSession(stats=, synth=SynthProfile(...), disk_slots=N)`.

**Measured on the mock itself:** tokens, acceptance per position and class, cache sources and LCP, User-Agent classes,
tool calls, reasoning vs answer tokens, loops, finish reasons.

**Synthesized** (only the engine-internal values the real engine will measure):
- **Per-card numbers:** each card's pass and exchange-wait share of the measured K3 pass (`card_split` 0.97 / 0.91,
  `wait_split` 0.05 / 0.11).
- **Link bytes:** 96 all-reduces x 2560 fp32 x 2 directions per token row (INFERRED).
- **The batched prefill:** every prompt pass but the last when the new tokens reach T* 48. It runs at 2,900 tok/s and
  is followed by a 2 ms hand-off.
- **Restores by source:** VRAM 0.6 ms, disk 260 ms.
- **Cache tier sizes:** 160 MiB per checkpoint and a 30 GB disk cap.
- **Bails:** an optional recovered bail every N passes.

**The stand-in disk tier.** `disk_slots` > 0 enables a second checkpoint tier: a checkpoint the VRAM LRU drops is
spilled there and can be unparked. With 0 (the default) the server tests' behaviour is unchanged.

**The deterministic test mode.** `SynthProfile(clock=stats.VirtualClock())`, with the same clock given to
`EngineStats` and `MockDrafter(clock=, ms=)`, makes every duration synthetic:
- K3 pass 10.45 / 13.2 / 15.0 / 16.5 ms for T = 1..4;
- sampler 0.12, host 0.35, drafter 2.0 ms;
- the batched prefill at 2,900 tok/s, the hand-off 2 ms, restores as above.

A fixed request sequence then gives exact values and byte-identical /metrics text (test_stats `TestDeterministic`).

**The launcher.** `tools/k3_serve.py` runs the mock with `SynthProfile` (sleeping its synthetic batched prefill,
hand-off and restores; `--no-synth` turns it off), `--mock-disk 32`, a recovered bail every 5,000 passes, and
`--draft` for DFlash-like drafting.

## 8. llm-top integration check (11-05)

llm-top was built from its repository (HEAD plus an uncommitted web view) with
`GOPROXY=off go build -o <scratch>/llm-top ./src`. It was run with `-once 120 40` against `tools/k3_serve.py --draft
--hw-fixture <scratch>`, under a scratch config:
- `server_url` = the mock;
- `backend` llama.cpp;
- `sessions_dir` an empty scratch directory;
- `lmstudio_logs` a scratch path;
- `server_process` `caminus`.

A client loop of shared-prefix requests ran meanwhile: two threads, Qwen Code's User-Agent, 8 requests, 7,840
prompt tokens of which 5,216 were reused, 960 generated.

**What the frame showed.** The mock appears as a llama.cpp server:
- **Title:** `k3-coder · Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002 · :<port>`, in the READING state with the
  progress gauge "94 % 178 new + 648 reused".
- **CONTEXT panel:** `build caminus-c58ed54`, "context 33k tokens · 1 slot".
- **THROUGHPUT panel:** read / write sparklines, "cache reuse 30.5 %", the `dflash` row (acceptance 4.3 % in the window,
  4.8 % since start) with its tok/draft and by-position bars, and "since start: 1.5k read · 648 reused · 320
  written".
- **MACHINE panel:** found the process by comm (`caminus memory 0.3 GB`), next to both cards' real VRAM.
- **ACTIVITY panel:** each finished request with new and cached tokens.

The low acceptance is the mock's: random logits over the 248K vocabulary at temperature 0.7.

**The server's own numbers in the same run** (`/caminus/status`):
- the floor 4.0863 ms;
- K3 pass p50 17.7 ms (10 ms sleep + the mock's logits), 23 % of the floor;
- TTFT split p50: queue 2,889 / restore 0.9 / batched 743 / hand-off 2.1 / passes 15.8 ms (two clients share one
  slot);
- cache sources: vram 6, cold 2; checkpoints 16, evicted 8, spilled 8;
- the fixture cards: XTX 20,205 MiB used, margin 3,083 MiB; XT 15,513, margin 4,337;
- energy by phase from the fixture's static power: 0.109 tokens / J in decode.

## 9. Left for M7 (done: M7 wiring and section 10)

- **Wire the real engine:**
  - `EngineSession` over `PromptCache` fills `PrefillStats` (source, exact, lcp, the TTFT split from `prepare()`'s
    `ms`);
  - the engine sets `pass_k3_ms` / `pass_sampler_ms` (device sampler) and `pass_cards` (per-card host timestamps
    around its launches and exchange waits, if they exist without a sync, else absent);
  - `bail` / `retry` from `FlashNextTP.verify`, `evicted` / `pgmigrate_add` from passpolice, `refusal` at the host
    refusals, `cache_update(pc.stats, ...)` after each request.
- **Hardware:** `HwReader(cards=<the engine devices' PCI slots>)` with its sampler thread, and `PhysicsConfig` from
  the engine's TP split.
- **The <= 3 % gate** with the metrics on (`EngineStats(enabled=False)` for the A/B).
- **The llm-top side** (its own repo): a `caminus` backend detected from `/props`' `engine`, `caminus` in findProc's
  names, and panels reading `/caminus/status`.

## 10. The real engine's metrics (M7 ops, impl:m7-ops)

Section 9's list, done on `K3Session` / `caminus serve`. Every value is still a host timestamp, a counter the loop has, or a
reading on another thread; no device sync is added.

**The scrape off the decode path.** Every hook that writes takes `EngineStats.lock` for its few adds (uncontended ~0.1 us; the
CPU cost test reads 4.8 us per recorded pass with the lock, was 4.2). A scrape takes `snapshot()` under the same lock: a copy of
the counters and the rings (~0.1 ms), and for `/metrics` llama-server's rate bucket taken and reset in the same critical
section (the old swap on the HTTP thread could lose the engine thread's adds). Quantiles and text are computed on the HTTP
thread from the copy, after the lock is released. The GIL is still shared: the gate is measured on the engine
(tests/test_serve_k3.py `scrape`: decode ms/token with and without a 10 Hz scraper of `/metrics` + `/caminus/status` + `/slots`
in a child process, interleaved; a difference over 0.05 ms fails).

**Per group:**
- **llama-server's series:** as before; `/slots`' `n_prompt_tokens_processed` now moves per batched-prefill chunk and per K3
  prompt pass (the cancel object `K3Session` hands to `pc.prepare` carries `progress(position)`, which `Prefill.run` calls after
  each chunk and the cache after each K3 prompt pass).
- **Speed vs physics:** the floor's per-card bytes come from the engine (`k3_session.engine_physics`: every weight the tapes
  read per card, `FlashNextTP.shards` + `.weights`, the routed experts at k_used / n_exp; `physics.source` `engine`); the
  bandwidths stay the measured 881 / 774 GB/s. `physics.pass_pct` divides by the target's own pass span (the slower card's
  `card_pass_ms`), not `pass_k3_ms` (which holds the drafter step on the device-accept path).
- **DFlash:** filler drafts are not drafts. `K3Session.drafts_real` is False for a plain pass (the policy's filler) and for
  the drafter's own filler (no draft launch: a window gap, a block past max_pos); `run_generation` then counts the pass with 0
  drafts (the per-class / per-position acceptance, `timings.drafted`, the plain pass time). The plain reference: measured when
  a plain pass ran, else the policy's estimate (a DFlash pass less the drafter step's device time; DFlash-always on this engine
  means plain passes are rare), `dflash.plain_source`.
- **Prompt cache:** as before; the cache's `refused` rises are also `refusals_total{kind="cache"}`.
- **Context:** the KV bytes per position from the prompt cache's layout (every attention layer's KV row on the card + the
  replicated IDXRAW row and a quarter pool row), not the spec's constants.
- **Reliability:** bails / retries as M7 wired them; refusals by kind (`k3_session.refusal_kind`: `RoutingError` ->
  `routing_flag`, `PrefillRefused` -> `prefill`, `ContextBoundsError` -> `bounds` (`context` for the request-level check of
  `prepare` and the HTTP layer's 400), `CapacityError` -> `capacity`, `CacheError` -> `cache`); `kfd_evicted_ms` and
  `pgmigrate` from the hardware sampler thread (`hw.HostCounters`, `EngineStats.host_sample`, 1 Hz).
- **TP / hardware:** `card_pass_ms` from device events (`FlashNextTP.enable_card_timing`, on when the stats hook records: a
  timing event before the pass's stage copy on each card's stream and `ev_done` made a timing event, read after `_pass_wait`'s
  synchronize, so no added sync; off by default, so gates and benches run the old calls); `link_bytes_total` the exchanges'
  payload (2 x 96 x T x 2560 x 4 B per pass); the hardware reader on the engine's own cards by PCI slot (`k3_session.engine_pci`
  from torch's device properties), started by `caminus serve` (`--hw-root /` is the default, `''` turns it off).
- **Exchange wait: not measured on the real engine.** The TP exchange waits inside the persistent kernel (OP_XCHG spins on the
  peer's flag), so the wait is inside each card's pass span; separating it needs device timestamps inside the kernel (the
  `*_trace` builds, tools/k3_trace.py), which the serving code objects do not have, and a kernel change is out of this step's
  scope. `caminus:exchange_wait_ms` is absent and `cards[].exchange_wait_note` gives the reason; nothing derived is labelled
  as a wait.

**The clock check** (`hw.clock_check`, `caminus serve` at startup, read only): per engine card the DPM level
(`power_dpm_force_performance_level`) and the PCI runtime PM control and status (`device/power/control`,
`power/runtime_status`). OK = `high` with runtime PM `on` (what the guard's `gpu-high` sets; tools/guard.sh runs every command
under it). `auto` is noted (decode ~13 % slower), `high` with runtime PM `auto` is noted as not held (after a runtime suspend /
resume the SMU runs at auto clocks while the level still reads `high`), a suspended card and unreadable files too. The result
is printed in the log and served as `clock_check` in `/caminus/status` and `caminus:clock_check_ok`; the per-card live values
are `cards[].perf_level` / `runtime_pm` / `runtime_status` and `caminus:runtime_suspend_allowed`.

## 11. The flight recorder (M7, user 11-05; impl:m7-ops)

**Where.** `serve/flight.py` `FlightRecorder`, `caminus serve --flight-dir` (default `~/.cache/caminus/flight`, `''` off;
`--flight-cap-gb 10`). The directory is 0700 and every file 0600 whatever the umask (`os.open(..., 0o600)`; an existing
directory is chmod-ed); one owner (an flock on `.lock`). Size cap (default 10 GB: the oldest bundles first, then the oldest JSONL
days, today's kept), retention 7 days, a 10 GB free-space floor for a bundle's state. Purge: `POST /caminus/flight/purge
{"older_than_days": D}` on the server, or `tools/flight.py purge [--dir] [--older-than-days D]` (refused while a server holds
the directory). Local only: never uploaded, never in the repo or a test fixture (tests use scratch directories).

**One JSONL line per generation request** (`requests-YYYYMMDD.jsonl`; errors and client-gone requests included), written on the
HTTP thread after the response (`K3Server.flight_row`): the access-log row (route, status, model, timings, tokens, finish) plus
`request_id`, `harness` (the User-Agent class, never the raw string), `profile`, `sampling` (temperature, top_k, top_p, min_p,
penalties, the resolved `seed`), `max_new`, `cache` (source, restore, cached, restore_ms, exact, lcp, slot: `main` or `side`,
the slot side-slot routing ran it in), `ttft_split_ms` (queue,
restore, batched, handoff, passes, total), `decode_ms_per_token`, `dflash` (drafted, accepted, acceptance, passes), `engine`
(the request's bails and recoveries: the session's counters around it), `tool_calls` (ok, failures by kind, lenient), `loop`
(flagged, repeated_lines), `verdict` (the engine thread's: `loop` / `tool_parse` / null) and `bundles`. No request text.

**Replay bundles** (`bundles/<time>-<request id>-<reason>/`): saved on the engine thread (the single owner of the engine):
- on demand: `POST /caminus/flight/bundle {"request_id": ID}` (default the most recent request; one of the last 8), queued
  FIFO with the requests; `tools/flight.py bundle [--url] [--request ID]`;
- automatically (at most 12 an hour): a pass that failed past its retries (`EngineError`, saved before the session reset drops
  the VRAM tier), a reasoning loop (`agentmon.LoopDetector` on the reply's text) or a tool-call parse failure (unclosed,
  undeclared, malformed: the same parser the HTTP thread runs, judged on the engine thread so nothing moves before the save).

A bundle holds `bundle.json` (format 1: the token log (prompt ids, generated ids), the sampling parameters with the Philox seed,
max_new / stops / forced tool-call ids / marks / thinking / profile, the engine key and its sha256, the cache-state reference
(source, cached positions, the prompt-end state (i0, commit), the state file)), `first_logits.npy` (the row the first token was
sampled from, and its sha256) and `state/` (a request that did not start cold: its prompt-end checkpoint with the KV + indexer
rows below it, written by the prompt cache's own writer as a one-checkpoint park entry, `PromptCache.export_checkpoint`; a cold
start needs none: the replay runs the same prefill). A side call takes no checkpoints, so its bundle has no state (the
cache-state note says why): a side call that started cold (the usual case: it shares less than the main line's root) replays
bitwise from cold; one that started from a main checkpoint does not. Its size is the rows below the prompt end plus one checkpoint (~160 MB +
~100 KB per position: ~0.2 GB at 1.5K positions, ~1.5 GB at 32K, ~7 GB at 262K).

**Replay** (`tools/replay.py BUNDLE`, GPU, under tools/guard.sh): the serving engine at the bundle's max_ctx, the bundle's state
linked into a scratch cache directory and loaded explicitly (`PromptCache.load_entry`: the unpark's key / layout / token-hash /
crc checks), then the turn through the server's own session code (`K3Session` + `run_generation`) with the same parameters and
seed; it diffs the generated tokens and the first logits row, bitwise expected (`RESULT BITWISE`, else `RESULT DIFF` with the
first difference). The engine key must match (a bundle saved by another build or another tree is `RESULT KEY MISMATCH`, exit
2: the key holds every build/*.hsaco and the package sources, so a code update makes old bundles refuse; `--force` replays it
anyway, not bitwise expected). Logits: the device-accept passes keep no later logits on the host, so every later position is
compared by its token.

## 12. Measured (impl:m7-ops, 11-05; guarded, both cards, max_ctx 33024, everything on: the hardware sampler, the flight recorder, card timing)

- **Every group populated by the real engine** (tests/test_serve_k3.py `metrics`, 9 / 9): the floor's per-card bytes from the
  engine 3.581 / 2.966 GB (the spec's 3.60 / 2.87; floor 4.065 ms), `pass_pct` 24.9 % over the target span (card p50 16.28 /
  16.36 ms; `pass_k3_ms` p50 17.55 with the drafter step), per-card pass counts == the 10,358 decode passes, link bytes ==
  passes x 7,864,320, KV 8,568 B per position per card, `/slots` progress at 2,048 / 4,096 / ... / 12,000 of a cold 12,001-token
  prompt, the plain reference measured (17.27 ms per token, speedup 1.275 on that mix), refusals context 1 / prefill 2, KFD
  evicted 8 ms and 174 page migrations over the run, the clock check `high` / `on` on both cards (under the guard's gpu-high;
  outside it, with the cards idle, both read `auto` / `auto`: not OK, both notes), VRAM == sysfs.
- **The scrape gate** (`scrape`: 7 x 512 tokens per arm, interleaved, a 10 Hz child scraping `/metrics` + `/caminus/status` +
  `/slots` with a new connection per request): 6.9849 ms/token without, 7.0392 with: **+0.0544 ms, FAIL** (bar 0.05). Reported
  controls in the same run: a second scraper-free arm -0.0005 (the noise); a child with the same 10 Hz wake-ups and ~4 ms of CPU
  per tick that sends nothing +0.0374, so the server's own share of the scrape is ~+0.017 ms/token. Earlier diagnostics (4 x 256
  tokens): `/health` alone +0.047, all three routes +0.062 (new connections) / +0.060 (keep-alive) / +0.059 (keep-alive with a
  0.5 ms GIL switch interval). The rendering is not the cost (a snapshot 0.06 ms, /metrics 0.43 ms, the status 0.56 ms on the
  CPU): most of the delay comes with any periodic process on the host. The gate as defined fails; whether it should bind on the
  server's share is the lead's call (gates are not changed here).
- **Server overhead with everything on** (`overhead`, the direct arm without the card-timing events): +0.95 % short, -1.16 % at
  32K (bar 3 %). Sampler 0.1487 ms per pass (bar 0.15).
- **Flight recorder:** three bundles (a decode-built start with T 1.0 seeded: a 198 MB start state exported in 71 ms; a cold
  start; retry exhaustion after a warm turn), each replayed by tools/replay.py in its own process: 3 / 3 RESULT BITWISE (tokens
  and the first logits row); the retry bundle's 12 tokens == the no-bail run's prefix. 119 JSONL lines, every file 0600 and
  directory 0700.

## 13. Side-slot routing (M7 finish, the lead's ruling of 11-05; impl:m7-finish)

`K3Session` hands every request to `PromptCache.prepare(ids, marks, side='auto', max_new=)` (and `plan(ids, side='auto',
max_new=)` first; the worker and `run_generation` pass the request's max_new to a session with `wants_max_new`).
`PromptCache.route` sends a request to the **side slot** when all of these hold, else to the main slot:
- it does not match the main line: it shares fewer tokens with the main line than the main line's root mark (the end of its
  system + tools block, from the last main request's template marks, else its `root` checkpoint), or than
  `promptcache.SIDE_MATCH_MIN` = 64 when no root is known (measured: unrelated chats share 2 .. 11 tokens under the template),
  and the main line's live state is not a prefix of it. A request that matches always goes to main;
- it fits: prompt + max_new + T <= side_rows (8192). A side pass past side_rows would drop the main slot, so a request without
  max_tokens (the server then allows the whole context) always goes to main;
- the main line holds more positions than the request's prompt (the deepest of its live state and its checkpoints): there is a
  conversation to protect. This condition is the implementer's addition to the ruling (a conversation whose first turn follows
  a short request would otherwise run in the side slot, which takes no checkpoints and cannot be parked, until it outgrows it);
- no conversation parked on the disk tier continues into it (`parked`: what `prepare` would unpark; a side call would start
  cold), e.g. a conversation's next turn after a long unrelated request displaced and auto-parked it. Also the implementer's
  addition.

The decision is counted in `caminus:cache_route_total{route,reason}` and `cache.route` (with the last decision's numbers), the
flight recorder's `cache.slot`, `PrefillStats.slot` and `K3Session.last_route`; `caminus:cache_side_active` / `cache.route.
side_active` say whether a side call holds the engine now. `caminus serve --no-side-route` sends every request to main.
A side call's failure loses only the side call: a pass that fails past its retries re-enters it (`PromptCache.recover(at,
ids)`: the side call ended with the main line intact, its prompt processed again), an abort whose device state is unknown
ends it (`drop_live()` during a side call: the main line back, its live state kept). The idle and shutdown parks end an
active side call first (`PromptCache.end_side()`), since `park()` is refused during one.

**Measured** (impl:m7-finish, 11-05, guarded, both cards, max_ctx 33024; tests/test_serve_k3.py `side` in the one-process run of
every stage): a tool conversation (20 synthetic tools, root 4,625 tokens) whose main requests T1 (cold, `no_main`), T2 (chat,
`match`, live 4,688), T3 (a token continuation, `match`, live 5,158), L (9,001 tokens, `no_fit`: the conversation auto-parked)
and T4 (`parked`, from disk, 5,365 cached) equal the same requests with no side calls, first logits and tokens bitwise; six side
calls (title, summary, raw tokens, next-speaker, title again, raw again) all `unrelated`, from cold, each equal to a cold run of
it alone; a side call whose prompt phase failed (HTTP 500) and one whose first decode pass bailed on every attempt (recovered,
HTTP 200, == cold) left the main line's log, state and VRAM checkpoints unchanged; main_lost 0, no reset; VRAM (sysfs) 23,138 /
19,171 MiB before and after; the idle park after a side call ended it and parked the conversation. Over the whole run: main
`match` 70, `no_main` 55, `no_fit` 2, `main_shorter` 3, `parked` 2, `off` 16 (abortx, routing off), side `unrelated` 21.
