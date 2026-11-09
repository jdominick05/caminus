# The prompt cache API (M6 -> M7 contract)

`caminus.cache.prompt.PromptCache` is the one object the server's engine worker uses for every prompt. It owns the live
slot of a `FlashNextTP` engine, its `Prefill` and its DFlash drafter, the VRAM checkpoint tier and the disk tier. Design,
policy and gates: docs/specs/spec-prefill-cache.md section 2 and the module docstring. This page is the contract M7 codes
against. The lead reviews it before M7 starts; changes after that need the lead.

## Ownership and threads

- **One owner.** The server's engine worker thread is the only caller. Every call that touches the engine runs on that
  thread.
- **The background park** (`park(background=True)`) is the one other thread. Every engine-touching call waits for it first
  (`fence()`).
- **No outside engine work.** The engine must not run passes or prefills the cache is not told about.
  - Passes the cache did not see raise `CacheStateError` on the next call. Report each accepted pass with `append()`
    (the server's session), or a generation's passes in bulk with `extend()` / `commit()`.
- **The live state has one owner: the cache** (`PromptCache.live`, caminus.cache.livestate: the token log, (i0, commit) and
  the engine counters it vouched for). The session keeps no copy and reads it through the cache (`live_tokens()`, `st`).
  - An engine `reset()` from outside drops the slot (counted in `stats['external_reset']`).
- **One process per cache directory.** An `flock` on `<dir>/.lock` refuses a second owner.

## Construction

```python
pc = PromptCache(m, pf, draft=None, cache_dir=None, cap_gb=30, retention_days=7, min_free_gb=10, disk=True,
                 n_slots=None, max_slots=64, side_rows=98304, auto_park_min=4096, every=8192, spill_min_gap=None)
```

- **Build order.** Build the cache last, after the engine (`FlashNextTP(..., vram_drafters=, vram_prefill=)`), the drafter
  (`DFlashK3Drafter`, or `None`) and the `Prefill`. The tier is sized from what those left free.
- **Tier size.** `n_slots=None` sizes the VRAM tier from each card's measured headroom below the guard's own kill line
  (`capacity.guard_line_mib`: tools/guard.sh's 23,800 MiB on the XTX, 19,850 on the XT; not 0.97 x total, which is 23 MiB
  higher on the XTX), with the use read as the guard reads it (`capacity.used_mib`: sysfs; hipMemGetInfo shows 163 / 353
  MiB less). The headroom is net of `capacity.CACHE_RESERVE_MIB` (the XTX keeps its 0.5 GB reserve) and `CACHE_WORK_MIB`;
  the slot count is the smaller of the two cards' fits, at most `max_slots`. Every other capacity check reads the same
  sysfs use against the same lines (`capacity.caps_mib`).
- **Counting.** The tier is counted by `capacity.cache_tier_mib` and checked by `capacity.check` before allocation:
  `n_slots` checkpoint slots, the io slot and, with `side_rows`, the side slot's checkpoint. The side slot's row area is
  host memory since 0.2.2, not in the tier (`pc.side_host_bytes`). `side_rows` is clamped to the engine's `max_ctx`
  (final round 4), and halved down to 8,192 when the area would not fit in MemAvailable with the host margin.
  `capacity.budget(cache=pc.tier)` adds it to a projection.
- **Refusals at construction.**
  - `capacity.CapacityError`: fewer than 2 slots fit.
  - `CacheError`: a drafter other than DFlash K3 (MTP needs per-pass rows a batched prefill does not give), or a prefill
    of another engine.
  - `CacheError`: a DFlash K3 drafter whose target (`df.tgt`) is not this engine (`extend()` reads that target's launch epoch
    to tell a generation through the drafter from one that bypassed it).
  - `CacheError`: the cache directory is held by another process.
- **Eviction.** When the VRAM tier is full, the root and the newest checkpoint stay and the checkpoint whose removal leaves
  the smallest gap goes. It is spilled to disk when at least `spill_min_gap` positions (default `every` = 8192) lie
  between it and the checkpoint below it. A closer one is dropped: a spill writes ~160 MB in the request's path. A spill
  that fails (an I/O error) drops the victim instead (`stats['spill_failed']`).
- **A full tier of pinned checkpoints** (every VRAM checkpoint a root or the newest, e.g. a tiny tier with many root
  marks): the new checkpoint is skipped and counted (`stats['ckpt_skipped']`, the reason in `pc.last_skip`), never
  raised. A checkpoint inside a batched run is taken in `Prefill.run`'s `at_stop`, where a raise would poison the engine.
- **The cache key** (`pc.key`, `pc.key_sha`) is computed once at construction. The prefill's numerics (`pf.S.moe`,
  `.q6k`, the glue's WMMA output and router mode) must not change afterwards: `prepare()` refuses with `CacheError` if
  they do.

## Calls

**`plan(ids, side=False, max_new=None) -> {'cached': int, 'source': str}`.** How many leading tokens `prepare(ids)` would
reuse, and from where. No side effect, except that it may read disk entries' token lists. Maps onto `EngineSession.plan`.
During a side call it answers for a main request (the main line's log, state and checkpoints: what `prepare(ids)` reuses after
ending the side call). `side=True`: what `prepare(ids, side=True)` reuses (the side log when it continues, else the main
checkpoint the side call would start from, else cold); `side='auto'`: as `route(ids, max_new)` decides (M7 finish). With
`side` True or `'auto'` the dict also holds `side` (bool) and `route` (the decision, or None).
`source` is one of:
- `'live'`: the live state continues;
- `'vram'`: a VRAM checkpoint;
- `'spill'`: a checkpoint of the live line held on disk (spilled from VRAM, kept from the park entry the line was
  unparked from, or, since final round 1, the live state a park of the line wrote: kind `park`); the name covers all three,
  so a `source_spill` count or a flight line's `spill` source is not only evictions;
- `'disk'`: a parked entry unparked;
- `'cold'`.

**`prepare(ids, marks=None, side=False, cancel=None, max_new=None) -> dict`.** The engine is made to hold `ids` up to its
prompt end. `side`: False (a main request), True (a side call) or `'auto'` (side-slot routing: `route(ids, max_new)` decides
after step 1, once an engine reset outside the cache has been taken in; see The side slot).

What it does:
1. Validates the whole request up front. `spec.check_capacity` and `Prefill.check_tokens` raise
   `ContextBoundsError` / `PrefillRefused`, and nothing changes.
2. Ends an active side call.
3. Picks the best state:
   - the live state, when the live log is a prefix of `ids`;
   - else the deepest VRAM or spilled checkpoint at or below the longest common prefix;
   - else a parked disk entry, when its match beats the local one by more than its load time;
   - else cold.
4. Auto-parks a live slot that would lose at least `auto_park_min` tokens, before anything overwrites it.
5. Restores the chosen state, then processes the rest (batched or K3 passes, routed by T*) with the policy's checkpoints.

`marks`: `{'root': int, 'user': int}`, token positions of the end of the system + tools block and of the start of the last
user turn. The server gets them from `PromptBuilder.build(..., marks=True).marks`. They are optional: without them only the
prompt-end and 8K checkpoints are taken.

Where the checkpoints go:
- Inside a batched run, a checkpoint position splits the run. Measured on a 32K prefill: +106 ms per stop on the chunk grid
  (the 8K ones: the hand-off and the segment's DFlash inject), +243 ms for an unaligned one (plus a partial chunk); the cache's
  cold 32K request with root + 3 x 8K + prompt-end checkpoints took +4.6 % over a stopless run.
- Positions closer than T* (48) to each other merge.
- A last-turn start within `USER_NEAR` (1024) tokens of the processing start becomes a checkpoint of the start state
  instead of a split. A split would cost every such request a partial chunk and an inject (+243 ms measured); the start
  state costs an edit of the last message at most 1024 re-prefilled tokens.
- One within T* of the prompt end is skipped.

Returns:
- `st`: `{'i0', 'commit'}`, the engine state the generation loop continues from. Positions `< i0` are processed and
  `ids[i0:]` are given tokens (1 .. T of them: the prompt's last pass). `i0 == len(ids)` with `commit >= 1` is an exact
  hit: `spec.loop` / `longctx.generate` rerun position n - 1 (`spec.resume_point`).
- `cached`: positions reused.
- `source`.
- `processed`.
- `limit`: passes must write below it. That is `max_ctx`, or `side_rows` in a side call.
- `ms`: `{'park', 'restore', 'process', 'total'}`.
- `stops`: the checkpoints taken, as `[(position, kind)]`.
- `side`.
- `route`: `route()`'s decision with `side='auto'`, else None.

**`route(ids, max_new=None, min_match=SIDE_MATCH_MIN) -> dict`** (M7 finish). Where `prepare(ids, side='auto', max_new=)`
sends the request; no side effect (it may read disk entries' token lists). Returns `route` (`'main'` | `'side'`), `reason`,
`shared` (tokens shared with the main line), `root` (the main line's root mark or None), `need` (prompt + max_new + T),
`main_n` (the deepest state the main line holds) and `side_rows`. The rule is in The side slot.

**Generation (the caller's).** Run the passes from `st` on `ids`: `longctx.generate(m, st, ids, n_new, draft)`. With the
cache's drafter built `DFlashK3Drafter(df)` (device_accept, the default) it runs the device-accept loop from that state
(`DFlashK3Drafter.resume`: the accept on the device, one host round trip per pass); with `device_accept=False`, or through
`spec.loop`, the generic loop. Both give the same tokens, the same (i0, commit) and the same engine state, bitwise
(tests/cache/test_prompt_cache.py resume: from a VRAM checkpoint, an unpark, the prompt-end checkpoint, an exact hit, past 2051,
every length mod 4; == plain greedy in tokens and in the logits of K3-routed positions after it). Measured 11-05 after a
VRAM restore + a 400-token append at depth ~4,000, 128 tokens: device accept 5.59 ms/token, the generic loop 5.62
(interleaved medians of 3; within 0.05 ms: not a speed claim); 200-token prompts at max_ctx 2048 (test_spec_policy gpu):
18.94 vs 19.15 ms per pass (prose). Bail recovery inside a pass is the engine's (`FlashNextTP.verify`'s bitwise retry).

**Generation that bypasses the drafter** (`longctx.generate(..., draft=None)`, `longctx.teacher_forced`, or any pass loop
without it, reported with `extend()` / `commit()`): the drafter has no valid row of those positions.
- **Detection, at `extend()`.** The drafter records the target's launch epoch each time it takes in a pass's features
  (`DFlashK3.fed`: at an inject, or when a step's inject completes). If a target pass ran after that, the reported passes
  bypassed it (`DFlashK3.followed()` is false). This also catches a DFlash generation followed by a draft=None one inside a
  single `extend()`; report each generation with its own `extend()` for the smallest give-up.
- **The give-up, host only, before any checkpoint or park can save the state.** The positions from the cache's previous state
  (the last state the drafter was in step with) to the new one are given up (`DFlashK3.give_up`: their slot tags
  invalidated, the range recorded in `gaps`; `stats['inject_gaps']` / `['inject_gap_positions']`).
- **A pending inject whose features the bypass rewrote is dropped, never launched** (`DFlashK3.drop_stale`,
  `df.stats['stale_pending']`). This is the generic loop's last inject. The give-up then starts at its position, so after a
  generic-loop generation a bypass gives up up to T more positions (the last pass's commit) than after a device-accept one
  (the lead's re-ratified bound, 11-05; this page said T - 1 before).
- **Drafting over the gap.** A draft whose window (2,048 positions) reads a given-up position is filler, never wrong
  (`DFlashK3.window_gap`, the drafter's `stats['gap_filler']`). A launch reading one would be refused by the slot tags before
  staging, and the device-accept loop checks its step on the host before the target's pass launches (`DFlashK3.step_check`).
  Drafting resumes once the window has passed the gap. Checkpoints, spills and parks carry the gaps with the window.
- **History.** Before the M6 fix-ups, the next request's first inject was refused after the target pass ran ('positions ..
  were never injected'), then `CacheStateError`. The first fix-up gave up only the positions from the drafter's `n_inj` at the
  next request. A bypass shorter than about T + 1 tokens ended below it and gave nothing up:
  - the device-accept loop's last draft block was then refused (`StaleRingError`) after the next request's target pass had
    launched;
  - the generic loop read rows of rejected drafts, and another pass's features from its pending inject.

  Fixed in fix:m6-fixups (tests/cache/test_prompt_cache.py cpu_bypass_short and the GPU stage `bypass`).

**`extend(toks, st)`.** Reports passes the caller ran from the last state the cache gave or recorded.
- `toks` is every token of the sequence and `st` the engine state after the passes (`longctx.generate`'s trimmed state).
- Nothing launches. Passes that bypassed the drafter are given up here (host only: see Generation that bypasses the drafter).
  Call it after each generation.
- The processed tokens must keep the live log as a prefix. Otherwise the slot is reset and `CacheStateError` raised.

**`append(toks, st)`.** Reports ONE pass the caller ran from the live state and accepted: `toks` are the tokens it adds
to the log (the pass's fed tokens and accepted drafts), `st` the state after it (`i0 + len(toks)`, commit). What `extend()`
checks for a continuation (the state continues the live log, commit in range, a side pass within `side_rows`, the drafter
bookkeeping), host only and O(len(toks)), without `extend()`'s LCP over the whole log; nothing launches. A state that does
not continue the live state resets the slot and raises `CacheStateError`, as in `extend()`. A side pass past `side_rows` is
caught at the pass that reaches it (before: at the request's end, when the session reported the whole generation).

**`live_tokens()`.** The live tokens (positions below the live state's i0): the cache's own list while the state is live,
read it and never modify it.

**`commit(toks, st, checkpoint=False)`.** Completion end: `extend()`. With `checkpoint=True`, it also takes a checkpoint
of the live state (kind `'end'`). The token log afterwards is prompt + reply[:-1], with the last sampled token not fed,
as in `EngineSession`.

**`park(background=False) -> name | None`.** Writes the live slot to the disk tier: tokens, KV + indexer rows `[0, i0)`,
every checkpoint, and the live state as the deepest checkpoint.
- `background=True` returns at once. The next engine-touching call waits for the write, and an error lands in
  `pc.park_error`.
- An unchanged slot is not written twice.
- It is refused (returns `None`, reason in `pc.park_error` / `pc.store.events`) when the entry would not fit the cap or
  would leave less than `min_free_gb` free.
- An older park whose tokens are a prefix of the new one is superseded and removed, but only when the new entry holds the
  older one's deepest state (final round 1, the lead's ruling of 11-06: a supersede must not lose a deeper restore point).
  A park's live state joins the line as a checkpoint backed by its entry (kind `park`, as an unpark's checkpoints do), so the
  line's dropping rule governs it and the next park of the line copies it into its own entry; the line's checkpoints backed
  by an older park then read from the new one. When the line restarted below it since (its checkpoint dropped), the new
  entry lacks it and the older entry is kept (`stats['superseded']` / `['supersede_kept']`). Before, an idle park at A,
  more turns and a shutdown park at B removed A's entry while B's held no state at A: a branch at A re-prefilled from a
  shallower checkpoint (G4-bounded, not bitwise). tests/cache/test_prompt_cache.py cpu (`cpu_supersede`), tests/e2e/test_serve_k3.py
  supersede.
- Call it when a conversation goes idle (after `commit`). `prepare` also parks automatically before a switch, in the
  request's path.

**`restore(ids) -> {'st', 'cached', 'source'}`.** Explicit unpark: loads whatever reuses the most of `ids` (a disk entry,
a checkpoint, the live state) with no processing. `prepare` does this by itself; `restore` exists for a server that
wants to warm a conversation ahead of its request. An active side call is ended first (the main line back), as `prepare`
does; the drafter's pending inject runs and any gap below the state is given up, so the caller's generation can start.

**`purge(all=True, older_than_s=None) -> [names]`.** Deletes disk entries: all of them, or those last used more than
`older_than_s` ago. The live line's checkpoints in them are dropped. The server's purge command calls it.

**`end_side() -> ms | None`** (M7 finish). Ends an active side call now: the main line's rows and state back, what the
next main request does first. The server's idle and shutdown parks call it, since `park()` is refused during a side call. None
without one.

**`fence()`, `reset()`, `close()`.**
- `fence()` waits for a background park.
- `reset()` drops the live slot and every VRAM checkpoint; the disk tier is kept. Use it after an `EngineError`, a
  poisoned engine, or a `CacheStateError`.
- `close()` drops every checkpoint (the main line's too during a side call) and every spill file of this process
  (`*-spill-<pid>-*.ckv`, tracked or not), then releases the directory.

## State guarantees (gate G3)

- **(a)** A checkpoint restored from disk equals the same checkpoint restored from VRAM, which equals the state when it
  was taken, bitwise. That covers the recurrent states, the KV / indexer rows below it, its pending rows, and the DFlash
  window and slot tags.
- **(b)** Resuming after a restore equals resuming without it, bitwise: state, logits and tokens. This holds for every
  prefix, decode-built spans included.
- **(c)** A restored prefix equals a cold batched prefill of the same tokens **only when the prefix was built by the
  batched prefill**.
  - Spans built by decode passes (the model's replies, short appends routed to K3 passes) are not bitwise a cold prefill:
    the decode matvec != MMQ. They are bounded by the G4b-style comparison instead.
  - The cache never re-prefills a decoded span to make them equal.
- **Corruption.** A corrupt, truncated or mismatched disk entry is refused (header, key and layout checks before
  anything is touched; a crc32 per 16 MiB piece while streaming). The request is then served cold. A checksum failure
  mid-load resets the slot rather than leaving it half restored. `pc.refusal` holds the reason and
  `stats['refused']` counts it.

## The side slot

- **Use.** `prepare(ids, side=True)` is for side calls (titles, summaries, a harness's background memory jobs) of at most
  `side_rows - T` positions. `side_rows` is 98,304 since 0.2.2 (8,192 before): Qwen Code 0.26.9 sends two background
  memory jobs (29,555 and 35,582 tokens, `max_tokens` 32,000) on every user message, and at 8,192 they took the main slot.
- **What it saves.**
  - The main live state goes into the side checkpoint slot.
  - The main rows from the side call's start (the deepest main checkpoint the side request shares, else 0) are copied to
    the side area, which is host memory laid out by position: first up to the prompt + T + `SIDE_CHUNK` (4,096) rows,
    then more before each pass writes there (`append()` covers the next pass's T rows, `SIDE_CHUNK` at a time), never
    past `min(main i0, side_rows)`. A side call copies the rows it overwrites, not the whole area; the area's pages are
    committed only where rows were copied.
- **During the call.** The call runs from that checkpoint, or from a cold start that keeps the rows in place, and takes
  no checkpoints.
- **Generation limit.** Generation must stay below `limit = side_rows`. A side pass past it drops the main slot
  (`CacheStateError` from `extend` / `commit`). Passes are reported one at a time (`append()`, as the server does): a
  multi-pass `extend()` / `commit()` whose passes wrote past the copied rows also drops the main slot.
- **Return to main.** The next `prepare(..., side=False)` puts the main rows and state back first. The main slot's next
  request is then served as if no side call had happened, bitwise.
- **Several side calls.** Consecutive side calls continue the side log when it is a prefix. Otherwise the main slot is
  restored and the side call re-entered.
- `park()` is refused while a side call is active (`end_side()` first).
- **Routing (M7 finish, the lead's ruling of 11-05).** `prepare(ids, side='auto', max_new=)` (`route()`) sends a request to the
  side slot when all of these hold, else to the main slot:
  - it does not match the main line: it shares fewer tokens with the main line than the main line's root mark (the last main
    request's `marks['root']`, kept while later main requests keep that prefix; else the line's `root` checkpoint), or than
    `SIDE_MATCH_MIN` = 64 when no root is known (measured 11-05 under the server's template: unrelated chats share 2 .. 11
    tokens), and the main line's live state is not a prefix of it. A request that matches always goes to main (`match`);
  - it fits: prompt + max_new + T <= side_rows, so its whole generation stays below side_rows (`no_fit` otherwise; a request
    without max_new, e.g. a chat without max_tokens, which the server lets fill the context, is never a side call);
  - the main line holds more positions than the request's prompt (`main_shorter` otherwise: there is no conversation to
    protect; without this a conversation whose first turn follows a short request would live in the side slot, which takes no
    checkpoints and cannot be parked, until it outgrew it). The implementer's addition to the ruling;
  - no conversation parked on the disk tier continues into it (`parked` otherwise: the main slot unparks it, a side call would
    start cold; e.g. a conversation's next turn after a long unrelated request displaced and auto-parked it). The implementer's
    addition;
  - and the main line exists (`no_main`: empty, reset outside the cache, or a poisoned engine) and the cache has a side slot
    (`off`).
  Counted in `stats['route_side']` / `['route_main']`.
- **A failure inside a side call loses only the side call** (M7 finish). `_drop_live()` during a side call ends it with the
  main line back intact, its live state kept (a side pass writes below side_rows only, and the main rows it can overwrite are in
  the side area; the drafter's pending inject of the failed call is dropped, never launched): `stats['side_dropped']`.
  `_recover_at(at, ids)` during a side call ends it the same way and enters it again for `ids` (the side request's prompt), which
  processes the prompt again from the same start to the same state `at`; without `ids`, None (a side call takes no
  checkpoints). Before, `drop_live()` (now `_drop_live`) also dropped the main line's live state and `recover()` returned None (the server then
  reset, dropping the VRAM tier).

## Errors

| Error | When | State after |
|---|---|---|
| `ContextBoundsError`, `PrefillRefused` | a request past max_ctx, a token out of range (`prepare`, up front) | unchanged |
| `CacheStateError` | passes the cache was not told about; `extend` that does not continue the log; a side pass past `side_rows`; a poisoned engine | slot reset (or unchanged for unreported passes: call `extend` / `commit` / `reset`) |
| `CacheRefused` | a disk entry refused | internal only: `prepare` / `restore` serve cold and count it |
| `CacheError` | changed numerics, another owner, park during a side call, no disk tier; at construction: a drafter or prefill of another engine | unchanged |
| (none) | a checkpoint with no VRAM slot to take (every one pinned or the newest); a spill that fails | the checkpoint skipped / the victim dropped, counted (`ckpt_skipped`, `spill_failed`) |
| `capacity.CapacityError` | the tier does not fit (construction) | nothing allocated |
| the engine's own (`RuntimeError` after bail retries, `ContextBoundsError` from the device bound guard) | during processing | the engine may be poisoned: `reset()` |

## Metrics

M7's /metrics and the llm-top cache panel read these:

- `pc.stats` (a Counter):
  - `requests`, `prompt_tokens`, `reused`, `source_<src>`;
  - `checkpoints`, `checkpoint_ms`, `evicted`, `spilled`, `spill_ms`, `spill_failed`, `ckpt_skipped`;
  - `parked`, `unparked`, `park_failed`, `refused`;
  - `side_requests`, `side_ends`, `main_lost`, `external_reset`; `route_side`, `route_main` (prepare's routing decisions),
    `side_dropped` (side calls dropped by `_drop_live()`);
  - `inject_gaps`, `inject_gap_positions` (generations that bypassed the drafter); the drafter's own `df.stats['gap_filler']`
    and `df.stats['stale_pending']` (pending injects dropped).
- `prepare()`'s `ms` breakdown and `cached` per request.
- `pc.n_slots`, `pc.tier_mib`, `pc.headroom`, and `len([c for c in pc.ckpts if c.slot is not None])` (VRAM fill).
- `pc.store.total_bytes()` against `pc.store.cap`, and `pc.store.events` (refusals, LRU / retention evictions, reaps).
- `pc.last_park`: name, bytes, ms.

## M7: the server's use (caminus.serve.k3_session.K3Session; the lead's pre-M7 additions)

(The device-accept loop from a cached state, listed here before the M6 fix-ups, is provided: `DFlashK3Drafter.resume`,
reached through `longctx.generate`; see Generation. The server runs its passes one at a time through
`DFlashK3Drafter.device_step`, the same pass `_loop` runs.)

- **Mapping onto `serve/engine.py`'s `EngineSession`.**
  - `plan` -> `plan`.
  - `prefill(ids, cancel, marks)` -> `prepare(ids, marks, cancel=cancel)`, then the prompt's last pass, which returns the
    first logits: `longctx.first_logits(m, st, ids, draft)`, then `extend(ids, st2)`. The marks come from
    `PromptBuilder.build(..., marks=True)` through the server's Job.
  - `verify` / `accept` / `sample_pass` -> the engine's passes; `accept(n)` reports the pass with `append` (live state
    step 2, 11-07: before, the session kept its own log and state and the cache heard only at `end`).
  - `end` -> `commit(toks, st)` with the cache's own live state (no 'end' checkpoint: the next request's prompt end takes one).
  - `reset` -> `reset`.
  - `prefill` -> `prompt(ids, marks, side, cancel, max_new, max_recoveries, hooks)` (live state step 3, 11-07: the cache
    runs the prompt phase itself: `prepare`, its bail retry, the prompt's last pass and its recovery; before, the session did,
    with its own copy of the state). A pass of the generation that fails after the engine's retries ->
    `recover_pass(err, toks, max_recoveries, hooks)`. `hooks` is the session's: `arm(on)` around each pass attempt (the test
    bail hook) and `event(kind, side)` after a recovery (its counters and stats).
  - `abort` (a host-side exception, between calls or inside one) -> the session drains what it had in flight
    (`launched()` says whether a target pass launched since the cache last vouched for the state), then reports once:
    `settle(returned, drained)`, and the CACHE decides what the device holds (returning 'reset' / 'dropped' /
    'dropped_side' / 'pass' / 'commit0' / 'kept'; the kept state committed, the live state and the VRAM tier kept):
    - a pass that returned, or was launched and did not return (the target's launch epoch `FlashNextTP.token` moved since the
      session's state was set), and was not accepted: dropped, the next pass at its position commits 0 (its launch applied
      the commit; reporting it again applies it twice: fix:m7-wiring, the verifier's KLD 0.197 / 0.067 on the next request);
    - a refusal before the launch: the state with its commit;
    - not known (a failure while the prompt is processed or during a bail recovery, a failed drain): the live state dropped
      (`_drop_live`), every checkpoint kept; a refusal before `prepare` touched anything keeps the previous state;
    - the engine was reset since the session's last completed operation (`FlashNextTP.seq` moved, e.g. inside `end`: the
      cache resets its slot when it refuses a state): the live state dropped (final round 4; before, `commit(toks, st)` told
      the cache the device held a state the reset had cleared, and the next request continued it).
    `reset()` only when the engine is poisoned.
  - A client disconnect mid-generation stops at the next pass boundary and ends through `end` -> `commit`: the accepted
    tokens are the live state, every checkpoint kept.
- **Cancellation between prefill chunks** (`prepare(..., cancel=)`, a callable or an object with `is_set()`): checked at every
  chunk boundary of the batched prefill (`Prefill.run(cancel=)`) and between K3 passes. Processing stops there with the
  hand-off of a run ending at that position; the live state is the processed prefix (commit 0 after a batched run), the
  result carries `cancelled=True`, no prompt-end checkpoint is taken, `stats['cancelled']` counts it, and the next request
  continues it (source `live`). The target state is bitwise a run that stopped there (G2b: a chunk boundary). The drafter's
  inject covers the cut segment's exported features below the cut (`[max(segment start, feat_base), cut)`: the export window
  is set from the segment's end); positions below that are given up (filler drafts while a window reads them, never wrong).
- **`_recover_at((i0, commit), ids=None)`** (private since live state step 3: `prompt()` / `recover_pass()` call it; bail recovery): after a pass that failed past the engine's retries the live state is
  undefined. The live state is dropped, every checkpoint kept (no pass wrote a row below the failed pass's first position),
  and the checkpoint of exactly that state (the request's prompt end: `prepare()`'s `st`) is restored; the caller replays its
  processed tokens with K3 passes and runs the failed pass again (bitwise the passes it lost). Returns `dict(st, cached,
  source)`, or `None` without that checkpoint (skipped at its creation, a poisoned engine): the caller resets.
  `stats['recovered']` / `['recover_failed']`. In a side call (no checkpoints) with `ids` = the side request's prompt: the side
  call is ended with the main line intact and entered again for `ids`, processed to the same state (The side slot); without
  `ids`, None.
- **`_drop_live()`** (private since live state step 3: `prompt()` / `settle()` call it): the live state is forgotten (a pass failed while the prompt was processed, or `abort` after a failure
  whose device state the session does not know), every checkpoint kept; the next `prepare` restores one (never `live`) or
  starts cold. During a side call only the side call's state is lost: it is ended, the main line back with its live state
  (returns `'side'`, else `'live'`; The side slot). An engine reset outside the cache since its last mark (the sequence
  epoch moved: the KV rows went too) voids every checkpoint: the slot is reset (cold, `stats['external_reset']`), as the
  next `prepare` would have done had `_drop_live` not re-marked the epochs.
- **Conversation identity.** None is needed: matching is by token prefix. Normalization (pinned billing headers, reasoning
  kept or stripped consistently) decides the hit rate (G6). The server parks the live conversation in the background after
  `idle_s` seconds without a request (`K3Session.idle`, the worker's idle timer; 8 hours by default in `caminus serve`, the
  user's ruling of 11-05), and only when the live log gained at least `auto_park_min` positions since the session's last park
  of the line (`K3Session.park_gain`). A clean shutdown (SIGTERM / SIGINT / a systemd stop) drains the engine thread and parks
  the live conversation when it changed since its last park (`K3Session.shutdown_park`, from the caller's thread once the
  worker exited; time-bounded by `caminus serve --park-s`), so a restart within the idle time loses nothing. A switch to
  another conversation before then still parks in the request's path (`prepare`'s auto-park); a slot the idle park already
  wrote is not written again (`park()` of an unchanged slot), so the switch then costs no park.
- **Side-slot routing** (M7 finish, the lead's ruling of 11-05): `K3Session` calls `plan(ids, side='auto', max_new=)` and
  `prepare(ids, marks, side='auto', cancel=, max_new=)` with the request's max_new (the worker and `run_generation` pass it to every
  session: the session contract, `serve/engine.py`), so harness title / summary / next-speaker calls run in the side slot and never displace the
  conversation; the next main request is served as if they had not happened, bitwise (tests/e2e/test_serve_k3.py side). The rule is
  in The side slot. Bail recovery and abort inside a side call use `_recover_at(at, ids)` / `_drop_live()` (the main line kept). The
  idle and shutdown parks call `end_side()` first. The decision: `K3Session.last_route`, `PrefillStats.slot`, the metrics'
  `caminus:cache_route_total{route,reason}` and `/caminus/status` `cache.route` (spec-metrics.md section 13).
  `caminus serve --no-side-route` (`K3Session(side_route=False)`) sends every request to main. A side call takes no
  checkpoints, so a flight-recorder bundle of one has no start state: one that started cold (the usual case) replays bitwise
  from cold.

## M7 ops: one checkpoint out and in (the flight recorder's replay bundles)

- **`export_checkpoint(at, root, cap_gb=, min_free_gb=) -> {name, bytes, ms} | None`.** The live line's checkpoint at
  `at = (pos, commit)` with the KV + indexer rows below it, written as a one-checkpoint 'park' entry (the same format, writer and
  checks as a park) into a DiskStore of its own directory (a bundle's `state/`; the store shares this cache's staging buffers).
  Host work and device-to-host copies only: no pass and no drafter launch. A pending DFlash inject stays pending and the launch
  epochs are unchanged, so the line continues as if nothing was exported (fix:m7: a flush of the pending inject here, without
  re-marking the epochs, made the conversation's next `prepare()` and `park()` raise CacheStateError). It works after a pass failed its retries (the rows below the failed pass's
  first position are intact, as `recover()` relies on). `None` with `pc.export_error` when the checkpoint is not in the live
  line, a side call is active, the engine is poisoned, or there is no room.
- **`load_entry(name, at=None) -> {st, cached, source='disk'}`.** Unparks the disk entry `name` of this cache's directory at its
  checkpoint `at` (default its deepest) explicitly, whatever the cost model says (`restore()` weighs a cold prefill against the
  load). The unpark's checks (key, layout, token hash, crc); `CacheRefused` leaves the file in place. tools/replay.py links a
  bundle's entry into a scratch cache directory and loads it with this.

## Disk format (version 1)

- **Header.** A 4 KiB preamble (magic `CAMINUS-KV-CACHE`, header length, header crc32, format), then a JSON header:
  - kind (`park` | `spill`), the full key and its sha256, the layout signature;
  - token count, rows count, the indexer's valid pools, the token hash;
  - the checkpoints' metadata;
  - the segment table (offset, bytes, a crc32 per 16 MiB piece).
- **Segments.** 4 KiB aligned: the tokens; per card and layer the KV rows, the IDXRAW rows and the IDXPOOL pools; per
  checkpoint and card its slot bytes.
- **Files.** File names start with the key hash. Files are 0600 and the directory 0700. Writes are atomic: a temp file in
  the same directory, fsync, rename, then fsync of the directory.
- **What is not in the key** because it changes no bit: chunk size, micro-chunks, HIP graphs.

## Measured (M6) and the lead's readings

- **G6 proxy** (tools/eval/cache_replay.py, CPU: a real PromptCache on a no-compute stand-in, the server's template; 366 recorded
  multi-turn agent episodes from the development machine's CTF benchmark runs (aggregate counts only; the episodes are not
  published), 10,267 requests, mean 26,098 / max 90,104 tokens): 97.85 % of follow-up request
  tokens reused with reasoning sent back, 95.75 % when the client drops it (every reasoning divergence served from the
  prompt-end checkpoint). **26** episodes end on invalid tool-call JSON (`refused_episodes`), not 61 as the M6 step reported
  (the verifier's correction); none of them loses a follow-up (the episode ends there).
- **Mid-draft stop vs plain** (the lead's reading, 11-05). A DFlash generation that stops inside a pass leaves the state at
  the last returned token (`longctx.generate` trims the accepted drafts past the total: i0 - extra, commit - extra), so its
  (i0, commit) and raw recurrent-state bytes can differ from a plain run's at the same position (different pass groupings
  and lazy-commit records). The gate holds for **K3-routed continuations**, which are bitwise (the next passes commit the
  same rows: the verifier's 30 teacher-forced positions after 17 / 18 / 20 new tokens bitwise equal; tests/cache/test_prompt_cache.py
  resume gates it with 8). A **batched continuation** restarts at i0 - commit, so when the spec and plain commits differ it
  re-prefills a different span: G4-bounded like any batched vs decode span, not bitwise (the verifier's 60-token batched
  continuation: logits equal after 17 new tokens, both at (3017, 1); not after 18, (3018, 2) vs (3018, 1)).
