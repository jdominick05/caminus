# Glossary

The project's own terms, as the code, the tests and the docs use them. Prefer these words over synonyms.

## Serving state

- **Live state.** What the engine's live slot holds right now: the **log** (the token ids of the processed positions), the
  state **(i0, commit)** the next pass starts from, and the engine counters the prompt cache vouched for. One owner: the
  prompt cache (`caminus.cache.livestate.LiveState`, held as `PromptCache.live`). The session keeps no copy; it reads the log
  and the state through the cache and reports every pass it accepts (`append`). While the state is live, `len(log) == i0`.
  After the live state is *dropped* the log stays (checkpoints are matched against it) and the state is (0, 0).
- **(i0, commit).** Positions below `i0` have been processed. The next pass starts at `i0` and first applies `commit` rows
  of the previous pass (see *lazy commit*). `commit` is at most T and at most i0.
- **Lazy commit.** A T-token verify pass writes its recurrent-state updates as pending records. The next pass applies the
  first `commit` of them (the accepted ones) before running, so a rejected draft costs nothing to undo.
- **Engine counters.** Every engine and test stand-in has them from construction: `token` (grows at every pass launch, a
  retry too), `seq` (grows at every `reset()`), `staged` (the position staged for the next launch). They tell the cache
  whether the device ran passes or was reset since it last vouched for the live state.
- **Vouched / known.** The cache *vouches* for the live state when an operation the session asked for completes
  (`LiveState.vouch`). The state is *not known* while the cache runs something that can leave the device undefined if it
  fails part way (the session's prompt phase, a bail recovery); a failure report then drops it.
- **Failure report.** After a host-side exception the session drains what it had in flight and reports once
  (`PromptCache.settle(returned, drained)`). The cache decides the outcome: `reset` (poisoned), `dropped` (reset since,
  not known, a failed drain), `pass` / `commit0` (a pass ran and was not accepted: its position, commit 0), `kept`.
- **Bail.** A pass that failed after the engine's bitwise retries ("a wait gave up"). **Bail recovery** restores the
  request's prompt-end checkpoint and replays the processed tokens with K3 passes; the client sees a delay, not an error.

## The prompt cache

- **Checkpoint.** A self-contained copy of the recurrent state at an (i0, commit), with the KV / indexer tail rows a later
  restart rewrites. Kinds: root (pinned), prompt end, user turn, every 8192 fresh tokens, end, park.
- **Prompt-end state.** The state right before a request's last prompt pass (`spec.prompt_end`): the checkpoint an exact
  hit and a bail recovery start from.
- **Tiers.** The VRAM tier (preallocated checkpoint slots), the disk tier (parked entries), the live slot itself.
- **Park / unpark.** Write the live conversation to the disk tier (idle, shutdown, or before a switch that would lose it),
  and load it back when a request continues it and the load is cheaper than re-prefilling.
- **Side slot / side call.** A short request unrelated to the conversation (a harness's title / summary call) runs in the
  side slot (at most 8192 positions). The main line's state is saved and comes back unchanged; a failure inside a side call
  loses only the side call.
- **Engine key.** What a cache entry must match to be used: the model, the prefill numerics, the TP layout, the drafter, the
  code objects and the engine-side Python sources (`build_hash`).

## The engine

- **K3.** The persistent decode megakernel engine: one launch per pass, every op of every layer on one tape.
- **Tape.** The list of segments one K3 launch runs.
- **Segment.** A run of tasks the persistent kernel's workgroups claim; segments wait on each other's done counters.
- **Module.** One compiled build of the K3 engine with its own op / format whitelist (`k3_gguf_gsq`, `k3_gguf_q35h`, ...).
- **Hand-off.** The batched prefill's end: the state left so the next K3 pass continues it (commit 0).
- **T.** Tokens per pass (4 on the Coder: one fed token + three drafts; 1 on the one-card engines).
- **Drafter.** Proposes the drafts a verify pass checks: DFlash (inside K3) or MTP. A **filler** draft is a placeholder when
  no real draft is available; it is never counted as a draft.

## Gates

- **Bitwise.** A change must leave logits, router ids and replicas identical to the previous reference.
- **D-281.** The bar for a change that is not bitwise by design: mean KLD against llama.cpp's one-token decode no worse than
  llama.cpp's own batch-vs-decode floor, top-1 within one point.
- **G2b / G3 / G4.** The batched prefill's gates: chunk geometry bitwise (G2b), a restored state continuing bitwise (G3),
  prefill-then-decode against llama.cpp's own floor (G4a against K3's one-token decode, G4b against llama.cpp's).
