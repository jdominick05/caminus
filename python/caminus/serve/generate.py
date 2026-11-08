"""One request's generation on an EngineSession (server spec section 2 'Stops', section 4; step S4).

TokenText: generated ids -> text, on the engine thread, so that a stop string ends the generation at the exact token:
  * EOS ids are never decoded (they end the generation in run_generation);
  * every other id is decoded, specials included (<think>, </think>, <tool_call> are text the output parser reads);
  * incomplete UTF-8 is held until it completes (tokenizer.StreamDecoder);
  * client stop strings are matched on the raw decoded text (reasoning included, as llama.cpp does). Text that could
    still become a stop string is held back: the longest suffix that is a proper prefix of some stop string, so at
    most max(len(stop)) - 1 characters. The match is the one that completes first (smallest end; then the earliest
    start), so the result does not depend on how the text arrives in pieces. On a match, the text before the stop is
    released, the rest is dropped, and the token whose text completed the match is the last one kept.

run_generation: prefill, then passes of [the last sampled token (+ forced tokens)] + drafts, sample-and-match on the
host (or the session's own sample_pass). It stops at an EOS id, after max_new tokens, on a stop string, or when
cancel is set (checked before every pass: a cancel lands at the next pass boundary). The session's lazy commit is
set by accept() to end exactly at the last emitted token's predecessor, so the state after a stop inside accepted
drafts equals a plain run stopped at the same token, and session.end() checkpoints it.

Forced tool calls (tool_choice required / named): `force` = (trigger_id, ids). When the model emits the trigger
(</think>) and no <tool_call> came before it, the forced ids are emitted right after it as if the model had written
them, and fed before the next sample. They count as generated tokens and are clipped to max_new like any other (the
generation then ends 'length'). With thinking off the caller appends them to the prompt instead.

Metrics: with `stats` (default: the session's), the loop reports the prompt (prefill_done), the first token
(first_token) and every decode pass (pass_done: from its drafter call to its emitted tokens' bookkeeping, the drafts
it accepted, the content class it started in), host timestamps only (docs/specs/spec-metrics.md)."""
import time
from dataclasses import dataclass, field

from .tokenizer import StreamDecoder


class StopMatcher:
    def __init__(self, stops):
        self.stops = [s for s in (stops or ()) if s]
        self.buf = ''
        self.matched = None         # the stop string that matched

    def _hold(self, s):
        """length of the longest suffix of s that is a proper prefix of a stop string"""
        best = 0
        for st in self.stops:
            for k in range(min(len(st) - 1, len(s)), best, -1):
                if s.endswith(st[:k]):
                    best = k
                    break
        return best

    def feed(self, text):
        """-> (released text, matched?)"""
        if self.matched is not None:
            return '', True
        if not self.stops:
            return text, False
        s = self.buf + text
        cut = None                  # the match that completes first (smallest end), then the longest
        for st in self.stops:
            p = s.find(st)
            if p != -1 and (cut is None or (p + len(st), p) < (cut[0] + len(cut[1]), cut[0])):
                cut = (p, st)
        if cut is not None:
            self.matched = cut[1]
            self.buf = ''
            return s[:cut[0]], True
        h = self._hold(s)
        self.buf = s[len(s) - h:] if h else ''
        return s[:len(s) - h], False

    def flush(self):
        out, self.buf = self.buf, ''
        return '' if self.matched is not None else out

    @property
    def held(self):
        return len(self.buf)


class TokenText:
    """ids -> released text with EOS skipping, UTF-8 holdback and stop-string holdback"""

    def __init__(self, tokenizer, stops=(), eos_ids=(), loop_guard=None):
        self.dec = StreamDecoder(tokenizer)
        self.stop = StopMatcher(stops)
        self.eos = frozenset(eos_ids)
        self.loop_guard = loop_guard
        self.loop_stopped = False

    def feed(self, ids):
        """-> (text, n_keep): retain through the token completing a stop string or the optional answer-loop guard"""
        out = []
        for j, t in enumerate(ids):
            if t in self.eos:
                continue
            txt = self.dec.feed(t)
            if not txt:
                continue
            rel, hit = self.stop.feed(txt)
            if rel:
                out.append(rel)
                if self.loop_guard is not None and self.loop_guard.feed(rel):
                    self.loop_stopped = True
                    return ''.join(out), j + 1
            if hit:
                return ''.join(out), j + 1
        return ''.join(out), None

    def finish(self):
        """end of the stream (not a stop match): the held bytes and characters"""
        if self.stop.matched is not None:
            return ''
        tail = self.dec.flush()
        rel, hit = self.stop.feed(tail) if tail else ('', False)
        out = rel + ('' if hit else self.stop.flush())
        if self.loop_guard is not None:
            self.loop_guard.feed(out)
            self.loop_stopped = self.loop_guard.finish()
        return out


@dataclass
class GenResult:
    finish: str = ''            # 'eos' | 'length' | 'stop' | 'cancel'
    gen_ids: list = field(default_factory=list)
    prompt_n: int = 0
    cached_n: int = 0
    prefill_ms: float = 0.0
    prefill_passes: int = 0
    restore: str = ''
    prefill_end_t: float = 0.0  # perf_counter when the prompt was processed
    first_token_t: float = 0.0  # perf_counter at the first emitted token
    decode_ms: float = 0.0      # prompt processed -> end (llama.cpp's predicted_ms: covers every generated token)
    passes: int = 0             # decode passes
    drafted: int = 0
    accepted: int = 0
    forced_n: int = 0
    stop_in_draft: int = 0
    prefill: dict = None        # the prompt's PrefillStats as a dict (source, restore / batched / hand-off / passes ms, exact, lcp)
    engine_counts: dict = None  # the session's counters over the request (bails, recoveries: the worker's, flight recorder)


def run_generation(sess, prompt, sampler, max_new, on_tokens, cancel=None, stop_ids=None, drafter=None,
                   n_draft=None, force=None, tool_open_id=None, stats=None, marks=None):
    """Generate on `sess` (EngineSession) after `prompt` (ids).
    on_tokens(ids) -> None | n_keep: called with each batch of emitted ids (forced ids included); returning n_keep
    stops the generation after the first n_keep ids of the batch (a stop string matched).
    drafter(tokens, k) -> k draft ids (deterministic); n_draft defaults to ntok - 1.
    force: (trigger_id, forced_ids) or None; tool_open_id: the <tool_call> id (a call before the trigger cancels the
    forcing). marks: the prompt cache's message boundaries, passed to sess.prefill when given. Raises EngineError from the
    session. Returns GenResult."""
    st = stats if stats is not None else sess.stats
    if st is not None and not st.enabled:
        st = None
    clock = st.clock if st is not None else None
    stop_ids = frozenset(sess.eos_ids if stop_ids is None else stop_ids)
    res = GenResult(prompt_n=len(prompt))
    row = sess.prefill(prompt, cancel, marks=marks, max_new=int(max_new))   # (max_new: the K3 session's side-slot routing)
    pf = sess.last_prefill
    res.cached_n, res.prefill_ms, res.prefill_passes, res.restore = pf.cached_n, pf.ms, pf.passes, pf.restore
    res.prefill = dict(source=pf.source, restore_ms=round(pf.restore_ms, 3), batched_ms=round(pf.batched_ms, 3),
                       handoff_ms=round(pf.handoff_ms, 3), passes_ms=round(pf.passes_ms, 3), exact=bool(pf.exact), lcp=int(pf.lcp),
                       slot=pf.slot)
    res.prefill_end_t = time.perf_counter()
    if st is not None:
        st.prefill_done(pf)
    if row is None:
        res.finish = 'cancel'
        sess.end()
        return res
    hist = list(prompt)
    k_max = (sess.ntok - 1) if n_draft is None else min(n_draft, sess.ntok - 1)
    force_pending = force is not None
    out = [sampler.sample(row, hist)]
    feed_len = 0                    # tokens fed by the pass that produced `out` (0: from prefill)
    drafts = []
    n_real = 0                      # the last pass's real drafts (0: none, or filler: a plain pass)
    t_first = None
    t_pass = draft_ms = 0.0         # metrics: the pass in flight's start (stats clock) and drafter ms
    cls = None
    while True:
        # ---- clip the batch: max_new, EOS, the forcing trigger
        out = out[:max_new - len(res.gen_ids)]
        done = None
        for j, t in enumerate(out):
            if t in stop_ids:
                out = out[:j + 1]
                done = 'eos'
                break
            if force_pending and t == tool_open_id:
                force_pending = False
            if force_pending and t == force[0]:
                out = out[:j + 1]
                break
        if done is None and len(res.gen_ids) + len(out) >= max_new:
            done = 'length'
        # ---- text, stop strings
        n_keep = on_tokens(out)
        if n_keep is not None:
            out = out[:n_keep]
            done = 'stop'
        if t_first is None:
            t_first = time.perf_counter()
            res.first_token_t = t_first
            if st is not None:
                st.first_token(len(out))
        acc0 = res.accepted
        if feed_len:
            sess.accept(feed_len - 1 + len(out))     # the fed tokens, then the accepted drafts (out[:-1])
            if n_real:                               # (filler drafts accepted by a plain pass are not accepted drafts)
                res.accepted += len(out) - 1
                if len(out) <= len(drafts) and out[-1] == drafts[len(out) - 1]:
                    res.accepted += 1                # the last emitted token was a draft too (emission ended there)
                    res.stop_in_draft += done in ('eos', 'stop')
        res.gen_ids += out
        hist += out
        if st is not None and feed_len:
            st.pass_done(t_pass, draft_ms, feed_len, n_real, res.accepted - acc0, len(out), cls)
        if done:
            break
        feed = [out[-1]]
        if force_pending and out[-1] == force[0]:
            force_pending = False
            fids = list(force[1])[:max_new - len(res.gen_ids)]     # never past max_new (nor the context room)
            res.forced_n += len(fids)
            n_keep = on_tokens(fids)
            if st is not None:                  # /slots n_decoded / n_remain and the position count them (M7)
                st.forced(len(fids) if n_keep is None else n_keep)
            if n_keep is not None:              # a stop string inside the forced text
                res.gen_ids += fids[:n_keep]
                hist += fids[:n_keep]
                res.finish = 'stop'
                break
            res.gen_ids += fids
            hist += fids
            feed += fids
            if len(res.gen_ids) >= max_new:
                res.finish = 'length'
                break
            while len(feed) > sess.ntok:        # forced tokens beyond one pass: process them like a prompt
                if cancel is not None and cancel.is_set():
                    break
                sess.verify(feed[:sess.ntok])
                sess.accept(sess.ntok)
                res.passes += 1
                feed = feed[sess.ntok:]
                if st is not None:
                    st.other_pass()
        if cancel is not None and cancel.is_set():
            res.finish = 'cancel'
            break
        k = min(k_max, sess.ntok - len(feed), max_new - len(res.gen_ids) - 1)
        if st is not None:
            t_pass = clock()
            cls = st.cls
        drafts = list(drafter(hist, k))[:k] if (drafter is not None and k > 0) else []
        if st is not None:
            draft_ms = (clock() - t_pass) * 1e3
        # a session that knows its drafts were filler (a plain pass, a drafter window gap) says so in drafts_real:
        # they are verified like drafts but count as none (the DFlash counters, the plain pass time)
        n_real = len(drafts) if sess.drafts_real else 0
        res.drafted += n_real
        out = sess.sample_pass(feed, drafts, sampler, hist, stop_ids)
        feed_len = len(feed)
        res.passes += 1
    if not res.finish:
        res.finish = done
    sess.end()
    res.decode_ms = (time.perf_counter() - res.prefill_end_t) * 1e3
    return res
