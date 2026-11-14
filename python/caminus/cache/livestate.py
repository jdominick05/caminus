"""The live state: what the engine's live slot holds right now, and the proof of it.

The prompt cache owns it (caminus.cache.prompt.PromptCache.live) and nothing else keeps a copy: the session reads the log
and the state through the cache and reports each pass it accepts (PromptCache.append); prefill and recovery report in bulk
(PromptCache.extend). It has three parts:

  log     the token ids of the processed positions. While the state is live, len(log) == st['i0']; after the live state
          is dropped (PromptCache.drop_live) the log stays, so checkpoints can still be matched against it, and st is (0, 0).
  st      dict(i0, commit): the next pass starts at position i0 and applies commit, the previous pass's lazy commit.
  marks   the engine counters (the engine's seq and token, the drafter's token) when the cache last vouched for log and st.
          A counter that moved since means the engine was reset outside the cache (seq) or ran passes the cache was not told
          about (token).

Before 2026-11-07 the session kept its own copy of the log and state, synced with the cache's by hand at the end of every
request, and worked out from its own epochs what the device held; four fixes (668dd7a, 03c0d93, facbdac, 500ff36) were
drift between those copies.
"""
from ..errors import NumericalFailure


def is_bail(e):
    """True for a pass that failed after the engine's bitwise retries (FlashNextTP / DFlashK3 'a wait gave up ...')"""
    if isinstance(e, NumericalFailure):
        return False   # terminal state corruption is never retryable, even when its diagnostic mentions a timeout
    return isinstance(e, RuntimeError) and 'a wait gave up' in str(e)


class LiveState:
    __slots__ = ('log', 'st', 'marks', 'known', 'vouched')

    def __init__(self):
        # known: False while the cache runs something that can leave the device state undefined if it fails part way (the
        # session's prompt phase, a bail recovery): a failure report then drops the live state (PromptCache.settle).
        # vouched: the engine's (seq, token) when an operation the session asked for last completed (vouch): what a failure
        # report compares with. The marks move inside an operation too (an extend the cache refuses resets the slot and
        # re-marks before it raises), and the report must still see the reset the session's request ran into
        self.log, self.st, self.marks, self.known, self.vouched = [], dict(i0=0, commit=0), None, True, None

    @staticmethod
    def counters(m, df):
        """the engine counters now: (m.seq, m.token, the drafter's token or 0). Every engine and stand-in has them from
        construction (CLAUDE.md, the engine counters)"""
        return (m.seq, int(m.token), int(df.token) if df is not None else 0)

    def mark(self, m, df):
        """the cache vouches for log and st as the device holds them now"""
        self.marks = self.counters(m, df)

    def vouch(self, m):
        """an operation the session asked for completed: the state it left is the one a failure report starts from"""
        self.vouched = (m.seq, int(m.token))

    def drift(self, m, df):
        """what moved since the mark: None, 'reset' (the engine's seq: a reset outside the cache, which took the KV rows with
        it) or 'passes' (a launch counter: passes the cache was not told about)"""
        seq, tok, dtok = self.counters(m, df)
        if seq != self.marks[0]:
            return 'reset'
        if (tok, dtok) != self.marks[1:]:
            return 'passes'
        return None

    def tokens(self):
        """the live tokens: positions < st['i0'] (the log itself while the state is live, nothing after a drop)"""
        i0 = int(self.st['i0'])
        return self.log if len(self.log) == i0 else self.log[:i0]
