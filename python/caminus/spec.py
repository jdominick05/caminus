"""Speculative decoding over a K3 engine with T-token verify passes (caminus.flashnext_tp.FlashNextTP(ntok=T)).

Each pass runs positions i0 .. i0 + T - 1: first `forced` given tokens (the last accepted token, or prompt tokens
during prefill), then drafts. The target's argmax at each position checks the next draft; the first mismatch ends the
pass, and the target's own token there is the bonus. The committed count (forced + accepted drafts) is passed to the
next verify as its lazy commit, so rejected drafts never reach the recurrent states (KV entries past the accepted
position are simply overwritten later). Greedy: the output equals plain one-token greedy decoding."""
import time
import torch
from .k3 import ContextBoundsError


class ExactHitRefused(ContextBoundsError):
    """a request the engine's state cannot continue bitwise, refused on the host before anything is launched (resume_point):
    zero new tokens from a state whose recurrent states hold every token already (commit 0), a state past the request's
    tokens, a lazy-commit count outside [0, min(T, i0)]"""


def resume_point(toks, i0, commit, T):
    """the (i0, commit) a pass loop resumes from for a request whose tokens are toks (loop(); longctx.generate), checked
    on the host before anything launches (ExactHitRefused):
    - i0 < len(toks): unchanged (toks[i0:] are given tokens still to run).
    - i0 == len(toks), the exact cache hit (a request with zero new tokens): its first logits are position i0 - 1's, which
      the state's last pass computed and did not keep. With commit >= 1 that pass's row for i0 - 1 is still uncommitted
      (lazy commit: the recurrent states hold position i0 - commit plus the pass's records), so the loop runs it again:
      (i0 - 1, commit - 1) commits the rows before it and reruns toks[i0 - 1] at i0 - 1 on the same state, the same token
      and position: bitwise the logits that pass had (a T-token pass's rows equal one-token steps, tools/gates/
      tp_multi.py), at the cost of one pass (the rest of the pass verifies drafts, as any pass). With commit 0 the states
      hold position i0 already (a fill or batched prefill of every token, or a checkpoint taken there): nothing gives those
      logits bitwise: refused. A prompt cache keeps the state before the prompt's last pass instead (prompt_end()).
    - i0 > len(toks) (the state holds positions the request does not have), commit outside [0, min(T, i0)], an empty
      request: refused."""
    n, i0, commit, T = len(toks), int(i0), int(commit), int(T)
    if n < 1: raise ExactHitRefused('resume: an empty request (no tokens)')
    if not 0 <= commit <= min(T, i0):
        raise ExactHitRefused(f'resume: lazy commit {commit} outside [0, min(T {T}, i0 {i0})]')
    if i0 > n:
        raise ExactHitRefused(f'resume: the state is at position {i0}, past the request\'s {n} tokens (it holds positions the '
                              f'request does not have: another sequence, or a longer conversation): restore an earlier checkpoint')
    if i0 < n: return i0, commit
    if commit < 1:
        raise ExactHitRefused(f'resume: zero new tokens from a state committed through position {i0 - 1} (commit 0: a fill or '
                              f'batched prefill of all {n} tokens, or a checkpoint taken there): the logits of position {i0 - 1} '
                              f'cannot be recomputed bitwise from it; restore a checkpoint before the prompt\'s last pass '
                              f'(spec.prompt_end) or an earlier one')
    return i0 - 1, commit - 1


def positions_needed(n_prompt, n_new, T):
    """the KV rows generate() may touch: a pass writes T positions from i0, and i0 reaches the last accepted token's
    position (at most n_prompt + n_new - 2 while tokens are still wanted; n_prompt - 1 in the prefill), so the engine's
    max_ctx (and a drafter's cache) must hold n_prompt + n_new + T - 2 positions (n_prompt - 1 + T without new tokens)"""
    return max(n_prompt - 1, n_prompt + n_new - 2) + T if n_new > 0 else 0


def check_capacity(m, n_prompt, n_new, draft=None):
    """refuse (ContextBoundsError) a generation the engine's KV cache, its pass limit (the dense tape's exact-attention
    limit, capacity.pass_limit) or the drafter's rows (draft.max_pos) cannot hold: before anything is reset or launched"""
    from .capacity import pass_limit
    need = positions_needed(n_prompt, n_new, m.ntok)
    lim = pass_limit(m)
    caps = [('engine max_ctx', m.max_ctx)] + ([(f'engine pass limit (exact attention up to {lim} positions)', lim)] if lim < m.max_ctx else []) + \
           ([('drafter max_pos', draft.max_pos)] if getattr(draft, 'max_pos', None) else [])
    for name, cap in caps:
        if need > cap:
            raise ContextBoundsError(f'spec.generate: {n_prompt} prompt + {n_new} new tokens with T = {m.ntok} need {need} '
                                     f'positions, {name} is {cap}')


def generate(m, prompt, n_new, draft, stats=None, prefill=None):
    """m: an engine with ntok = T; prompt: token ids; draft(tokens, k) -> k proposed next tokens after tokens, for any
    1 <= k <= T - 1 (an object with inject(pos0, features) also receives each pass's exported target features, as DFlash
    needs; one with prefill(tokens, n) is told after each prompt pass that the target has run positions < n, as MTP needs;
    one with flush() is told before a pass that asks it for no drafts (k = 0: T given tokens), so it can take the last
    pass's features before this pass rewrites them, as DFlash needs; one with reset() is reset with the engine; one with
    device_accept true and generate(m, prompt, n_new, stats) runs the whole generation itself).
    Returns prompt + n_new greedy tokens. stats (dict) gets passes, accepted drafts, time, and per generation pass
    the three serial phases: draft (host wall time to the drafts on the host), verify (the target pass to its argmax on
    the host) and inject (the drafter's feature intake, to its device's completion when the drafter has sync()).
    prefill: a caminus.prefill.Prefill of m: the prompt's positions [0, min(n - 1, prefill.limit)) through the batched
    prefill when they reach its crossover (route()), validated up front with the whole prompt; the last prompt token (and
    any prompt past the limit) then runs as K3 passes from that committed state (commit 0), so the first logits come from
    decode numerics. None: every prompt token as K3 passes (as before)."""
    if getattr(draft, 'device_accept', False) and hasattr(draft, 'generate'):   # (caminus.dflash_k3: the S3 device-accept loop)
        return draft.generate(m, prompt, n_new, stats) if prefill is None else draft.generate(m, prompt, n_new, stats, prefill=prefill)
    if not prompt: raise ValueError('spec.generate: an empty prompt')
    check_capacity(m, len(prompt), n_new, draft)            # before the reset: nothing runs if it cannot fit
    P = prefill_span(prefill, prompt, draft)
    m.reset()
    if hasattr(draft, 'reset'): draft.reset()
    i0 = 0
    if P:
        pst = prefill.run(list(prompt), 0, 0, P, draft, validated=True); i0 = P
        if stats is not None: stats['prefill'] = pst
    toks, _, _ = loop(m, list(prompt), i0, 0, len(prompt) + n_new, draft, stats)
    return toks[:len(prompt) + n_new]


def prompt_end(m, prompt, draft=None, prefill=None, stats=None):
    """generate()'s cold path up to its last prompt pass (the pass that runs the prompt's last token, and the first drafts):
    reset m (and draft), the batched prefill of [0, P) when routed (prefill_span), then the prompt passes of T tokens before
    the last one. Returns the state dict(i0, commit) with 1 <= len(prompt) - i0 <= T: the prompt cache's prompt-end
    checkpoint (M6). Restoring it and continuing (longctx.generate with the same prompt: zero new tokens past it) runs
    exactly the passes the cold run runs from there, so an exact cache hit's first logits and tokens are the cold run's,
    bitwise (G3: the restored state is the cold state), with no logits stored"""
    if getattr(draft, 'device_accept', False):
        raise ValueError('spec.prompt_end: the device-accept loop resets inside generate(); pass a generic-loop drafter '
                         '(DFlashK3Drafter(df, device_accept=False))')
    if not prompt: raise ValueError('spec.prompt_end: an empty prompt')
    check_capacity(m, len(prompt), 1, draft)
    P = prefill_span(prefill, prompt, draft)
    m.reset()
    if hasattr(draft, 'reset'): draft.reset()
    i0 = 0
    if P:
        pst = prefill.run(list(prompt), 0, 0, P, draft, validated=True); i0 = P
        if stats is not None: stats['prefill'] = pst
    _, i0, commit = loop(m, list(prompt), i0, 0, len(prompt) + 1, draft if draft is not None else _Filler(), stats, until_last=True)
    return dict(i0=i0, commit=commit)


class _Filler:
    def __call__(self, toks, k):
        return [int(toks[-1])] * k


def prefill_span(prefill, prompt, draft=None):
    """the prompt positions [0, P) a generation sends through the batched prefill (0: none): every prompt token but the
    last (it goes through K3), at most prefill.limit (the engine's: max_ctx with QSA, else QSA_EXACT), and only from the
    crossover on (prefill.route); validated here, with the whole prompt, before the caller resets anything
    (caminus.prefill.PrefillRefused)"""
    if prefill is None: return 0
    P = min(len(prompt) - 1, prefill.limit)
    if P < 1 or not prefill.route(P):
        prefill.check_tokens(list(prompt), draft)            # (routed to K3 passes: the whole prompt still checked up front)
        return 0
    prefill.validate(list(prompt), 0, P, draft)
    return P


def loop(m, toks, i0, commit, total, draft, stats=None, until_last=False):
    """generate()'s pass loop from any engine state: positions < i0 have run (the last pass's first `commit` tokens not yet
    committed: the lazy commit), toks[i0:] are given tokens still to run; runs until len(toks) >= total. A fresh start is
    (i0, commit) = (0, 0) after m.reset(); caminus.longctx resumes a filled context (a snapshot) with its (i0, commit).
    A request with zero given tokens (i0 == len(toks): an exact cache hit) reruns the last pass's row of position i0 - 1
    when it is still uncommitted, else is refused before anything launches (resume_point). until_last: stop before the
    pass that would run the last given token (prompt_end()). Returns (toks, i0, commit): every token so far (it may pass
    total by the last pass's accepted drafts) and the state after the last pass, for another loop() on the same sequence"""
    T = m.ntok; passes = acc = 0; t0 = time.perf_counter(); n_given = len(toks)
    t_gen = None; t_draft = t_verify = t_inject = 0.0; gen_passes = 0
    if len(toks) < total or until_last: i0, commit = resume_point(toks, i0, commit, T)   # (host only: before any launch)
    while len(toks) < total:
        if until_last and len(toks) - i0 <= T: break       # (the next pass would run the last given token)
        forced = min(T, len(toks) - i0)                     # given tokens at i0 .. (prompt, or the last bonus)
        if i0 + T > m.max_ctx: raise ContextBoundsError(f'spec.generate: a pass at {i0} .. {i0 + T - 1} past max_ctx {m.max_ctx}')
        k = T - forced
        if k and t_gen is None: t_gen = time.perf_counter()  # generation starts (the prompt is in)
        if k and hasattr(m, 'prefetch'): m.prefetch(toks, range(i0, i0 + forced))   # their PLE reads overlap the drafting
        td = time.perf_counter()
        if k: d = list(draft(toks, k))
        else:                                               # T given tokens, no drafts: the drafter's pending intake runs now
            d = []                                          # (DFlash: the last pass's inject, before this pass rewrites the
            if hasattr(draft, 'flush'): draft.flush()       # features; a no-op after a prompt pass, whose prefill flushed)
        t_draft += time.perf_counter() - td
        gen_passes += k > 0
        seq = toks[:i0 + forced] + d
        tv = time.perf_counter()
        if hasattr(m, 'verify_tokens'): a = m.verify_tokens(seq, i0, commit)   # target's next token after each position
        else: a = m.verify(seq, i0, commit).argmax(dim=1).tolist()
        ti = time.perf_counter()
        if hasattr(draft, 'inject'):
            draft.inject(i0, m.features)                    # target features of every processed position
            if hasattr(draft, 'sync'): draft.sync()
        te = time.perf_counter()
        if k: t_verify += ti - tv; t_inject += te - ti
        j = 0
        while j < k and d[j] == a[forced - 1 + j]: j += 1  # accepted drafts
        if i0 + forced < len(toks):                         # still in the prompt: nothing to accept or add yet
            if hasattr(draft, 'prefill'): draft.prefill(toks, i0 + forced)   # (MTP: pair the prompt rows whose h is known)
            commit = forced; i0 += forced; passes += 1; continue
        toks = toks[:i0 + forced] + d[:j] + [a[forced - 1 + j]]
        commit = forced + j; i0 += forced + j; passes += 1; acc += j
    if stats is not None:
        t1 = time.perf_counter()
        stats.update(passes=passes, accepted=acc, seconds=t1 - t0, gen_seconds=t1 - (t_gen or t1), gen_passes=gen_passes,
                     draft_seconds=t_draft, verify_seconds=t_verify, inject_seconds=t_inject, gen_tokens=len(toks) - n_given)
    return toks, i0, commit


class DFlashDrafter:
    """adapts caminus.dflash.DFlash2 to generate(): the anchor is the last token, at position len(tokens) - 1"""
    def __init__(self, df):
        self.df = df
        self.max_pos = df.K.shape[1]   # the drafter's K / V rows (generate() checks a run fits)

    def inject(self, i0, feats):
        self.df.inject(i0, feats)

    def sync(self):
        """wait for the drafter's device (the inject's time is its own, not the next draft's)"""
        torch.cuda.synchronize(self.df.dev)

    def __call__(self, toks, k):
        return self.df.draft_graph(toks[-1], len(toks) - 1, k)
