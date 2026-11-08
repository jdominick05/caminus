"""The prompt path of an engine with no batched prefill yet (qwen3next: Qwen3NextTP, Qwen3-Coder-Next on both cards): a
prefill driver (caminus.prefill.driver.PrefillDriver: the prompt cache's and K3Session's contract) with no crossover, so the
routing rule (batched_span) sends every prompt part to K3 passes (bitwise the one-token decode; spec / longctx
teacher-forced passes). run() is never reached through the routing; called directly it refuses before anything is written."""
from .driver import PrefillDriver, PrefillRefused


class KPassOnly(PrefillDriver):
    """no batched prefill: crossover None (route() / span() never batch: the K3 passes take every prompt)"""
    crossover = None
    chunk = 0

    def __init__(self, m):
        self.m = m
        self.n_vocab = int(m.T['token_embd.weight'].shape[1]) if hasattr(m, 'T') else None

    def numerics(self):
        """the engine key's prefill part: none (no batched prefill)"""
        return dict(wmma_out='none', router_logits='none', moe='none', q6k='none')

    def validate(self, toks, p0, p1, draft=None, n_total=None):
        raise PrefillRefused(f'{type(self.m).__name__} has no batched prefill: every prompt runs as K3 passes')

    def run(self, toks, i0, commit, p1, draft=None, **kw):
        raise PrefillRefused(f'{type(self.m).__name__} has no batched prefill: every prompt runs as K3 passes')
