"""errors shared by the engines and the server (no torch import)"""


class NonFiniteLogits(RuntimeError):
    """an engine produced non-finite logits (NaN / inf in a row: its weights or states are corrupt). The engine (where it
    checks) marks itself broken - sticky, reset() keeps it - and the server stops serving: a poisoned process must fail,
    never stream the argmax of NaN (token 0). Seen 2026-11-13: one served load in ~15 had NaN from block ~52 on, every
    prompt, every request, until the process ended"""
