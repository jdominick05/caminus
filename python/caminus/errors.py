"""errors shared by the engines and the server (no torch import)"""


class NumericalFailure(RuntimeError):
    """Terminal numerical state corruption: refuse further work until process restart.

    Reset, retry, abort recovery or parking cannot make partially written or
    corrupt state usable. The server marks the worker broken and exits 75.
    """


class NonFiniteLogits(NumericalFailure):
    """an engine produced non-finite logits (NaN / inf in a row: its weights or states are corrupt). The engine (where it
    checks) marks itself broken - sticky, reset() keeps it - and the server stops serving: a poisoned process must fail,
    never stream the argmax of NaN (token 0). Seen 2026-11-13: one served load in ~15 had NaN from block ~52 on, every
    prompt, every request, until the process ended"""


class InvalidKVEncoding(NumericalFailure):
    """A Q8 encoder refused a block; other cache blocks may already be written.

    The whole append and its cache state are unusable, even when the refused
    block itself was left untouched. Never recover by resetting or parking it.
    """
