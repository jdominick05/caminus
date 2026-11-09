"""Host-side buffer checks shared by the batched prefill's launchers (caminus.prefill.mmq, caminus.prefill.glue): every launch is
refused (ValueError) before anything runs unless each buffer covers what the kernel touches through it and no buffer the
kernel writes shares a byte with another of its buffers (except the exact in-place uses a kernel is written for).

Why the view and not the storage: the prefill's workspaces (mmq.Workspace, prefill_ops.GlueWorkspace) are one allocation
each, carved into views; a short view inside the block passes a check against the end of the storage and the kernel
then writes into the next buffer. So a buffer must hold what the kernel touches within its OWN elements (view_bytes).
The one exception is deliberate and stays a storage check (mmq's weights): MMQ reads past the last row of a K = 384
shard into the zeroed tail FlashNextTP gives it (mmq.overread_bytes), which is outside the shard's view by design.

Why overlap matters: the kernels run many workgroups at once, so a written buffer that shares bytes with one another
workgroup reads gives an order-dependent result (or, partially aliased, garbage); only an exact in-place use (the same
bytes, read and written by the same thread or workgroup, as rms_mul's y = x) is safe, and only where the kernel says so.

Two forms. The element-indexed launchers (caminus.prefill.glue / attn / gdn / tp) state each buffer once, as (tensor, dtype,
elements[, alignment]), and `LaunchCheck` derives both the per-buffer check and the overlap check from that one statement.
The byte-indexed GEMM launchers (caminus.prefill.mmq / moe / dense) keep their own per-buffer checks, whose refusals name each
buffer's dimensions (rows x columns, the quantized block bytes, the weights' over-read: mmq.weight_bytes), and call
`disjoint` with the byte counts those checks computed.

ALIGN is the prefill workspaces' block alignment: every buffer of a workspace layout (mmq.workspace_layout, glue_layout,
attn_layout, tp.xchg_layout, dense.bf16_layout) starts on it (`aligned`)."""
import torch

ALIGN = 256


def aligned(layout, a=ALIGN):
    """a workspace layout {name: bytes} with each buffer rounded up to whole a-byte blocks"""
    return {n: (b + a - 1) // a * a for n, b in layout.items()}


def view_bytes(t):
    """bytes a contiguous tensor's own elements cover (its view, not its storage)"""
    return t.numel() * t.element_size()


def span(t, nbytes):
    """the device byte range [lo, hi) a kernel touches through t's pointer"""
    lo = t.data_ptr()
    return lo, lo + int(nbytes)


def disjoint(op, bufs, writes, same_ok=()):
    """refuse (ValueError) a launch whose written buffers overlap any other of its buffers.
    bufs: {name: (tensor or None, bytes the kernel touches from the tensor's pointer)}; writes: the names the kernel writes;
    same_ok: pairs {a, b} that may be the exact same bytes (same start, same length: an in-place use the kernel is written
    for). A partial overlap is always refused; two read-only buffers may overlap freely"""
    _overlap(op, {n: span(t, b) for n, (t, b) in bufs.items() if t is not None and int(b) > 0}, writes, same_ok)


def _overlap(op, R, writes, same_ok):
    """disjoint's rule on the byte ranges R {name: (lo, hi)} (dict order: the first overlapping partner is the one named)"""
    ok = {frozenset(p) for p in same_ok}
    for w in writes:
        if w not in R: continue
        a = R[w]
        for n, b in R.items():
            if n == w or not (a[0] < b[1] and b[0] < a[1]): continue
            if a == b and frozenset((w, n)) in ok: continue
            how = 'is the same buffer as' if a == b else 'partially overlaps'
            raise ValueError(f'{op}: {w} [{a[0]:#x}, {a[1]:#x}) {how} {n} [{b[0]:#x}, {b[1]:#x})'
                             + (' (exact in-place allowed only for ' + ', '.join('/'.join(sorted(p)) for p in ok) + ')' if ok else
                                ' (this kernel has no in-place form)'))



def _host_mapped(t):
    """a CPU tensor inside a caminus.device.hip.host_mapped() buffer (pinned, mapped into every card: the 1M-context mode's
    KV cache) passes the device check; any other host tensor does not"""
    from .hip import is_host_mapped
    return is_host_mapped(t)

class LaunchCheck:
    """the host checks of one launch with each buffer stated once (one per launcher module: its message prefix, its
    default pointer alignment, how it names dtypes). check(dev, op, bufs, writes, same_ok=(), opt=(), names=None):
    bufs {name: (tensor, dtype, n[, align])}, in the order the checks run: each tensor must be a contiguous tensor of dtype
    on dev with >= n elements in its own view (view_bytes) and its pointer align-aligned (default: this module's), and the
    kernel touches its first n elements; a name in opt may be None (an absent optional buffer: skipped), any other None is
    refused. Then disjoint's overlap rule over those n-element spans (op, writes, same_ok). names: {name: what} where a
    refusal names the buffer otherwise than its key (the per-buffer refusals only; the overlap refusal uses the keys)"""
    __slots__ = ('mod', 'align', 'short')

    def __init__(self, mod, align=1, short_dtype=False):
        """mod: the per-buffer refusals' prefix; align: the default pointer alignment (1: none); short_dtype: name a dtype
        'float32' rather than 'torch.float32'"""
        self.mod, self.align, self.short = mod, int(align), bool(short_dtype)

    def __call__(self, dev, op, bufs, writes, same_ok=(), opt=(), names=None):
        mod, R = self.mod, {}
        for n, b in bufs.items():
            t = b[0]
            if t is None and n in opt: continue
            what = names.get(n, n) if names else n
            host = isinstance(t, torch.Tensor) and t.device.type == 'cpu' and _host_mapped(t)   # (device.hip.host_mapped)
            if not isinstance(t, torch.Tensor) or (t.device != dev and not host) or not t.is_contiguous():
                raise ValueError(f'{mod}: {what} must be a contiguous tensor on {dev}')
            if t.dtype != b[1]:
                raise ValueError(f'{mod}: {what} must be {str(b[1])[6:] if self.short else b[1]} (is {t.dtype})')
            k = int(b[2])
            if t.numel() < k: raise ValueError(f'{mod}: {what} holds {t.numel()} < {b[2]} elements (its view; {view_bytes(t)} B)')
            a = b[3] if len(b) > 3 else self.align
            lo = t.data_ptr()
            if a > 1 and lo % a: raise ValueError(f'{mod}: {what} must be {a}-byte aligned')
            if k > 0: R[n] = (lo, lo + k * t.element_size())
        _overlap(op, R, writes, same_ok)
