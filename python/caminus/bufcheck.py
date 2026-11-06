"""Host-side buffer checks shared by the batched prefill's launchers (caminus.mmq, caminus.prefill_ops): every launch is
refused (ValueError) before anything runs unless each buffer covers what the kernel touches through it and no buffer the
kernel writes shares a byte with another of its buffers (except the exact in-place uses a kernel is written for).

Why the view and not the storage: the prefill's workspaces (mmq.Workspace, prefill_ops.GlueWorkspace) are one allocation
each, carved into views; a short view inside the block passes a check against the end of the storage and the kernel
then writes into the next buffer. So a buffer must hold what the kernel touches within its OWN elements (view_bytes).
The one exception is deliberate and stays a storage check (mmq's weights): MMQ reads past the last row of a K = 384
shard into the zeroed tail FlashNextTP gives it (mmq.overread_bytes), which is outside the shard's view by design.

Why overlap matters: the kernels run many workgroups at once, so a written buffer that shares bytes with one another
workgroup reads gives an order-dependent result (or, partially aliased, garbage); only an exact in-place use (the same
bytes, read and written by the same thread or workgroup, as rms_mul's y = x) is safe, and only where the kernel says so."""


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
    R = {n: span(t, b) for n, (t, b) in bufs.items() if t is not None and int(b) > 0}
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
