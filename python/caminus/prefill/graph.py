"""HIP-graph capture of the batched prefill's chunks (M6, DeepInfer's zero-overhead scheduling lever): each card's compute
of a ChunkStack chunk captured once per launch structure and replayed as HIP graphs, instead of ~4,800 Python launches per
chunk (the ~190 ms host-enqueue floor that set T* = 48 and kept short prompts at parity with llama.cpp).

Per-card graphs, split at every exchange (measured 11-04, HIP 7.2 / torch 2.14, tests/prefill/test_prefill_graph.py's probes):
- one graph spanning both cards (card 1's streams joined into card 0's capture by events) captures and instantiates but
  SEGFAULTS in hipGraphLaunch, with or without copy nodes, with or without packet capture: not usable;
- hipMemcpyPeerAsync is NOT captured: during a capture it runs at once, outside the graph (a captured graph of a kernel and
  a peer copy has one node): the exchange's pushes can never be inside a graph;
so a graph holds one card's compute of one half-layer of one micro-chunk (ChunkStack._half: everything between two of
its exchange points; and the chunk's last hc_post), and the exchange runs between the graph launches exactly as in an eager
chunk: the same prefill_tp.Exchange calls on the same streams (hipMemcpyPeerAsync on the copy streams, HIP events only;
every push waits the peer's last read of its receive rows, every in-place sum waits both pushes, card 0's partial first:
both invariants hold by construction, the exchange code is not touched). A graph launch on a card's compute stream is
ordered like the launches it replaces (after the feed, the sums, the events), so nothing else changes.

Device memory: with HIP's default packet capture every instantiated graph takes a 2 MiB kernel-argument pool (200 graphs of
25 nodes: 402 MiB), ~390 MiB per chunk structure at ~194 graphs; with DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 the same 200 graphs
take 2 MiB (10 KiB each; host launch 3.4 ms for their 5,000 nodes, GPU 17.7 vs 16.0 ms). The variable must be set before the
process initializes HIP (the entry point's environment); Graphs refuses to run without it (ENV). Every structure's
instantiation is measured as this process's own device memory (proc_vram: KFD's per-process counter; hipMemGetInfo is
device-wide, and once read +16 MiB on the XTX for a structure while this process's counter stayed flat over 783
instantiations; ChunkGraph.mib), a structure above REFUSE_MIB per card is dropped and its chunks run
eager (the packet-capture signature is ~390 MiB; HIP's own pools grow in 16 MiB steps), and each structure held is counted
at capacity.PREFILL_GRAPH_MIB (prefill_need_mib(graphs=)). A capture launches nothing, so a refused capture (the caps, the
memory check) leaves the chunk to the eager path.

What a structure is (ChunkStack.structure): the micro-chunk bounds (so the chunk size), the exchange mode, the routed-expert
/ dense Q6_K / attention paths and the Glue modes, and per piece the attention core's dense and QSA rows, whether it pools
(QSA) and whether it exports DFlash features. The rest of the launch sequence depends on the chunk's position: the attention
core (q / k RoPE positions, the KV / IDXRAW rows written, attend's first rows and split counts, i.e. its grids; the pools,
the score sub-blocks and grids, the selection) and the feature export's rows and offsets. Those launches are recorded per
region (ChunkStack._pos: a piece's attention core on a card, its feature export) with their graph nodes; a replay, right
before it launches a segment, reruns exactly that segment's regions' Python at the new position with the launches
intercepted (caminus.device.hip.RECORD; the cards run the earlier segments meanwhile): every launcher's
own checks run again (the KV / IDXRAW rows, the dense rows below QSA_EXACT, the pools: check_span and the rest; nothing is
skipped), and each node whose grid or arguments differ is updated in its instantiated graph (hipGraphExecKernelNodeSetParams,
which copies the arguments; a launch already queued keeps the ones it was launched with: probe 'inflight'). The static
buffers (workspace views, weights, states) are the same objects at every replay of a structure, so the view and overlap
checks made at capture hold. Every launch during a capture must be on the capturing stream (checked: a launch elsewhere would
run at once, outside the graph). A chunk is never padded to a size: the recurrences would run over garbage tokens and KV
rows past the chunk would be written.

After the replays the caller synchronizes both devices and checks every flag word exactly as after eager chunks
(caminus.prefill.coder.Prefill.run: the routing flags, the feed's clamps, the attention core's row refusals; sticky words)."""
import ctypes, os, time
import torch
from ..device import hip as _hip_mod
from ..device.hip import hip, check

c_void_p, c_size_t, c_uint, c_int = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int
EV_DISABLE_TIMING = 0x2
CAPTURE_GLOBAL = 0          # hipStreamCaptureModeGlobal
ENV = {'DEBUG_CLR_GRAPH_PACKET_CAPTURE': '0'}   # set before HIP initializes (see the module notes): 10 KiB per graph, not 2 MiB
REFUSE_MIB = 64     # one structure's instantiation measured above this (per card, this process's own memory) is refused
STATUS_ACTIVE = 1           # hipStreamCaptureStatusActive


class Dim3(ctypes.Structure):
    _fields_ = [('x', c_uint), ('y', c_uint), ('z', c_uint)]


class KParams(ctypes.Structure):   # hipKernelNodeParams (hip/hip_runtime_api.h)
    _fields_ = [('blockDim', Dim3), ('extra', c_void_p), ('func', c_void_p), ('gridDim', Dim3), ('kernelParams', c_void_p),
                ('sharedMemBytes', c_uint)]


_api = None


def api():
    global _api
    if _api is None:
        H = hip()
        H.hipStreamBeginCapture.argtypes = [c_void_p, c_int]
        H.hipStreamEndCapture.argtypes = [c_void_p, ctypes.POINTER(c_void_p)]
        H.hipGraphInstantiate.argtypes = [ctypes.POINTER(c_void_p), c_void_p, c_void_p, c_void_p, c_size_t]
        H.hipGraphLaunch.argtypes = [c_void_p, c_void_p]
        H.hipStreamGetCaptureInfo_v2.argtypes = [c_void_p, ctypes.POINTER(c_int), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(c_void_p),
                                                 ctypes.POINTER(ctypes.POINTER(c_void_p)), ctypes.POINTER(c_size_t)]
        H.hipGraphKernelNodeGetParams.argtypes = [c_void_p, ctypes.POINTER(KParams)]
        H.hipGraphExecKernelNodeSetParams.argtypes = [c_void_p, c_void_p, ctypes.POINTER(KParams)]
        H.hipGraphGetNodes.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_size_t)]
        H.hipGraphDestroy.argtypes = [c_void_p]; H.hipGraphExecDestroy.argtypes = [c_void_p]
        H.hipEventCreateWithFlags.argtypes = [ctypes.POINTER(c_void_p), c_uint]
        H.hipEventRecord.argtypes = [c_void_p, c_void_p]; H.hipStreamWaitEvent.argtypes = [c_void_p, c_void_p, c_uint]
        H.hipEventDestroy.argtypes = [c_void_p]
        _api = H
    return _api


def proc_vram(devs):
    """this process's own device memory per device in devs, MiB (KFD's /sys/class/kfd/kfd/proc/PID/vram_GPUID, matched to
    the device by its PCI location): unlike hipMemGetInfo it excludes other processes (the XTX drives the desktop; one
    structure read +16 MiB through hipMemGetInfo, 11-04, while this counter stayed flat); None where it cannot be read"""
    out = []
    try:
        loc = {}
        base = '/sys/class/kfd/kfd/topology/nodes'
        for n in os.listdir(base):
            props = dict(l.split() for l in open(f'{base}/{n}/properties') if len(l.split()) == 2)
            gid = int(open(f'{base}/{n}/gpu_id').read())
            if gid: loc[int(props.get('location_id', -1))] = gid
    except OSError:
        loc = {}
    for d in devs:
        try:
            pr = torch.cuda.get_device_properties(d)
            gid = loc[(int(pr.pci_bus_id) << 8) | (int(pr.pci_device_id) << 3)]
            out.append(int(open(f'/sys/class/kfd/kfd/proc/{os.getpid()}/vram_{gid}').read()) / 2 ** 20)
        except (OSError, KeyError, AttributeError, ValueError):
            out.append(None)
    return out


def _key(vals):
    """a launch's arguments as comparable values (pointers, ints, floats as their ctypes value)"""
    return tuple(v.value if hasattr(v, 'value') else v for v in vals)


class Recorder:
    """caminus.device.hip.RECORD while a chunk is captured ('capture': the regions' launches recorded with their graph nodes) or a
    graph's regions are rerun at a new position ('patch': the launches intercepted, each compared with its node's and the
    node updated when the grid or an argument differs)"""
    def __init__(self, mode, graph=None):
        self.mode, self.g = mode, graph
        self.regions = [] if mode == 'capture' else None
        self.cur = None; self.i = 0; self.updated = 0; self.compared = 0
        self.seg, self.stream = None, None      # (capture) the segment being captured and its stream
        self.nlaunch = 0                         # (capture) Kernel launches recorded in the segment

    # -- the stack's regions (ChunkStack._pos) --------------------------------------------------------------------------
    def region(self, key, fn):
        if self.mode != 'capture': raise RuntimeError('prefill_graph: a region outside a capture')
        self.cur = dict(key=key, fn=fn, launches=[], seg=self.seg)
        try: fn()
        finally: self.regions.append(self.cur); self.cur = None

    def patch(self, regions):
        for r in regions:
            self.cur, self.i = r, 0
            r['fn']()
            if self.i != len(r['launches']):
                raise RuntimeError(f'prefill_graph: region {r["key"]} reran {self.i} launches, captured {len(r["launches"])}')
        self.cur = None

    # -- caminus.device.hip.Kernel.__call__'s hooks ----------------------------------------------------------------------------
    def intercept(self, k, grid, block, vals, shmem, stream):
        if self.mode != 'patch': return False
        if self.cur is None: raise RuntimeError(f'prefill_graph: launch {k.name} outside a patched region')
        L = self.cur['launches']
        if self.i >= len(L): raise RuntimeError(f'prefill_graph: region {self.cur["key"]}: launch {self.i} ({k.name}) past the {len(L)} captured')
        rec = L[self.i]; self.i += 1
        if rec['fn'] != k.fn.value or rec['block'] != tuple(block) or rec['shmem'] != shmem:
            raise RuntimeError(f'prefill_graph: region {self.cur["key"]}: launch {self.i - 1} is {k.name}, captured {rec["name"]}')
        key = _key(vals); self.compared += 1
        if rec['grid'] == tuple(grid) and rec['args'] == key: return True
        p = KParams(); p.blockDim = Dim3(*block); p.extra = None; p.func = rec['func']; p.gridDim = Dim3(*grid); p.sharedMemBytes = shmem
        arr = (c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), c_void_p) for v in vals])
        p.kernelParams = ctypes.cast(arr, c_void_p)
        check(api().hipGraphExecKernelNodeSetParams(self.g.segs[self.cur['seg']][1], rec['node'], ctypes.byref(p)))   # (HIP copies the arguments)
        rec['grid'], rec['args'] = tuple(grid), key; self.updated += 1
        return True

    def launched(self, k, grid, block, vals, shmem, stream):
        if self.mode != 'capture': return
        if stream != self.stream:
            raise RuntimeError(f'prefill_graph: {k.name} launched on stream {stream:#x} during the capture of {self.seg} on {self.stream:#x} '
                               f'(it ran at once, outside the graph)')
        self.nlaunch += 1
        if self.cur is None: return
        H = api(); st, cid, gr, deps, nd = c_int(), ctypes.c_ulonglong(), c_void_p(), ctypes.POINTER(c_void_p)(), c_size_t()
        check(H.hipStreamGetCaptureInfo_v2(c_void_p(stream), ctypes.byref(st), ctypes.byref(cid), ctypes.byref(gr), ctypes.byref(deps), ctypes.byref(nd)))
        if st.value != STATUS_ACTIVE or nd.value != 1:
            raise RuntimeError(f'prefill_graph: {k.name} captured with status {st.value}, {nd.value} dependencies (one expected: its node)')
        node = c_void_p(deps[0])
        p = KParams(); check(H.hipGraphKernelNodeGetParams(node, ctypes.byref(p)))
        self.cur['launches'].append(dict(name=k.name, fn=k.fn.value, func=p.func, node=node, grid=tuple(grid), block=tuple(block), shmem=shmem,
                                         args=_key(vals)))


class ChunkGraph:
    """one captured chunk structure: per segment (card, kind, layer, piece) its instantiated graph, the partial buffer each
    half returns (the exchange's input), the position regions with their nodes, the device memory the instantiations took
    per card (measured)"""
    def __init__(self, struct):
        self.struct, self.segs, self.parts, self.regions, self.by_seg = struct, {}, {}, [], {}
        self.mib, self.nodes, self.capture_ms, self.inst_ms = [0.0, 0.0], 0, 0.0, 0.0
        self.replays = 0; self.last_patch = (0, 0)

    def destroy(self):
        H = api()
        for g, e in self.segs.values():
            if e: check(H.hipGraphExecDestroy(e))
            if g: check(H.hipGraphDestroy(g))
        self.segs = {}


class Graphs:
    """the captured chunk graphs of one ChunkStack, by launch structure (ChunkStack.structure), at most max_graphs structures
    (the least recently used dropped first). policy: which uncaptured structures run() captures ('always'; 'never': replay
    only what is held; a set of chunk sizes: the chunks of those sizes). Refused (RuntimeError) unless the process runs with
    ENV (2 MiB per graph otherwise)"""
    def __init__(self, stack, max_graphs=8, policy='always'):
        for k, v in ENV.items():
            if os.environ.get(k) != v:
                raise RuntimeError(f'prefill_graph: run the process with {k}={v} (set before HIP initializes): with packet capture every '
                                   f'instantiated graph takes 2 MiB of device memory, ~390 MiB per chunk structure')
        self.S, self.max, self.policy = stack, int(max_graphs), policy
        self.g = {}             # structure -> ChunkGraph (insertion order = LRU order)
        self.refused = set()    # structures whose capture the caps or the memory check refused (eager from then on)
        self.stats = dict(captures=0, replays=0, eager=0, patch_ms=0.0, launch_ms=0.0, updated=0, refused=0)
        self.total = [0.0, 0.0]     # the process's device memory measured over every instantiation, MiB per card (destroyed ones included)

    def mib(self):
        """the device memory of every structure held, per card (measured at instantiation)"""
        return [sum(g.mib[c] for g in self.g.values()) for c in (0, 1)]

    def want(self, struct):
        if self.policy == 'always': return True
        if self.policy == 'never': return False
        return (struct[0][-1][1]) in self.policy          # (the chunk's size: the last piece's end)

    def run(self, S, B, inputs, mode):
        st = S.structure(B, mode)
        g = self.g.pop(st, None)
        if g is None and self.want(st) and st not in self.refused:
            from ..device import capacity
            try: g = self.capture(S, st, B, inputs, mode)
            except capacity.CapacityError as e:   # (a capture launches nothing: the chunk runs eager, the structure is not retried)
                self.refused.add(st); self.stats['refused'] += 1; g = None
                import sys; print(f'prefill_graph: structure not captured, eager: {e}', file=sys.stderr, flush=True)
        if g is None:
            self.stats['eager'] += 1
            return S._body(B, inputs, mode)
        self.g[st] = g                                      # (most recently used last)
        self.replay(S, g, B, inputs, mode)

    def capture(self, S, st, B, inputs, mode):
        """capture every segment of the chunk (nothing runs on the GPU), instantiate them, record the position regions' nodes;
        the device memory measured over the instantiations (checked against the caps after: over one, dropped at once)"""
        from ..device import capacity
        H = api(); ks = S.ks
        while len(self.g) >= self.max:                      # (the least recently used structure out first)
            old = self.g.pop(next(iter(self.g))); self._idle(S); old.destroy()
        capacity.check([d.index for d in S.devs], list(capacity.PREFILL_GRAPH_MIB), f'prefill_graph: a captured chunk structure')
        g = ChunkGraph(st)
        rec = Recorder('capture'); S.rec = rec; _hip_mod.RECORD = rec
        for d in S.devs: torch.cuda.synchronize(d)
        devs = [d.index for d in S.devs]
        own0 = proc_vram(devs); free0 = [torch.cuda.mem_get_info(d)[0] for d in S.devs]
        t0 = time.perf_counter(); tc = 0.0
        def seg(c, key, fn):
            nonlocal tc
            began = False
            with torch.cuda.device(S.devs[c]):
                rec.seg, rec.stream, rec.nlaunch = key, ks[c].cuda_stream, 0
                try:
                    check(H.hipStreamBeginCapture(c_void_p(ks[c].cuda_stream), CAPTURE_GLOBAL)); began = True
                    out = fn()
                    gr = c_void_p(); began = False
                    check(H.hipStreamEndCapture(c_void_p(ks[c].cuda_stream), ctypes.byref(gr)))
                except BaseException:
                    if began:
                        try:
                            gr = c_void_p(); H.hipStreamEndCapture(c_void_p(ks[c].cuda_stream), ctypes.byref(gr))
                            if gr: H.hipGraphDestroy(gr)
                        except Exception: pass
                    raise
                t1 = time.perf_counter()
                exe = c_void_p()
                try: check(H.hipGraphInstantiate(ctypes.byref(exe), gr, None, None, 0))
                except BaseException:
                    H.hipGraphDestroy(gr); raise
                tc += time.perf_counter() - t1
                n = c_size_t(); check(H.hipGraphGetNodes(gr, None, ctypes.byref(n))); g.nodes += n.value
                g.segs[key] = (gr, exe)
                if n.value < rec.nlaunch:   # (a call that ran eagerly instead of being captured would leave fewer nodes)
                    raise RuntimeError(f'prefill_graph: segment {key}: {n.value} graph nodes for {rec.nlaunch} kernel launches')
            return out
        try:
            for kind, il in S.halves():
                for j, (r0, r1) in enumerate(B):
                    parts = [seg(c, (c, kind, il, j), lambda c=c: S._half(c, kind, il, r0, r1 - r0, inputs[c], None)) for c in (0, 1)]
                    g.parts[(kind, il, j)] = parts
            for j, (r0, r1) in enumerate(B):
                for c in (0, 1): seg(c, (c, 'post', -1, j), lambda c=c: S._post(c, r0, r1 - r0))
        except BaseException:
            g.destroy(); raise
        finally:
            _hip_mod.RECORD = None; S.rec = None; rec.seg = rec.stream = None
            torch.cuda.set_device(S.devs[0])
        for d in S.devs: torch.cuda.synchronize(d)
        own1 = proc_vram(devs)
        g.mib = [(b - a) if a is not None and b is not None else (f - torch.cuda.mem_get_info(d)[0]) / capacity.MIB
                 for a, b, f, d in zip(own0, own1, free0, devs)]
        g.regions = rec.regions
        for r in g.regions: g.by_seg.setdefault(r['seg'], []).append(r)
        g.capture_ms, g.inst_ms = (time.perf_counter() - t0 - tc) * 1e3, tc * 1e3
        rows = capacity.caps_mib([d.index for d in S.devs])
        if any(u > cap for cap, u in rows):                 # (measured after: over a cap, the structure is dropped at once)
            g.destroy(); raise capacity.CapacityError(f'prefill_graph: the structure took {g.mib} MiB; a card is past its cap {rows}')
        if any(g.mib[c] > REFUSE_MIB for c in (0, 1)):
            # (fail closed on the measurement, not the environment string: HIP may have initialized with packet capture on before
            # ENV was set, 2 MiB per graph, ~390 MiB per structure; the runtime's own pools grow in 16 MiB steps, so one
            # structure may read +16 while the amortized cost stays within PREFILL_GRAPH_MIB)
            ns = len(g.segs); g.destroy()
            raise capacity.CapacityError(f'prefill_graph: the structure took {[round(x, 1) for x in g.mib]} MiB > {REFUSE_MIB} per card '
                                         f'({ns} graphs; HIP packet capture on?)')
        self.total = [self.total[c] + g.mib[c] for c in (0, 1)]
        self.stats['captures'] += 1
        return g

    def replay(self, S, g, B, inputs, mode):
        """the chunk from g: each card's segments launched on its compute stream in _body's order, the exchange between them as
        in an eager chunk; a segment's position regions are patched at the stack's current position right before it is
        launched, so the host patches the later segments while the cards run the earlier ones (patching all first left the
        cards idle for it: a 2048-token chunk 672 vs 636 ms eager, measured). A failure mid-chunk propagates (Prefill.run
        poisons the engine: earlier segments ran) and the structure is dropped"""
        t0 = time.perf_counter()
        H = api(); ks = S.ks
        rec = Recorder('patch', g); tp = [0.0]
        by_seg = g.by_seg
        def launch(c, kind, il, j):
            key = (c, kind, il, j)
            regs = by_seg.get(key)
            if regs:
                t1 = time.perf_counter(); S.rec = rec; _hip_mod.RECORD = rec
                try: rec.patch(regs)
                finally: _hip_mod.RECORD = None; S.rec = None; tp[0] += time.perf_counter() - t1
            check(H.hipGraphLaunch(g.segs[key][1], c_void_p(ks[c].cuda_stream)))
            return g.parts[(kind, il, j)][c] if kind != 'post' else None
        try: S._body(B, inputs, mode, launch=launch)
        except BaseException:
            # (some segments may be patched and launched, others not: the structure is never replayed again)
            self.g.pop(g.struct, None); self._idle(S); g.destroy(); raise
        g.replays += 1; g.last_patch = (rec.compared, rec.updated)
        self.stats['replays'] += 1; self.stats['patch_ms'] += tp[0] * 1e3; self.stats['launch_ms'] += (time.perf_counter() - t0 - tp[0]) * 1e3
        self.stats['updated'] += rec.updated

    def clear(self):
        if self.g: self._idle(self.S)
        for g in self.g.values(): g.destroy()
        self.g = {}

    @staticmethod
    def _idle(S):
        """both cards idle before an exec is destroyed (its launches may still be queued)"""
        for d in S.devs: torch.cuda.synchronize(d)


