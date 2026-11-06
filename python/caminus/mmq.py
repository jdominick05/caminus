# This module is a Python port of ggml's host launch code for the MMQ kernels (ggml-cuda/mmq.cu ggml_cuda_mul_mat_q and
# mmq.cuh launch_mul_mat_q: the argument layout, the J choice, the q8_1 buffer layout and padding, mul_mat_id's grouping) from
# llama.cpp (https://github.com/ggml-org/llama.cpp, through the ignite-llama fork), which is distributed under the following
# license (THIRD_PARTY.md):
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Quantized GEMMs for the batched prefill (docs/agentic/spec-prefill-cache.md S2): llama.cpp's ggml-cuda MMQ, vendored in
kernels/ggml_mmq (MIT; tools/vendor_mmq.py) and built by build.sh into build/prefill_mmq_<format>.hsaco (one per format)
and build/prefill_mmq_glue.hsaco (the activation quantizer, mul_mat_id's grouping). This module is ggml's host launch
code for them (ggml-cuda/mmq.cu ggml_cuda_mul_mat_q, mmq.cuh launch_mul_mat_q), in Python, on the K3 TP shards:

  mm(fmt, w, rows, K, y, ncols, dst)            dst[c, r] = sum_k W[r, k] * x[c, k]   (y: x quantized by quantize())
  quantize(x, K, ncols, y, layout)              x fp32 [ncols][K] -> y, ggml's block_q8_1_mmq layout ([K_pad / 128][ncols])
  mul_mat_id(...)                               the MoE projections: tokens grouped by expert on the device (mm_ids_helper:
                                                one wave per expert, a scan in token order, no host sync per expert), one
                                                MMQ launch for all groups (grid z = expert), results scattered back by slot

Numerics: the same kernels as llama.cpp (instructions equal to ignite's shipped objects: tests/test_prefill_mmq.py
codegen), launched with the same arguments, the same J rule (the fewest column tiles; mul_mat_id on RDNA3 against the mean
tokens per expert) and stream_k off (RDNA3's config: one workgroup per output tile, K in order), so an output element
depends only on its row and its column's activation, not on T, J, the tile it lands in or the chunk's expert groups
(gate G2b, measured there).

Memory: every buffer is the caller's (Workspace preallocates them for a maximum chunk; nothing here allocates per call).
Bounds, checked on the host before each launch (refused with ValueError, never clamped), against each buffer's OWN view
(caminus.bufcheck: a short view of a Workspace block is refused, not written past; the weights' over-read below is the
one deliberate read past a view), and no written buffer may overlap another buffer of the launch:
- the weights: MMQ loads K in K_vram (256) steps, so a row whose K is not a multiple of 256 (the card-0 shard of the MoE
  down projections: 384 = 1.5 x 256) is read past its end into the next row, and the last row past the tensor: up to
  overread_bytes(fmt, K) more bytes must be allocated (and finite: they are multiplied by the activations' zero padding).
  ggml pads every quantized tensor for this (ggml-cuda.cu get_alloc_size: MATRIX_ROW_PADDING, zeroed in init_tensor:
  pad_bytes); caminus.flashnext_tp gives its K-split shards a zeroed tail of exactly overread_bytes(fmt, K)
  (FlashNextTP.ksplit_pad: only the K = 384 shards have one);
- the activations: K padded to MATRIX_ROW_PADDING (512) with zeros, and Y_TAIL bytes after the last column (a J-wide
  column tile reads up to J - 1 columns past the end: ggml's src1_q8_1_padding);
- rows a multiple of 128 (the fallback=false kernels have no row bound checks: ggml_cuda_mmq_needs_fallback);
- mul_mat_id's ids_dst J - 1 entries past T * k (read unconditionally by the last group's tile).
Routing ids (group): ggml's mm_ids_helper (vendored verbatim) assumes every id in [0, n_exp) and distinct within its token;
otherwise ids_src1 / ids_dst entries are never written and the groups' bounds count columns nobody fills, so the
quantizer and MMQ after it index with stale words (out-of-bounds writes: tests/test_prefill_safety.py's CPU emulation).
group() therefore always runs k3_mmq_ids_clamp first, on the same stream: every id clamped into range and made distinct
on the device (in place; valid ids are not written), the sticky flag word err |= ROUTE_RANGE / ROUTE_DUP
(caminus.prefill_ops), which the host checks (prefill_ops.check_routing) before it uses the chunk."""
import ctypes, os, re, struct
import torch
from . import flashnext   # noqa: F401  (puts gguf-py on the path)
from .bufcheck import disjoint, view_bytes
from .hip import BUILD, Kernel, Module, check, hip
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as _Q

FORMATS = ('IQ2_S', 'IQ3_S', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS', 'Q2_0', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0')
TYPE_ID = {f: int(_Q[f]) for f in FORMATS}                      # ggml_type
BLOCK = {f: GGML_QUANT_SIZES[_Q[f]] for f in FORMATS}           # (values, bytes) per block
MATRIX_ROW_PADDING = 512                                        # ggml-cuda common.cuh
QK8_1_MMQ, Y_BLOCK = 128, 144                                   # block_q8_1_mmq: 128 int8 + 4 x 4 B of scales / sums
QUANT_BLOCK = 128                                               # CUDA_QUANTIZE_BLOCK_SIZE_MMQ
J_MAX = 128
Y_TAIL = 18432                                                  # >= ggml's src1_q8_1_padding for every J <= 128 (J * 144
                                                                # rounded up to nthreads * 4; 128 * 144 = 18 x 1024)
SMPBO = 65536                                                   # gfx1100 LDS per workgroup (ggml's smpbo)
D4, DS4 = 0, 1                                                  # mmq_q8_1_ds_layout (D2S6: Q2_K only, not built)
CFG_WORDS = 12
K_VRAM = 256
IDS_KMAX = 16                                                   # k3_mmq_ids_clamp's K3_IDS_KMAX


class _U3(ctypes.Structure):
    _fields_ = [('x', ctypes.c_uint32), ('y', ctypes.c_uint32), ('z', ctypes.c_uint32)]


def fastdiv(d):
    """ggml-cuda common.cuh init_fastdiv_values: <mp, L, d> for the kernels' fastdiv / fastmodulo"""
    d = int(d); assert 0 < d < (1 << 32), d
    L = 0
    while L < 32 and (1 << L) < d: L += 1
    mp = ((1 << 32) * ((1 << L) - d) // d + 1) & 0xFFFFFFFF
    return _U3(mp, L, d)


def k_pad(K):
    return (int(K) + MATRIX_ROW_PADDING - 1) // MATRIX_ROW_PADDING * MATRIX_ROW_PADDING


def y_bytes(K, ncols):
    """bytes of a quantized activation buffer for ncols columns of K values (block_q8_1_mmq, K padded to 512, + Y_TAIL)"""
    return k_pad(K) // QK8_1_MMQ * int(ncols) * Y_BLOCK + Y_TAIL


def row_bytes(fmt, K):
    qk, ts = BLOCK[fmt]
    if K % qk: raise ValueError(f'mmq: K {K} is not a multiple of {fmt} blocks ({qk})')
    return K // qk * ts


def overread_bytes(fmt, K):
    """bytes MMQ reads past the end of the last row of a [rows][K] matrix of fmt: the last K_vram load of a row covers
    ceil(K / 256) * 256 values"""
    qk, ts = BLOCK[fmt]; nb, bpi = K // qk, K_VRAM // qk
    return ((nb + bpi - 1) // bpi * bpi - nb) * ts


def pad_bytes(fmt, K):
    """the zeroed tail a [rows][K] tensor of fmt gets so MMQ may read it (>= overread_bytes; ggml's rule: the row padded to
    a multiple of MATRIX_ROW_PADDING values)"""
    qk, ts = BLOCK[fmt]
    r = (MATRIX_ROW_PADDING - K % MATRIX_ROW_PADDING) % MATRIX_ROW_PADDING
    p = r // qk * ts
    assert p >= overread_bytes(fmt, K), (fmt, K)
    return p


def extent(t):
    """bytes from t's first byte to the end of its STORAGE: only for the weights' over-read (MMQ reads past the last
    row of a K = 384 shard into the zeroed tail beyond its view). Every other buffer is checked against its view
    (bufcheck.view_bytes): a short view of a Workspace block must be refused, not written past"""
    s = t.untyped_storage()
    return s.nbytes() - (t.data_ptr() - s.data_ptr())


def _fn(mod, name):
    """a kernel of a loaded module by its (mangled) symbol name (hip.Module's attribute access refuses names starting
    with '_', which every mangled C++ name does)"""
    f = ctypes.c_void_p()
    check(hip().hipModuleGetFunction(ctypes.byref(f), mod.handle, name.encode()))
    return Kernel(f, name)


def _elf_funcs(path):
    """the FUNC symbol names of an ELF64 code object (.symtab / .dynsym)"""
    b = open(path, 'rb').read()
    assert b[:4] == b'\x7fELF' and b[4] == 2 and b[5] == 1, path
    shoff, = struct.unpack_from('<Q', b, 0x28); shentsize, shnum = struct.unpack_from('<HH', b, 0x3A)
    secs = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
    names = set()
    for s in secs:
        if s[1] not in (2, 11): continue                         # SHT_SYMTAB, SHT_DYNSYM
        stro = secs[s[6]][4]; off, size, ent = s[4], s[5], s[9]
        for k in range(size // ent):
            nm, info = struct.unpack_from('<IB', b, off + k * ent)
            if info & 0xF == 2:                                  # STT_FUNC
                e = b.index(b'\0', stro + nm); names.add(b[stro + nm:e].decode())
    return names


class MMQ:
    """the prefill GEMM kernels on one device; every launch runs with that device current (its module's kernels, its
    current stream, or the stream given)"""
    _RE_MM = re.compile(r'^_ZL9mul_mat_qIL9ggml_type(\d+)ELi(\d+)ELb([01])EL9ggml_prec(\d+)EE')
    _RE_Q = re.compile(r'^_ZL17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout(\d)ELb([01])EE')
    _RE_IDS = re.compile(r'^_ZL13mm_ids_helperILi(\d+)EE')

    def __init__(self, device, formats=FORMATS):
        self.dev = torch.device(device)
        assert self.dev.type == 'cuda', device
        self.cfg, self.kmm, self.mods = {}, {}, {}
        with torch.cuda.device(self.dev):
            name = 'prefill_mmq_glue'; self.glue = Module(name); self.mods[name] = self.glue
            self.kq, self.kids = {}, {}
            for s in _elf_funcs(os.path.join(BUILD, name + '.hsaco')):
                m = self._RE_Q.match(s)
                if m: self.kq[(int(m.group(1)), m.group(2) == '1')] = _fn(self.glue, s)
                m = self._RE_IDS.match(s)
                if m: self.kids[int(m.group(1))] = _fn(self.glue, s)
            assert set(self.kq) == {(D4, False), (D4, True), (DS4, False), (DS4, True)} and 10 in self.kids, (self.kq, self.kids)
            self.kclamp = self.glue.k3_mmq_ids_clamp
            for f in formats: self._load(f)

    def _load(self, fmt):
        name = f'prefill_mmq_{fmt.lower()}'
        with torch.cuda.device(self.dev):
            mod = Module(name); self.mods[name] = mod
            out = torch.zeros(64 * CFG_WORDS, dtype=torch.int32, device=self.dev); cnt = torch.zeros(1, dtype=torch.int32, device=self.dev)
            mod.k3_mmq_configs((1,), (1,), out, cnt); torch.cuda.synchronize(self.dev)
            n = int(cnt.item()); rows = out[:n * CFG_WORDS].view(n, CFG_WORDS).tolist()
            syms = {}
            for s in _elf_funcs(os.path.join(BUILD, name + '.hsaco')):
                m = self._RE_MM.match(s)
                if m: syms[(int(m.group(1)), int(m.group(2)), m.group(3) == '1', int(m.group(4)))] = s
            for r in rows:
                t, J, fb, nthreads, occ, I, stride, kv, sk, nbytes, qk, lay = r
                assert t == TYPE_ID[fmt] and fb == 0 and sk == 0 and kv == K_VRAM and qk == BLOCK[fmt][0], (fmt, r)
                assert nbytes <= SMPBO, (fmt, J, nbytes)
                sym = syms.get((t, J, False, 30))                  # GGML_PREC_Q8 = 30
                assert sym is not None, f'{name}: no kernel for J={J}'
                self.cfg[(fmt, J)] = dict(nthreads=nthreads, I=I, nbytes=nbytes, layout=lay)
                self.kmm[(fmt, J)] = _fn(mod, sym)
            assert len(rows) == len(syms), (fmt, len(rows), len(syms))

    def _launch(self, k, *a, **kw):
        with torch.cuda.device(self.dev): k(*a, **kw)

    def layout(self, fmt):
        return self.cfg[(fmt, 16)]['layout']

    def choose_J(self, fmt, ncols_opt):
        """llama.cpp's rule (mmq.cu ggml_cuda_mul_mat_q): the J with the fewest column tiles, the smallest on a tie,
        stopping at one tile"""
        best, nt_best = 0, 1 << 62
        for J in range(8, J_MAX + 1, 8):
            if nt_best <= 1: break
            c = self.cfg.get((fmt, J))
            if c is None or c['nbytes'] > SMPBO: continue
            nt = (int(ncols_opt) + J - 1) // J
            if nt < nt_best: best, nt_best = J, nt
        assert best, (fmt, ncols_opt)
        return best

    # -- the activation quantizer -------------------------------------------------------------------------------------------
    def quantize(self, x, K, ncols, y, layout, s01=None, ids=None, stream=None):
        """quantize_mmq_q8_1: column c of y = x row (ids[c] if ids else c) (x fp32, row stride s01 floats, default K); y in
        ggml's [K_pad / 128][ncols] block_q8_1_mmq layout, the K padding zeros"""
        K, ncols = int(K), int(ncols); s01 = K if s01 is None else int(s01)
        self._chk_f32(x, 'x'); self._chk_dev(y, 'y')
        if K % 4 or ncols < 1 or s01 < K: raise ValueError(f'mmq.quantize: K {K} (a multiple of 4), ncols {ncols}, s01 {s01}')
        if view_bytes(y) < y_bytes(K, ncols): raise ValueError(f'mmq.quantize: y holds {view_bytes(y)} B < {y_bytes(K, ncols)}')
        # with ids (group's forward map: values < ncols by construction, the ids clamped first) x needs ncols rows as well
        xb = 4 * ((ncols - 1) * s01 + K)
        if view_bytes(x) < xb: raise ValueError(f'mmq.quantize: x holds {view_bytes(x)} B < {ncols} rows ({xb} B)')
        if ids is not None: self._chk_i32(ids, ncols, 'ids')
        disjoint('mmq.quantize', dict(x=(x, xb), ids=(ids, 4 * ncols), y=(y, y_bytes(K, ncols))), ('y',))
        Kp = k_pad(K)
        self._launch(self.kq[(layout, False)], (ncols, (Kp + 4 * QUANT_BLOCK - 1) // (4 * QUANT_BLOCK), 1), (QUANT_BLOCK,), x,
                                 ids if ids is not None else ctypes.c_void_p(0), y, ctypes.c_int64(K), ctypes.c_int64(s01),
                                 ctypes.c_int64(0), ctypes.c_int64(0), ctypes.c_int64(Kp), ctypes.c_int(ncols), ctypes.c_int(1),
                                 ctypes.c_int(0), stream=stream)

    def quantize_scatter(self, x, K, ntok, ids_inv, k, y, layout, s_tok=None, stream=None):
        """quantize_mmq_q8_1 (scatter): each token's row of x quantized once and written to its k compact columns
        ids_inv[t * k + slot] of y ([K_pad / 128][ntok * k])"""
        K, ntok, k = int(K), int(ntok), int(k); s_tok = K if s_tok is None else int(s_tok)
        self._chk_f32(x, 'x'); self._chk_dev(y, 'y'); self._chk_i32(ids_inv, ntok * k, 'ids_inv')
        if K % 4 or ntok < 1 or k < 1 or s_tok < K: raise ValueError(f'mmq.quantize_scatter: K {K}, ntok {ntok}, k {k}, s_tok {s_tok}')
        xb = 4 * ((ntok - 1) * s_tok + K)
        if view_bytes(x) < xb: raise ValueError(f'mmq.quantize_scatter: x holds {view_bytes(x)} B < {ntok} rows ({xb} B)')
        if view_bytes(y) < y_bytes(K, ntok * k): raise ValueError(f'mmq.quantize_scatter: y holds {view_bytes(y)} B < {y_bytes(K, ntok * k)}')
        disjoint('mmq.quantize_scatter', dict(x=(x, xb), ids_inv=(ids_inv, 4 * ntok * k), y=(y, y_bytes(K, ntok * k))), ('y',))
        Kp = k_pad(K)
        self._launch(self.kq[(layout, True)], (ntok, (Kp + 4 * QUANT_BLOCK - 1) // (4 * QUANT_BLOCK), 1), (QUANT_BLOCK,), x, ids_inv, y,
                                ctypes.c_int64(K), ctypes.c_int64(0), ctypes.c_int64(s_tok), ctypes.c_int64(0), ctypes.c_int64(Kp),
                                ctypes.c_int(ntok * k), ctypes.c_int(1), ctypes.c_int(k), stream=stream)

    # -- dense ---------------------------------------------------------------------------------------------------------------
    def mm(self, fmt, w, rows, K, y, ncols, dst, J=None, stream=None):
        """dst[c * rows + r] = sum_k W[r, k] x[c, k] for c < ncols (W: [rows][K] of fmt, raw GGUF rows; y: quantize() of x
        in this format's layout). J: the column tile (default: llama.cpp's choice for ncols)"""
        rows, K, ncols = int(rows), int(K), int(ncols)
        J = self.choose_J(fmt, ncols) if J is None else int(J)
        c = self._chk_mm(fmt, w, rows, K, 1, J)
        if ncols < 1: raise ValueError(f'mmq.mm: ncols {ncols}')
        self._chk_dev(y, 'y')
        if view_bytes(y) < y_bytes(K, ncols): raise ValueError(f'mmq.mm: y holds {view_bytes(y)} B < {y_bytes(K, ncols)} ({ncols} columns)')
        self._chk_f32(dst, 'dst')
        if view_bytes(dst) < 4 * rows * ncols: raise ValueError(f'mmq.mm: dst holds {view_bytes(dst)} B < {rows} x {ncols} fp32')
        disjoint('mmq.mm', dict(w=(w, rows * row_bytes(fmt, K) + overread_bytes(fmt, K)), y=(y, y_bytes(K, ncols)), dst=(dst, 4 * rows * ncols)), ('dst',))
        qk = BLOCK[fmt][0]; ntx = (ncols + J - 1) // J
        s12 = ncols * k_pad(K) * 36 // (32 * 4)
        self._launch(self.kmm[(fmt, J)], (rows // c['I'], ntx, 1), (32, c['nthreads'] // 32), w, y, ctypes.c_void_p(0), ctypes.c_void_p(0), dst,
                           ctypes.c_void_p(0), ctypes.c_void_p(0), fastdiv(K // qk), ctypes.c_int(rows), ctypes.c_int(ncols),
                           ctypes.c_int(K // qk), ctypes.c_int(ncols), ctypes.c_int(rows),
                           fastdiv(1), fastdiv(1), ctypes.c_int(rows * K // qk), ctypes.c_int(s12), ctypes.c_int(ncols * rows),
                           fastdiv(1), fastdiv(1), ctypes.c_int(rows * K // qk), ctypes.c_int(s12), ctypes.c_int(ncols * rows),
                           fastdiv(ntx), shmem=c['nbytes'], stream=stream)
        return J

    # -- mul_mat_id ---------------------------------------------------------------------------------------------------------
    def group(self, ids, ntok, k, n_exp, ids_src1, ids_dst, bounds, per_slot, err, stream=None):
        """mm_ids_helper: the router's choices ids [ntok][k] (int32 expert ids) -> expert_bounds [n_exp + 1] (group e is
        compact columns bounds[e] .. bounds[e + 1] - 1, in token order), ids_dst[compact] = t * k + slot (the output row),
        and ids_src1: per_slot False (gate / up: one activation row per token) the inverse map [t * k + slot] -> compact
        column (for quantize_scatter); per_slot True (down: one activation row per slot) [compact] -> t * k + slot.
        First, on the same stream, k3_mmq_ids_clamp: ids clamped IN PLACE into [0, n_exp) and made distinct per token (a
        valid id is never changed or written), err (int32 [>= 1], the sticky routing flag word) |= ROUTE_RANGE / ROUTE_DUP
        for any id it had to replace; mm_ids_helper only ever sees the clamped ids, so every table entry the later launches
        read is written. The caller checks err (prefill_ops.check_routing) before it uses the chunk"""
        ntok, k, n_exp = int(ntok), int(k), int(n_exp)
        if k != 10: raise ValueError(f'mmq.group: k {k} (mm_ids_helper is built for the Coder\'s 10)')
        if not 0 < ntok < (1 << 22): raise ValueError(f'mmq.group: ntok {ntok}')
        if not k <= n_exp < (1 << 16) or k > IDS_KMAX: raise ValueError(f'mmq.group: n_exp {n_exp}, k {k} (k <= n_exp, k <= {IDS_KMAX})')
        if ntok * 4 > SMPBO: raise ValueError(f'mmq.group: ntok {ntok} needs {ntok * 4} B of LDS > {SMPBO}')
        self._chk_i32(ids, ntok * k, 'ids'); self._chk_i32(ids_src1, ntok * k, 'ids_src1')
        self._chk_i32(ids_dst, ntok * k + J_MAX - 1, 'ids_dst'); self._chk_i32(bounds, n_exp + 1, 'bounds'); self._chk_i32(err, 1, 'err')
        disjoint('mmq.group', dict(ids=(ids, 4 * ntok * k), ids_src1=(ids_src1, 4 * ntok * k), ids_dst=(ids_dst, 4 * (ntok * k + J_MAX - 1)),
                                   bounds=(bounds, 4 * (n_exp + 1)), err=(err, 4)), ('ids', 'ids_src1', 'ids_dst', 'bounds', 'err'))
        self._launch(self.kclamp, ((ntok + 255) // 256,), (256,), ids, ctypes.c_int(ntok), ctypes.c_int(k), ctypes.c_int(n_exp), err, stream=stream)
        self._launch(self.kids[10], (n_exp,), (32,), ids, ids_src1, ids_dst, bounds, ctypes.c_int(ntok), ctypes.c_int(k),
                      ctypes.c_int(k if per_slot else 1), ctypes.c_int(k), ctypes.c_int(k if per_slot else 1),
                      ctypes.c_bool(not per_slot), shmem=ntok * 4, stream=stream)

    def mm_id(self, fmt, w, n_exp, rows, K, y, ntok, k, ids_dst, bounds, dst, J=None, stream=None):
        """the grouped GEMM: for each compact column j of expert e (bounds), dst[ids_dst[j] * rows + r] = sum_k W_e[r, k] y_j
        (W: [n_exp][rows][K] of fmt; y: the compact quantized activations, quantize_scatter / quantize with ids_src1).
        J default: llama.cpp's RDNA3 choice for mul_mat_id, against ceil(ntok * k / n_exp) columns per expert"""
        rows, K, ntok, k, n_exp = int(rows), int(K), int(ntok), int(k), int(n_exp)
        J = self.choose_J(fmt, (ntok * k + n_exp - 1) // n_exp) if J is None else int(J)
        c = self._chk_mm(fmt, w, rows, K, n_exp, J)
        if ntok < 1 or k < 1: raise ValueError(f'mmq.mm_id: ntok {ntok}, k {k}')
        self._chk_dev(y, 'y')
        if view_bytes(y) < y_bytes(K, ntok * k): raise ValueError(f'mmq.mm_id: y holds {view_bytes(y)} B < {y_bytes(K, ntok * k)}')
        self._chk_i32(ids_dst, ntok * k + J - 1, 'ids_dst'); self._chk_i32(bounds, n_exp + 1, 'bounds')
        self._chk_f32(dst, 'dst')
        if view_bytes(dst) < 4 * rows * ntok * k: raise ValueError(f'mmq.mm_id: dst holds {view_bytes(dst)} B < {rows} x {ntok * k} fp32')
        disjoint('mmq.mm_id', dict(w=(w, n_exp * rows * row_bytes(fmt, K) + overread_bytes(fmt, K)), y=(y, y_bytes(K, ntok * k)),
                                   ids_dst=(ids_dst, 4 * (ntok * k + J - 1)), bounds=(bounds, 4 * (n_exp + 1)), dst=(dst, 4 * rows * ntok * k)), ('dst',))
        qk = BLOCK[fmt][0]; ntx = (ntok + J - 1) // J; n = ntok * k
        s12 = 1 * k_pad(K) * 36 // (32 * 4)
        self._launch(self.kmm[(fmt, J)], (rows // c['I'], ntx, n_exp), (32, c['nthreads'] // 32), w, y, ids_dst, bounds, dst, ctypes.c_void_p(0),
                           ctypes.c_void_p(0), fastdiv(K // qk), ctypes.c_int(rows), ctypes.c_int(n), ctypes.c_int(K // qk),
                           ctypes.c_int(n), ctypes.c_int(rows),
                           fastdiv(1), fastdiv(n_exp), ctypes.c_int(rows * K // qk), ctypes.c_int(s12), ctypes.c_int(k * rows),
                           fastdiv(1), fastdiv(1), ctypes.c_int(n_exp * rows * K // qk), ctypes.c_int(ntok * s12), ctypes.c_int(n * rows),
                           fastdiv(ntx), shmem=c['nbytes'], stream=stream)
        return J

    # -- checks -------------------------------------------------------------------------------------------------------------
    def _chk_dev(self, t, what):
        if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
            raise ValueError(f'mmq: {what} must be a contiguous tensor on {self.dev}')

    def _chk_f32(self, t, what):
        self._chk_dev(t, what)
        if t.dtype != torch.float32: raise ValueError(f'mmq: {what} must be float32')

    def _chk_i32(self, t, n, what):
        self._chk_dev(t, what)
        if t.dtype != torch.int32 or t.numel() < n: raise ValueError(f'mmq: {what} must be int32 with >= {n} entries')

    def _chk_mm(self, fmt, w, rows, K, n_exp, J):
        c = self.cfg.get((fmt, J))
        if c is None: raise ValueError(f'mmq: no {fmt} kernel for J={J}')
        self._chk_dev(w, 'w')
        if w.dtype != torch.uint8: raise ValueError('mmq: w must be the raw uint8 GGUF rows')
        if rows < 1 or rows % 128: raise ValueError(f'mmq: rows {rows} must be a multiple of 128 (no fallback kernels)')
        rb = row_bytes(fmt, K); need = n_exp * rows * rb + overread_bytes(fmt, K)
        if w.numel() < n_exp * rows * rb: raise ValueError(f'mmq: w has {w.numel()} B < {n_exp} x {rows} x {rb}')
        if extent(w) < need:
            raise ValueError(f'mmq: {fmt} K={K}: MMQ reads {overread_bytes(fmt, K)} B past the last row; w\'s allocation ends '
                             f'{extent(w) - n_exp * rows * rb} B after it (pad the tensor: pad_bytes)')
        return c


# -- the prefill GEMM workspace -----------------------------------------------------------------------------------------
ALIGN = 256


def card_dims(geom, card, gdn_keys=10, moe_blocks=12):
    """the per-card GEMM shapes of FlashNextTP's split (flashnext_tp._build_tapes): GDN by key heads (gdn_keys on card 0
    of the 16), attention one kv head per card, every expert's intermediate split at moe_blocks x 32 (card 0's share)"""
    kc = gdn_keys if card == 0 else geom['ssm_groups'] - gdn_keys
    per_v = geom['ssm_heads'] // geom['ssm_groups']                     # value heads per key head (3)
    sd = geom['ssm_state']                                               # 128
    nbt = geom['ffn'] // 32; ffc = 32 * (moe_blocks if card == 0 else nbt - moe_blocks)
    nh = geom['nh'] // geom['nkv']                                       # query heads per card (one kv head each)
    return dict(D=geom['D'], qkv=(2 + per_v) * sd * kc, z=per_v * sd * kc, ssm_K=per_v * sd * kc, q=2 * nh * geom['hd'],
                kv=geom['hd'], ao_K=nh * geom['hd'], ffc=ffc, k=geom['k_used'], n_exp=geom['n_exp'])


MOE_J_MIN, MOE_TILE_BYTES = 16, 16   # caminus.moe's smallest column tile and its tile list entry (int4)


def moe_tiles(ncols, n_exp):
    """caminus.moe.max_tiles at its smallest column tile: the longest tile list a chunk of ncols compact columns can have"""
    return int(n_exp) + (int(ncols) + MOE_J_MIN - 1) // MOE_J_MIN


WMMA_ROWS = 128   # gemm.hip's token tile (prefill_ops.TPAD): a WMMA GEMM writes whole 128-row tiles


def workspace_layout(T, d):
    """name -> bytes of one card's prefill GEMM workspace for chunks of up to T tokens (d: card_dims): the quantized
    activations (dense, both layouts; the MoE gate / up columns and the down projection's per-slot columns), mul_mat_id's
    grouping tables and the routing flag word (err), every GEMM output of a layer (the attention half's and the FFN
    half's, not shared: the largest is the routed down projection's per-slot partials [T, k, D] fp32), caminus.moe's two tile
    lists and their counts (moe_ta / moe_tb / moe_nt: the grouping writes them, the fused gate / up and the down kernel read
    them; o_gate / o_up / x_moe_down are used by the MMQ path only) and the fp32 inputs
    of the four K-side projections, which no other workspace holds (x_ssm_out [T, ssm_K]: the GDN output; x_attn_out
    [T, ao_K]: the attention output; x_sh_down [T, ffc] / x_moe_down [T, k, ffc]: the shared / routed experts' swiglu;
    the D-wide input is the glue workspace's `mixed`). Until impl:prefill-safety (11-04) S2's tp test allocated those four
    as test tensors outside every count (+82 MiB measured on the XT at chunk 2048); the chunk loop (S4 / S6 / S7) writes
    them, and may alias them onto outputs it no longer needs (then they leave this layout and the count). The dense GEMMs'
    outputs (o_qkv .. o_sh_down) hold whole 128-row WMMA tiles (tpad(T) rows: the same bytes at T a multiple of 128), since
    ChunkStack(q6k='wmma') writes them on gemm.hip (its other buffers: caminus.dense.Bf16Workspace). Each buffer is
    ALIGN-aligned in one allocation"""
    T, k = int(T), d['k']
    Tw = -(-T // WMMA_ROWS) * WMMA_ROWS                 # the dense outputs at whole WMMA row tiles (ChunkStack(q6k='wmma'))
    kmax = max(d['D'], d['ssm_K'], d['ao_K'])
    L = dict(y_d4=y_bytes(kmax, T), y_ds4=y_bytes(kmax, T), y_moe=y_bytes(d['D'], T * k), y_down=y_bytes(d['ffc'], T * k),
             ids_inv=4 * T * k, ids_fwd=4 * T * k, ids_dst=4 * (T * k + J_MAX - 1), bounds=4 * (d['n_exp'] + 1), err=4,
             o_qkv=4 * Tw * d['qkv'], o_z=4 * Tw * d['z'], o_q=4 * Tw * d['q'], o_k=4 * Tw * d['kv'], o_v=4 * Tw * d['kv'],
             o_blk=4 * Tw * d['D'], o_sh_gate=4 * Tw * d['ffc'], o_sh_up=4 * Tw * d['ffc'], o_sh_down=4 * Tw * d['D'],
             o_gate=4 * T * k * d['ffc'], o_up=4 * T * k * d['ffc'], o_down=4 * T * k * d['D'],
             x_ssm_out=4 * T * d['ssm_K'], x_attn_out=4 * T * d['ao_K'], x_sh_down=4 * T * d['ffc'], x_moe_down=4 * T * k * d['ffc'],
             moe_ta=MOE_TILE_BYTES * moe_tiles(T * k, d['n_exp']), moe_tb=MOE_TILE_BYTES * moe_tiles(T * k, d['n_exp']), moe_nt=8)
    return {n: (b + ALIGN - 1) // ALIGN * ALIGN for n, b in L.items()}


def workspace_bytes(T, d):
    return sum(workspace_layout(T, d).values())


class Workspace:
    """one card's preallocated prefill GEMM buffers for chunks of up to T tokens: a single uint8 allocation, carved by
    workspace_layout (attributes .y_d4, .o_down, ... as uint8 / int32 / fp32 views). Checked against the card's cap
    (capacity.check) before it allocates; nothing on the prefill path allocates after it. Zeroed once: every table
    mm_ids_helper fills (ids_inv / ids_fwd / ids_dst / bounds) starts at 0, so a word it does not write in some chunk
    (ids_dst's J - 1 padding, read by MMQ's last tile and never used) is 0 or an earlier chunk's in-range value, never
    garbage; with the clamped ids it writes every other entry. err: the routing flag word (check_routing)"""
    def __init__(self, device, T, dims, check=True):
        import torch
        from . import capacity
        self.dev, self.T, self.dims = torch.device(device), int(T), dict(dims)
        self.layout = workspace_layout(self.T, self.dims); self.nbytes = sum(self.layout.values())
        if check: capacity.check([self.dev.index], [self.nbytes / capacity.MIB], f'prefill workspace (T={self.T})')
        self.block = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.dev)
        off = 0
        for n, b in self.layout.items():
            v = self.block[off:off + b]
            if n.startswith('ids') or n.startswith('moe_') or n in ('bounds', 'err'): v = v.view(torch.int32)
            elif n.startswith('o_') or n.startswith('x_'): v = v.view(torch.float32)
            setattr(self, n, v); off += b
        self.err = self.err[:1]                                           # (its 256-byte slot: one flag word)

    def check_routing(self, what='prefill routing'):
        """raise prefill_ops.RoutingError if this card's routing flag word is set (the clamp replaced an id)"""
        from .prefill_ops import check_routing
        return check_routing(self.err, what=what)
