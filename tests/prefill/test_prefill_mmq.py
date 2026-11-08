"""Gates of the batched prefill's quantized GEMMs (docs/specs/spec-prefill-cache.md S2; caminus.prefill.mmq, kernels/prefill_mmq.hip,
the vendored ggml MMQ in kernels/ggml_mmq).

  python -I tests/prefill/test_prefill_mmq.py CODER STAGE... [--scratch DIR] [--ref BIN] [--ref-stock BIN] [--json OUT]

Stages (several may be given; 'gpu' = g1 g2 speed, on cuda:0: HIP_VISIBLE_DEVICES picks the card):
  codegen  CPU: every kernel of build/prefill_mmq_*.hsaco has the instructions of ignite's shipped one (build-hip-therock's
           mmq-instance-<format>.cu.o, quantize.cu.o, mmid.cu.o, unbundled; PC-relative literals masked)
  shards   CPU: the TP split of every quantized GEMM tensor (flashnext_tp: gdn_keys 10, moe_blocks 12): column split (each
           card owns whole output rows) or K split (partial sums), the K-split shards' K ranges on their format's blocks,
           the rows a multiple of 128, MMQ's over-read past a shard (caminus.prefill.mmq.overread_bytes)
  g1       GPU (cuda:0; HIP_VISIBLE_DEVICES=1 = the XT): real Coder tensors, one of each (kind, format), T = 2048:
           column split: each card's row shard == llama.cpp's MMQ on the full tensor, those rows, bitwise;
           K split: each card's partial == llama.cpp's MMQ on that card's K shard (a standalone tensor), and card 0 +
           card 1 (OP_XCHG's order) == the same sum of llama.cpp's partials, bitwise; max relative error of the sum vs
           llama.cpp's full-K MMQ (reported; not bitwise by construction);
           mul_mat_id (gate / up / down experts on each card's shard, skewed top-10 routing): == llama.cpp's mul_mat_id on
           the same shard, bitwise, and == every expert's rows through the plain MMQ, bitwise
  g2       GPU: G2a two runs bitwise; G2b the same 2048 tokens at T = 2048 / 1024 / 512 / 777 + 1271 / 1000 + 1048 /
           512 + 788 + 748 (each chunk quantized and, for mul_mat_id, grouped on its own) bitwise per position, and one
           column set through every compiled J bitwise
  speed    GPU: per op (quantize + MMQ, mul_mat_id with its grouping) at T = 2048 per format and per-card shape: ms and
           TFLOPS (median of 20), and llama.cpp's own time for the op (--ref-stock: its real kernel choice; --ref: MMQ)
  xcard    both cards: the same inputs on the XTX and the XT give the same bits (dense Q6_K / Q4_K, a K-split Q6_K, a gate
           mul_mat_id, the K=384 down mul_mat_id)
  tp       both cards, the loaded model: FlashNextTP (T=4, --max-ctx, DFlash as the smoke builds it) with the prefill counted
           in its VRAM check (vram_prefill=--chunk), STEP 0 on the loaded shards (FlashNextTP.shards: every K-split shard on
           its blocks, covering MMQ's over-read, zero tail; G1 tensors' shard bytes == the GGUF slices), the prefill
           buffers (mmq.Workspace per card + the two-card feed) and every quantized GEMM of the 48 layers on K3's own
           shards at T = --chunk, both cards at once and each alone (ms per chunk); device-level VRAM at each step (the
           prefill buffers include S3's glue workspace, caminus.prefill.glue.GlueWorkspace; the GEMMs' inputs live in the
           workspaces, nothing outside the counted buffers), per part against its count (workspace, feed, glue, the
           prefill modules), and the check: idle + the engine's projection (FlashNextTP.vram_plan, at its own split) >=
           the measured device-level peak, on both cards.
           --alloc-only: no GEMM stack (the 262,144 allocation check)
--ref: gg_mmq_ref linked against a GGML_CUDA_FORCE_MMQ=ON fork build (MMQ at every T: tools/reference/build_gg_mmq_ref.sh; default
$ARTIFACTS/impl-prefill-s2/ggml-fmmq/gg_mmq_ref, built from 842653396 (published as e4e3f8eda; identical ggml/, common/, src/) with its libraries
beside it);
--ref-stock: against build-hip-therock (default build/gg_mmq_ref). --scratch: where the reference's input / output files and
the unbundled code objects go (default $ARTIFACTS/impl-prefill-s2/scratch; fixed names, emptied after each reference run, so
reruns do not accumulate; ~15 GB during g1). Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.engine import flashnext
from caminus.prefill import mmq   # noqa: E402  (flashnext puts gguf-py on the path)
from gguf import GGUFReader   # noqa: E402
from gguf.constants import GGML_QUANT_SIZES   # noqa: E402

TR = os.path.join(os.environ.get('ROCM_PATH') or os.path.expanduser('~/rocm-therock/10.2.0a20261011'), 'lib', 'llvm', 'bin', '')   # as build.sh
IGN = os.environ.get('IGNITE') or os.path.expanduser('~/src/ignite-llama')                                # as build.sh
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
OBJ = os.path.join(IGN, 'build-hip-therock/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda')
FAILS, NUMBERS = [], {}


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


# ---- the TP split (flashnext_tp._build_tapes, gdn_keys=10, moe_blocks=12, 12 query heads / 1 kv head per card) --------
GK, MB, NH, HD = 10, 12, 12, 256
KC, KS = [GK, 16 - GK], [0, GK]
VRUNS = [[(16 * g + KS[c], 16 * g + KS[c] + KC[c]) for g in range(3)] for c in (0, 1)]
MBR = [(0, MB), (MB, 20)]


def split_of(kind):
    """('rows' | 'cols', per-card ranges in the original tensor: rows, or K values)"""
    if kind == 'attn_qkv':
        return 'rows', [[(128 * KS[c], 128 * (KS[c] + KC[c])), (2048 + 128 * KS[c], 2048 + 128 * (KS[c] + KC[c]))] +
                        [(4096 + 128 * a, 4096 + 128 * b) for a, b in VRUNS[c]] for c in (0, 1)]
    if kind == 'attn_gate': return 'rows', [[(128 * a, 128 * b) for a, b in VRUNS[c]] for c in (0, 1)]
    if kind == 'attn_q': return 'rows', [[(NH * 2 * HD * c, NH * 2 * HD * (c + 1))] for c in (0, 1)]
    if kind in ('attn_k', 'attn_v'): return 'rows', [[(HD * c, HD * (c + 1))] for c in (0, 1)]
    if kind in ('ffn_gate_shexp', 'ffn_up_shexp', 'ffn_gate_exps', 'ffn_up_exps'): return 'rows', [[(32 * a, 32 * b)] for a, b in MBR]
    if kind == 'ssm_out': return 'cols', [[(128 * a, 128 * b) for a, b in VRUNS[c]] for c in (0, 1)]
    if kind == 'attn_output': return 'cols', [[(NH * HD * c, NH * HD * (c + 1))] for c in (0, 1)]
    if kind in ('ffn_down_shexp', 'ffn_down_exps'): return 'cols', [[(32 * a, 32 * b)] for a, b in MBR]
    return None, None


def gemm_tensors(rd):
    """{(kind, fmt): (name, tensor)} first layer of each kind and format; and the full list"""
    first, allt = {}, []
    for t in rd.tensors:
        m = re.match(r'blk\.(\d+)\.(\w+)\.weight$', t.name)
        if not m or t.tensor_type.name not in mmq.FORMATS: continue
        kind = m.group(2)
        if split_of(kind)[0] is None: continue
        allt.append((t.name, kind, t.tensor_type.name, t))
        first.setdefault((kind, t.tensor_type.name), (t.name, t))
    return first, allt


# ---- stage: codegen ---------------------------------------------------------------------------------------------------
def _dis(path, sym):
    out = subprocess.run([TR + 'llvm-objdump', '-d', '--disassemble-symbols=' + sym, path], capture_output=True, text=True).stdout
    L = []
    for l in out.splitlines()[5:]:
        l = re.sub(r'//.*', '', l).strip()
        if not l or l == '...' or l.startswith('<') or l.endswith('>:'): continue
        L.append(re.sub(r'^(s_add_u32|s_addc_u32)( s\d+, s\d+, )(0x[0-9a-f]+|lit\(0x[0-9a-f]+\)|-?\d+)$', r'\1\2REL', l))
    return L


def _funcs(path):
    return mmq._elf_funcs(path)


def stage_codegen(scratch):
    d = os.path.join(scratch, 'codegen'); os.makedirs(d, exist_ok=True)     # fixed: a rerun overwrites the same files
    def unbundle(obj):
        b = os.path.basename(obj); fb, co = os.path.join(d, b + '.fatbin'), os.path.join(d, b + '.co')
        subprocess.run([TR + 'llvm-objcopy', '--dump-section=.hip_fatbin=' + fb, obj, '/dev/null'], check=True)
        subprocess.run([TR + 'clang-offload-bundler', '--unbundle', '--type=o', '--input=' + fb,
                        '--targets=hipv4-amdgcn-amd-amdhsa--gfx1100', '--output=' + co], check=True)
        return co
    pairs = [(f'prefill_mmq_{f.lower()}', os.path.join(OBJ, 'template-instances', f'mmq-instance-{f.lower()}.cu.o')) for f in mmq.FORMATS]
    pairs += [('prefill_mmq_glue', os.path.join(OBJ, 'quantize.cu.o')), ('prefill_mmq_glue', os.path.join(OBJ, 'mmid.cu.o'))]
    tot = 0
    for mod, obj in pairs:
        mine = os.path.join(mmq.BUILD, mod + '.hsaco'); ship = unbundle(obj); sf = _funcs(ship)
        syms = [s for s in sorted(_funcs(mine)) if s.startswith('_ZL') and s in sf]
        if mod == 'prefill_mmq_glue':
            syms = [s for s in syms if ('quantize' in s) == ('quantize' in obj)]
        bad = [s for s in syms if _dis(mine, s) != _dis(ship, s)]
        want = len([s for s in _funcs(mine) if s.startswith('_ZL') and (mod != 'prefill_mmq_glue' or ('quantize' in s) == ('quantize' in obj))])
        tot += len(syms)
        check(not bad and len(syms) == want, f'codegen {mod} vs {os.path.basename(obj)}: {len(syms) - len(bad)} / {want} kernels '
              f'with ignite\'s instructions' + (f' (differ: {[b[:60] for b in bad]})' if bad else ''))
    NUMBERS['codegen_kernels'] = tot


# ---- stage: shards ----------------------------------------------------------------------------------------------------
def stage_shards(rd):
    first, allt = gemm_tensors(rd); rows = {}
    for name, kind, fmt, t in allt:
        how, r = split_of(kind); qk = GGML_QUANT_SIZES[t.tensor_type][0]
        if how == 'rows':
            R = [sum(b - a for a, b in r[c]) for c in (0, 1)]; K = int(t.shape[0])
            ok = all(x % 128 == 0 for x in R) and K % qk == 0
            key = (kind, fmt, 'column split', f'rows/card {R[0]} / {R[1]}, K {K}', ok)
        else:
            Kc = [sum(b - a for a, b in r[c]) for c in (0, 1)]
            al = all(a % qk == 0 and b % qk == 0 for c in (0, 1) for a, b in r[c]); rows_ok = int(t.shape[1]) % 128 == 0
            ov = [mmq.overread_bytes(fmt, K) for K in Kc]
            key = (kind, fmt, 'K split', f'card 0 K {r[0]}, card 1 K {r[1]} (K/card {Kc[0]} / {Kc[1]}; K mod 256 {Kc[0] % 256} / '
                   f'{Kc[1] % 256}); {qk}-blocks: {al}; MMQ over-read past the shard = FlashNextTP.ksplit_pad tail {ov[0]} / {ov[1]} B',
                   al and rows_ok)
        rows[key] = rows.get(key, 0) + 1
    for (kind, fmt, how, desc, ok), n in sorted(rows.items()):
        check(ok, f'shards {kind} {fmt} x{n}: {how}: {desc}')


# ---- reference runs ---------------------------------------------------------------------------------------------------
class Ref:
    """a batch of gg_mmq_ref jobs (files under dir), run at once"""
    def __init__(self, binary, d, tag):
        self.bin, self.d, self.tag, self.jobs, self.outs = binary, d, tag, [], []
        os.makedirs(d, exist_ok=True)

    def _f(self, name, arr):
        p = os.path.join(self.d, name); np.ascontiguousarray(arr).tofile(p); return p

    def mm(self, key, fmt, w, rows, K, x, reps=0):
        n = len(self.jobs); o = os.path.join(self.d, f'{self.tag}_{n}.out')
        self.jobs.append(f'mm {fmt} {rows} {K} {x.shape[0]} {self._f(f"{self.tag}_{n}.w", w)} {self._f(f"{self.tag}_{n}.x", x.astype(np.float32))} {o} {reps}')
        self.outs.append((key, o, (x.shape[0], rows)))

    def mmid(self, key, fmt, w, nexp, rows, K, x, ids, per_slot, reps=0):
        n = len(self.jobs); o = os.path.join(self.d, f'{self.tag}_{n}.out'); T, k = ids.shape
        self.jobs.append(f'mmid {fmt} {nexp} {rows} {K} {T} {k} {int(per_slot)} {self._f(f"{self.tag}_{n}.w", w)} '
                         f'{self._f(f"{self.tag}_{n}.x", x.astype(np.float32))} {self._f(f"{self.tag}_{n}.ids", ids.astype(np.int32))} {o} {reps}')
        self.outs.append((key, o, (T, k, rows)))

    def run(self):
        jf = os.path.join(self.d, f'{self.tag}.jobs'); open(jf, 'w').write('\n'.join(self.jobs) + '\n')
        r = subprocess.run([self.bin, jf], capture_output=True, text=True, timeout=1800)
        print(r.stdout.splitlines()[0] if r.stdout else '', flush=True)
        if r.returncode != 0: raise RuntimeError(f'gg_mmq_ref failed ({r.returncode}): {r.stderr[-2000:]}')
        res, times = {}, {}
        lines = [l for l in r.stdout.splitlines() if l.startswith('JOB ')]
        assert len(lines) == len(self.outs), (len(lines), len(self.outs))
        for (key, o, shape), l in zip(self.outs, lines):
            res[key] = np.fromfile(o, dtype=np.float32).reshape(shape)
            m = re.search(r'ms median ([\d.]+) min ([\d.]+)', l)
            if m: times[key] = (float(m.group(1)), float(m.group(2)))
        for f in os.listdir(self.d):                                     # the inputs / outputs read: emptied (the next run
            if f.startswith(self.tag + '_'): open(os.path.join(self.d, f), 'w').close()   # writes the same names again)
        self.jobs, self.outs = [], []
        return res, times


# ---- the device side --------------------------------------------------------------------------------------------------
class Dev:
    def __init__(self, torch, dev):
        self.torch, self.dev = torch, dev
        self.M = mmq.MMQ(dev)

    def put(self, a, dt=None):
        t = self.torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        return t if dt is None else t.to(dt)

    def weights(self, raw):
        """raw uint8 rows -> a device tensor with the zeroed tail ggml (and flashnext_tp's K-split shards) give it"""
        torch = self.torch; nb = raw.size
        b = torch.zeros(nb + 512, dtype=torch.uint8, device=self.dev); b[:nb] = self.put(raw.reshape(-1)); return b[:nb]

    def dense(self, fmt, w, rows, K, x, J=None):
        """x: device fp32 [n][K] -> device fp32 [n][rows]"""
        torch = self.torch; n = x.shape[0]
        y = torch.empty(mmq.y_bytes(K, n), dtype=torch.uint8, device=self.dev); out = torch.empty(n, rows, device=self.dev)
        self.M.quantize(x, K, n, y, self.M.layout(fmt))
        self.M.mm(fmt, w, rows, K, y, n, out, J=J)
        return out

    def moe(self, fmt, w, nexp, rows, K, x, ids, per_slot, J=None):
        """x: device fp32 [T][K] (per_slot False) or [T][k][K]; ids: device int32 [T][k] -> device fp32 [T][k][rows]"""
        torch = self.torch; T, k = ids.shape
        s1 = torch.zeros(T * k, dtype=torch.int32, device=self.dev); sd = torch.zeros(T * k + mmq.J_MAX - 1, dtype=torch.int32, device=self.dev)
        bd = torch.zeros(nexp + 1, dtype=torch.int32, device=self.dev); err = torch.zeros(1, dtype=torch.int32, device=self.dev)
        y = torch.empty(mmq.y_bytes(K, T * k), dtype=torch.uint8, device=self.dev); out = torch.empty(T, k, rows, device=self.dev)
        self.M.group(ids, T, k, nexp, s1, sd, bd, per_slot, err)
        if int(err.item()): raise AssertionError(f'mmq.group flagged the test routing ({int(err.item()):#x}): it must be valid')
        lay = self.M.layout(fmt)
        if per_slot: self.M.quantize(x, K, T * k, y, lay, ids=s1)
        else: self.M.quantize_scatter(x, K, T, s1, k, y, lay)
        self.M.mm_id(fmt, w, nexp, rows, K, y, T, k, sd, bd, out, J=J)
        return out, bd, sd

    def per_expert(self, fmt, w, nexp, rows, K, x, ids, per_slot):
        """mul_mat_id the slow way: each expert's (token, slot) pairs through the plain MMQ, in token order"""
        torch = self.torch; T, k = ids.shape; out = torch.zeros(T, k, rows, device=self.dev)
        rb = mmq.row_bytes(fmt, K); idc = ids.cpu().numpy()
        for e in range(nexp):
            t, s = np.nonzero(idc == e)
            if len(t) == 0: continue
            xs = x[torch.from_numpy(t).to(self.dev)] if not per_slot else x[torch.from_numpy(t).to(self.dev), torch.from_numpy(s).to(self.dev)]
            o = self.dense(fmt, w[e * rows * rb:(e + 1) * rows * rb], rows, K, xs.contiguous())
            out[torch.from_numpy(t).to(self.dev), torch.from_numpy(s).to(self.dev)] = o
        return out


def bits(a):
    a = a.cpu().numpy() if hasattr(a, 'cpu') else a
    return np.ascontiguousarray(a, dtype=np.float32).view(np.int32)


def same(a, b):
    return a.shape == b.shape and np.array_equal(bits(a), bits(b))


def rel(a, b):
    a = a.cpu().numpy() if hasattr(a, 'cpu') else a; b = b.cpu().numpy() if hasattr(b, 'cpu') else b
    return float(np.abs(a.astype(np.float64) - b).max() / max(np.abs(b).max(), 1e-30))


def routing(rng, T, nexp=256, k=10, skew=1.1):
    """top-k distinct experts per token from a skewed popularity (a few experts get hundreds of tokens, some none)"""
    p = 1.0 / np.arange(1, nexp + 1) ** skew; p = p[rng.permutation(nexp)]; p /= p.sum()
    g = -np.log(-np.log(rng.random((T, nexp))))                     # Gumbel top-k = sampling without replacement
    return np.argsort(-(np.log(p)[None, :] + g), axis=1)[:, :k].astype(np.int32)


def activations(rng, T, K):
    x = rng.standard_normal((T, K)).astype(np.float32)
    x[rng.random((T, K)) < 0.002] *= 30.0                            # a few outliers, as real activations have
    return x


def tensor_raw(t):
    return np.asarray(t.data).reshape(-1).view(np.uint8)


def kslice_raw(t, ranges, rows, rb_full):
    """the K-split shard of a [rows][K] tensor (ranges in K values; blocks concatenated) as raw [rows][...] bytes"""
    bs, ts = GGML_QUANT_SIZES[t.tensor_type]
    full = tensor_raw(t).reshape(rows, rb_full // ts, ts)
    return np.concatenate([full[:, a // bs:b // bs] for a, b in ranges], axis=1).reshape(rows, -1)


def kslice_x(x, ranges):
    return np.concatenate([x[..., a:b] for a, b in ranges], axis=-1)


def row_sel(ranges):
    return np.concatenate([np.arange(a, b) for a, b in ranges])


# ---- stage: g1 --------------------------------------------------------------------------------------------------------
def stage_g1(rd, D, ref_bin, scratch, T=2048, seed=1):
    torch = D.torch; rng = np.random.default_rng(seed); first, _ = gemm_tensors(rd)
    R = Ref(ref_bin, os.path.join(scratch, 'g1'), 'g1'); mine = {}; meta = {}
    ids = routing(rng, T); idt = D.put(ids)
    for (kind, fmt), (name, t) in sorted(first.items()):
        how, r = split_of(kind); K = int(t.shape[0]); raw = tensor_raw(t)
        if kind.endswith('_exps'):
            E, rows_full = int(t.shape[2]), int(t.shape[1]); rb = mmq.row_bytes(fmt, K)
            if how == 'rows':                                        # gate / up: one activation row per token
                x = activations(rng, T, K); xt = D.put(x)
                R.mmid((kind, fmt, 'full'), fmt, raw, E, rows_full, K, x, ids, False)
                for c in (0, 1):
                    (r0, r1), = r[c]; sh = raw.reshape(E, rows_full, rb)[:, r0:r1].copy(); w = D.weights(sh)
                    o, _, _ = D.moe(fmt, w, E, r1 - r0, K, xt, idt, False); mine[(kind, fmt, c)] = o.cpu().numpy()
                    if c == 1: mine[(kind, fmt, 'pe', c)] = D.per_expert(fmt, w, E, r1 - r0, K, xt, idt, False).cpu().numpy()
                    R.mmid((kind, fmt, 'shard', c), fmt, sh, E, r1 - r0, K, x, ids, False)
                meta[(kind, fmt)] = ('exp_rows', name, r, rows_full)
            else:                                                    # down: one activation row per (token, slot)
                x = activations(rng, T * 10, K).reshape(T, 10, K); rows = rows_full
                R.mmid((kind, fmt, 'full'), fmt, raw, E, rows, K, x, ids, True)
                for c in (0, 1):
                    (a, b), = r[c]; Kc = b - a; bs, ts = GGML_QUANT_SIZES[t.tensor_type]
                    sh = raw.reshape(E, rows, K // bs, ts)[:, :, a // bs:b // bs].copy(); w = D.weights(sh)
                    xc = np.ascontiguousarray(x[..., a:b]); o, _, _ = D.moe(fmt, w, E, rows, Kc, D.put(xc), idt, True)
                    mine[(kind, fmt, c)] = o.cpu().numpy()
                    if c == 0: mine[(kind, fmt, 'pe', c)] = D.per_expert(fmt, w, E, rows, Kc, D.put(xc), idt, True).cpu().numpy()
                    R.mmid((kind, fmt, 'shard', c), fmt, sh, E, rows, Kc, xc, ids, True)
                meta[(kind, fmt)] = ('exp_cols', name, r, rows)
        else:
            rows_full = int(t.shape[1]); x = activations(rng, T, K); xt = D.put(x)
            if how == 'rows':
                R.mm((kind, fmt, 'full'), fmt, raw, rows_full, K, x)
                rb = mmq.row_bytes(fmt, K); full = raw.reshape(rows_full, rb)
                for c in (0, 1):
                    sh = np.concatenate([full[a:b] for a, b in r[c]]); w = D.weights(sh)
                    mine[(kind, fmt, c)] = D.dense(fmt, w, sh.shape[0], K, xt).cpu().numpy()
                meta[(kind, fmt)] = ('rows', name, r, rows_full)
            else:
                R.mm((kind, fmt, 'full'), fmt, raw, rows_full, K, x)
                for c in (0, 1):
                    sh = kslice_raw(t, r[c], rows_full, mmq.row_bytes(fmt, K)); Kc = sum(b - a for a, b in r[c]); xc = kslice_x(x, r[c])
                    mine[(kind, fmt, c)] = D.dense(fmt, D.weights(sh), rows_full, Kc, D.put(xc)).cpu().numpy()
                    R.mm((kind, fmt, 'shard', c), fmt, sh, rows_full, Kc, xc)
                meta[(kind, fmt)] = ('cols', name, r, rows_full)
    t0 = time.time(); ref, _ = R.run(); print(f'reference: {len(ref)} llama.cpp ops in {time.time() - t0:.1f} s', flush=True)
    relerr = {}
    for (kind, fmt), (how, name, r, rows) in sorted(meta.items()):
        tag = f'{kind} {fmt} ({name})'
        if how == 'rows':
            sel = [row_sel(r[c]) for c in (0, 1)]
            check(all(same(mine[(kind, fmt, c)], ref[(kind, fmt, 'full')][:, sel[c]]) for c in (0, 1)),
                  f'G1 column split {tag}: both cards\' row shards == llama.cpp MMQ full tensor rows, bitwise (T={T})')
        elif how == 'exp_rows':
            ok = all(same(mine[(kind, fmt, c)], ref[(kind, fmt, 'shard', c)]) for c in (0, 1))
            ok2 = all(same(mine[(kind, fmt, c)], ref[(kind, fmt, 'full')][:, :, r[c][0][0]:r[c][0][1]]) for c in (0, 1))
            check(ok and ok2, f'G1 mul_mat_id column split {tag}: both cards == llama.cpp mul_mat_id on the shard and on the full '
                  f'tensor\'s rows, bitwise (T={T}, k=10, skewed routing)')
            check(same(mine[(kind, fmt, 'pe', 1)], mine[(kind, fmt, 1)]), f'G1 mul_mat_id {tag} card 1 == every expert through the plain MMQ, bitwise')
        else:
            p = [mine[(kind, fmt, c)] for c in (0, 1)]; q = [ref[(kind, fmt, 'shard', c)] for c in (0, 1)]
            s = p[0] + p[1]; sq = q[0] + q[1]
            check(same(p[0], q[0]) and same(p[1], q[1]) and same(s, sq),
                  f'G1 K split {tag}: each card\'s partial == llama.cpp MMQ on its K shard, card 0 + card 1 == the same sum of llama.cpp\'s, bitwise')
            e = rel(s, ref[(kind, fmt, 'full')]); relerr[f'{kind} {fmt}'] = e
            print(f'     max relative error of the split sum vs llama.cpp full-K MMQ: {e:.3e} (|max| normalized)', flush=True)
            if how == 'exp_cols':
                check(same(mine[(kind, fmt, 'pe', 0)], mine[(kind, fmt, 0)]), f'G1 mul_mat_id {tag} card 0 (K 384) == every expert through the plain MMQ, bitwise')
    NUMBERS['g1_ksplit_relerr_vs_fullK'] = relerr


# ---- stage: g2 --------------------------------------------------------------------------------------------------------
SPLITS = [[2048], [1024, 1024], [512] * 4, [777, 1271], [1000, 1048], [512, 788, 748]]


def stage_g2(rd, D, seed=2):
    torch = D.torch; rng = np.random.default_rng(seed); first, _ = gemm_tensors(rd); T = 2048
    picks = {}
    for (kind, fmt), v in sorted(first.items()):                 # one tensor per format (G2b covers every format), plus
        moe = kind.endswith('_exps')                             # every format's mul_mat_id (gate exps; down on both cards)
        if kind in ('ffn_up_exps',): continue
        for c in ((0, 1) if kind == 'ffn_down_exps' else (0,)):
            picks.setdefault((fmt, moe, c), (kind, v))
    ids = routing(rng, T); idt = D.put(ids)
    for (fmt, moe, card), (kind, (name, t)) in sorted(picks.items()):
        K = int(t.shape[0]); raw = tensor_raw(t); how, r = split_of(kind)
        if moe:
            E, rows = int(t.shape[2]), int(t.shape[1]); rb = mmq.row_bytes(fmt, K)
            if how == 'rows':
                (r0, r1), = r[card]; w = D.weights(raw.reshape(E, rows, rb)[:, r0:r1].copy()); rows = r1 - r0; per_slot = False
                x = D.put(activations(rng, T, K))
            else:
                (a, b), = r[card]; bs, ts = GGML_QUANT_SIZES[t.tensor_type]
                w = D.weights(raw.reshape(E, rows, K // bs, ts)[:, :, a // bs:b // bs].copy()); K = b - a; per_slot = True
                x = D.put(activations(rng, T * 10, K).reshape(T, 10, K))
            run = lambda p0, p1, J=None: D.moe(fmt, w, E, rows, K, x[p0:p1].contiguous(), idt[p0:p1].contiguous(), per_slot, J=J)[0]
        else:
            rows = int(t.shape[1]); w = D.weights(raw); x = D.put(activations(rng, T, K))
            run = lambda p0, p1, J=None: D.dense(fmt, w, rows, K, x[p0:p1].contiguous(), J=J)
        base = run(0, T); name = f'{name}{f", card {card} K {K}" if moe and how == "cols" else ""}'
        check(same(base, run(0, T)), f'G2a {fmt} ({name}): two runs at T=2048 bitwise')
        ok, detail = True, []
        for sp in SPLITS[1:]:
            outs, p0 = [], 0
            for n in sp: outs.append(run(p0, p0 + n)); p0 += n
            cat = torch.cat(outs)
            if not same(cat, base):
                ok = False; bad = int((bits(cat) != bits(base)).reshape(T, -1).any(axis=1).sum()); detail.append(f'{sp}: {bad} positions differ')
        check(ok, f'G2b {fmt} ({name}{", mul_mat_id: groups per chunk" if moe else ""}): chunks '
              f'{[s for s in SPLITS[1:]]} == one T=2048 pass, bitwise per position' + (f' ({detail})' if detail else ''))
        Js = sorted(J for (f, J) in D.M.cfg if f == fmt); jo = [run(0, 300, J=J) for J in Js]
        check(all(same(o, jo[0]) for o in jo), f'G2b {fmt}: 300 columns through every compiled J {Js} bitwise')


# ---- stage: speed -----------------------------------------------------------------------------------------------------
def stage_speed(rd, D, ref_bins, scratch, T=2048, reps=20, seed=3):
    """per op at T=2048: the per-card shapes of every (kind, format), mine (quantize + MMQ; mul_mat_id with grouping) and
    llama.cpp's (each --ref binary: its graph compute for the op, at the card-0 shape)"""
    torch = D.torch; rng = np.random.default_rng(seed); first, allt = gemm_tensors(rd)
    ids = routing(rng, T); idt = D.put(ids); res = {}
    refs = {tag: Ref(b, os.path.join(scratch, 'speed'), f'speed_{tag}') for tag, b in ref_bins.items() if b}
    def timeit(fn):
        for _ in range(3): fn()
        torch.cuda.synchronize(D.dev); ts = []
        for _ in range(reps):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
        return float(np.median(ts)), float(np.min(ts))
    for (kind, fmt), (name, t) in sorted(first.items()):
        how, r = split_of(kind); K = int(t.shape[0]); raw = tensor_raw(t)
        for c in (0, 1):
            if kind.endswith('_exps'):
                E, rows = int(t.shape[2]), int(t.shape[1]); rb = mmq.row_bytes(fmt, K); bs, ts = GGML_QUANT_SIZES[t.tensor_type]
                if how == 'rows':
                    (r0, r1), = r[c]; sh = raw.reshape(E, rows, rb)[:, r0:r1].copy(); Rc, Kc, per_slot = r1 - r0, K, False
                    xn = activations(rng, T, K)
                else:
                    (a, b), = r[c]; sh = raw.reshape(E, rows, K // bs, ts)[:, :, a // bs:b // bs].copy(); Rc, Kc, per_slot = rows, b - a, True
                    xn = activations(rng, T * 10, Kc).reshape(T, 10, Kc)
                w = D.weights(sh); x = D.put(xn)
                s1 = torch.zeros(T * 10, dtype=torch.int32, device=D.dev); sd = torch.zeros(T * 10 + 127, dtype=torch.int32, device=D.dev)
                bd = torch.zeros(E + 1, dtype=torch.int32, device=D.dev); y = torch.empty(mmq.y_bytes(Kc, T * 10), dtype=torch.uint8, device=D.dev)
                gerr = torch.zeros(1, dtype=torch.int32, device=D.dev)
                out = torch.empty(T, 10, Rc, device=D.dev); lay = D.M.layout(fmt)
                def op():
                    D.M.group(idt, T, 10, E, s1, sd, bd, per_slot, gerr)
                    if per_slot: D.M.quantize(x, Kc, T * 10, y, lay, ids=s1)
                    else: D.M.quantize_scatter(x, Kc, T, s1, 10, y, lay)
                    return D.M.mm_id(fmt, w, E, Rc, Kc, y, T, 10, sd, bd, out)
                J = op(); tm = timeit(lambda: D.M.mm_id(fmt, w, E, Rc, Kc, y, T, 10, sd, bd, out)); to = timeit(op)
                assert int(gerr.item()) == 0, f'mmq.group flagged the speed routing ({int(gerr.item()):#x})'
                flop = 2.0 * Rc * Kc * T * 10
                for tag, R in refs.items():
                    if c == 0: R.mmid((kind, fmt), fmt, sh, E, Rc, Kc, xn, ids, per_slot, reps=reps)
            else:
                rows = int(t.shape[1])
                if how == 'rows':
                    rb = mmq.row_bytes(fmt, K); sh = np.concatenate([raw.reshape(rows, rb)[a:b] for a, b in r[c]]); Rc, Kc = sh.shape[0], K
                    xn = activations(rng, T, K)
                else:
                    sh = kslice_raw(t, r[c], rows, mmq.row_bytes(fmt, K)); Rc, Kc = rows, sum(b - a for a, b in r[c]); xn = activations(rng, T, Kc)
                w = D.weights(sh); x = D.put(xn); y = torch.empty(mmq.y_bytes(Kc, T), dtype=torch.uint8, device=D.dev)
                out = torch.empty(T, Rc, device=D.dev); lay = D.M.layout(fmt)
                def op():
                    D.M.quantize(x, Kc, T, y, lay); return D.M.mm(fmt, w, Rc, Kc, y, T, out)
                J = op(); tm = timeit(lambda: D.M.mm(fmt, w, Rc, Kc, y, T, out)); to = timeit(op)
                flop = 2.0 * Rc * Kc * T
                for tag, R in refs.items():
                    if c == 0: R.mm((kind, fmt), fmt, sh, Rc, Kc, xn, reps=reps)
            res[(kind, fmt, c)] = dict(rows=Rc, K=Kc, J=J, mmq_ms=tm[0], op_ms=to[0], mmq_tflops=flop / tm[0] / 1e9, op_tflops=flop / to[0] / 1e9)
            print(f'SPEED {kind:15s} {fmt:7s} card {c} rows {Rc:5d} K {Kc:5d} J {J:3d}: MMQ {tm[0]:7.3f} ms ({flop / tm[0] / 1e9:6.2f} TFLOPS), '
                  f'op {to[0]:7.3f} ms ({flop / to[0] / 1e9:6.2f})', flush=True)
            del w, x, out
    for tag, R in refs.items():
        _, times = R.run()
        for (kind, fmt), (med, mn) in sorted(times.items()):
            m = res[(kind, fmt, 0)]; m[f'llama_{tag}_ms'] = med
            flop = 2.0 * m['rows'] * m['K'] * T * (10 if kind.endswith('_exps') else 1)
            print(f'LLAMA[{tag}] {kind:15s} {fmt:7s} card-0 shape: {med:7.3f} ms ({flop / med / 1e9:6.2f} TFLOPS); mine op {m["op_ms"]:7.3f} ms', flush=True)
    # the projected per-chunk time of every quantized GEMM of the 48 layers, per card (each tensor at its own kind / format's
    # measured op time), and the chunk's rate if the cards run their shares at once (TP: the slower card sets it)
    tot = [0.0, 0.0]; missing = set()
    for name, kind, fmt, t in allt:
        for c in (0, 1):
            m = res.get((kind, fmt, c))
            if m is None: missing.add((kind, fmt)); continue
            tot[c] += m['op_ms']
    NUMBERS['speed'] = {f'{k[0]} {k[1]} card{k[2]}': v for k, v in res.items()}
    NUMBERS['stack_gemm_ms_per_chunk'] = tot
    print(f'PROJECTION all quantized GEMMs of the 48 layers at T={T}: card 0 shapes {tot[0]:.1f} ms, card 1 shapes {tot[1]:.1f} ms '
          f'(this device); TP chunk >= {max(tot):.1f} ms -> <= {T / max(tot) * 1000:.0f} tok/s from the GEMMs alone'
          + (f' (missing {sorted(missing)})' if missing else ''), flush=True)


# ---- stage: xcard (both cards) ----------------------------------------------------------------------------------------
def stage_xcard(rd, torch, seed=4):
    """the same inputs on the XTX (96 CUs) and the XT (84): bitwise equal outputs (dense Q6_K, a gate mul_mat_id, the K=384
    down mul_mat_id, a K-split Q6_K): no device dependence (stream_k off, one workgroup per output tile)"""
    rng = np.random.default_rng(seed); first, _ = gemm_tensors(rd); T = 2048
    Ds = [Dev(torch, torch.device('cuda:0')), Dev(torch, torch.device('cuda:1'))]
    print('cards', [torch.cuda.get_device_name(d.dev) for d in Ds], flush=True)
    ids = routing(rng, T)
    cases = [('attn_qkv', 'Q6_K'), ('ffn_gate_exps', 'IQ2_S'), ('ffn_down_exps', 'IQ4_NL'), ('ssm_out', 'Q6_K'), ('ffn_up_shexp', 'Q4_K')]
    for kind, fmt in cases:
        name, t = first[(kind, fmt)]; K = int(t.shape[0]); raw = tensor_raw(t); how, r = split_of(kind)
        if kind.endswith('_exps'):
            E, rows = int(t.shape[2]), int(t.shape[1]); rb = mmq.row_bytes(fmt, K)
            if how == 'rows':
                (r0, r1), = r[0]; sh = raw.reshape(E, rows, rb)[:, r0:r1].copy(); x = activations(rng, T, K); args = (E, r1 - r0, K)
                outs = [d.moe(fmt, d.weights(sh), *args, d.put(x), d.put(ids), False)[0] for d in Ds]
            else:
                (a, b), = r[0]; bs, ts = GGML_QUANT_SIZES[t.tensor_type]; sh = raw.reshape(E, rows, K // bs, ts)[:, :, a // bs:b // bs].copy()
                x = activations(rng, T * 10, b - a).reshape(T, 10, b - a)
                outs = [d.moe(fmt, d.weights(sh), E, rows, b - a, d.put(x), d.put(ids), True)[0] for d in Ds]
        elif how == 'cols':
            rows = int(t.shape[1]); sh = kslice_raw(t, r[0], rows, mmq.row_bytes(fmt, K)); Kc = sum(b - a for a, b in r[0]); x = activations(rng, T, Kc)
            outs = [d.dense(fmt, d.weights(sh), rows, Kc, d.put(x)) for d in Ds]
        else:
            rows = int(t.shape[1]); x = activations(rng, T, K)
            outs = [d.dense(fmt, d.weights(raw), rows, K, d.put(x)) for d in Ds]
        check(same(outs[0], outs[1]), f'xcard {kind} {fmt} ({name}): the XTX and the XT give the same bits (T={T})')


# ---- stage: tp (both cards, the loaded model) ------------------------------------------------------------------------
def stage_tp(rd, coder, torch, chunk=2048, max_ctx=32768, reps=3, seed=6, alloc_only=False):
    """FlashNextTP (T=4, max_ctx, DFlash as the smoke builds it) with the prefill counted in its VRAM check
    (vram_prefill=chunk), then the prefill's buffers (mmq.Workspace per card, the two-card feed) and every quantized GEMM of
    the 48 layers on K3's own shards, both cards at once (each card its share, as TP runs them):
    - STEP 0 on the loaded shards: every K-split shard's ranges on its format's blocks, its allocation covering MMQ's
      over-read, its tail zero; the shard bytes of each G1 tensor == the GGUF slices G1 tested;
    - the per-chunk time of the whole stack (ms, both cards concurrently and each alone) at T = chunk;
    - VRAM: the device-level peak (sysfs) after load, after the prefill buffers and after the stack"""
    from caminus.device import capacity
    from caminus.decode import dflash as dk
    from caminus.engine.flashnext_tp import FlashNextTP
    from caminus.prefill.feed import PrefillFeed
    SYSFS = ('/sys/class/drm/card1/device', '/sys/class/drm/card2/device')
    used = lambda: [int(open(f'{d}/mem_info_vram_used').read()) / capacity.MIB for d in SYSFS]
    print('compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
    vram = dict(idle=used())
    g = capacity.geometry(coder)
    drd = GGUFReader(dk.DRAFTER, 'r'); f = {x.name: x for x in drd.fields.values()}
    layers = [int(v) for v in f['dflash.target_layers'].contents()]; del drd, f
    dg = dk.geom(dk.DRAFTER)
    m = FlashNextTP(coder, ntok=4, max_ctx=max_ctx, feature_layers=layers, vram_drafters=(('dflash', dg, None),), vram_prefill=chunk)
    df = dk.DFlashK3(m, dk.DRAFTER); dr = dk.DFlashK3Drafter(df)
    print('vram plan (MiB per card):', {k: v for k, v in m.vram_plan.items() if k != 'rows'}, flush=True)
    toks = [248044, 760, 22414, 5285, 315, 13, 11, 13, 760, 22414, 5285, 315]
    for p in range(3): m.verify(toks, p * 4, 0 if p == 0 else 4)
    dr.generate(m, toks, 32, stats={})
    for d in (0, 1): torch.cuda.synchronize(d)
    vram['loaded'] = used()
    # STEP 0 on the loaded shards
    first, allt = gemm_tensors(rd)
    nks = nok = 0; bad = []
    for name, kind, fmt, t in allt:
        sh = m.shards.get(name)
        if sh is None: bad.append(f'{name}: no shard record'); continue
        how, r = split_of(kind); qk = GGML_QUANT_SIZES[t.tensor_type][0]
        want_kind = {'rows': ('rows', 'exp_rows'), 'cols': ('cols', 'exp_cols')}[how]
        if sh['kind'] not in want_kind or [list(map(tuple, x)) for x in sh['ranges']] != [list(map(tuple, x)) for x in r]:
            bad.append(f'{name}: {sh["kind"]} {sh["ranges"]} vs the split {how} {r}'); continue
        for c in (0, 1):
            info = sh['infos'][c]; w = info['w']; E = int(t.shape[2]) if kind.endswith('_exps') else 1
            if how == 'cols':
                nks += 1
                Kc = info['cols']; ok = all(a % qk == 0 and b % qk == 0 for a, b in sh['ranges'][c]) and Kc == sum(b - a for a, b in r[c])
                need = E * info['rows'] * info['rb'] + mmq.overread_bytes(fmt, Kc)
                ok &= mmq.extent(w) >= need and w.numel() == E * info['rows'] * info['rb']
                base = torch.empty(0, dtype=torch.uint8, device=w.device).set_(w.untyped_storage())
                off = w.data_ptr() - w.untyped_storage().data_ptr(); tail = base[off + w.numel():]
                ok &= bool((tail == 0).all().item())
                nok += ok
                if not ok: bad.append(f'{name} card {c}: K {Kc}, extent {mmq.extent(w)} < {need} or tail not zero')
    check(not bad, f'STEP 0 on the loaded shards: {nok} / {nks} K-split shards (both cards) on their blocks, covering MMQ\'s over-read, '
          f'zero tails; every GEMM tensor\'s record matches the split' + (f' ({bad[:4]})' if bad else ''))

    def shard_bytes():
        """every GEMM tensor's shards on both cards == the GGUF tensor's slices, bytewise (each tensor read again with O_DIRECT
        onto card 0; run last: the reads leave blocks in torch's cache on the XTX)"""
        bad, nb = [], 0
        for name, kind, fmt, t in allt:
            sh = m.shards[name]; how, r = split_of(kind)
            full = m._full(name); rb = mmq.row_bytes(fmt, int(t.shape[0])); bs, ts = GGML_QUANT_SIZES[t.tensor_type]
            E = int(t.shape[2]) if kind.endswith('_exps') else 1; R = int(t.shape[1])
            for c in (0, 1):
                if kind.endswith('_exps'):
                    exp = (full.view(E, R, rb)[:, r[c][0][0]:r[c][0][1]] if how == 'rows' else
                           full.view(E, R, rb // ts, ts)[:, :, r[c][0][0] // bs:r[c][0][1] // bs])
                else:
                    exp = (torch.cat([full.view(R, rb)[a:b] for a, b in r[c]]) if how == 'rows' else
                           torch.cat([full.view(R, rb // ts, ts)[:, a // bs:b // bs] for a, b in r[c]], dim=1))
                w = sh['infos'][c]['w']; nb += w.numel()
                if not torch.equal(w.to(full.device).view(-1), exp.contiguous().view(-1)): bad.append(f'{name} card {c}')
            del full
        check(not bad, f'STEP 0 on the loaded shards: all {len(allt)} GEMM tensors\' shards on both cards ({nb / 2**30:.1f} GiB) == the '
              f'GGUF tensors\' slices, bytewise' + (f' (differ: {bad[:4]})' if bad else ''))
    # the prefill buffers, each checked against its card's cap before it allocates, at the ENGINE's split; then every
    # prefill module the chunk loop loads (MMQ, glue, gemm; the feed's in PrefillFeed). Device-level VRAM after each
    devs = [torch.device('cuda:0'), torch.device('cuda:1')]
    from caminus.prefill import glue as prefill_ops
    from caminus.device import gemm as _gemm
    split = dict(gdn_keys=m.gdn_keys, moe_blocks=m.moe_blocks)
    sync = lambda: [torch.cuda.synchronize(d) for d in (0, 1)]
    W = [mmq.Workspace(devs[c], chunk, mmq.card_dims(g, c, **split)) for c in (0, 1)]; sync(); vram['workspace'] = used()
    feed = PrefillFeed(coder, devices=(devs[capacity.PREFILL_TABLE_CARD], devs[1 - capacity.PREFILL_TABLE_CARD]), max_chunk=chunk)
    sync(); vram['feed'] = used()
    GW = [prefill_ops.GlueWorkspace(devs[c], chunk, prefill_ops.glue_dims(g, c, m.gdn_keys)) for c in (0, 1)]; sync(); vram['glue'] = used()
    Ms = [mmq.MMQ(d) for d in devs]; Gs = []
    for d in devs:
        with torch.cuda.device(d): Gs.append(prefill_ops.Glue(d)); _gemm._m()
    sync(); vram['buffers'] = used()
    plan = list(capacity.prefill_need_mib(g, chunk, **split))
    parts = dict(workspace=[W[c].nbytes / capacity.MIB for c in (0, 1)], feed=[x / capacity.MIB for x in capacity.prefill_feed_bytes(g, chunk)],
                 glue=[GW[c].nbytes / capacity.MIB for c in (0, 1)], modules=list(capacity.PREFILL_CODE_MIB))
    prev = 'loaded'
    for ph in ('workspace', 'feed', 'glue', 'buffers'):
        dv = [vram[ph][c] - vram[prev][c] for c in (0, 1)]; pp = parts['modules' if ph == 'buffers' else ph]
        print(f'  prefill {"modules (MMQ x 11, glue, gemm)" if ph == "buffers" else ph}: device-level +{dv[0]:.1f} / +{dv[1]:.1f} MiB '
              f'(counted {pp[0]:.1f} / {pp[1]:.1f})', flush=True)
        prev = ph
    print(f'prefill buffers + modules: device-level +{vram["buffers"][0] - vram["loaded"][0]:.0f} / +{vram["buffers"][1] - vram["loaded"][1]:.0f} MiB '
          f'(capacity.prefill_need_mib at split {split["gdn_keys"]} / {split["moe_blocks"]}: {plan[0]:.0f} / {plan[1]:.0f})', flush=True)
    # the GEMMs' fp32 inputs and the routing, in the workspaces' own buffers (the GEMM workspace's x_* and the glue
    # workspace's mixed / ids: nothing allocated outside the counted buffers; S2's test allocated these as test tensors)
    rng = np.random.default_rng(seed); ids = routing(rng, chunk)
    X = []
    for c in (0, 1):
        d = mmq.card_dims(g, c, **split); w_ = W[c]
        with torch.cuda.device(devs[c]):
            x = dict(mixed=GW[c].mixed[:chunk * d['D']].view(chunk, d['D']), gdn=w_.x_ssm_out[:chunk * d['ssm_K']].view(chunk, d['ssm_K']),
                     att=w_.x_attn_out[:chunk * d['ao_K']].view(chunk, d['ao_K']), hsh=w_.x_sh_down[:chunk * d['ffc']].view(chunk, d['ffc']),
                     hmoe=w_.x_moe_down[:chunk * d['k'] * d['ffc']].view(chunk, d['k'], d['ffc']), ids=GW[c].ids[:chunk * d['k']].view(chunk, d['k']))
            for k_ in ('mixed', 'gdn', 'att', 'hsh', 'hmoe'): x[k_].normal_()
            x['ids'].copy_(torch.from_numpy(ids)); X.append(x)
    sync(); vram['inputs'] = used()
    nl = 1 + max(int(n.split('.')[1]) for n, _, _, _ in allt)
    by_layer = {}
    for name, kind, fmt, t in allt: by_layer.setdefault(int(name.split('.')[1]), []).append((kind, name))
    def layer_ops(c, il):
        """yield callables: the layer's quantized GEMMs on card c, in prefill order"""
        M, w_, x, d, T = Ms[c], W[c], X[c], mmq.card_dims(g, c, **split), chunk
        ops = dict(by_layer[il]); info = lambda n: m.shards[n]['infos'][c]
        def dense(nm, xin, K, out, y):
            i = info(nm); fmt = i['fmt']; lay = M.layout(fmt); yb = y[lay]
            M.quantize(xin, K, T, yb, lay); M.mm(fmt, i['w'], i['rows'], K, yb, T, out)
        ys = {mmq.D4: w_.y_d4, mmq.DS4: w_.y_ds4}
        if 'attn_qkv' in ops:
            yield lambda: dense(ops['attn_qkv'], x['mixed'], d['D'], w_.o_qkv, ys)
            yield lambda: dense(ops['attn_gate'], x['mixed'], d['D'], w_.o_z, ys)
            yield lambda: dense(ops['ssm_out'], x['gdn'], d['ssm_K'], w_.o_blk, ys)
        else:
            for k_, o in (('attn_q', w_.o_q), ('attn_k', w_.o_k), ('attn_v', w_.o_v)): yield (lambda k_=k_, o=o: dense(ops[k_], x['mixed'], d['D'], o, ys))
            yield lambda: dense(ops['attn_output'], x['att'], d['ao_K'], w_.o_blk, ys)
        yield lambda: dense(ops['ffn_gate_shexp'], x['mixed'], d['D'], w_.o_sh_gate, ys)
        yield lambda: dense(ops['ffn_up_shexp'], x['mixed'], d['D'], w_.o_sh_up, ys)
        yield lambda: dense(ops['ffn_down_shexp'], x['hsh'], d['ffc'], w_.o_sh_down, ys)
        def moe_gu():
            gi, ui = info(ops['ffn_gate_exps']), info(ops['ffn_up_exps']); assert gi['fmt'] == ui['fmt']
            M.group(x['ids'], T, d['k'], d['n_exp'], w_.ids_inv, w_.ids_dst, w_.bounds, False, w_.err)
            lay = M.layout(gi['fmt']); M.quantize_scatter(x['mixed'], d['D'], T, w_.ids_inv, d['k'], w_.y_moe, lay)
            M.mm_id(gi['fmt'], gi['w'], d['n_exp'], gi['rows'], d['D'], w_.y_moe, T, d['k'], w_.ids_dst, w_.bounds, w_.o_gate)
            M.mm_id(ui['fmt'], ui['w'], d['n_exp'], ui['rows'], d['D'], w_.y_moe, T, d['k'], w_.ids_dst, w_.bounds, w_.o_up)
        def moe_down():
            di = info(ops['ffn_down_exps'])
            M.group(x['ids'], T, d['k'], d['n_exp'], w_.ids_fwd, w_.ids_dst, w_.bounds, True, w_.err)
            lay = M.layout(di['fmt']); M.quantize(x['hmoe'], d['ffc'], T * d['k'], w_.y_down, lay, ids=w_.ids_fwd)
            M.mm_id(di['fmt'], di['w'], d['n_exp'], di['rows'], d['ffc'], w_.y_down, T, d['k'], w_.ids_dst, w_.bounds, w_.o_down)
        yield moe_gu; yield moe_down
    def stack(cards):
        gens = {c: [op for il in range(nl) for op in layer_ops(c, il)] for c in cards}
        for i in range(max(len(v) for v in gens.values())):
            for c in cards:
                if i < len(gens[c]):
                    with torch.cuda.device(devs[c]): gens[c][i]()
        for c in cards: torch.cuda.synchronize(devs[c])
    res = {}
    for tag, cards in (() if alloc_only else (('both', (0, 1)), ('xtx_card0', (0,)), ('xt_card1', (1,)))):
        stack(cards); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); stack(cards); ts.append((time.perf_counter() - t0) * 1e3)
        res[tag] = (float(np.median(ts)), float(np.min(ts)))
        print(f'STACK {tag}: all quantized GEMMs of the {nl} layers at T={chunk}: median {res[tag][0]:.1f} ms, min {res[tag][1]:.1f} '
              f'({chunk / res[tag][0] * 1000:.0f} tok/s from the GEMMs alone)', flush=True)
    for d in (0, 1): torch.cuda.synchronize(d)
    vram['after_stack'] = used()
    for c in (0, 1): W[c].check_routing(f'tp stack card {c}')            # (valid test routing: the clamp replaced nothing)
    vram['torch_peak_reserved'] = [torch.cuda.max_memory_reserved(d) / capacity.MIB for d in (0, 1)]
    print('VRAM (device-level MiB [XTX, XT]):', {k: [round(v, 0) for v in vv] for k, vv in vram.items()}, flush=True)
    # the projection against the measurement: the plan this engine checked (fixed + ctx + drafters + prefill at its split)
    # plus what the card held before this process started (idle: the desktop, the driver), against the highest
    # device-level use seen at any step (guard.sh's 1 Hz log has the run's own peak, reported by the caller too)
    vp = m.vram_plan; proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] for c in (0, 1)]
    peak = [max(vv[c] for k, vv in vram.items() if k != 'torch_peak_reserved') for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(vram['idle'][c] + proj[c] >= peak[c], f'VRAM projection >= measurement, {nm}: idle {vram["idle"][c]:.0f} + projected {proj[c]:.0f} '
              f'(fixed {vp["fixed_mib"][c]:.0f}, ctx {vp["ctx_mib"]:.0f}, drafters {vp["drafters_mib"][c]:.0f}, prefill {vp["prefill_mib"][c]:.0f}) = '
              f'{vram["idle"][c] + proj[c]:.0f} MiB vs the measured device-level peak {peak[c]:.0f} MiB (margin {vram["idle"][c] + proj[c] - peak[c]:+.0f}; '
              f'net of idle: measured {peak[c] - vram["idle"][c]:.0f} vs projected {proj[c]:.0f})')
    NUMBERS['tp_projection'] = dict(projected=proj, idle=vram['idle'], peak=peak, split=split)
    NUMBERS['tp'] = dict(stack_ms=res, vram=vram, plan_prefill_mib=plan, parts_mib=parts, workspace_mib=[W[c].nbytes / capacity.MIB for c in (0, 1)],
                         glue_mib=[GW[c].nbytes / capacity.MIB for c in (0, 1)],
                         chunk=chunk, max_ctx=max_ctx, vram_plan={k: v for k, v in m.vram_plan.items() if k != 'rows'})
    feed.close()
    t0 = time.time(); shard_bytes(); print(f'(shard bytes checked in {time.time() - t0:.1f} s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--scratch', default=None, help='default $ARTIFACTS/impl-prefill-s2/scratch (fixed names, never /tmp)')
    ap.add_argument('--ref', default=devpaths.artifacts('impl-prefill-s2', 'ggml-fmmq', 'gg_mmq_ref'))
    ap.add_argument('--ref-stock', default=os.path.join(ROOT, 'build', 'gg_mmq_ref')); ap.add_argument('--json', default=None)
    ap.add_argument('--chunk', type=int, default=2048); ap.add_argument('--max-ctx', type=int, default=32768)
    ap.add_argument('--alloc-only', action='store_true')
    a = ap.parse_args()
    stages = []
    for s in a.stages: stages += ['g1', 'g2', 'speed'] if s == 'gpu' else [s]
    scratch = a.scratch or devpaths.need('ARTIFACTS', 'impl-prefill-s2', 'scratch')
    os.makedirs(scratch, exist_ok=True)
    rd = GGUFReader(a.coder, 'r')
    if 'codegen' in stages: stage_codegen(scratch)
    if 'shards' in stages: stage_shards(rd)
    if 'xcard' in stages or 'tp' in stages:
        import torch
        if 'xcard' in stages: stage_xcard(rd, torch)
        if 'tp' in stages: stage_tp(rd, a.coder, torch, chunk=a.chunk, max_ctx=a.max_ctx, alloc_only=a.alloc_only)
    gpu = [s for s in stages if s in ('g1', 'g2', 'speed')]
    if gpu:
        import torch
        D = Dev(torch, torch.device('cuda:0')); print('device', torch.cuda.get_device_name(D.dev), flush=True)
        if 'g1' in gpu:
            if not a.ref or not os.path.exists(a.ref): sys.exit(f'g1 needs --ref (a GGML_CUDA_FORCE_MMQ=ON gg_mmq_ref: tools/reference/build_gg_mmq_ref.sh); {a.ref} missing')
            stage_g1(rd, D, a.ref, scratch)
        if 'g2' in gpu: stage_g2(rd, D)
        if 'speed' in gpu: stage_speed(rd, D, {'stock': a.ref_stock, 'mmq': a.ref}, scratch)
    if a.json:
        json.dump(dict(fails=FAILS, numbers={k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in NUMBERS.items()}),
                  open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
