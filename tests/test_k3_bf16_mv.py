"""Gates G2 / G2b of the GSQ port (docs/agentic/spec-gsq-port.md S2): OP_BF16_MV (kernels/k3_gguf_ops.h) against llama.cpp's
own BF16 x F32 matvec, mul_mat_vec_f<nv_bfloat16, float, ncols_dst, block_size> (ignite-llama ggml-cuda/mmvf.cu: what
ignite's HIP build runs for every BF16 matrix at ne11 = 1, and at ne11 = 4 for the matrices MMF does not take: hc inject,
ssm alpha / beta, the shared-expert gate), on real GSQ-RCO Coder tensors read with O_DIRECT (shard 1 only).
References (both must agree bitwise, and the engine with both):
  shipped  the gfx1100 code object of ignite's build (build-hip-therock .../ggml-cuda/mmvf.cu.o, unbundled), whose bf16
           kernel bodies are checked byte for byte inside bin/libggml-hip.so (what llama-server loads); the silu input of
           the hc up comes from the same build's unary.cu.o: scale_unary_kernel<op_silu> (ggml_scale + ggml_silu fused)
  source   build/gg_mmvf_ref.hsaco: mmvf.cu compiled from the ignite tree by build.sh with ignite's flags
checks
  G2   disassembly of the shipped bf16 kernels: the col2 loop does 2 FMAs per token (v_fmac_f32 / v_fma_f32), no
       v_mul_f32 / v_dot2 / v_pk_fma (fp contraction as assumed), and the float mode of k3_run's descriptor equals theirs
  G2b  (a) T = 1 (module k3_gguf_gsq): engine == mul_mat_vec_f<..., 1, bs> bitwise, at 3 / 5 / 6 / 7 / 8 workgroups per
           WGP and rows_per_task 8 / 16 / 5 / 1 / 2 / 3 / 4, every card; K = 10240 (hc down, inject, output_hc down), 2560 (router, shared
           gate, alpha, beta, ple key / value, indexer q), 320 (hc up and output_hc up through the silu path)
       (b) T = 2 / 3 / 4 (module k3_gguf_gsq_mt, rows_per_task 16 (5 at 5 WG/WGP) / 1 / 3): every token's row == T = 1's, and T = 4 == mul_mat_vec_f<..., 4, bs>
       (c) the alpha path llama.cpp runs fused (mul_mat_vec_f<..., 1, 256, has_fusion> with x_bias = dt bias) == the
           engine's output + the bias (one fp32 add), so K3's separate add stays bitwise
  tools/guard.sh LOG 900 python -I tests/test_k3_bf16_mv.py [CODER-00001-of-00002.gguf] [--cards 0,1] [--wgp 3,5,6,7,8]
      [--module k3_gguf_gsq | k3_gguf_gsq_trace] [--ignite ~/src/ignite-llama/build-hip-therock]"""
import ctypes, os, re, struct, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus import gguf as G, k3
from caminus.hip import hip, check as hcheck, Kernel, BUILD
from caminus.flashnext import DirectReader

a = sys.argv[1:]
def opt(name, default):
    if name in a:
        i = a.index(name); v = a[i + 1]; del a[i:i + 2]; return v
    return default
CARDS = [int(v) for v in opt('--cards', '0,1').split(',')]
WGP = [int(v) for v in opt('--wgp', '3,5,6,7,8').split(',')]
M1 = opt('--module', 'k3_gguf_gsq'); assert M1 in ('k3_gguf_gsq', 'k3_gguf_gsq_trace'), M1
MT = M1.replace('k3_gguf_gsq', 'k3_gguf_gsq_mt')
IGN = os.path.expanduser(opt('--ignite', '~/src/ignite-llama/build-hip-therock'))
TR = os.path.join(os.environ.get('ROCM_PATH') or os.path.expanduser('~/rocm-therock/10.2.0a20261011'), 'llvm', 'bin', '')   # as build.sh
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = a[0] if a else devpaths.coder()
RD = G.GGUFReader(PATH, 'r'); TT = {t.name: t for t in RD.tensors}
DR = DirectReader(PATH)
HC = 4
fails = []


def check(cond, what):
    if not cond: fails.append(what); print('  FAIL ' + what, flush=True)
    return cond


# ---- the shipped code objects ----
OBJ = os.path.join(IGN, 'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda')
TMP = tempfile.mkdtemp(prefix='bf16mv_')


def unbundle(obj):
    fb, co = os.path.join(TMP, os.path.basename(obj) + '.fatbin'), os.path.join(TMP, os.path.basename(obj) + '.gfx1100.co')
    subprocess.run([TR + 'llvm-objcopy', '--dump-section=.hip_fatbin=' + fb, obj, '/dev/null'], check=True)
    subprocess.run([TR + 'clang-offload-bundler', '--unbundle', '--type=o', '--input=' + fb,
                    '--targets=hipv4-amdgcn-amd-amdhsa--gfx1100', '--output=' + co], check=True)
    return co


def symbols(path):
    """{name: bytes} of the FUNC / OBJECT symbols of a code object"""
    data = open(path, 'rb').read(); secs, out = {}, {}
    for l in subprocess.run([TR + 'llvm-readelf', '-SW', path], capture_output=True, text=True).stdout.splitlines():
        f = l.replace('[ ', '[').split()
        if len(f) > 5 and f[0].startswith('[') and f[0].strip('[]').isdigit(): secs[int(f[0].strip('[]'))] = (int(f[3], 16), int(f[4], 16))
    for l in subprocess.run([TR + 'llvm-readelf', '-sW', path], capture_output=True, text=True).stdout.splitlines():
        f = l.split()
        if len(f) >= 8 and f[3] in ('FUNC', 'OBJECT') and f[6].isdigit() and int(f[2]) > 0:
            ad = int(f[1], 16); addr, off = secs[int(f[6])]; out[f[7]] = data[ad - addr + off: ad - addr + off + int(f[2])]
    return out


def mmvf_name(nc, bs, fusion=False):
    return (f'_ZL13mul_mat_vec_fI14__hip_bfloat16fLi{nc}ELi{bs}ELb{int(fusion)}ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_'
            'devicePfi15HIP_vector_typeIjLj3EEiiiSB_iiiSB_iiii')


SILU = '_ZL18scale_unary_kernelIXadL_ZL7op_silufEEEvPKfPflff'
CO_MMVF, CO_UNARY = unbundle(os.path.join(OBJ, 'mmvf.cu.o')), unbundle(os.path.join(OBJ, 'unary.cu.o'))
USED = [(nc, bs, fu) for nc in (1, 4) for bs in (160, 256) for fu in (False,)] + [(1, 256, True)]


def provenance():
    """the kernels the test launches are byte for byte in the library llama-server loads"""
    so = open(os.path.join(IGN, 'bin/libggml-hip.so'), 'rb').read()
    sm, su = symbols(CO_MMVF), symbols(CO_UNARY)
    for nm, tab in [(mmvf_name(*u), sm) for u in USED] + [(SILU, su)]:
        for part in (nm, nm + '.kd'):
            check(part in tab and so.count(tab[part]) >= 1, f'provenance: {part[:60]}... not found in libggml-hip.so')
    print(f'provenance: {len(USED)} mul_mat_vec_f<bf16> kernels + scale_unary_kernel<op_silu> (bodies and descriptors) found in libggml-hip.so', flush=True)


def g2():
    """G2: the fp ops of the shipped bf16 kernels, and the float mode of their descriptors vs k3_run's"""
    dis = subprocess.run([TR + 'llvm-objdump', '-d', '--mcpu=gfx1100', CO_MMVF], capture_output=True, text=True).stdout
    for nc, bs, fu in USED:
        nm = mmvf_name(nc, bs, fu)
        m = re.search(r'<' + re.escape(nm) + r'>:\n(.*?)\n\n', dis, re.S); body = m.group(1) if m else ''
        ins = [l.split('//')[0] for l in body.splitlines()]
        fma = sum(len(re.findall(r'v_(?:dual_)?fmac?_f32', l)) for l in ins)
        bad = sum(len(re.findall(r'v_(?:dual_)?mul_f32|dot2|pk_fma|v_mad_f32|v_fmamk|v_fmaak', l)) for l in ins)
        add = sum(len(re.findall(r'v_(?:dual_)?add_f32', l)) for l in ins)
        if fu:   # the fused variant also holds the gate's fmas and the GLU epilogues (silu / gelu / swiglu_oai): counts only
            print(f'  G2 mul_mat_vec_f<bf16, float, 1, 256, has_fusion> (alpha + dt bias): {fma} fma, {bad} mul-type, {add} add '
                  f'(gate + GLU epilogue code included; gated by (c) instead)', flush=True)
            continue
        exp_fma = 2 * nc
        check(fma == exp_fma and bad == 0, f'G2 {nm[:52]}: {fma} fma (want {exp_fma}), {bad} mul/dot2/pk_fma/fmamk')
        print(f'  G2 mul_mat_vec_f<bf16, float, {nc}, {bs}>: {fma} v_fma(c)_f32 (2 per token, in the col2 loop), '
              f'{bad} v_mul_f32 / dot2 / pk_fma / fmamk, {add} v_add_f32 (2 x 5 butterfly levels per token)', flush=True)
    def rsrc1(path, name):
        kd = symbols(path)[name + '.kd']; r1, = struct.unpack_from('<I', kd, 48)
        return dict(round32=(r1 >> 12) & 3, denorm32=(r1 >> 16) & 3, denorm16_64=(r1 >> 18) & 3, dx10=(r1 >> 21) & 1, ieee=(r1 >> 23) & 1)
    ref = rsrc1(CO_MMVF, mmvf_name(1, 256))
    for m in (M1, MT):
        mine = rsrc1(os.path.join(BUILD, f'{m}.hsaco'), 'k3_run')
        check(mine == ref, f'G2 float mode: {m} k3_run {mine} != mul_mat_vec_f {ref}')
    print(f'  G2 float mode (compute_pgm_rsrc1) of mul_mat_vec_f and k3_run ({M1}, {MT}): {ref}', flush=True)


class Ref:
    """one code object's mul_mat_vec_f<bf16> launches on the current device (the 216-byte kernarg packed by hand from the
    code object's metadata: x, y, ids, fusion (104 B), dst, ncols2, nchannels_y, stride_row, stride_col_y2, stride_col_dst,
    channel_ratio, 3 channel strides, sample_ratio, 3 sample strides, ids_stride)"""
    def __init__(self, path):
        data = open(path, 'rb').read(); self._img = ctypes.create_string_buffer(data, len(data)); self.h = ctypes.c_void_p()
        hcheck(hip().hipModuleLoadData(ctypes.byref(self.h), self._img)); self.fns = {}

    def fn(self, name):
        if name not in self.fns:
            f = ctypes.c_void_p(); hcheck(hip().hipModuleGetFunction(ctypes.byref(f), self.h, name.encode())); self.fns[name] = f
        return self.fns[name]

    def mmvf(self, w, x, N, K, nc, bias=None):
        """w bf16 bytes [N, K], x fp32 [nc, K] -> fp32 [nc, N]"""
        bs = 32 * k3.mmvf_warps(K); out = torch.full((nc * N,), float('nan'), device=x.device)
        fus = bytearray(104)
        if bias is not None: fus[0:8] = struct.pack('<Q', bias.data_ptr())
        ka = (struct.pack('<QQQ', w.data_ptr(), x.data_ptr(), 0) + bytes(fus) + struct.pack('<Qi', out.data_ptr(), K // 2)
              + struct.pack('<3I', 0, 0, 0) + struct.pack('<3i', K, K // 2, N) + struct.pack('<3I', 1, 0, 1) + struct.pack('<3i', 0, 0, 0)
              + struct.pack('<3I', 1, 0, 1) + struct.pack('<4i', 0, 0, 0, 0))
        assert len(ka) == 216
        buf = ctypes.create_string_buffer(ka, 216); size = ctypes.c_size_t(216)
        extra = (ctypes.c_void_p * 5)(1, ctypes.cast(buf, ctypes.c_void_p), 2, ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), 3)
        s = torch.cuda.current_stream().cuda_stream
        hcheck(hip().hipModuleLaunchKernel(self.fn(mmvf_name(nc, bs, bias is not None)), N, 1, 1, bs, 1, 1, 128 * (2 if bias is not None else 1),
                                           ctypes.c_void_p(s), None, extra))
        torch.cuda.synchronize(); return out

    def scale_silu(self, x, s):
        """ignite's fused ggml_scale + ggml_silu: silu(s * x + 0)"""
        out = torch.full_like(x, float('nan')); n = x.numel()
        Kernel(self.fn(SILU), SILU)(((n + 255) // 256,), (256,), x, out, ctypes.c_int64(n), ctypes.c_float(s), ctypes.c_float(0.0))
        torch.cuda.synchronize(); return out


def grid(module, n):
    return k3.resident_grid(module, n)


# rows per task: the shipped tables' values and the split-K / short paths' batch tails (R = 2: odd counts)
RPT1 = (8, 16, 5, 1, 2, 3, 4)
RPTM = (1, 3)
CASES = [  # (tensor, silu): every BF16 shape class of the Coder's decode tape
    ('blk.0.hc_attn_down.weight', False), ('blk.0.hc_attn_inject.weight', False), ('output_hc_down.weight', False),   # K 10240
    ('blk.0.ffn_gate_inp.weight', False), ('blk.0.ffn_gate_inp_shexp.weight', False), ('blk.0.ssm_alpha.weight', False),  # K 2560
    ('blk.0.ssm_beta.weight', False), ('blk.1.ple_key.weight', False), ('blk.1.ple_value.weight', False),
    ('blk.3.indexer.q_proj.weight', False),
    ('blk.0.hc_attn_up.weight', True), ('blk.3.hc_ffn_up.weight', True), ('output_hc_up.weight', True),               # K 320, silu
]


def run_card(card):
    dev = f'cuda:{card}'; n_ok = n_all = 0
    refs = {'shipped': Ref(CO_MMVF), 'source': Ref(os.path.join(BUILD, 'gg_mmvf_ref.hsaco'))}
    silu_ref = Ref(CO_UNARY)
    for name, silu in CASES:
        t = TT[name]; K = int(t.shape[0]); N = int(t.shape[1]) if len(t.shape) > 1 else 1
        w = DR.read(int(t.data_offset), int(t.n_bytes), device=dev); assert w.numel() == N * K * 2
        gen = torch.Generator(device=dev).manual_seed(4000 + N + K)
        X = torch.randn(4, K, device=dev, generator=gen) * (2.0 if silu else 0.5)
        X[0, :7] = torch.tensor([0.0, -0.0, 0.0, -0.0, 1e-40, -1e-40, 0.0], device=dev)   # zeros of both signs, denormals
        X = X.contiguous()
        xa = silu_ref.scale_silu(X.reshape(-1), 1.0 / HC).reshape(4, K) if silu else X   # the matvec's input in llama.cpp
        R1 = {k: torch.stack([r.mmvf(w, xa[j].contiguous(), N, K, 1) for j in range(4)]) for k, r in refs.items()}
        R4 = {k: r.mmvf(w, xa, N, K, 4).reshape(4, N) for k, r in refs.items()}
        n_all += 3; n_ok += check(torch.equal(R1['shipped'].view(torch.int32), R1['source'].view(torch.int32)), f'card {card} {name}: shipped != source (ncols_dst 1)')
        n_ok += check(torch.equal(R4['shipped'].view(torch.int32), R4['source'].view(torch.int32)), f'card {card} {name}: shipped != source (ncols_dst 4)')
        n_ok += check(torch.equal(R4['shipped'].view(torch.int32), R1['shipped'].view(torch.int32)), f'card {card} {name}: ncols_dst 4 != ncols_dst 1 per token')
        ref = R1['shipped'].view(torch.int32)
        for nw in WGP:
            for rpt in RPT1:
                e = k3.Engine(module=M1, grid=grid(M1, nw)); out = torch.full((N,), float('nan'), device=dev)
                e.bf16_mv(w, N, K, X[0].contiguous(), out, rows_per_task=rpt, silu_scale=1.0 / HC if silu else None)
                e.build().run()
                n_all += 1; n_ok += check(torch.equal(out.view(torch.int32), ref[0]), f'card {card} {name} wgp {nw} rpt {rpt} T=1: engine != mul_mat_vec_f '
                                          f'({int((out.view(torch.int32) != ref[0]).sum())} of {N} rows, max |d| {float((out - R1["shipped"][0]).abs().max()):.3e})')
            for T in (2, 3, 4):
              for rpt in (16 if nw != 5 else 5,) + RPTM:
                e = k3.Engine(module=M1, grid=grid(MT, nw), T=T); assert e.module == MT
                out = torch.full((T * N,), float('nan'), device=dev)
                e.bf16_mv(w, N, K, X[:T].contiguous().reshape(-1), out, rows_per_task=rpt, silu_scale=1.0 / HC if silu else None)
                e.build().run()
                n_all += 1; n_ok += check(torch.equal(out.view(torch.int32).reshape(T, N), ref[:T]), f'card {card} {name} wgp {nw} T={T} rpt {rpt}: rows != T=1 / mul_mat_vec_f')
        if name == 'blk.0.ssm_alpha.weight':   # (c) llama.cpp's fused bias add (the dt bias of the alpha path)
            bias = torch.randn(N, device=dev, generator=gen)
            fz = {k: r.mmvf(w, X[0].contiguous(), N, K, 1, bias=bias) for k, r in refs.items()}
            e = k3.Engine(module=M1, grid=grid(M1, 8)); out = torch.full((N,), float('nan'), device=dev)
            e.bf16_mv(w, N, K, X[0].contiguous(), out); e.build().run()
            n_all += 1; n_ok += check(all(torch.equal((out + bias).view(torch.int32), v.view(torch.int32)) for v in fz.values()),
                                      f'card {card} {name}: engine + bias != mul_mat_vec_f<..., has_fusion> (x_bias)')
            print(f'  card {card} {name}: fused-bias kernel == engine + fp32 add', flush=True)
        print(f'  card {card} {name:32s} N {N:5d} K {K:5d} W {k3.mmvf_warps(K)}{" silu" if silu else ""}: '
              f'refs agree; T=1 at {WGP} x rpt {RPT1}, T=2/3/4 x rpt (16 | 5){RPTM} done', flush=True)
        del w
    return n_ok, n_all


provenance(); g2()
tot_ok = tot = 0
for card in CARDS:
    torch.cuda.set_device(card)
    print(f'card {card}: {torch.cuda.get_device_name(card)}, grids T=1 {[grid(M1, n) for n in WGP]}', flush=True)
    ok_, all_ = run_card(card); tot_ok += ok_; tot += all_
    print(f' card {card}: {ok_}/{all_} bitwise checks pass', flush=True)
    torch.cuda.empty_cache()
print(f'checks: {tot_ok}/{tot}; failures: {len(fails)}')
print('PASS' if not fails else 'FAIL'); sys.exit(0 if not fails else 1)
