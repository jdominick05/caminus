"""DFlash K3 drafter (python/caminus/dflash_k3.py) vs llama.cpp's dflash refdump (tools/dflash_dump.cpp via
tools/dflash_refdump.sh -> $ARTIFACTS/drafters/df-refdump/<text>-n3[-N128]), on the XT alone (solo build: no target, the
head's two row shards both on the XT, no XTX launch, no exchange).
  ops   : every drafter op fed llama.cpp's own input tensor, its output compared with llama.cpp's (bitwise unless noted)
  e2e   : llama.cpp's call sequence replayed (its injected K / V written into the ring, or K3's own inject from the dumped target
          features with --own-inject); every draft (passes 0..) compared with llama.cpp's drafts
  accept: the device accept vs the host rule; adv: OP_DF_ACCEPT / OP_DF_SELECT on garbage exchanged ids (flags, clamps)
usage (XT alone): HIP_VISIBLE_DEVICES=1 DF_XT=0 guard.sh LOG SECS python -I tests/test_dflash_k3_ops.py ops code
                  ... e2e-all prose-n3-N128,code-n3-N128,shell-n3-N128      (e2e TEXT [--own-inject]: one text)"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import k3
from caminus import dflash_k3 as dk

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
CODER = devpaths.coder()
REF = devpaths.artifacts('drafters', 'df-refdump')
XT = int(os.environ.get('DF_XT', '1'))   # HIP_VISIBLE_DEVICES=1 DF_XT=0: the XT alone


class Dump:
    def __init__(self, d):
        self.d = d; self.ix = {}
        for l in open(os.path.join(d, 'index.jsonl')):
            r = json.loads(l)
            self.ix.setdefault((r['ctx'], r['eval'], r['name']), r)
        self.passes = [json.loads(l) for l in open(os.path.join(d, 'passes.jsonl'))]

    def has(self, ctx, ev, name): return (ctx, ev, name) in self.ix

    def get(self, ctx, ev, name):
        r = self.ix[(ctx, ev, name)]; ne = r['ne']
        a = np.fromfile(os.path.join(self.d, r['file']), dtype=np.float32)
        n = ne[0] * ne[1] * ne[2] * ne[3]; assert a.size == n, (name, a.size, ne)
        return a.reshape(ne[3] * ne[2] * ne[1], ne[0]) if ne[0] else a


def dev(a, dt=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a)).to(f'cuda:{XT}', dt)


def eng():
    with torch.cuda.device(XT):
        return k3.Engine(module=dk.MODULE, grid=k3.resident_grid(dk.MODULE), T=4, limit_us=200_000)


def run(E):
    with torch.cuda.device(XT):
        E.build(); E.run(check=True); torch.cuda.synchronize()


RES = []


def cmp(name, got, ref, bitwise=True):
    g = got.detach().cpu().numpy().astype(np.float32).reshape(-1) if torch.is_tensor(got) else np.asarray(got, np.float32).reshape(-1)
    r = (ref.detach().float().cpu().numpy() if torch.is_tensor(ref) else np.asarray(ref, np.float32)).astype(np.float32).reshape(-1)
    assert g.size == r.size, (name, g.size, r.size)
    same = np.array_equal(g.view(np.uint32), r.view(np.uint32))
    nd = int((g.view(np.uint32) != r.view(np.uint32)).sum())
    md = float(np.max(np.abs(g.astype(np.float64) - r))) if not same else 0.0
    rel = md / max(1e-30, float(np.max(np.abs(r))))
    ok = same if bitwise else rel < 2e-3
    RES.append((name, ok, bitwise, nd, g.size, md, rel))
    print(f'{"PASS" if ok else "FAIL"} {name:40s} {"bitwise" if same else f"diff {nd}/{g.size} max {md:.3g} rel {rel:.2e}"}'
          f'{"" if bitwise else " (tolerance)"}', flush=True)
    return same


def q8(E, x):
    n = x.numel(); q = torch.zeros((n + 31) // 32 * 36, dtype=torch.uint8, device=x.device)
    return E.gg_q8_1(x, q), q


def ops(text):
    D = Dump(os.path.join(REF, text if '-n3' in text else f'{text}-n3'))
    df = dk.DFlashK3(None, T=4, target_gguf=CODER, devs=(XT, XT))
    L, T, Dm, hd = df.L, 4, df.D, df.hd
    B = lambda *s: torch.zeros(*s, device=f'cuda:{XT}')
    pos = torch.zeros(1, dtype=torch.int32, device=f'cuda:{XT}')
    # ---- inject (evals 2, 3: 4-row verify passes) ----
    for ev in (2, 3):
        p0 = D.passes[ev - 2]['pos']; pos.fill_(p0)
        feats = np.stack([D.get('tgt', ev, f'layer_inp-{l}') for l in df.layers_tgt])        # [5, T, D]
        f = dev(feats); fq = torch.zeros(T * 5 * Dm // 32 * 36, dtype=torch.uint8, device=f.device); g0, gn = B(T, Dm), B(T, Dm)
        E = eng(); s = dk.q8_perm(E, f, fq, 5, Dm); s = df._mv(E, df.fc, fq, g0, s); df._rms(E, g0, df.enc_norm, gn, dep=s); run(E)
        cmp(f'inject{ev}.fc+rms (inp_g_embeddings)', gn, D.get('inj', ev, 'inp_g_embeddings'))
        gref = dev(D.get('inj', ev, 'inp_g_embeddings'))
        for il in range(df.nl):
            E = eng(); s, gq = q8(E, gref); kr, vr, ko = B(T, 1024), B(T, 1024), B(T, 1024)
            k_ = df._mv(E, L[il]['k'], gq, kr, s); df._mv(E, L[il]['v'], gq, vr, s)
            df._qkn(E, kr, df.nkv, L[il]['k_norm'], pos, ko, dep=k_); run(E)
            cmp(f'inject{ev}.K{il} (Kcur_injected)', ko, D.get('inj', ev, f'Kcur_injected-{il} (view)'))
            cmp(f'inject{ev}.V{il} (Vcur_injected)', vr, D.get('inj', ev, f'Vcur_injected-{il} (view)'))
    # ---- draft (evals 2, 3) ----
    for ev in (2, 3):
        ps = D.passes[ev - 2]; n0, anchor = ps['pos'], ps['last']; pos.fill_(n0)
        ids = torch.tensor([anchor] + [df.mask_id] * 3, dtype=torch.int32, device=f'cuda:{XT}')
        x = B(T, Dm); E = eng(); dk.df_embed(E, df.tok_embd, ids, x, T, Dm, df.emb_vocab, df.emb_rb); run(E)
        cmp(f'draft{ev}.embed (inp_noise_embd)', x, D.get('dft', ev, 'inp_noise_embd'))
        ctx = ring_from_dump(df, D, ev - 1)                         # llama.cpp's injected K / V for every position < n0
        src = D.get('dft', ev, 'inp_noise_embd')
        for il in range(df.nl):
            Li = L[il]; g = lambda n: D.get('dft', ev, f'{n}-{il}')
            xin = dev(src); nn = B(T, Dm); nnq = torch.zeros(T * Dm // 32 * 36, dtype=torch.uint8, device=xin.device)
            E = eng(); df._rms(E, xin, Li['attn_norm'], nn, q8=nnq); run(E); cmp(f'draft{ev}.L{il} attn_norm', nn, g('noise_norm'))
            nref = dev(g('noise_norm')); dyn, nc = B(T, 640), B(T, Dm)
            E = eng(); s, q_ = q8(E, nref); s = df._mv(E, Li['ca_proj'], q_, dyn, s)
            dk.df_conv(E, nref, dyn, Li['ca_base'], nc, 0, df.ks, df.gs, dep=s); run(E); cmp(f'draft{ev}.L{il} conv_proj+conv0', nc, g('attn_conv_in'))
            cref = dev(g('attn_conv_in')); qf, kf, vf, qo, ko = B(T, 4096), B(T, 1024), B(T, 1024), B(T, 4096), B(T, 1024)
            E = eng(); s, cq = q8(E, cref); a = df._mv(E, Li['q'], cq, qf, s); b = df._mv(E, Li['k'], cq, kf, s); df._mv(E, Li['v'], cq, vf, s)
            df._qkn(E, qf, df.nh, Li['q_norm'], pos, qo, dep=a)
            df._qkn(E, kf, df.nkv, Li['k_norm'], pos, ko, dep=b); run(E)
            cmp(f'draft{ev}.L{il} q mv+norm+rope', qo, g('Qcur')); cmp(f'draft{ev}.L{il} k mv+norm+rope', ko, g('Kcur'))
            cmp(f'draft{ev}.L{il} v mv', vf, g('Vcur'))
            # the fused forms: one q | k | v matvec, q norm + RoPE, k norm + RoPE with K / V written into the ring (f16)
            qkv, qo2 = B(T, 6144), B(T, 4096); E = eng(); s, cq = q8(E, cref); m_ = df._mv(E, Li['qkv'], cq, qkv, s)
            df._qkn(E, qkv, df.nh, Li['q_norm'], pos, qo2, dep=m_, tok_stride=6144)
            df._qkn(E, qkv.view(-1)[4096:], df.nkv, Li['k_norm'], pos, None, dep=m_, tok_stride=6144, ring=(df.cache[il], qkv.view(-1)[5120:])); run(E)
            cmp(f'draft{ev}.L{il} fused qkv: q', qo2, g('Qcur'))
            rows = torch.tensor([(n0 + t) & df.rmask for t in range(T)], device=qkv.device)
            cr = df.cache[il].view(torch.float16).view(df.ring, 2048)[rows].float()
            cmp(f'draft{ev}.L{il} fused ring K (f16 of Kcur)', cr[:, :1024], torch.from_numpy(g('Kcur').reshape(T, 1024)).half().float())
            cmp(f'draft{ev}.L{il} fused ring V (f16 of Vcur)', cr[:, 1024:], torch.from_numpy(g('Vcur').reshape(T, 1024)).half().float())
            # attention on llama.cpp's own context and block K / V (not bitwise: llama.cpp's flash-attention vec kernel sums
            # in its own order; tolerance 2e-3 of the max)
            set_ring(df, il, ctx[il], n0, dev(g('Kcur')).view(T, -1), dev(g('Vcur')).view(T, -1))
            att = B(T, 4096); E = eng()
            dk.attn_df(E, dev(g('Qcur')), df.cache[il], pos, att, df.nh, df.nkv, hd, df.scale, df.rmask, df.window); run(E)
            cmp(f'draft{ev}.L{il} attention (kqv_out)', att, g('kqv_out'), bitwise=False)
            att2 = B(T, 4096); attq2 = torch.zeros(T * 4096 // 32 * 36, dtype=torch.uint8, device=att.device); E = eng()
            dk.attn_df(E, dev(g('Qcur')), df.cache[il], pos, att2, df.nh, df.nkv, hd, df.scale, df.rmask, df.window, q8=attq2)
            s, aq_ref = q8(E, att2); E.segs[s].dep = s - 1; run(E)
            cmp(f'draft{ev}.L{il} attention q8_1 emit == gg_q8_1', attq2.view(torch.int32).float(), aq_ref.view(torch.int32).float())
            # the split form (partial pass + combine: the tapes' default, DFlashK3.ATTN_SPLIT)
            att3 = B(T, 4096); attq3 = torch.zeros(T * 4096 // 32 * 36, dtype=torch.uint8, device=att.device); E = eng()
            dk.attn_df(E, dev(g('Qcur')), df.cache[il], pos, att3, df.nh, df.nkv, hd, df.scale, df.rmask, df.window, q8=attq3, part=df.S['attp'])
            s, aq_ref3 = q8(E, att3); E.segs[s].dep = s - 1; run(E)
            cmp(f'draft{ev}.L{il} split attention (kqv_out)', att3, g('kqv_out'), bitwise=False)
            cmp(f'draft{ev}.L{il} split attention q8_1 emit == gg_q8_1', attq3.view(torch.int32).float(), aq_ref3.view(torch.int32).float())
            aref = dev(g('kqv_out')); ob, co, fi = B(T, Dm), B(T, Dm), B(T, Dm)
            E = eng(); s, aq = q8(E, aref); s = df._mv(E, Li['o'], aq, ob, s)
            s2 = dk.df_conv(E, ob, dyn_of(df, Li['ca_proj'], nref), Li['ca_base'], co, 1, df.ks, df.gs, dep=s)
            dk.df_conv(E, ob, dyn_of(df, Li['ca_proj'], nref), Li['ca_base'], fi, 1, df.ks, df.gs, res=xin, dep=s2); run(E)
            cmp(f'draft{ev}.L{il} o mv+conv1', co, g('attn_conv_out')); cmp(f'draft{ev}.L{il} +residual (ffn_inp)', fi, g('ffn_inp'))
            firef = dev(g('ffn_inp')); hn = B(T, Dm)
            E = eng(); df._rms(E, firef, Li['ffn_norm'], hn); run(E); cmp(f'draft{ev}.L{il} ffn_norm', hn, g('ffn_norm'))
            href = dev(g('ffn_norm')); dynf, hc = B(T, 640), B(T, Dm)
            E = eng(); s, q_ = q8(E, href); s = df._mv(E, Li['cf_proj'], q_, dynf, s)
            dk.df_conv(E, href, dynf, Li['cf_base'], hc, 0, df.ks, df.gs, dep=s); run(E); cmp(f'draft{ev}.L{il} ffn conv_proj+conv0', hc, g('ffn_conv_in'))
            hcref = dev(g('ffn_conv_in')); hh, hu, ff = B(T, df.ffn), B(T, df.ffn), B(T, Dm)
            hhq = torch.zeros(T * df.ffn // 32 * 36, dtype=torch.uint8, device=hh.device)
            E = eng(); s, q_ = q8(E, hcref); a1 = df._mv(E, Li['gate'], q_, hh, s); a2 = df._mv(E, Li['up'], q_, hu, s)
            s = E.also(E.gg_q8_1(hh, hhq, dep=a1, mul=hu), a2); df._mv(E, Li['down'], hhq, ff, s); run(E)
            cmp(f'draft{ev}.L{il} gate/up (swiglu q8)/down (ffn_out)', ff, g('ffn_out'))
            # the fused forms: OP_DF_MVCONV (down + conv1 + residual -> l_out), from llama.cpp's own swiglu input
            lo2 = B(T, Dm); dyf = dynf_of(df, Li['cf_proj'], href)
            E = eng(); s = E.also(E.gg_q8_1(hh, hhq, mul=hu), -1) if False else E.gg_q8_1(hh, hhq, mul=hu)
            dk.df_mvconv(E, Li['down'], hhq, lo2, dyf, Li['cf_base'], firef, 1, df.ks, df.gs, dep=s); run(E)
            cmp(f'draft{ev}.L{il} MVCONV down+conv1+res (l_out)', lo2, g('l_out'))
            lo3 = B(T, Dm); E = eng(); s, aq = q8(E, aref)
            dk.df_mvconv(E, Li['o'], aq, lo3, dyn_of(df, Li['ca_proj'], nref), Li['ca_base'], xin, 1, df.ks, df.gs, dep=s); run(E)
            cmp(f'draft{ev}.L{il} MVCONV o+conv1+res (ffn_inp)', lo3, g('ffn_inp'))
            ffref = dev(g('ffn_out')); co2, lo = B(T, Dm), B(T, Dm)
            E = eng(); s2 = dk.df_conv(E, ffref, dynf_of(df, Li['cf_proj'], href), Li['cf_base'], co2, 1, df.ks, df.gs)
            dk.df_conv(E, ffref, dynf_of(df, Li['cf_proj'], href), Li['cf_base'], lo, 1, df.ks, df.gs, res=firef, dep=s2); run(E)
            cmp(f'draft{ev}.L{il} ffn conv1', co2, g('ffn_conv_out')); cmp(f'draft{ev}.L{il} l_out', lo, g('l_out'))
            src = g('l_out')
        emb = B(T, Dm); E = eng(); df._rms(E, dev(src), df.out_norm, emb); run(E); cmp(f'draft{ev}.output_norm', emb, D.get('dft', ev, 'result_norm'))
        eref = dev(D.get('dft', ev, 'result_norm'))
        lgs = [B(T, df.head[c]['rows']) for c in (0, 1)]
        E = eng(); s, eq = q8(E, eref)
        for c in (0, 1): df._mv(E, df.head[c], eq, lgs[c], s, rpt=64)
        run(E)
        cmp(f'draft{ev}.head (result_output)', torch.cat(lgs, 1), D.get('dft', ev, 'result_output'))
        # top-k per shard + selector + walk from llama.cpp's logits and output norm
        lref = D.get('dft', ev, 'result_output'); gate = B(T, df.rank); out = torch.zeros_like(df.S['out'])
        tkx = df.H[1]['tkx']; tkx.zero_()
        anc = torch.tensor([anchor], dtype=torch.int32, device=f'cuda:{XT}')
        E = eng(); s, eq = q8(E, eref); gt = df._mv(E, df.sel_hidden, eq, gate, s); ms = []
        for c in (0, 1):
            hw = df.head[c]; lc = dev(lref[:, df.head_off[c]:df.head_off[c] + hw['rows']])
            sc, nch = dk.df_topk_scan(E, lc, hw['rows'], 1, df.H[c]['tk0'], df.topk, df.head_off[c])
            ms.append(dk.df_topk_merge(E, df.H[c]['tk0'], df.k, nch * df.topk, tkx[c * df.k * df.topk * 2:(c + 1) * df.k * df.topk * 2], df.topk, dep=sc))
        E.also(ms[1], ms[0])
        sl = dk.df_select(E, tkx, 2, df.sel_prev, df.sel_next, gate, anc, out, df.rank, df.topk, df.sel_rb, df.vocab, dep=ms[1]); E.also(sl, gt); run(E)
        hw = df.head[0]
        o = out.cpu().numpy(); k, kc = df.k, df.topk
        lat = D.get('dft', ev, 'dflash2_lattice')                     # [T][n_embd]: row i: cand [kc] (float), scores [kc][kc]
        cand_k3 = o[k:k + k * kc].reshape(k, kc); un_k3 = o[k + k * kc:k + 2 * k * kc].view(np.float32).reshape(k, kc)
        sc_k3 = o[k + 2 * k * kc:].view(np.float32).reshape(k, kc)
        lg_ref = D.get('dft', ev, 'result_output')
        ok_sets = all(set(cand_k3[i].tolist()) == set(lat[i + 1, :kc].astype(np.int64).tolist()) for i in range(k))
        un_ok = all(np.array_equal(un_k3[i], lg_ref[i + 1][cand_k3[i]]) for i in range(k))
        print(f'{"PASS" if ok_sets else "FAIL"} draft{ev}.topk candidate sets == llama.cpp lattice', flush=True); RES.append(('topk sets', ok_sets, True, 0, 0, 0, 0))
        print(f'{"PASS" if un_ok else "FAIL"} draft{ev}.topk unary == logits at the candidates (bitwise)', flush=True); RES.append(('topk unary', un_ok, True, 0, 0, 0, 0))
        # walked score rows vs the lattice's (llama.cpp's F32 matmul: its own order -> tolerance), and the drafts
        pr = 0; walk = []; worst = 0.0
        for i in range(k):
            row = lat[i + 1]; ids_ref = row[:kc].astype(np.int64)
            sref = row[kc + pr * kc:kc + pr * kc + kc]
            # align llama.cpp's candidate order to K3's
            perm = [int(np.where(ids_ref == c)[0][0]) for c in cand_k3[i]]
            worst = max(worst, float(np.max(np.abs(sref[perm] - sc_k3[i]))) / max(1e-9, float(np.max(np.abs(sref)))))
            pr = int(np.argmax(sref)); walk.append(int(ids_ref[pr]))
        dk3 = o[:k].tolist()
        print(f'{"PASS" if worst < 1e-4 else "FAIL"} draft{ev}.selector scores rel diff {worst:.2e} (tolerance)', flush=True)
        RES.append(('scores', worst < 1e-4, False, 0, 0, worst, worst))
        same = dk3 == walk == ps['draft']
        print(f'{"PASS" if same else "FAIL"} draft{ev}.drafts K3 {dk3} llama.cpp lattice walk {walk} pass log {ps["draft"]}', flush=True)
        RES.append(('drafts', same, True, 0, 0, 0, 0))
    nb = sum(1 for r in RES if r[2]); nbo = sum(1 for r in RES if r[2] and r[1]); nt = sum(1 for r in RES if not r[2]); nto = sum(1 for r in RES if not r[2] and r[1])
    print(f'RESULT ops {text}: bitwise {nbo}/{nb}, tolerance {nto}/{nt}', flush=True)
    return nbo == nb and nto == nt


def accept_test(n_cases=24, seed=0):
    """OP_DF_TOPK work 2 + OP_DF_ACCEPT (kind 'step's accept, solo: both verify shards scanned on the XT) vs the host rule
    (caminus.flashnext_tp._run: torch.max per shard, card 0 on a tie, a NaN counts as the maximum) on synthetic logits: random
    rows, ties inside a shard / across the shards / across chunks, +0 vs -0, +-inf, NaN (both signs); then the accept (F, j,
    the bonus, pos_d) vs the host loop for random drafts"""
    df = dk.DFlashK3(None, T=4, target_gguf=CODER, devs=(XT, XT), max_pos=2051)
    T, vh, V = 4, df.head[0]['rows'], df.vocab
    st = df.engines['step'][1]; E = st.E
    # the accept segment: the tape's step kind has it after the scans (index of OP_DF_ACCEPT)
    ia = next(i for i, sg in enumerate(E.segs) if sg.op == k3.OP_DF_ACCEPT)
    rng = np.random.default_rng(seed); ok_all = True; nck = 0
    def host_tok(L):
        out = []
        lg = [torch.from_numpy(L[:, :vh]).to(f'cuda:{XT}'), torch.from_numpy(L[:, vh:]).to(f'cuda:{XT}')]
        (v0, x0), (v1, x1) = [tuple(t.cpu().numpy() for t in torch.max(l_, dim=1)) for l_ in lg]
        return [int(x0[t]) if (v0[t] >= v1[t] or v0[t] != v0[t]) else vh + int(x1[t]) for t in range(T)]
    for case in range(n_cases):
        L = rng.standard_normal((T, V)).astype(np.float32) * 3
        kind = case % 8
        for t in range(T):
            m = float(L[t].max()) + 1.0
            if kind == 1: L[t, [5, 77777]] = m                              # a tie inside shard 0 (two chunks)
            if kind == 2: L[t, [vh - 1, vh + 3]] = m                        # a tie across the shards
            if kind == 3: L[t, :] = -np.abs(L[t]); L[t, 1000] = -0.0; L[t, vh + 9] = 0.0          # -0 first, +0 later: equal
            if kind == 4: L[t, :] = -np.abs(L[t]); L[t, vh + 50] = 0.0; L[t, 60000] = -0.0
            if kind == 5: L[t, [vh + 7, 3000]] = np.inf                     # +inf in both shards
            if kind == 6: L[t, vh + 11] = np.nan; L[t, 4] = np.float32(np.uint32(0xFFC00000).view(np.float32))   # NaN, -NaN
            if kind == 7: L[t, 12345] = np.nan; L[t, vh + 1] = np.nan; L[t, 100] = np.inf
        for c in (0, 1):
            df.H[c]['vlg'].copy_(torch.from_numpy(np.ascontiguousarray(L[:, c * vh:(c + 1) * vh] if c == 0 else L[:, vh:])))
        ref = host_tok(L)
        F = int(rng.integers(1, T + 1)); i0 = int(rng.integers(0, 2051 - 2 * T))
        drafts = [int(ref[F - 1 + j]) if rng.random() < 0.6 else int(rng.integers(0, V)) for j in range(T - F)]
        toks = [int(rng.integers(0, V)) for _ in range(F)] + drafts
        df.reset(); df._test_mark_injected(0, i0)   # (the accept on whatever the ring holds: rows declared injected)
        df._launch('step', i0=i0, F=F, toks=toks)
        r = df.record()
        j = 0
        while j < T - F and toks[F + j] == ref[F - 1 + j]: j += 1
        ok = r['a'] == ref and r['j'] == j and r['anchor'] == ref[F - 1 + j] and r['pos_d'] == i0 + F + j and r['flag'] == 0
        ok_all &= ok; nck += 1
        print(f'{"PASS" if ok else "FAIL"} accept case {case} (kind {kind}): device a {r["a"]} host {ref}; F {F} j {r["j"]}/{j} '
              f'bonus {r["anchor"]} pos_d {r["pos_d"]} (i0 {i0}) flag {r["flag"]}', flush=True)
    # an out-of-range F (host check bypassed): the device clamps and flags, the host refuses
    df.reset(); df.n_inj = 100
    try:
        df._start('step', i0=100, F=9, toks=[1, 2, 3, 4]); df._wait('step'); bad_ok = False
    except k3.ContextBoundsError as e:
        bad_ok = 'flag 1' in str(e)
    print(f'{"PASS" if bad_ok else "FAIL"} accept: F = 9 (> T) flagged on the device and refused', flush=True)
    df.reset()
    print(f'RESULT accept: {"PASS" if ok_all and bad_ok else "FAIL"} ({nck} cases)', flush=True)
    return ok_all and bad_ok


def adv_test():
    """OP_DF_ACCEPT / OP_DF_SELECT on garbage exchanged values (a corrupted exchange or a bailed task's stale slot; never
    produced by the scans), XT alone: every id outside [0, vocab) that reaches a row of a (NaN, +-inf, negative, >= vocab,
    huge) sets flag 2, whether it lands on the anchor or on another row; the clean pairs give flag 0; OP_DF_SELECT's drafts
    stay in [0, vocab) whatever its candidate ids / anchor"""
    df = dk.DFlashK3(None, T=4, target_gguf=CODER, devs=(XT, XT), max_pos=2051)
    T, V, D_ = df.T, df.vocab, f'cuda:{XT}'; ok_all = True
    def rep(name, cond, info=''):
        nonlocal ok_all
        ok_all &= bool(cond); print(f'{"PASS" if cond else "FAIL"} adv {name} {info}', flush=True)
    E = eng(); n4 = T * df.nch_v * 2
    pairs = torch.zeros(2 * n4, device=D_); blk = torch.zeros(64, dtype=torch.int32, device=D_); rec = torch.zeros(T + 4, dtype=torch.int32, device=D_)
    dk.df_accept(E, blk, rec, 1, V, 2051, pairs=pairs, nch=df.nch_v)
    with torch.cuda.device(XT): E.build()
    def accept(row, bad_id, toks, F):
        """every pair (value -1, id 5 + row), then card 0's chunk 0 of `row` wins that row with id bad_id (None: id 77)"""
        p = torch.full((2 * n4,), -1.0)
        p[1::2] = torch.tensor([5.0 + (q // df.nch_v) % T for q in range(2 * T * df.nch_v)])   # row r's ids: 5 + r
        q = row * df.nch_v; p[2 * q] = 10.0; p[2 * q + 1] = 77.0 if bad_id is None else bad_id
        pairs.copy_(p.to(D_))
        h = torch.zeros(64, dtype=torch.int32); h[16] = 100; h[20] = F; h[21:21 + T] = torch.tensor(toks, dtype=torch.int32)
        blk.copy_(h.to(D_))
        with torch.cuda.device(XT): E.run(check=True); torch.cuda.synchronize()
        return rec.cpu().tolist(), int(blk[19].item())
    for row, F, toks, where in ((0, 1, [1, 2, 3, 4], 'the anchor (F 1, j 0)'), (3, 1, [1, 0, 0, 0], 'row 3, not the anchor (F 1, j 0)'),
                                (1, 4, [1, 2, 3, 4], 'row 1, not the anchor (F 4)')):
        r, b19 = accept(row, None, toks, F)
        want = [77 if t == row else 5 + t for t in range(T)]
        rep(f'clean pairs, winner at {where}', r[:T] == want and r[T + 3] == 0, f'rec {r}')
        for nm, bad in (('NaN', float('nan')), ('-NaN', float(np.uint32(0xFFC00000).view(np.float32))), ('+inf', float('inf')),
                        ('-inf', -float('inf')), ('-3', -3.0), ('-0.5', -0.5), ('vocab', float(V)), ('1e9', 1e9), ('1e30', 1e30)):
            r, b19 = accept(row, bad, toks, F)
            rep(f'id {nm} at {where}: flag 2, anchor in [0, vocab)', (r[T + 3] & 2) != 0 and 0 <= r[T + 1] < V and 0 <= b19 < V, f'rec {r}')
    # OP_DF_SELECT: garbage candidate ids / anchor -> drafts clamped into [0, vocab)
    E = eng(); k, kc, H = T - 1, df.topk, 3
    cand = torch.zeros(H * k * kc * 2, device=D_); gate = torch.randn(T * df.rank, device=D_)
    anchor = torch.zeros(1, dtype=torch.int32, device=D_); out = torch.zeros(k + 3 * k * kc, dtype=torch.int32, device=D_)
    dk.df_select(E, cand, H, df.sel_prev, df.sel_next, gate, anchor, out, df.rank, kc, df.sel_rb, df.vocab)
    with torch.cuda.device(XT): E.build()
    garb = [float('nan'), -5.0, float(V + 100), float(2 ** 30), float('inf'), -float('inf'), -1.0, float(V), 1e30, -0.5]
    for trial, an in enumerate((-7, V + 3, 2 ** 31 - 1, 5, 0)):
        c = torch.randn(H * k * kc, 2)
        c[:, 1] = torch.tensor([garb[(i + trial) % len(garb)] if (i + trial) % 3 else float(i * 7 % V) for i in range(H * k * kc)])
        if trial == 4: c[:, 1] = torch.tensor([garb[i % len(garb)] for i in range(H * k * kc)])   # every id garbage
        cand.copy_(c.reshape(-1).to(D_)); anchor.fill_(an)
        with torch.cuda.device(XT): E.run(check=True); torch.cuda.synchronize()
        o = out.cpu().tolist()
        rep(f'select garbage ids / anchor {an}: drafts in [0, vocab)', all(0 <= x < V for x in o[:k]), f'drafts {o[:k]}')
    df.reset()
    print(f'RESULT adv: {"PASS" if ok_all else "FAIL"}', flush=True)
    return ok_all


def ring_test():
    """OP_KV_APPEND ring form (the remote inject's XT side: K / V rows received fp32 -> f16 ring rows) vs torch's f16 of the
    same rows, at positions wrapping the ring; XT alone"""
    df = dk.DFlashK3(None, T=4, target_gguf=CODER, devs=(XT, XT), max_pos=2051)
    M = df.nkv * df.hd; ok = True
    for p0 in (0, 1000, df.ring - 2, 2047):
        K = torch.randn(4, M, device=f'cuda:{XT}'); V = torch.randn(4, M, device=f'cuda:{XT}')
        K[0, 5] = -0.0; V[1, 7] = 1e-40   # (a -0 and an fp32 denormal: f16 conversion as torch's)
        pos = torch.tensor([p0], dtype=torch.int32, device=f'cuda:{XT}')
        E = eng(); dk.kv_ring(E, K, V, df.cache[0], pos, df.rmask); run(E)
        c = df.cache[0].view(torch.float16).view(df.ring, 2 * M)
        rows = torch.tensor([(p0 + t) & df.rmask for t in range(4)], device=c.device)
        got = c[rows]; ref = torch.cat([K, V], 1).half()
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16)); ok &= same
        print(f'{"PASS" if same else "FAIL"} ring append at {p0} (rows {rows.tolist()}) == torch f16 bitwise', flush=True)
    print(f'RESULT ring: {"PASS" if ok else "FAIL"}', flush=True)
    return ok


def dyn_of(df, W, nref):
    E = eng(); s, q_ = q8(E, nref); dyn = torch.zeros(4, 640, device=nref.device); df._mv(E, W, q_, dyn, s); run(E); return dyn


dynf_of = dyn_of


def ring_from_dump(df, D, last_inject):
    """{layer: {pos: (K row, V row)}} from llama.cpp's inject evals 1 .. last_inject (later rows overwrite)"""
    ctx = [dict() for _ in range(df.nl)]
    for ev in range(1, last_inject + 1):
        n = D.ix[('inj', ev, 'inp_g_embeddings')]['ne'][1]
        p0 = 0 if ev == 1 else D.passes[ev - 2]['pos']
        for il in range(df.nl):
            K = D.get('inj', ev, f'Kcur_injected-{il} (view)'); V = D.get('inj', ev, f'Vcur_injected-{il} (view)')
            for j in range(n): ctx[il][p0 + j] = (K[j], V[j])
    return ctx


def set_ring(df, il, ctx_l, n0, Kb=None, Vb=None):
    """ring rows of layer il: positions < n0 from ctx_l (f16, as llama.cpp's cpy to its f16 cache), the block's from Kb / Vb.
    The positions written below n0 are declared injected to the drafter's slot tags (the host refuses a draft whose window
    reads a slot without its own injected row; these rows are llama.cpp's, written here)"""
    M = df.nkv * df.hd; c = df.cache[il].view(torch.float16).view(df.ring, 2 * M)
    ps = sorted(p for p in ctx_l if p < n0)[-df.ring:]
    if ps:
        K = np.stack([ctx_l[p][0] for p in ps]); V = np.stack([ctx_l[p][1] for p in ps])
        rows = torch.tensor([p & df.rmask for p in ps], device=c.device)
        c[rows] = torch.cat([dev(K), dev(V)], 1).to(torch.float16)
        if il == df.nl - 1:   # (once per position: after the last layer's rows)
            a = ps[0]
            for i in range(1, len(ps) + 1):   # contiguous runs
                if i == len(ps) or ps[i] != ps[i - 1] + 1: df._test_mark_injected(a, ps[i - 1] + 1); a = ps[i] if i < len(ps) else a
    if Kb is not None:
        rows = torch.tensor([(n0 + t) & df.rmask for t in range(4)], device=c.device)
        c[rows] = torch.cat([Kb, Vb], 1).to(torch.float16)


def e2e(text, own_inject):
    D = Dump(os.path.join(REF, text if '-n3' in text else f'{text}-n3'))
    df = dk.DFlashK3(None, T=4, target_gguf=CODER, devs=(XT, XT))
    df.reset()
    n_ok = n_tot = 0; mism = []
    ninj = max(e for (c, e, n) in D.ix if c == 'inj')
    times = []
    for p, ps in enumerate(D.passes):
        ev_inj = p + 1                                   # the inject before pass p's draft
        if ev_inj > ninj: break
        if own_inject:
            if ev_inj == 1:
                n = D.ix[('inj', 1, 'inp_g_embeddings')]['ne'][1]
                F = np.stack([D.get('tgt', 1, f'layer_inp-{l}') for l in df.layers_tgt])   # [5, n, D]
                for i0 in list(range(0, n - 3, 4)) + ([n - 4] if n % 4 else []):
                    df.feats.copy_(dev(F[:, i0:i0 + 4])); df.inject(i0); df.flush()
            else:
                F = np.stack([D.get('tgt', ev_inj, f'layer_inp-{l}') for l in df.layers_tgt])   # [5, n <= 4, D]
                if F.shape[1] < 4:   # a short verify (fewer drafts): pad; the padded rows' K / V are rewritten by the next block
                    F = np.concatenate([F, np.zeros((F.shape[0], 4 - F.shape[1], F.shape[2]), np.float32)], 1)
                df.feats.copy_(dev(F)); df.inject(D.passes[ev_inj - 2]['pos'])   # pending: fused into the draft
        else:
            ctx = ring_from_dump(df, D, ev_inj) if p == 0 else None
            if p == 0:
                for il in range(df.nl): set_ring(df, il, ctx[il], ps['pos'])
            else:
                p0 = D.passes[ev_inj - 2]['pos']
                for il in range(df.nl):
                    K = D.get('inj', ev_inj, f'Kcur_injected-{il} (view)'); V = D.get('inj', ev_inj, f'Vcur_injected-{il} (view)')
                    set_ring(df, il, {p0 + j: (K[j], V[j]) for j in range(K.shape[0])}, p0 + K.shape[0])
            df.n_inj = max(df.n_inj, ps['pos'])
        torch.cuda.synchronize(XT); t0 = time.perf_counter()
        d = df.draft(ps['last'], ps['pos'])
        times.append(time.perf_counter() - t0)
        n_tot += 1; ok = d == ps['draft']; n_ok += ok
        if not ok:
            cand, un, sc = df.details()
            mism.append((p, d, ps['draft'], [float(np.sort(s)[-1] - np.sort(s)[-2]) for s in sc], ll_margin(D, p, d, ps['draft'])))
    # a mismatch is excused when llama.cpp's draft is a prefix of K3's (llama.cpp shortened it at the -n limit) or when, at the
    # first differing position, llama.cpp's own walked score row has K3's pick within TIE of its own pick (a near-tie: the
    # attention's half-precision KQ in llama.cpp's flash-attention kernel moves the scores by ~1e-3 relative)
    TIE = 0.25
    trunc = [m for m in mism if m[1][:len(m[2])] == m[2]]
    ties = [m for m in mism if m not in trunc and m[4] is not None and m[4] <= TIE]
    other = [m for m in mism if m not in trunc and m not in ties]
    print(f'RESULT e2e {text} ({"K3 inject from the target features" if own_inject else "llama.cpp K / V"}): drafts equal {n_ok}/{n_tot}; '
          f'differ {len(mism)} = {len(trunc)} llama.cpp truncations + {len(ties)} near-ties (llama.cpp margin <= {TIE}) + {len(other)} other; '
          f'launch median {1e3 * np.median(times):.3f} ms', flush=True)
    for m in mism: print(f'  pass {m[0]}: K3 {m[1]} llama.cpp {m[2]} (K3 walked score margins {["%.3g" % v for v in m[3]]}; '
                         f'llama.cpp margin at the first difference {m[4] if m[4] is None else "%.3g" % m[4]})')
    return n_ok, n_tot, len(other)


def ll_margin(D, p, d_k3, d_ll):
    """llama.cpp's walked score row at the first position where the drafts differ: score(llama pick) - score(K3 pick); None
    if K3's pick is not among llama.cpp's candidates there (or the drafts do not differ within llama.cpp's length)"""
    ev = p + 2
    if not D.has('dft', ev, 'dflash2_lattice'): return None
    lat = D.get('dft', ev, 'dflash2_lattice'); kc = 16; pr = 0
    for i in range(min(len(d_k3), len(d_ll))):
        row = lat[i + 1]; ids = row[:kc].astype(np.int64); sc = row[kc + pr * kc:kc + pr * kc + kc]
        pick = int(np.argmax(sc))
        if d_k3[i] != d_ll[i]:
            w = np.where(ids == d_k3[i])[0]
            return None if not len(w) else float(sc[pick] - sc[w[0]])
        pr = pick
    return None


if __name__ == '__main__':
    mode, text = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(XT)
    for a in sys.argv[3:]:
        if a.startswith('--rms='): dk.DFlashK3.RMS_VAR = int(a[6:])
        if a.startswith('--qk='): dk.DFlashK3.QK_VAR = int(a[5:])
    print(f'variants rms {dk.DFlashK3.RMS_VAR} qk {dk.DFlashK3.QK_VAR}', flush=True)
    if mode == 'ops': sys.exit(0 if ops(text) else 1)
    if mode == 'accept': sys.exit(0 if accept_test() else 1)
    if mode == 'adv': sys.exit(0 if adv_test() else 1)
    if mode == 'ring': sys.exit(0 if ring_test() else 1)
    if mode == 'e2e': e2e(text, '--own-inject' in sys.argv)
    if mode == 'e2e-all':
        bad = 0
        for t in text.split(','):
            for own in (False, True): bad += e2e(t, own)[2]
        print(f'RESULT e2e-all: {bad} unexcused draft differences', flush=True)
        sys.exit(1 if bad else 0)
