"""CPU cross-check of the Coder's float64 reference (caminus.ref_coder) against llama.cpp's own intermediates (llama.cpp
here: the ignite-llama fork's libllama that build/gg_golden links): a gg_golden dump of the GSQ-RCO Coder (one token per
llama_decode, positions 0 .. NS - 1). Every check of a block starts from llama.cpp's own input of that block, so no drift from
the layers before hides or fakes a difference; then the whole forward on the dump's tokens.
Per dumped layer L (inputs: hc_init, or llama.cpp's l_last of layer L - 1 when it is dumped; its hc_mixed; its hc_combine):
  - PLE (the ple.layers layer): from l_last of the layer before, the gated value (the value times each stream's gate)
    and the conv's output (4 taps, 3 apart, from a fresh history: its oldest tap reaches a real row from position 9 on)
  - each hyper-connection mix (before the mixer, before the FFN): hc_mixed and the inject logits from the streams in
  - the mixer (Gated DeltaNet with its SIGMOID output gate, or attention) on llama.cpp's hc_mixed, from a fresh state
    over the dump's positions: ||d|| / ||llama|| of linear_attn_out / attn_output
  - the combine: llama.cpp's streams + its block output * 2 sigmoid(inject / 4) == its hc_combine / l_last
  - the FFN on llama.cpp's hc_mixed: the SAME 10 experts at every position, each expert's renormalized weight against
    llama.cpp's ffn_moe_weights_norm matched by expert id (|d| < 1e-5: catches a wrong router weighting, which a logits
    KLD does not), the output's ||d|| / ||llama||
  - the head (when layer 47 is dumped): result_norm (the output_hc mix) from l_last-47
Then the whole forward (CoderRef.feed on the dump's tokens): the streams after each dumped layer (||ref - llama|| /
||ref|| under the layer's line), the forward's top-10 against llama.cpp's, and the logits (mean KLD(ref || llama) < 0.01,
the same top token at 7 of 8 positions or more).
The stored q8_0 dump of a code text (8 positions, layers 0, 1, 6-11, 20, 21) is another text: run it with --text none.
Coverage: the dump must hold NEED (the PLE layer and its neighbours, attention layers early and late, both expert-down
formats IQ4_NL and Q2_0 deep in the stack, and layer 47's odd formats: IQ4_XS gate / up, Q8_0 shared down); a dump without
them, or with fewer than 10 positions, FAILs that check (the rest still runs). With --text the dump's tokens are checked against the own BPE of that text.
The dump (a GPU run on both cards: the Coder does not fit one; ~2 min with the load, 12 one-token steps):
  tools/guard.sh LOG 900 build/gg_golden $CODER GOLDEN_DIR "$(python -I -c 'import sys;
      sys.path[:0] = ["python", "tools/reference"]; from q3moe_ref import TEXT; print(TEXT)')" 12 0,1,2,3,20,21,35,36,45,46,47 f16
CPU only, but the whole forward reads every routed expert of the 8 positions (a few GB of the file through the page cache):
never run it beside a GPU timing or gate. The reader is caminus.ggufio; the n-gram hash is caminus.prefill.feed's.
  usage: python -I tests/engine/test_coder_golden_cpu.py GOLDEN_DIR GGUF [--text public|none] [--threads N]"""
import argparse
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), os.path.join(HERE, '..', '..', 'tools', 'reference')]
from caminus.ref_coder import CoderRef   # noqa: E402

ap = argparse.ArgumentParser(usage=__doc__); ap.add_argument('golden'); ap.add_argument('gguf')
ap.add_argument('--text', default='public', help="'public': tools/reference/q3moe_ref.py's TEXT; 'none': skip the token check")
ap.add_argument('--threads', type=int, default=None)
A = ap.parse_args()
G = A.golden
rows = [l.rstrip('\n').split('\t') for l in open(f'{G}/index.tsv')]
RES = []
NEED = (0, 1, 2, 3, 20, 21, 35, 36, 45, 46, 47)
# lines: about twice the worst values measured on llama.cpp's stored q8_0-KV dump of layers 0, 1, 6-11, 20, 21 (8 positions of
# a code text; 2026-11-19): mix / inject 1.4e-7, PLE 2.0e-7, combine 9.0e-8 (float32 rounding: no quantized step), the mixer
# 2.0e-2 (attention) / 1.2e-2 (GDN), the FFN 1.2e-2 (llama.cpp's q8_1 activations), the forward's drift 3.7e-2 at layer 21,
# logits KLD 2.8e-4 with 8 / 8. The head and the layers past 21 were not in that dump: RESID is set wider (4x) for them
TOL = dict(mix=1e-6, inject=1e-6, ple=1e-6, combine=1e-6, mixer=0.05, ffn=0.03, head=1e-6)
RESID = 0.15          # the forward's streams vs llama.cpp's l_last at any dumped layer


def ok(name, cond, info=''):
    RES.append(bool(cond)); print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def get(step, name, k=0):
    """the k-th tensor named `name` at `step` (names repeat: a layer's two hc mixes), numpy [ne3, ne2, ne1, ne0] squeezed"""
    hits = [r for r in rows if int(r[0]) == step and r[2] == name]
    if len(hits) <= k or hits[k][8].startswith('skipped'): return None
    r = hits[k]; ne = [int(v) for v in r[4:8]]
    a = np.fromfile(f'{G}/{r[8]}', dtype=np.int32 if r[3] == 'i32' else np.float32)
    return a.reshape(ne[::-1]).squeeze()


def stack(name, k=0):
    """[NS, ...] float64 of a tensor over the dump's steps (None when a step lacks it)"""
    v = [get(st, name, k) for st in range(NS)]
    return None if any(x is None for x in v) else np.stack(v).astype(np.float64)


def nrel(a, b):
    """max over positions of ||a - b|| / ||b||"""
    a = np.asarray(a, np.float64).reshape(len(a), -1); b = np.asarray(b, np.float64).reshape(len(b), -1)
    return float(max(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30) for x, y in zip(a, b)))


ref = CoderRef(A.gguf, threads=A.threads)
toks = [int(v) for v in open(f'{G}/tokens.txt').read().split()]
NS = len(toks); HC, D = ref.HC, ref.D
if A.text != 'none':
    from caminus.bpe import BPE
    from q3moe_ref import TEXT
    mine = BPE(ref.F.meta).encode(TEXT)[:NS]
    ok(f'the dump\'s {NS} tokens are the own BPE\'s ids of the public text (llama.cpp\'s tokenizer == ours)', mine == toks, f'({toks} / {mine})')
layers = sorted({int(r[2].rsplit('-', 1)[1]) for r in rows if r[2].startswith('l_last-')})
ok(f'the dump holds layers {list(NEED)} (it has {layers})', set(NEED) <= set(layers) and NS >= 10, f'({NS} positions, >= 10)')
ref.toks = list(toks)          # (the n-gram context of the PLE checks: the dump's tokens)
hc_init = stack('hc_init')


def streams_in(L):
    """llama.cpp's streams entering layer L's attention mix, or None (the PLE layer's are made below; not dumped)"""
    if L == 0: return hc_init
    return stack(f'l_last-{L - 1}') if L - 1 in layers else None


worst = {k: 0.0 for k in TOL}; ids_same = pairs = 0; werr = 0.0
for L in layers:
    x = streams_in(L)
    if L in ref.ple_layers and x is not None:
        ref.reset(); ref.toks = list(toks)
        gv, co = ref.ple_parts(L, x, 0)
        gv_l, co_l = stack(f'ple_gated_value-{L}'), stack(f'ple_conv_out-{L}')
        e = max(nrel(gv, gv_l), nrel(co, co_l)); worst['ple'] = max(worst['ple'], e)
        ok(f'layer {L}: PLE on llama.cpp\'s l_last-{L - 1}: gated value + conv output (4 taps, {ref.ngram.n_gram} apart)', e < TOL['ple'], f'({e:.1e})')
        x = x + gv_l + co_l          # llama.cpp's PLE output: the attention mix's input
    for kind, nth in (('attn', 0), ('ffn', 1)):
        xin = x if kind == 'attn' else stack(f'hc_combine-{L}')
        hm_l, inj_l = stack(f'hc_mixed-{L}', nth), stack(f'hc_inject-{L}', nth)
        if xin is not None:
            hm, inj = ref.hc_mix(xin, f'blk.{L}.hc_{kind}')
            e1, e2 = nrel(hm, hm_l), nrel(inj, inj_l)
            worst['mix'] = max(worst['mix'], e1); worst['inject'] = max(worst['inject'], e2)
            ok(f'layer {L}: the {kind} hyper-connection mix (hc_mixed, inject) from llama.cpp\'s streams', e1 < TOL['mix'] and e2 < TOL['inject'], f'({e1:.1e}, {e2:.1e})')
        if kind == 'attn':
            ref.reset()
            if ref.gdn[L]: blk = ref._gdn(L, hm_l); blk_l = stack(f'linear_attn_out-{L}')
            else:
                blk = ref._attn(L, hm_l, np.arange(NS, dtype=np.float64))
                blk_l = np.stack([[a for a in [get(st, f'attn_output-{L}', k) for k in range(3)] if a is not None and a.size == D][-1] for st in range(NS)]).astype(np.float64)
            e = nrel(blk, blk_l); worst['mixer'] = max(worst['mixer'], e)
            ok(f'layer {L}: the {"Gated DeltaNet (sigmoid gate)" if ref.gdn[L] else "attention"} block on llama.cpp\'s hc_mixed', e < TOL['mixer'], f'(||d|| / ||llama|| {e:.2e})')
            after = stack(f'hc_combine-{L}')
        else:
            y, top, w = ref._moe(L, hm_l); blk_l = stack(f'ffn_out-{L}')
            e = nrel(y, blk_l); worst['ffn'] = max(worst['ffn'], e)
            same = 0
            for st in range(NS):
                tk = get(st, f'ffn_moe_topk-{L}').reshape(-1); wn = get(st, f'ffn_moe_weights_norm-{L}').reshape(-1).astype(np.float64)
                pairs += 1
                if len(tk) == len(top[st]) and set(tk.tolist()) == set(top[st].tolist()):
                    same += 1; ids_same += 1; at = {int(v): i for i, v in enumerate(top[st])}
                    werr = max(werr, float(np.abs(np.array([w[st][at[int(v)]] for v in tk]) - wn).max()))
            ok(f'layer {L}: the FFN on llama.cpp\'s hc_mixed: the same {ref.KU} experts at {same} of {NS} positions; output', same == NS and e < TOL['ffn'],
               f'(||d|| / ||llama|| {e:.2e})')
            after = stack(f'l_last-{L}')
        if xin is not None and after is not None:
            e = nrel(ref.hc_combine(xin, blk_l, inj_l), after); worst['combine'] = max(worst['combine'], e)
            ok(f'layer {L}: the {kind} combine (streams + block * 2 sigmoid(inject / {HC}))', e < TOL['combine'], f'({e:.1e})')
ok(f'every FFN on llama.cpp\'s input: the same experts at {ids_same} of {pairs} (layer, position) pairs; the renormalized '
   f'weights against ffn_moe_weights_norm by expert id, max |d| {werr:.1e} (< 1e-5)', pairs > 0 and ids_same == pairs and werr < 1e-5)
if ref.L - 1 in layers:
    hN, _ = ref.hc_mix(stack(f'l_last-{ref.L - 1}'), 'output_hc', inject=False)
    e = nrel(hN, stack('result_norm')); worst['head'] = e
    ok(f'the head\'s mix (output_hc) from llama.cpp\'s l_last-{ref.L - 1} == result_norm', e < TOL['head'], f'({e:.1e})')
print('worst per kind (llama.cpp\'s own inputs; 0 = not in the dump): ' + ', '.join(f'{k} {v:.2e}' for k, v in worst.items()), flush=True)

# ---- the whole forward ----
ref.reset(); tr, ro = [], []; lg = ref.feed(toks, trace=tr, routes=ro)
same = pairs = 0; least = ref.KU
for L in layers:
    e = nrel(tr[L], stack(f'l_last-{L}'))
    ok(f'layer {L}: the forward\'s streams vs llama.cpp\'s l_last, max ||d|| / ||llama|| {e:.2e} (< {RESID})', e < RESID)
    for st in range(NS):
        n = len(set(ro[L][0][st].tolist()) & set(get(st, f'ffn_moe_topk-{L}').reshape(-1).tolist()))
        same += n == ref.KU; pairs += 1; least = min(least, n)
ok(f'the forward\'s top-{ref.KU} experts: the same set at {same} of {pairs} (layer, position) pairs (>= 3 / 4), at least {least} in common (>= {ref.KU - 3})',
   same * 4 >= pairs * 3 and least >= ref.KU - 3)
kl, t1 = [], 0
for st in range(NS):
    lr = lg[st] - lg[st].max(); lr -= np.log(np.exp(lr).sum())
    ll = get(st, 'result_output').astype(np.float64); ll -= ll.max(); ll -= np.log(np.exp(ll).sum())
    kl.append(float(np.sum(np.exp(lr) * (lr - ll)))); t1 += int(np.argmax(lr) == np.argmax(ll))
ok(f'logits: mean KLD(ref || llama.cpp) {np.mean(kl):.2e} (< 0.01; max {max(kl):.2e}), the same top token at {t1} of {NS} (>= 7 / 8)',
   np.mean(kl) < 0.01 and t1 * 8 >= NS * 7)
print(f'RESULT {sum(RES)} ok, {len(RES) - sum(RES)} FAIL')
sys.exit(0 if all(RES) else 1)
