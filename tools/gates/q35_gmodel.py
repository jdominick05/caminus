"""The G-model gate of the qwen35 line and the Coder on own code (docs/specs/spec-own-quant.md): the engine's logits and
llama.cpp's against the float64 reference (tools/reference/q35_ref.py: caminus.ref_q35, or caminus.ref_coder for a qwen4exp
file), teacher-forced over the reference's tokens. Per position KLD(ref || candidate), top-1 agreement and |logit
difference|; the gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's. The GGUF's architecture picks the engine:
  qwen35 (dense): Dense35Own, or with --cards 2 Dense35TP (devices 0 and 1: also the replicated residual bitwise on both
      cards after every token); F16 KV
      tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz [--llama LLAMA] [--module M]
          [--cards 2 [--ffn-share F]]
  qwen4exp (the Coder): FlashNextTP, always both cards (the Coder does not fit one), its replicas checked after every token;
      --kv q8 (default: the served engine's Q8 KV) or f16; --llama is required and must be llama.cpp's run with the same KV
      type (--llama-kv: q8_0 with --kv q8, f16 with --kv f16; another pairing is refused). The load dominates (the Coder's
      smoke runs under a 600 s guard); 256 positions not yet timed
      tools/guard.sh LOG 900 python -I tools/gates/q35_gmodel.py $CODER --ref REF.npz --llama LLAMA [--kv q8|f16]
  python -I tools/gates/q35_gmodel.py --ref REF.npz --llama LLAMA --ref-only     # the reference against llama.cpp, no card
  qwen35moe (Moe35Own, one card, --kv f16 or q8): --attn-from 0 scores every row on the sliced pass with the decode attention
      kernel (engine/dec_q35m_attn.py; --attn-splits N forces its splits, so a short reference exercises its merge)
LLAMA: llama.cpp's one-token-decode logits rows of the same tokens: a .bin of f32 [N, vocab] (build/gg_golden MODEL --lc JOBS
with the lines 'nctx N+64', 'kv f16' (or 'kv q8_0'), 'onecard' (dense only: the Coder needs both cards), 'tfx ref TOKENS.txt 0
N 1 1 OUT'), or a gg_golden dump directory (stepI's result_output; its tokens.txt must be the reference's tokens: the stored
short dumps e2e_{code,prose,shell}_decode, q8_0 KV, which tools/gates/common.py names). A .bin carries no tokens: its rows
are trusted to be the reference's.
The reference's own reading of the model is confirmed by llama.cpp: llama.cpp's mean KLD against it under --confirm (0.01: a
gross misreading of a block costs 0.1 or more) and its top-1 agreeing at 15 positions of 16 or more. --ref-only runs that
alone; with an engine it is one of the gate's lines too, so a llama.cpp run of other tokens or another model fails instead
of loosening the relative rule. The line sits near llama.cpp's own noise on prose and shell text (on the Coder its
prefill-then-decode against its one-token decode scores 9.9e-3 / 9.5e-3 there, code 1.1e-3: tools/gates/g4_floor.py
short, n_ubatch 2048; llama.cpp's quantized activations cost about 1e-3 on the dense line). A subtle misreading can pass it:
on Qwen3.6-35B-A3B, sigmoid instead of softmax router weights (KLD 1.6e-3) and 7 experts instead of 8 (9.2e-3) both do, and
on the Coder the sigmoid router (KLD 2.4e-4 on 8 positions), so a MoE reference also needs its golden test (each block from
llama.cpp's own input: tests/engine/test_q36_golden_cpu.py, tests/engine/test_coder_golden_cpu.py).
Prints PASS / FAIL and RESULT; exit 0 only on PASS."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
from q3moe_gmodel import line, score


def llama_rows(path, seq):
    """llama.cpp's logits rows of the tokens seq: from a gg_golden dump directory (stepI's result_output; refused unless its
    tokens.txt starts with seq) or an f32 [len(seq), vocab] .bin"""
    n = len(seq)
    if os.path.isdir(path):
        toks = [int(v) for v in open(os.path.join(path, 'tokens.txt')).read().split()]
        if toks[:n] != list(seq):
            bad = next((i for i, (a, b) in enumerate(zip(toks, seq)) if a != b), min(len(toks), n))
            raise SystemExit(f'FAIL --llama {path}: its tokens differ from the reference\'s at position {bad} '
                             f'({len(toks)} tokens, the reference {n})\nRESULT FAIL')
        got = {}
        for l in open(os.path.join(path, 'index.tsv')):
            r = l.rstrip('\n').split('\t')
            if r[2] == 'result_output' and int(r[0]) < n: got[int(r[0])] = r[8]
        if sorted(got) != list(range(n)): raise SystemExit(f'--llama {path}: result_output for {len(got)} of the {n} steps')
        return np.stack([np.fromfile(os.path.join(path, got[i]), dtype=np.float32) for i in range(n)])
    a = np.fromfile(path, dtype=np.float32)
    if a.size == 0 or a.size % n: raise SystemExit(f'--llama {path}: {a.size} values is not {n} rows')
    return a.reshape(n, -1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='?'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--module', default=None); ap.add_argument('--ref-only', action='store_true')
    ap.add_argument('--confirm', type=float, default=0.01)
    ap.add_argument('--kv', default=None, choices=('f16', 'q8'), help="the engine's KV cache (the Coder: q8 by default; the dense line: f16 only)")
    ap.add_argument('--llama-kv', default=None, choices=('f16', 'q8_0'), help="llama.cpp's KV type in --llama (default: the engine's)")
    ap.add_argument('--cards', type=int, default=1, choices=(1, 2)); ap.add_argument('--ffn-share', type=float, default=0.5)
    ap.add_argument('--dump', default=None, help='save our logits rows (npy) for a bitwise comparison of two builds')
    ap.add_argument('--yarn', type=float, default=None,
                    help='the YaRN factor (qwen35moe only): the engine builds the long-context rope from it, so the '
                         'reference and llama.cpp must be run in the same mode for the rows to be comparable')
    ap.add_argument('--speed', type=int, default=0, help='then this many greedy passes (step_token) after the rows: median ms per token')
    ap.add_argument('--attn-from', type=int, default=None,
                    help='qwen35moe only: Moe35Own(attn_from=): from this position on the passes run sliced with the decode '
                         'attention as its own kernel (0: every row; default the engine\'s switch)')
    ap.add_argument('--attn-splits', type=int, default=None, help='qwen35moe only: the attention kernel\'s splits (default: by depth)')
    A = ap.parse_args()
    arch = None
    if not A.ref_only:
        if not A.gguf: ap.error('GGUF')
        from caminus import ggufio
        F = ggufio.GGUFFile(A.gguf); arch = F.arch; F.close()
        coder = arch == 'qwen4exp'
        kv = A.kv or ('q8' if coder else 'f16')
        if not coder and kv != 'f16' and arch != 'qwen35moe':
            ap.error(f'--kv {kv}: the dense line\'s gate runs the F16 cache only')
        if coder and not A.llama: ap.error('the Coder\'s gate needs --llama (its bound and the reference\'s confirmation)')
        lkv = A.llama_kv or {'q8': 'q8_0', 'f16': 'f16'}[kv]
        if A.llama and {'q8': 'q8_0', 'f16': 'f16'}[kv] != lkv:
            ap.error(f'--kv {kv} against llama.cpp\'s --llama-kv {lkv}: the bound must be llama.cpp with the same KV type')
    else:
        lkv = A.llama_kv or 'f16'
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    sl, conf = None, True
    if A.llama:
        ll = llama_rows(A.llama, seq)
        if ll.shape != ref.shape: raise SystemExit(f'llama.cpp rows {ll.shape} != the reference\'s {ref.shape}')
        sl = score(ref, ll); print(line('llama.cpp', sl) + f'  (its KV {lkv})', flush=True)
        conf = sl['mean'] < A.confirm and sl['top1'] * 16 >= sl['n'] * 15
        print(('PASS' if conf else 'FAIL') + f' the reference is confirmed by llama.cpp: mean KLD {sl["mean"]:.3e} < {A.confirm}, '
              f'top-1 {sl["top1"]}/{sl["n"]} >= 15/16', flush=True)
    if A.ref_only:
        if sl is None: ap.error('--ref-only needs --llama')
        print('RESULT', 'PASS' if conf else 'FAIL'); sys.exit(0 if conf else 1)
    import torch
    rep_ok = True
    if coder:
        from caminus.engine.flashnext_tp import FlashNextTP
        print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
        eng = FlashNextTP(A.gguf, max_ctx=len(seq) + 8 + A.speed, module=A.module, q8=kv == 'q8')
        print(f'module {eng.module}; KV {kv}', flush=True)
        rows, bad = [], []
        for i in range(len(seq)):
            rows.append(eng.step(seq, i).float().cpu().numpy())
            if not all(eng.check_replicas()): bad.append(i)
        ours = np.stack(rows); rep_ok = not bad
        print(('PASS' if rep_ok else 'FAIL') + f' the replicas (residual, router ids) are the same bits on both cards after every token'
              + ('' if rep_ok else f' (differ at positions {bad[:8]})'), flush=True)
    elif A.cards == 2:
        from caminus.engine.q35_dense_tp import Dense35TP
        print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
        eng = Dense35TP(A.gguf, max_ctx=len(seq) + 8 + A.speed, module=A.module, ffn_share=A.ffn_share)
        print(f'module {eng.module}; ffn rows {eng.f0} / {eng.ffn - eng.f0}; head rows {eng.vh} / {eng.vocab - eng.vh}', flush=True)
        rows, bad = [], []
        for i, t in enumerate(seq):
            rows.append(eng.step(t).float().cpu().numpy())
            if not eng.check_replicas()[0]: bad.append(i)
        ours = np.stack(rows); rep_ok = not bad
        print(('PASS' if rep_ok else 'FAIL') + f' the replicated residual is the same bits on both cards after every token'
              + ('' if rep_ok else f' (differs at positions {bad[:8]})'), flush=True)
    elif arch == 'qwen35moe':
        # (the MoE line on own code: Moe35Own's tape. --yarn F builds the long-context rope the same way the serve does,
        #  so a YaRN run is gated against a YaRN reference and a YaRN llama.cpp, never against native rows)
        from caminus.engine.q35_moe import Moe35Own
        print(torch.cuda.get_device_name(0), flush=True)
        eng = Moe35Own(A.gguf, max_ctx=len(seq) + 8 + A.speed, module=A.module, yarn=A.yarn, q8=A.kv == 'q8',
                       attn_from=A.attn_from, attn_splits=A.attn_splits)
        print(f'module {eng.module}; rope {"YaRN x%g" % A.yarn if A.yarn else "native"}; kv {getattr(eng, "kv_format_id", None) or "f16"}; '
              f'sliced from position {eng.attn_from} (attention kernel splits {A.attn_splits or "by depth"})', flush=True)
        ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq])
    else:
        from caminus.engine.q35_dense import Dense35Own
        print(torch.cuda.get_device_name(0), flush=True)
        eng = Dense35Own(A.gguf, max_ctx=len(seq) + 8 + A.speed, module=A.module)   # (its own VRAM check; this script keeps nothing else on the card)
        ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq])
    so = score(ref, ours); print(line('caminus', so), flush=True)
    if A.dump: np.save(A.dump, ours)
    if A.speed:   # greedy passes after the rows (step_token: the argmax read back, as the served decode), each synchronized
        import time
        toks, ts = list(seq), []
        nxt = int(np.argmax(ours[-1]))
        for i in range(A.speed):
            toks.append(nxt)
            t0 = time.perf_counter(); nxt = eng.step_token(toks, len(toks) - 1); ts.append(time.perf_counter() - t0)
        med = float(np.median(ts)) * 1e3
        print(f'decode speed: {med:.2f} ms per token ({1e3 / med:.1f} tok/s), median of {A.speed} greedy passes at positions '
              f'{len(seq)} .. {len(seq) + A.speed - 1}', flush=True)
    ok = rep_ok and conf
    if sl is not None:
        c1 = so['mean'] <= sl['mean']; c2 = so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' mean KLD vs float64: caminus {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' top-1 agreement: caminus {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = ok and c1 and c2
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
