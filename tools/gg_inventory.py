"""Tape inventory of a Flash-Next (qwen4exp) GGUF for the K3 TP engine: which (op, format) pairs its tapes need, per op,
with the layers that need them; checked against a module's capability set (caminus.k3.MODULE_CAPS) and printed as the
per-op format whitelists a module build takes (spec-gsq-port.md 4b).

  python -I tools/gg_inventory.py MODEL.gguf [--module k3_gguf_gsq] [--tape [--ntok T]] [--json OUT]

Header mode (default, CPU only): maps every tensor the way python/caminus/flashnext_tp.py's tapes use it. --tape also
builds the tapes on the GPU (FlashNextTP local_only, tape_only: nothing is launched) and checks that the tapes'
formatted segments are exactly the header inventory, and that every pair is in the module's set. Exit 0 = all inside.
Host-side tensors (token_embd, the PLE table: dequantized by gguf-py) are listed apart, with gguf-py support."""
import json, os, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))
sys.path.insert(0, os.path.expanduser('~/src/ignite-llama/gguf-py'))
from gguf import GGUFReader, quants   # noqa: E402

# the GG matvec call sites of flashnext_tp (BF16 there: OP_BF16_MV; F32 is not a GG format)
MV_NAMES = ('attn_qkv', 'attn_gate', 'ssm_out', 'ssm_alpha', 'ssm_beta', 'attn_q', 'attn_k', 'attn_v', 'attn_output',
            'ffn_down_shexp', 'ple_key', 'ple_value', 'hc_attn_down', 'hc_attn_inject', 'hc_attn_up', 'hc_ffn_down',
            'hc_ffn_inject', 'hc_ffn_up')
ROUTER_NAMES = ('ffn_gate_inp', 'ffn_gate_inp_shexp')


def mv_kind(fmt):
    return ('BF16_MV', 'BF16') if fmt == 'BF16' else ('F32_MV', 'F32') if fmt == 'F32' else ('MV', fmt)


def header_inventory(path, caps):
    """{(op, fmt): sorted layer list ('out' = the head)}, host tensors {name: fmt}, errors"""
    rd = GGUFReader(path, 'r'); T = {t.name: t.tensor_type.name for t in rd.tensors}
    f = {x.name: x for x in rd.fields.values()}
    nl = int(f['qwen4exp.block_count'].contents())
    inv, errs = defaultdict(set), []
    add = lambda k, l: inv[k].add(l)
    for il in range(nl):
        p = f'blk.{il}.'
        for n in MV_NAMES:
            if p + n + '.weight' in T: add(mv_kind(T[p + n + '.weight']), il)
        for n in ROUTER_NAMES:
            fm = T[p + n + '.weight']
            if fm not in ('F32', 'BF16'): errs.append(f'{p}{n}: {fm} (router / shared gate: F32 or BF16 only)')
            add(mv_kind(fm), il)
        g, u = T[p + 'ffn_gate_shexp.weight'], T[p + 'ffn_up_shexp.weight']
        if g == u and ('GU_DENSE', g) in caps: add(('GU_DENSE', g), il)
        else: add(mv_kind(g), il); add(mv_kind(u), il); add(('Q8_1', 'SWIGLU'), il)
        ge, ue = T[p + 'ffn_gate_exps.weight'], T[p + 'ffn_up_exps.weight']
        if ge != ue: errs.append(f'{p}ffn_gate/up_exps: {ge} / {ue} (one format decodes both)')
        add(('MOE_GU', ge), il); add(('MOE_DOWN', T[p + 'ffn_down_exps.weight']), il)
    for n in ('output_hc_down', 'output_hc_up', 'output'):
        add(mv_kind(T[n + '.weight']), 'out')
    host = {n: T[n] for n in ('token_embd.weight',) if n in T}
    # the other shards (split GGUF): only the PLE table is used, on the host
    from caminus.flashnext import split_tensors
    for n, x in split_tensors(path, rd).items(): host[n] = x['qtype'].name
    if 'per_layer_token_embd.weight' in T: host['per_layer_token_embd.weight'] = T['per_layer_token_embd.weight']
    for n, fm in host.items():
        from gguf.constants import GGMLQuantizationType
        if GGMLQuantizationType[fm] not in quants._type_traits: errs.append(f'{n}: {fm} has no gguf-py dequantizer')
    return {k: sorted(v, key=lambda l: (isinstance(l, str), l)) for k, v in inv.items()}, host, errs


def layers_str(ls):
    """[0, 1, 2, 5, 'out'] -> '0-2,5,out'"""
    out, run = [], []
    for l in ls + [None]:
        if isinstance(l, int) and run and l == run[-1] + 1: run.append(l); continue
        if run: out.append(f'{run[0]}-{run[-1]}' if len(run) > 2 else ','.join(map(str, run)))
        run = [l] if isinstance(l, int) else []
        if isinstance(l, str): out.append(l)
    return ','.join(out)


def main():
    a = sys.argv[1:]
    def opt(name, default, f=str):
        if name in a:
            i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
        return default
    module, out_json, ntok = opt('--module', 'k3_gguf_gsq'), opt('--json', None), opt('--ntok', 1, int)
    tape = '--tape' in a
    if tape: a.remove('--tape')
    path = a[0]
    from caminus import k3
    caps = k3.MODULE_CAPS[k3.module_for(module, ntok)]
    inv, host, errs = header_inventory(path, caps)
    pend = k3.PENDING.get(k3.module_for(module, ntok), frozenset())
    print(f'{os.path.basename(path)}: module {module}{f" (no device code yet: {sorted(pend)})" if pend else ""}')
    bad = sorted(set(inv) - caps)
    for k in sorted(inv):
        print(f'  {k[0]:9s} {k[1]:8s} {"ok " if k in caps else "NOT IN MODULE"}  layers {layers_str(inv[k])}')
    for n, fm in host.items(): print(f'  host      {fm:8s} {n} (gguf-py dequantize)')
    wl = defaultdict(list)
    for op, fm in sorted(inv):
        if op in ('MV', 'MOE_GU', 'MOE_DOWN', 'GU_DENSE'): wl[op].append(f'GG_{fm}')
    print('  whitelist defines: ' + ' '.join(f'-DK3_FMTS_{op}={",".join(v)}' for op, v in sorted(wl.items())))
    print(f'  format instances: {sum(len(v) for v in wl.values())} '
          f'({", ".join(f"{op} {len(v)}" for op, v in sorted(wl.items()))}); unsupported: {bad or "none"}')
    for e in errs: print('  ERROR ' + e)
    res = dict(model=path, module=module, inventory={f'{k[0]}:{k[1]}': v for k, v in inv.items()}, host=host,
               unsupported=[list(k) for k in bad], errors=errs)
    ok = not bad and not errs
    if tape:
        import time
        import torch
        from caminus.flashnext_tp import FlashNextTP
        for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
        t0 = time.time()
        m = FlashNextTP(path, module=module, local_only=True, tape_only=True, ntok=ntok)
        tl = time.time() - t0
        used = [torch.cuda.memory_allocated(d) / 2**30 for d in (0, 1)]
        tinv = [m.engines[c].inventory() for c in (0, 1)]
        tk = set(tinv[0]) | set(tinv[1])
        glue = {('Q8_1', 'PLAIN'), ('Q8_1', 'SILU')}   # generic glue (every module has it): not in the header map
        same = tk - glue == set(inv) - glue and set(tinv[0]) == set(tinv[1])
        print(f'  tape (ntok {ntok}, local_only, tape_only): built in {tl:.1f} s; segments {[len(e.segs) for e in m.engines]}; '
              f'torch allocated {used[0]:.2f} / {used[1]:.2f} GiB; modules {[e.module for e in m.engines]}')
        for k in sorted(tk): print(f'    tape {k[0]:9s} {k[1]:8s} segments card0 {tinv[0].get(k, 0):4d} card1 {tinv[1].get(k, 0):4d}')
        print(f'  tape pairs == header pairs (both cards): {same}; tape unsupported: {sorted(tk - caps) or "none"}; '
              f'launch refused: {m.engines[0].refuse}')
        if m.engines[0].refuse:   # a tape with pairs the module lacks (or has no device code for) must not launch
            try:
                m.verify([0] * ntok, 0, 0); print('  ERROR: the tape launched'); ok = False
            except RuntimeError as ex: print(f'  verify() refused: {ex}')
        res.update(tape=dict(ntok=ntok, segments=[len(e.segs) for e in m.engines], gib=used, load_s=tl,
                             counts=[{f'{k[0]}:{k[1]}': v for k, v in t.items()} for t in tinv], equal_header=same))
        ok &= same and not (tk - caps)
    if out_json: json.dump(res, open(out_json, 'w'), indent=1)
    print('INVENTORY OK' if ok else 'INVENTORY FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
