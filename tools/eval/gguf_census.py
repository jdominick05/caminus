"""What a decode token reads from each GGUF given, for an MoE model read with caminus.ggufio: per tensor role (the name
without its layer) the formats, the file's bytes and the bytes one token reads at short context: every matrix once, of
an expert tensor the experts a token uses (expert_used_count of expert_count), of the token embedding one row, no
norm vector counted apart (they are whole). The bound is those bytes over --gbs (the card's sustained bandwidth).
  python -I tools/eval/gguf_census.py GGUF [GGUF ...] [--gbs 858]"""
import argparse, collections, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus import ggufio
ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='+'); ap.add_argument('--gbs', type=float, default=858.0)
A = ap.parse_args()
for path in A.gguf:
    F = ggufio.GGUFFile(path)
    used, of = int(F.hp('expert_used_count') or 1), int(F.hp('expert_count') or 1)
    roles = collections.OrderedDict()
    for t in F.tensors:
        role = re.sub(r'^blk\.\d+\.', '', t.name)
        if role == 'token_embd.weight': per = t.n_bytes / (t.n_values // t.shape[0])
        elif '_exps.' in role: per = t.n_bytes * used / of
        else: per = t.n_bytes
        r = roles.setdefault(role, [collections.Counter(), 0, 0.0])
        r[0][t.type] += 1; r[1] += t.n_bytes; r[2] += per
    tok = sum(r[2] for r in roles.values()); exp = sum(r[2] for k, r in roles.items() if '_exps.' in k)
    print(f'{os.path.basename(path)}: {F.arch}, {len(F.tensors)} tensors, {os.path.getsize(path) / 1e9:.3f} GB; a token uses {used} of {of} experts')
    for role, (types, nb, per) in roles.items():
        if nb < 1 << 20: continue
        print(f'  {role:24s} {", ".join(f"{k} {v}" for k, v in sorted(types.items())):28s} {nb / 1e9:8.3f} GB  per token {per / 1e6:9.2f} MB')
    print(f'  per token {tok / 1e9:.3f} GB (experts {exp / 1e9:.3f} GB): {tok / A.gbs / 1e6:.2f} ms at {A.gbs:.0f} GB/s, {1e3 / (tok / A.gbs / 1e6):.0f} tok/s')
