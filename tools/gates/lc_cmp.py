"""Compare two tools/eval/lc_baseline.py results (OUT/k3_qsa.json or k3_baseline.json) on everything that is not a timing: per
depth the teacher-forced numbers against llama.cpp (KLD mean / max / per 64 positions, top-1, llama.cpp's own drift: exact
float equality, so the same logits give the same numbers and any drift in them shows) and every probe's greedy answer
(the text, copy found, == llama.cpp, the first divergence and llama.cpp's gap there). Fill / pass / snapshot times, the
police counters and compaction_proactiveness are printed, never compared. CPU only.
  python -I tools/gates/lc_cmp.py NEW.json STORED.json      exit 0 when every compared field is equal
Stored results of the current tree (MANIFEST.md): $ARTIFACTS/depth/split/gates/lc/k3_qsa.json (depths 1792, 8192) and
lc32/k3_qsa.json (32768), both from the split-attention gates of 11-03 (the verifier's: depth/verify-split/gates/lc*)."""
import json, sys

TOP = ('model', 'T', 'max_ctx', 'qsa')
TF = ('n', 'kld_mean', 'kld_max', 'top1', 'llama_drift_kld_mean', 'llama_drift_kld_max', 'llama_drift_top1', 'kld_by_64')
PROBE = ('answer', 'item_pos', 'k3_found', 'llama_found', 'same_as_llama', 'first_divergence', 'divergence_gap', 'k3_text', 'llama_text')
INFO = ('fill_s', 'fill_tok_s', 'fill_max_pass_ms', 'fill_max_pass_net_ms', 'fill_evicted_passes', 'fill_evicted_ms',
        'compaction_proactiveness', 'snapshot_mib')


def main(a):
    if len(a) != 2: print(__doc__); return 2
    x, y = (json.load(open(p)) for p in a)
    bad = []
    for k in TOP:
        if x.get(k) != y.get(k): bad.append(f'{k}: {x.get(k)!r} vs {y.get(k)!r}')
    dx = {d['D']: d for d in x['depths']}; dy = {d['D']: d for d in y['depths']}
    if sorted(dx) != sorted(dy): bad.append(f'depths {sorted(dx)} vs {sorted(dy)}')
    for D in sorted(set(dx) & set(dy)):
        p, q = dx[D], dy[D]; nb = len(bad)
        for k in TF:
            if p['tf'].get(k) != q['tf'].get(k): bad.append(f'D={D} tf.{k}: {p["tf"].get(k)!r} vs {q["tf"].get(k)!r}')
        if sorted(p['probes']) != sorted(q['probes']): bad.append(f'D={D} probes {sorted(p["probes"])} vs {sorted(q["probes"])}')
        for name in sorted(set(p['probes']) & set(q['probes'])):
            for k in PROBE:
                if p['probes'][name].get(k) != q['probes'][name].get(k):
                    bad.append(f'D={D} {name}.{k}: {p["probes"][name].get(k)!r} vs {q["probes"][name].get(k)!r}')
        print(f'D={D:>6}: tf mean KLD {p["tf"]["kld_mean"]!r} vs {q["tf"]["kld_mean"]!r}, top-1 {p["tf"]["top1"]!r} vs {q["tf"]["top1"]!r}; '
              f'{len(p["probes"])} probes; ' + ('EQUAL' if len(bad) == nb else f'{len(bad) - nb} fields differ'))
        print('   info (not compared): ' + '; '.join(f'{k} {p.get(k)!r} vs {q.get(k)!r}' for k in INFO if k in p or k in q))
    for b in bad: print('DIFF', b)
    print('RESULT', 'EQUAL' if not bad else f'NOT EQUAL ({len(bad)} fields)')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
