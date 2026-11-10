"""The DFlash drafter's offline acceptance, phase 2 (the drafter alone, fp32 torch: caminus.decode.dflash_q3m_torch): every
anchor position of the generated text in phase 1's files, the block [anchor, mask x n] over the features of every earlier
position (window 2048); a draft row accepted while it equals the target's greedy token and every earlier row did.
Per-position acceptance and the mean accepted drafts for 7 and 3 drafts, both feature conventions.
  tools/guard.sh LOG 1200 python -I tools/eval/q3m_dflash_accept.py DRAFTER_DIR OUT"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.decode.dflash_q3m_torch import DFlashQ3MRef
from caminus.device import capacity
R = DFlashQ3MRef(sys.argv[1], 'cuda')   # (projects and checks its own weights and one window's pass)
t2d = R.w['t2d']; print('d2t as offsets matches t2d:', torch.equal(R.d2t_ids, torch.nonzero(t2d).flatten()), flush=True)
files = sorted(glob.glob(sys.argv[2] + '_*.npz'))
# beside the drafter: one file's features on the card at a time, as stored and as fp32, and their context rows (a fifth of
# the features: 5 tapped layers in, 1 row out): 4 x the largest feature array bounds it
feat_mib = 4 * max(max(np.load(f)[conv].nbytes for conv in ('fa', 'fb')) for f in files) / 2 ** 20 + 64
capacity.check([0], [feat_mib], f'q3m_dflash_accept features ({feat_mib:.0f} MiB projected beside the drafter)')
for conv in ('fa', 'fb'):
    for nd in (7, 3):
        hits = np.zeros(nd); n_anch = 0; lens = []
        for f in files:
            z = np.load(f); seq = z['seq'].tolist(); n0 = int(z['n_prompt']); feats = torch.from_numpy(z[conv]).cuda()
            ctx_all = R.context(feats)
            for p in range(n0, len(seq) - nd - 1):
                lo = max(0, p - 2048)
                lg = R.logits(ctx_all[lo:p], torch.arange(lo, p, device='cuda'), [seq[p]] + [R.c['mask_token']] * nd, p)
                d = R.d2t_ids[lg[1:].argmax(-1)].tolist()
                k = 0
                while k < nd and d[k] == seq[p + 1 + k]: k += 1
                hits[:k] += 1; n_anch += 1; lens.append(k)
        acc = hits / n_anch
        print(f'{conv} ({"entering" if conv == "fa" else "leaving"} layers 1/12/23/34/45), {nd} drafts, {n_anch} anchors: '
              f'per-position acceptance ' + ' '.join(f'{a * 100:.1f}%' for a in acc) + f'; mean accepted {np.mean(lens):.2f} (+1 bonus)', flush=True)
print('RESULT DONE')
