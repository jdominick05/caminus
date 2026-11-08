"""Every Flash-Next (REAP-308) layer in the K3 engine vs llama.cpp, teacher forced: for each layer, the engine runs
that layer alone (caminus.engine.flashnext.FlashNextK3.build_layers) over all dumped positions, each position starting from
llama.cpp's own input residual (l_last of the layer before; hc_init for layer 0) and carrying the engine's own
recurrent states; the output residual is compared with llama.cpp's l_last of the layer. A correct layer is at the
per-op level (~1e-7 relative) except where an activation-code rounding flip shows as a ~1e-3 step (see the end).
Also checks the host-side token embedding (gguf-py's dequantization of the token_embd row: REAP-308 Q6_K, the GSQ-RCO
Coder IQ4_XS) against llama.cpp's inp_embd (model.input_embed) and hc_init: a gate, bitwise for IQ4_XS (G3 of
docs/specs/spec-gsq-port.md), else within 1e-6.
  HIP_VISIBLE_DEVICES=0 python tests/engine/test_golden_layers.py GOLDEN_DIR (gg_golden with layers 0..47) [model.gguf]
  LAYERS=0,1,3,17,47 ...: only those layers (each needs the layer before it in the dump, for its input residual)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.engine.flashnext import FlashNextK3, quants

GD = sys.argv[1]
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[2] if len(sys.argv) > 2 else devpaths.reap()
idx = [l.rstrip('\n').split('\t') for l in open(f'{GD}/index.tsv')]
def golden(step, name):
    r = [r for r in idx if int(r[0]) == step and r[2] == name][0]
    return np.fromfile(f'{GD}/{r[8]}', dtype=np.float32)
tokens = [int(v) for v in open(f'{GD}/tokens.txt').read().split()]
steps = sorted({int(r[0]) for r in idx})
torch.cuda.set_per_process_memory_fraction(0.85, 0)
m = FlashNextK3(PATH, layers=(0, 1))
emb_rel, emb_bit, efmt = 0.0, True, m.T['token_embd.weight'].tensor_type.name
for i in steps:
    row = np.ascontiguousarray(np.asarray(m.T['token_embd.weight'].data)[tokens[i]][None])
    x = quants.dequantize(row, m.T['token_embd.weight'].tensor_type).astype(np.float32).reshape(-1)
    for ref, ours in ((golden(i, 'model.input_embed'), x), (golden(i, 'hc_init'), np.tile(x, m.hc))):
        emb_bit &= np.array_equal(ours.view(np.uint32), ref.view(np.uint32))
        emb_rel = max(emb_rel, float(np.abs(ours - ref).max() / np.abs(ref).max()))
emb_ok = emb_bit if efmt == 'IQ4_XS' else emb_rel <= 1e-6
print(f'token embedding ({efmt}) vs inp_embd / hc_init: {"bitwise" if emb_bit else f"max rel {emb_rel:.1e}"} '
      f'(gate {"bitwise" if efmt == "IQ4_XS" else "<= 1e-6"}: {"ok" if emb_ok else "FAIL"})')
worst, bad = 0.0, []
LAYERS = [int(v) for v in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else range(m.nl)
for il in LAYERS:
    if il or LAYERS[0] != 0: m.build_layers(il, il + 1)
    rels = []
    for i in steps:
        inp = golden(i, 'hc_init' if il == 0 else f'l_last-{il - 1}')
        out = m.run_layers(torch.from_numpy(inp.reshape(m.hc, m.D)), tokens, i).reshape(-1).cpu().numpy()
        ref = golden(i, f'l_last-{il}')
        rels.append(float(np.abs(out - ref).max() / np.abs(ref).max()))
    kind = 'attn' if il % m.fai == m.fai - 1 else 'gdn'
    flag = '' if max(rels) < 2e-6 else '  <--'
    if flag: bad.append(il)
    worst = max(worst, max(rels))
    print(f'layer {il:2d} ({kind}{"+ple" if il in m.ple_layers else ""}): max rel {max(rels):.1e}  per step ' + ' '.join(f'{r:.0e}' for r in rels) + flag)
print(f'worst {worst:.1e}; layers over 2e-6: {bad if bad else "none"}; token embedding {"ok" if emb_ok else "FAIL"}')
print('A diagnostic, not a gate: a 1e-7 input difference sometimes moves one q8_1 activation code across a rounding boundary\n'
      '(~1e-3 on that step), GDN state carries it forward, and llama.cpp\'s flash attention holds V in f16 (~1e-4).\n'
      'Inspect a jump with the per-op golden tests; the end-to-end gate is tests/engine/test_flashnext_e2e.py.')
sys.exit(0 if emb_ok else 1)
