"""On a real fla GatedDeltaNet layer (the S1 shape: d 896, 7 heads x 128), caminus.train.gdn.install() must route the
backward state recurrence to our kernel (fla's layer uses state_v_first=True; a silent fallback once hid this), keep
the forward identical, and keep every gradient within 1e-2 relative of fla's.
  python tests/train/test_gdn_layer.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
import fla.ops.gated_delta_rule.chunk as chunk
from fla.layers import GatedDeltaNet
from caminus.train import gdn

torch.manual_seed(0)
layer = GatedDeltaNet(hidden_size=896, expand_v=1, head_dim=128, num_heads=7, layer_idx=0).cuda().bfloat16()
x0 = torch.randn(4, 2048, 896, device='cuda', dtype=torch.bfloat16)
gy = torch.randn(4, 2048, 896, device='cuda', dtype=torch.bfloat16)


def run():
    for p in layer.parameters(): p.grad = None
    x = x0.clone().requires_grad_(); y = layer(x)[0]; y.backward(gy)
    return y.detach(), [x.grad.float()] + [p.grad.float().clone() for p in layer.parameters()]


y_ref, g_ref = run()
gdn.install()
taken = {'ours': 0, 'fallback': 0}
dispatch = chunk.chunk_gated_delta_rule_bwd_dhu


def counting(*a, **kw):
    taken['ours' if gdn.bwd_dhu_supported(*a, **kw) else 'fallback'] += 1
    return dispatch(*a, **kw)


chunk.chunk_gated_delta_rule_bwd_dhu = counting
y_new, g_new = run()
names = ['x'] + [n for n, _ in layer.named_parameters()]
errs = {n: ((a - b).norm() / b.norm().clamp_min(1e-12)).item() for n, a, b in zip(names, g_new, g_ref)}
ok = taken == {'ours': 1, 'fallback': 0} and torch.equal(y_ref, y_new) and max(errs.values()) < 1e-2
print(f"dispatch {taken}; forward identical {torch.equal(y_ref, y_new)}; max grad rel err {max(errs.values()):.1e} "
      f"({max(errs, key=errs.get)})")
print('PASS' if ok else 'FAIL')
