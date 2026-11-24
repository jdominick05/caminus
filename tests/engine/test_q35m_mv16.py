"""The qwen35moe decode module's Q8_0 matvecs (k3_q35_moe, OP_OWN_MV16) against float64, on Qwen3.6-35B-A3B's own weights:
every Q8_0 tensor kind of the file (layer 0's and the first attention layer's, and the vocab head), random fp16
activations, through both tiles the module holds: the wave-per-row tile (kernels/own_quant.h own_row_q8_0_w, Seg.pad1
bit 16: Q8_0 at K = 1024 n, rows in tasks of about one per workgroup as Moe35Own sizes them) and the 4-rows tile
(own_rows_q8_0_1rt, rows_per_task 32). Per row |y - W x| / sum |w x| with W the blocks decoded on the host
(caminus.own_dequant); the gate: within 1e-6 for both (fp32 sums of exact products: ~1e-8 measured; a layout bug is
O(1)), and the row tile's tasks at two sizes bitwise (a row's sum does not depend on the task size).
  tools/guard.sh LOG 300 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/engine/test_q35m_mv16.py GGUF
Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus import ggufio, own_dequant  # noqa: E402
from caminus.device import capacity  # noqa: E402
from caminus.engine import k3  # noqa: E402

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def main():
    F = ggufio.GGUFFile(sys.argv[1]); rng = np.random.default_rng(3)
    print(torch.cuda.get_device_name(0), flush=True)
    kinds = {}
    for t in F.tensors:   # (one tensor per kind: layer 0's, the first layer that has it, the head)
        if t.type == 'Q8_0': kinds.setdefault(t.name.split('.', 2)[-1] if t.name.startswith('blk.') else t.name, t)
    # one matrix on the card at a time (the vocab head the largest, 515 MiB): its blocks, x, the fp32 rows of the arms
    # alive at once, each arm's tape; checked before anything is allocated
    peak = max(t.n_bytes + int(t.shape[0]) * 2 + 4 * t.rows()[0] * 4 for t in kinds.values())
    capacity.check([torch.cuda.current_device()], [peak / 2 ** 20 + 16], 'test_q35m_mv16: one Q8_0 matrix, its activations and outputs')
    grid = k3.Engine(module='k3_q35_moe').grid   # (the default grid: 4 workgroups per WGP)
    for kind, t in kinds.items():
        R, _ = t.rows(); K = int(t.shape[0])
        x = (rng.standard_normal(K) * 2).astype(np.float16); x16 = torch.from_numpy(x).cuda()
        w = torch.from_numpy(np.ascontiguousarray(t.pread(0, t.n_bytes))).cuda()
        Wd = own_dequant.rows(t, 0, R).astype(np.float64); xd = x.astype(np.float64)
        ref, den = Wd @ xd, np.maximum(np.abs(Wd) @ np.abs(xd), 1e-30)
        arms = [('4-rows', 32, False)]
        if K % 1024 == 0:
            rpt = -(-(-(-R // grid)) // 8) * 8
            arms += [('row', rpt, True), ('row', 8, True)]
        outs = []
        for name, rpt, row in arms:
            E = k3.Engine(module='k3_q35_moe'); out = torch.zeros(R, device='cuda')
            E.own_matvec16(w, 'Q8_0', R, K, x16, out, rows_per_task=rpt, row=row); E.build().run(); torch.cuda.synchronize()
            o = out.double().cpu().numpy(); outs.append(out.cpu())
            err = float((np.abs(o - ref) / den).max())
            check(err < 1e-6, f'{kind} ({R} x {K}) {name} tile, {rpt} rows per task: max |y - W x| / sum |w x| {err:.2e} (< 1e-6)')
        if K % 1024 == 0:
            check(torch.equal(outs[1].view(torch.int32), outs[2].view(torch.int32)), f'{kind}: the row tile at {arms[1][1]} and 8 rows per task bitwise')
    print('RESULT ' + ('FAIL' if FAILS else 'PASS'), flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
