"""Launcher of benchmarks/iq2s_load_probe.hip: projects the probe's VRAM (--copies x (records + d) + slack) and checks it
through caminus.device.capacity on the chosen card before the probe allocates anything, then runs it on that card alone.

  tools/guard.sh LOG 180 python -I benchmarks/iq2s_load_probe.py --probe PROBE --card N [--reps 50] [--rows 17408]
      [--k 5120] [--copies 8]
PROBE: the probe built from benchmarks/iq2s_load_probe.hip (its header has the command). N: the HIP index of the card,
the non-display card first (`caminus ps`), as torch / HIP index it under the current HIP_VISIBLE_DEVICES. Prints the probe's output and its RESULT."""
import argparse, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from caminus.device import capacity

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--probe', required=True)
ap.add_argument('--card', type=int, required=True)
ap.add_argument('--reps', type=int, default=50)
ap.add_argument('--rows', type=int, default=17408)
ap.add_argument('--k', type=int, default=5120)
ap.add_argument('--copies', type=int, default=8)
A = ap.parse_args()
if A.k % 256 or A.rows < 8 or A.copies < 1: raise SystemExit('--k a multiple of 256, --rows >= 8, --copies >= 1')
n = A.rows * (A.k // 32)
need = A.copies * ((n * 10 + 64) + (n // 8 * 2 + 64)) + 4
mib = need / 2 ** 20 + 64.0   # (+ the HIP context's own buffers)
capacity.check([A.card], [mib], f'iq2s_load_probe ({A.copies} copies of {A.rows} x {A.k}: {mib:.0f} MiB)')
print(f'capacity: {mib:.0f} MiB on device {A.card}: within its cap', flush=True)
# (the same device visibility as the check above: --card indexes the devices this process sees)
r = subprocess.run([A.probe, '--card', str(A.card), '--reps', str(A.reps), '--rows', str(A.rows), '--k', str(A.k),
                    '--copies', str(A.copies)])
sys.exit(r.returncode)
