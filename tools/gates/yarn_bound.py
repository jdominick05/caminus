"""The long-context mode's (YaRN) extra checks for the Qwen3-30B-A3B G-model gates (q3moe_gmodel.py, q3moe_depth_gmodel.py).

The gates' rule, caminus's mean KLD against the reference <= llama.cpp's, is relative: a wrong YaRN reference or a
misconfigured llama.cpp run (no 'yarn' job line, or the wrong original context) grows llama.cpp's KLD and loosens the bar
instead of failing. Under --yarn two checks pin it:
  - an absolute bar on llama.cpp's mean KLD against the YaRN reference (--llama-max-kld; the default is twice llama.cpp's
    native figure at that length: 2.83e-03 at 128 tokens, 5.73e-03 for the last 64 of 40,896, docs/dev/qwen3-30b-a3b.md).
    The defaults are not measured in the mode yet: a failure on the first run is a finding to read, not proof of a bug.
  - --native-ref NATIVE.npz (the native reference of the same tokens: q3moe_ref.py / q3moe_ref32.py without --yarn): both
    engines must sit closer to the YaRN reference than to the native one. At 128 tokens the two references differ by a
    mean KLD of 3.7e-02, so a run without YaRN shows here at once.
Pure numpy: tests/engine/test_q3moe_yarn_cpu.py runs it on synthetic rows."""
import numpy as np

LLAMA_MAX_KLD = {'short': 5.7e-03, 'depth': 1.15e-02}   # twice llama.cpp's native mean KLD (128 tokens; 40,896 deep)


def add_args(ap, kind):
    ap.add_argument('--llama-max-kld', type=float, default=None,
                    help=f'with --yarn and --llama: the most llama.cpp\'s mean KLD against the YaRN reference may be (default '
                         f'{LLAMA_MAX_KLD[kind]:.2e}, twice its native figure; not yet measured in the mode)')
    ap.add_argument('--native-ref', default=None,
                    help='with --yarn: the native reference of the same tokens; both engines must be closer to the YaRN one')


def check_args(ap, A):
    if not A.yarn and (A.llama_max_kld is not None or A.native_ref):
        ap.error('--llama-max-kld and --native-ref check the long-context mode: they need --yarn')


def load_native(path, seq, n_rows):
    """the native reference's logits for the same tokens and rows (refused otherwise)"""
    z = np.load(path)
    if 'yarn' in z.files: raise SystemExit(f'--native-ref {path}: a YaRN reference (its yarn entry); it needs a native one')
    if [int(v) for v in z['seq']] != list(seq): raise SystemExit(f'--native-ref {path}: other tokens than the --ref reference')
    if z['logits'].shape[0] != n_rows: raise SystemExit(f'--native-ref {path}: {z["logits"].shape[0]} rows, the --ref has {n_rows}')
    return z['logits']


def gate(score, kind, ours, so, ll=None, sl=None, native=None, bar=None):
    """(ok, lines): the absolute bar on llama.cpp (when its rows sl were scored) and, with the native reference's logits,
    the closer-to-YaRN check for caminus (ours, so) and llama.cpp (ll, sl)"""
    ok, out = True, []
    if sl is not None:
        b = LLAMA_MAX_KLD[kind] if bar is None else float(bar)
        c = sl['mean'] <= b
        out.append(('PASS' if c else 'FAIL') + f' llama.cpp against the YaRN reference: mean KLD {sl["mean"]:.3e} <= {b:.2e} '
                   f'(an absolute bar{"" if bar is not None else ", twice its native figure, not yet measured in the mode"})')
        ok &= c
    if native is not None:
        for name, rows, s in (('caminus', ours, so), ('llama.cpp', ll, sl)):
            if rows is None: continue
            sn = score(native, rows)
            c = s['mean'] < sn['mean']
            out.append(('PASS' if c else 'FAIL') + f' {name}: closer to the YaRN reference (mean KLD {s["mean"]:.3e}) than to the '
                       f'native one ({sn["mean"]:.3e}, top-1 {sn["top1"]}/{sn["n"]}): YaRN took effect')
            ok &= c
    return ok, out
