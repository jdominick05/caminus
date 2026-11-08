"""tools/eval/sweep_262k.py cmp with missing / short inputs (CPU, nothing launched; the sweep review, 11-05): a gate whose inputs are
absent or short is reported NOT RUN and the RESULT is not PASS (it used to be skipped silently).

  HIP_VISIBLE_DEVICES= python -I tests/tools/test_sweep_cmp.py --scratch <job scratch dir>/<step>/sweepcmp

The sweep's paths are pointed at fresh directories under --scratch (refused under /tmp, ~/.cache/caminus or $ARTIFACTS); nothing
under $ARTIFACTS/sweep-262k is read or written. Cases: an empty tree (every binding gate NOT RUN, exit 3); a short K3 logits
file (NOT RUN 'short'); probes missing from our run or without llama.cpp's tokens; curves without the plain tokens of a text;
no guard logs; a FAIL beside a NOT RUN (RESULT FAIL); nothing missing and nothing failed (RESULT PASS)."""
import argparse
import io
import json
import os
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'eval')); sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates')); sys.path.insert(0, os.path.join(ROOT, 'python'))
FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def point(SW, root):
    """the sweep module's paths under root (a fresh tree)"""
    SW.S = root; SW.LCS = os.path.join(root, 'lc'); SW.SEQ = os.path.join(SW.LCS, 'seq.ids')
    SW.K3OUT = os.path.join(root, 'k3'); SW.G4D = os.path.join(root, 'g4')
    for d in (SW.LCS, SW.K3OUT, SW.G4D): os.makedirs(d, exist_ok=True)


def run(SW):
    del SW.FAILS[:]; del SW.NOT_RUN[:]; del SW.REPORTED_NOT_RUN[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): SW.cmd_cmp(None)
    line, rc = SW.cmp_result()
    return line, rc, buf.getvalue(), list(SW.NOT_RUN), list(SW.FAILS)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    import sweep_262k as SW
    from common import ARTIFACTS
    root = os.path.realpath(os.path.expanduser(a.scratch))
    for bad in ('/tmp', os.path.realpath(os.path.expanduser('~/.cache/caminus')), os.path.realpath(ARTIFACTS)):
        if root == bad or root.startswith(bad + os.sep): raise SystemExit(f'--scratch {root}: refused (under {bad})')
    os.makedirs(root, exist_ok=True)
    k = 0

    def fresh():
        nonlocal k
        k += 1
        d = os.path.join(root, f'case{k}-{os.getpid()}')
        os.makedirs(d, exist_ok=False); point(SW, d)
        return d

    # (1) an empty tree: every binding gate NOT RUN, RESULT NOT RUN, exit 3
    fresh()
    line, rc, out, nr, fl = run(SW)
    gates = [g for g, _ in nr]
    check(rc == 3 and line.startswith('RESULT NOT RUN') and not fl and any('64K pooled floor' in g for g in gates)
          and any(g.startswith('G4b 81920') for g in gates) and any(g.startswith('G4b 131072') for g in gates)
          and any(g.startswith('G4b 261632') for g in gates) and any('probes' in g for g in gates) and any('DFlash == plain' in g for g in gates)
          and any('guard peaks' in g for g in gates) and out.count('NOT RUN ') >= len(nr),
          f'empty tree: {line} (exit {rc}); {len(nr)} binding gates NOT RUN, each printed: {gates}')
    check(not any('one-token D-281' in g for g in gates) and 'one-token D-281 at 261,632' in out,
          'empty tree: the dropped item 4 (one-token D-281 at ~262K) is printed NOT RUN as reported, not counted as a gate')
    # (2) a short K3 logits file: NOT RUN 'short', never read
    d = fresh()
    with open(os.path.join(SW.K3OUT, 'k3_pd_N131072.npy'), 'wb') as f: f.write(b'\x93NUMPY' + b'\0' * 100)
    line, rc, out, nr, fl = run(SW)
    sh = [w for g, w in nr if g.startswith('G4b 131072')]
    check(rc == 3 and sh and 'short' in sh[0] and 'missing' in sh[0], f'short input: G4b 131072 NOT RUN ({sh[0] if sh else None}); {line}')
    # (3) probes: one missing from our run, one without llama.cpp's tokens
    fresh()
    json.dump({'depths': [{'D': 65536, 'probes': [{'name': 'q_vault_a', 'answer': '1234-5678', 'item_pos': 100},
                                                   {'name': 'q_tag_a', 'answer': 'RARE', 'item_pos': 200}]}]},
              open(os.path.join(SW.LCS, 'probes.json'), 'w'))
    json.dump({'q_vault_a_65536': {'found': True, 'tokens': [1, 2], 'text': '1234-5678'}}, open(os.path.join(SW.K3OUT, 'probes.json'), 'w'))
    line, rc, out, nr, fl = run(SW)
    pg = sorted(g for g, _ in nr if g.startswith('probe '))
    check(rc == 3 and len(pg) == 2 and any('q_tag_a_65536' in g for g in pg) and any('q_vault_a_65536' in g for g in pg),
          f'probes: the one missing from our run and the one without llama.cpp tokens are NOT RUN: {pg}; {line}')
    # (4) curves with a text missing from plain.json; equal where both exist
    fresh()
    json.dump({'depths': {'2048': {'dflash': {t: {'tokens': [5, 6, 7]} for t in SW.TEXTS}}}}, open(os.path.join(SW.K3OUT, 'curves.json'), 'w'))
    json.dump({'depths': {'2048': {'prose': {'tokens': [5, 6, 7]}, 'code': {'tokens': [5, 6, 7]}}}}, open(os.path.join(SW.K3OUT, 'plain.json'), 'w'))
    line, rc, out, nr, fl = run(SW)
    cg = [g for g, _ in nr if g.startswith('DFlash == plain 2048')]
    check(rc == 3 and cg == ['DFlash == plain 2048 shell'] and 'PASS curves' in out and not fl,
          f'curves: the shell text without plain tokens NOT RUN ({cg}), the two present texts compared (PASS); {line}')
    # (5) a FAIL beside a NOT RUN: RESULT FAIL (exit 1), the NOT RUN still listed
    fresh()
    json.dump({'depths': {'2048': {'dflash': {'prose': {'tokens': [5, 6, 7]}}}}}, open(os.path.join(SW.K3OUT, 'curves.json'), 'w'))
    json.dump({'depths': {'2048': {'prose': {'tokens': [5, 9, 7]}}}}, open(os.path.join(SW.K3OUT, 'plain.json'), 'w'))
    line, rc, out, nr, fl = run(SW)
    check(rc == 1 and line.startswith('RESULT FAIL') and 'NOT RUN' in line and fl, f'a failing gate beside NOT RUN ones: {line} (exit {rc})')
    # (6) the result rule itself: nothing missing, nothing failed -> PASS
    del SW.FAILS[:]; del SW.NOT_RUN[:]
    check(SW.cmp_result() == ('RESULT PASS', 0), f'no NOT RUN and no FAIL: {SW.cmp_result()}')
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
