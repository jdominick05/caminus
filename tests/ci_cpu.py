"""The CPU suites in one run, as CI runs them (.github/workflows/ci.yml) and as anyone can run them locally:

  python -I tests/ci_cpu.py --scratch DIR [--only NAME,...] [--jobs N] [--list]

Each suite runs when what it needs is here and is SKIPPED with the reason otherwise (a missing model or library is never a
failure); nothing touches a GPU (HIP_VISIBLE_DEVICES is emptied for every child). What a suite may need:
  torch   PyTorch importable (any build: CI installs the CPU wheel; nothing here launches a kernel)
  vocab   $CODER names a GGUF with the Coder's metadata: the tokenizer and the chat template (tests/serve). The weights are not
          needed: CI writes a vocab-only GGUF from the first 11 MiB of shard 1 (tools/release/gguf_vocab_only.py)
  coder   $CODER names the Coder's real shard 1 (its tensors; shard 2 beside it): the prefill launchers' CPU refusals
DIR (required, never under /tmp; it must not exist or be empty) gets one subdirectory and one log per suite. Exit 1 when any
suite that ran failed; for each failure its failure reports (FAIL / ERROR lines, tracebacks) and its log's tail are printed.
Suites run with PYTHONUNBUFFERED=1, so a report sits in the log where it happened.
--jobs N suites run at once (default: the CPU count, at most 8; 1 = one after another). Every suite is its own process with
its own scratch directory and log, and the results print in the list's order whatever finishes first. The suites that
need `coder` never overlap one another at any N: each loads the Coder's tensors into host memory."""
import argparse
import concurrent.futures
import contextlib
import importlib.util
import os
import re
import struct
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (name, argv with {coder} / {scratch} placeholders, needs)
SUITES = [
    ('serve', ['tests/serve/run_all.py'], {'vocab'}),
    ('packaging', ['tests/packaging/test_packaging.py', '--scratch', '{scratch}'], set()),
    ('arch', ['tests/test_arch.py', '--scratch', '{scratch}'], set()),
    ('host', ['tests/device/test_host.py', '--scratch', '{scratch}'], set()),
    ('manage', ['tests/cmd/test_manage.py', '--scratch', '{scratch}'], set()),
    ('advisor', ['tests/harness/test_advisor.py', '--scratch', '{scratch}'], set()),
    ('context_note', ['tests/harness/test_context_note.py', '--scratch', '{scratch}'], set()),
    ('docs', ['tests/docs/test_docs.py'], set()),
    ('own_code', ['tests/test_own_code.py'], set()),
    ('own_dequant', ['tests/io/test_own_dequant.py'], set()),
    ('own_formats', ['tests/io/test_own_formats.py'], set()),
    ('ref_q35', ['tests/test_ref_q35.py', '--scratch', '{scratch}'], set()),
    ('q35_tape', ['tests/engine/test_q35_tape_cpu.py', '--scratch', '{scratch}'], {'torch'}),
    ('q35_tp', ['tests/engine/test_q35_tp_cpu.py', '--scratch', '{scratch}'], {'torch'}),
    ('q35_tp_ver', ['tests/engine/test_q35_verify_tp_cpu.py', '--scratch', '{scratch}'], {'torch'}),
    ('pass_police', ['tests/engine/test_pass_police.py'], set()),
    ('sweep_cmp', ['tests/tools/test_sweep_cmp.py', '--scratch', '{scratch}'], set()),
    ('serve_smoke1', ['tests/tools/test_serve_smoke1.py'], set()),
    ('nightly_plan', ['tests/tools/test_nightly_plan.py', '--scratch', '{scratch}'], set()),
    ('objcheck', ['tests/tools/test_objcheck.py', '--scratch', '{scratch}'], set()),
    ('pr_ready', ['tests/tools/test_pr_ready.py', '--scratch', '{scratch}'], set()),
    ('bounds_host', ['tests/engine/test_bounds_host.py'], {'torch'}),
    ('kv_host', ['tests/device/test_kv_host_cpu.py'], {'torch'}),
    ('ple_aio', ['tests/engine/test_ple_aio.py'], {'torch'}),
    ('spec_policy', ['tests/decode/test_spec_policy.py', 'cpu'], {'torch'}),
    ('dflash35_adapter', ['tests/decode/test_dflash35_adapter.py'], {'torch'}),
    ('prompt_cache', ['tests/cache/test_prompt_cache.py', 'cpu', '--cache-dir', '{scratch}'], {'torch'}),
    ('dflash_ring_fuzz', ['tests/decode/test_dflash_ring_fuzz.py'], {'torch'}),
    ('prefill_driver', ['tests/prefill/test_prefill_driver.py'], {'torch'}),
    ('prefill_feed', ['tests/prefill/test_prefill_feed.py', '{coder}', '--cpu'], {'torch', 'coder'}),
    ('prefill_safety', ['tests/prefill/test_prefill_safety.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_gdn', ['tests/prefill/test_prefill_gdn.py', '{coder}', 'cpu', 'vram'], {'torch', 'coder'}),
    ('prefill_glue', ['tests/prefill/test_prefill_glue.py', '{coder}', 'vram'], {'torch', 'coder'}),
    ('prefill_dense', ['tests/prefill/test_prefill_dense.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_moe', ['tests/prefill/test_prefill_moe.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_moe_canary', ['tests/prefill/test_prefill_moe_canary.py', '{coder}', 'review'], {'torch', 'coder'}),
    ('prefill_tp', ['tests/prefill/test_prefill_tp.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_s6', ['tests/prefill/test_prefill_s6.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_attn', ['tests/prefill/test_prefill_attn.py', '{coder}', 'cpu'], {'torch', 'coder'}),
    ('prefill_qsa', ['tests/prefill/test_prefill_qsa.py', '{coder}', 'cpu'], {'torch', 'coder'}),
]


def gguf_tensor_count(path):
    """the GGUF's tensor count, or None when path is not a readable GGUF"""
    try:
        with open(path, 'rb') as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b'GGUF':
        return None
    return struct.unpack_from('<Q', head, 8)[0]


def available():
    """need -> None when present, else why not"""
    coder = os.environ.get('CODER', '')
    n = gguf_tensor_count(coder) if coder else None
    torch_ok = importlib.util.find_spec('torch') is not None
    return {
        'torch': None if torch_ok else 'PyTorch is not installed',
        'vocab': None if n is not None else ('$CODER is not set' if not coder else f'$CODER {coder} is not a readable GGUF'),
        'coder': None if n else ('$CODER is not set' if not coder else
                                 f'$CODER {coder} has no tensors (a vocab-only GGUF)' if n == 0 else f'$CODER {coder} is not a readable GGUF'),
    }


def default_jobs(per_gib=2):
    """suites at once by default: the CPU count, at most 8, and at most one per per_gib GiB of the host's MemAvailable (a
    suite takes up to ~1.5 GiB; with a served model holding host memory, 8 at once took MemAvailable under the GPU
    guard's 1500 MiB floor, and the guard self-test's real guard killed its run)"""
    avail = None
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    avail = int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    n = min(8, os.cpu_count() or 1)
    return n if avail is None else max(1, min(n, int(avail // per_gib)))


def failure_lines(lines, tail=60, around=30, most=12):
    """what a failed suite's log shows: every failure report (a line starting FAIL or ERROR, unittest's FAIL: / ERROR:
    blocks, a Traceback) with the lines after it, at most `most` of them, then the last `tail` lines; '    ...' where
    lines are left out"""
    keep = set(range(max(0, len(lines) - tail), len(lines)))
    marks = [i for i, x in enumerate(lines) if re.match(r'(FAIL|ERROR)[: ]|Traceback \(most recent call last\)', x)]
    for i in marks[:most]:
        keep.update(range(i, min(i + around, len(lines))))
    out, prev = [], -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append('    ...')
        out.append('    ' + lines[i]); prev = i
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scratch', required=False)
    ap.add_argument('--only', default='', help='comma-separated suite names')
    ap.add_argument('--jobs', type=int, default=default_jobs(), help='suites run at once (1 = one after another; default: the CPU '
                    'count, at most 8, and at most one per 2 GiB of MemAvailable)')
    ap.add_argument('--list', action='store_true', help='print the suites and what each needs, then exit')
    a = ap.parse_args(argv)
    have = available()
    if a.list:
        for name, args, needs in SUITES:
            miss = [f'{n}: {have[n]}' for n in sorted(needs) if have[n]]
            print(f'{name:20s} needs {",".join(sorted(needs)) or "-":14s} {"runs" if not miss else "SKIP (" + "; ".join(miss) + ")"}')
        return 0
    if not a.scratch:
        ap.error('--scratch is required')
    S = os.path.abspath(a.scratch)
    if S == '/tmp' or S.startswith('/tmp/'):
        raise SystemExit('ci_cpu: --scratch under /tmp refused')
    if os.path.exists(S) and os.listdir(S):
        raise SystemExit(f'ci_cpu: --scratch {S} is not empty')
    os.makedirs(S, exist_ok=True)
    only = {x for x in a.only.split(',') if x}
    unknown = only - {n for n, _, _ in SUITES}
    if unknown:
        raise SystemExit(f'ci_cpu: unknown suites {sorted(unknown)}')
    env = dict(os.environ, HIP_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')   # (a suite's prints and its tracebacks in the order written)
    env.setdefault('TMPDIR', os.path.join(S, 'tmp'))
    os.makedirs(env['TMPDIR'], exist_ok=True)
    if a.jobs < 1:
        raise SystemExit('ci_cpu: --jobs must be at least 1')
    heavy = threading.Lock()   # (the suites that load the Coder's tensors: one at a time)

    def run(name, args, needs):
        sub = os.path.join(S, name)
        argv_ = [x.replace('{coder}', os.environ.get('CODER', '')).replace('{scratch}', sub) for x in args]
        log = os.path.join(S, f'{name}.log')
        with heavy if 'coder' in needs else contextlib.nullcontext():
            t0 = time.time()
            with open(log, 'w') as f:
                rc = subprocess.run([sys.executable, '-I'] + argv_, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc == 0 and verdict(log) == 'FAIL': rc = 1   # (a suite that printed RESULT FAIL and still exited 0)
            return rc, time.time() - t0, log

    def verdict(log):
        """the last RESULT line of a suite's log ('PASS' / 'FAIL' / other), None without one: the plain-script suites print
        their verdict and some exit 0 either way, so the exit code alone passed a suite that printed RESULT FAIL"""
        last = None
        with open(log, errors='replace') as f:
            for line in f:
                if line.startswith('RESULT '): last = line.split()[1] if len(line.split()) > 1 else ''
        return last

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        todo = []
        for name, args, needs in SUITES:
            if only and name not in only:
                continue
            miss = [f'{n}: {have[n]}' for n in sorted(needs) if have[n]]
            todo.append((name, '; '.join(miss), None if miss else pool.submit(run, name, args, needs)))
        for name, miss, fut in todo:   # (in the list's order: a line prints when its suite and every earlier one is done)
            if fut is None:
                results.append((name, 'SKIP', miss))
                print(f'SKIP {name}: {miss}', flush=True)
                continue
            rc, dt, log = fut.result()
            results.append((name, 'PASS' if rc == 0 else 'FAIL', f'{dt:.0f} s' + ('' if rc == 0 else f', exit {rc}, {log}')))
            print(f'{"PASS" if rc == 0 else "FAIL"} {name} ({dt:.0f} s)', flush=True)
            if rc != 0:
                with open(log, errors='replace') as f:
                    print(failure_lines(f.read().splitlines()), flush=True)
    fails = [n for n, r, _ in results if r == 'FAIL']
    print(f'ci_cpu: {sum(r == "PASS" for _, r, _ in results)} passed, {len(fails)} failed, '
          f'{sum(r == "SKIP" for _, r, _ in results)} skipped' + (f' (failed: {", ".join(fails)})' if fails else ''))
    print(f'RESULT {"PASS" if not fails else "FAIL"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
