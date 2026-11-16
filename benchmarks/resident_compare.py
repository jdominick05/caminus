"""Compare two real agent CLIs on identical goals and independently checked effects.

This runner also supports CPU fixture CLIs for its own tests. Model-backed runs
must use the normal GPU guard/window; this script never starts or stops a server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


def argv(text):
    value = json.loads(text)
    if not isinstance(value, list) or not value or not all(isinstance(s, str) for s in value):
        raise argparse.ArgumentTypeError('command must be a nonempty JSON array of strings')
    return value


def command(args, values):
    return [replace(arg, values) for arg in args]


def replace(arg, values):
    for key, value in values.items():
        arg = arg.replace('{'+key+'}', str(value))
    return arg


def invoke(args, workspace, goal, output, timeout, env):
    """Time the agent, not its verifier. No shell interpolation or shared workspace."""
    started = time.perf_counter()
    timed_out = False
    with (output.with_suffix('.stdout')).open('wb') as out, (output.with_suffix('.stderr')).open('wb') as err:
        try:
            p = subprocess.Popen(args, cwd=workspace, env=env, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
        except OSError as e:
            err.write(str(e).encode())
            return dict(seconds=time.perf_counter()-started,rc=127,timed_out=False,error=str(e))
        try:
            p.communicate(goal.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Only this invocation's process group, never the model server.
            try:os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
    return dict(seconds=time.perf_counter()-started, rc=p.returncode, timed_out=timed_out)


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', type=Path, required=True, help='goal.txt, seed/ and independent verify.py')
    ap.add_argument('--out', type=Path, required=True, help='fresh output directory')
    for name in ('qwen','resident'):
        ap.add_argument('--'+name+'-argv', type=argv, required=True, help='JSON argv; goal on stdin; {workspace}, {run_dir}, {goal_file} placeholders')
        ap.add_argument('--'+name+'-prepare', type=argv, help='optional setup outside the measured agent interval')
    ap.add_argument('--profile', type=Path, required=True, help='JSON recording actual model, endpoint, context, sampling and hardware settings for both CLIs')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--timeout', type=float, default=600)
    a = ap.parse_args(args)
    if a.repeats<1 or a.timeout<=0:ap.error('repeats and timeout must be positive')
    case, out = a.case.resolve(), a.out.resolve()
    goal = (case/'goal.txt').read_text()
    seed, verify = case/'seed', case/'verify.py'
    if not seed.is_dir() or not verify.is_file():ap.error('case must contain seed/ and verify.py')
    profile = json.loads(a.profile.read_text())
    if out.exists() and any(out.iterdir()):ap.error('--out must be absent or empty')
    out.mkdir(parents=True, exist_ok=True)
    (out/'goal.txt').write_text(goal)
    digest = hashlib.sha256()
    for p in sorted(seed.rglob('*')):
        if p.is_symlink():ap.error('seed must not contain symlinks')
        if p.is_file():digest.update(str(p.relative_to(seed)).encode()+b'\0'+p.read_bytes())
    report = dict(goal_sha256=hashlib.sha256(goal.encode()).hexdigest(), seed_sha256=digest.hexdigest(),
                  verifier_sha256=hashlib.sha256(verify.read_bytes()).hexdigest(), profile=profile,
                  uid=os.getuid(), repeats=a.repeats, timeout=a.timeout, runs=[], summary={})
    report['commands']={name:dict(argv=getattr(a,name+'_argv'),prepare=getattr(a,name+'_prepare')) for name in ('qwen','resident')}
    for repeat in range(a.repeats):
        for name in (('qwen','resident') if repeat%2==0 else ('resident','qwen')):
            run_dir=out/f'{repeat+1:02d}-{name}';run_dir.mkdir()
            workspace=run_dir/'workspace';shutil.copytree(seed,workspace)
            values=dict(workspace=workspace, run_dir=run_dir, goal_file=out/'goal.txt')
            env=dict(os.environ, BENCH_WORKSPACE=str(workspace), BENCH_RUN_DIR=str(run_dir), BENCH_GOAL_FILE=str(out/'goal.txt'))
            row=dict(harness=name, repeat=repeat+1, workspace=str(workspace))
            prep=getattr(a,name+'_prepare')
            if prep:
                row['prepare']=invoke(command(prep,values),workspace,'',run_dir/'prepare',a.timeout,env)
            if not prep or row['prepare']['rc']==0:
                row.update(invoke(command(getattr(a,name+'_argv'),values),workspace,goal,run_dir/'agent',a.timeout,env))
                row['verification']=invoke([sys.executable,'-I',str(verify),str(workspace)],run_dir,'',run_dir/'verify',min(a.timeout,60),env)
                row['correct']=row['rc']==0 and not row['timed_out'] and row['verification']['rc']==0 and not row['verification']['timed_out']
            else:row['correct']=False
            report['runs'].append(row)
            (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    for name in ('qwen','resident'):
        rows=[r for r in report['runs'] if r['harness']==name]
        passed=[r['seconds'] for r in rows if r['correct']]
        report['summary'][name]=dict(passed=len(passed), total=len(rows), median_correct_seconds=statistics.median(passed) if passed else None)
    if all(r['correct'] for r in report['runs']):
        report['resident_speedup']=report['summary']['qwen']['median_correct_seconds']/report['summary']['resident']['median_correct_seconds']
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'],resident_speedup=report.get('resident_speedup'))),flush=True)
    return 0 if all(r['correct'] for r in report['runs']) else 1


if __name__=='__main__':raise SystemExit(main())
