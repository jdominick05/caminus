# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Lab sessions: a long-lived lab agent to work WITH while designing the harness.

  open    a fresh resident home (on the agent host by default: the agent has a real account and real tools), its daemon
          running, a brain behind it; optionally a suite's fixture in its work folder
  say     send it a goal or a message, wait, print what it did (its replies, tool calls, gate refusals, the judge)
  turn    with the driven brain YOU are the model: print the request waiting for an answer (the conversation tail, the
          tools offered, the newest tool result)
  reply   answer that request: a tool call (--call NAME --args JSON) or text
  events  what happened since you last looked; status: the resident's status
  close   stop the daemon, its job workers (a box and the Chrome in it outlive the daemon by design) and the driven brain:
          TERM, then KILL after CLOSE_GRACE seconds for whatever is left, and say what it had to kill (the home stays)

Ask the agent what it can and cannot do from inside its account, have it try a tool and report what got in the way, or
drive it yourself to see exactly what a model sees. Everything stays in the session's own home (agent host: ~/lab/,
never ~/homes), and the daemon exits after --ttl.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import brains as brains_mod
import hosts as hosts_mod
import runner

HERE = Path(__file__).resolve().parent
TERMINAL = runner.DONE_STATES | {'completed'}


def _dir(out, name):
    return Path(out).resolve() / 'sessions' / name


def _load(out, name):
    p = _dir(out, name) / 'session.json'
    if not p.exists():
        raise SystemExit(f'no lab session {name!r} under {out} (lab.py open first)')
    return json.loads(p.read_text())


def _save(out, s):
    p = _dir(out, s['name']) / 'session.json'
    p.write_text(json.dumps(s, indent=1))


def _host(s):
    h = hosts_mod.make(s['host'], s['out'], s['checkout'])
    if s['host'] == 'agent':
        h.programs = {s['arm']: (s['program'], s.get('runtime_sha'))}
    return h


def _agent(s, h, op, *args, secs=120):
    arm = runner.Arm(s['arm'], s['checkout'])
    return h.run([*h.runtime(arm), op, 'lab', '--home', s['home'], '--json', *args], secs)


def fmt(e, width=500):
    d = e.get('data') or {}
    k = e.get('kind')
    cut = lambda v: (v if len(v) <= width else v[:width] + f' ... [{len(v)} chars]')
    if k == 'assistant':
        text = d.get('text') or d.get('content') or ''
        calls = d.get('tool_calls') or d.get('calls') or []
        names = ', '.join(f"{(c.get('function') or {}).get('name', '?')}({(c.get('function') or {}).get('arguments', '')[:200]})"
                          for c in calls if isinstance(c, dict))
        return f'[{e.get("seq")}] agent: {cut(str(text)).strip()}' + (f'  -> {names}' if names else '')
    if k == 'tool_result':
        res = d.get('result')
        body = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
        return f'[{e.get("seq")}]   {d.get("name", "?")} => {cut(body)}'
    return f'[{e.get("seq")}] {k}: {cut(json.dumps(d, ensure_ascii=False))}'


def _events(s, h, after):
    out = []
    while True:
        page = runner._json_tail(_agent(s, h, 'events', '--after', str(after), secs=180).stdout)
        if not isinstance(page, list) or not page:
            return out
        out += page
        after = max(int(e.get('seq') or after) for e in page)
        if len(page) < 200:
            return out


def open_session(a):
    out = Path(a.out).resolve()
    d = _dir(out, a.name)
    if (d / 'session.json').exists() and not a.replace:
        raise SystemExit(f'lab session {a.name} exists (lab.py close {a.name}, or open --replace)')
    d.mkdir(parents=True, exist_ok=True)
    arm = runner.load_arms([f'{a.arm}={a.checkout}'] if a.checkout else [a.arm], None)[0]
    h = hosts_mod.make(a.host, out, arm.checkout)
    if a.host == 'local' and not a.brain.startswith(('driven', 'scripted')) and not a.unsafe_local:
        raise SystemExit('a real model on the local host gets a shell as you: use --host agent (or --unsafe-local)')
    h.prepare([arm])
    rdir = h.fresh(f'sessions/{a.name}')
    home, work, bridge = f'{rdir}/home', f'{rdir}/home/work', f'{rdir}/bridge'
    for sub in ('req', 'res'):
        h.write(f'{bridge}/{sub}/.keep', b'')
    if a.fixture:
        suite = runner.load_suite(a.fixture)
        h.put_tree(suite['dir'] / 'fixture', work)
    else:
        h.write(f'{work}/.keep', b'')
    s = dict(name=a.name, out=str(out), host=a.host, checkout=str(arm.checkout), arm=arm.name, commit=arm.commit,
             dirty=arm.dirty, home=home, work=work, brain=a.brain, seen=0, opened=time.time(), ttl=a.ttl)
    if a.host == 'agent':
        s['program'], s['runtime_sha'] = h.programs[arm.name]
    if a.brain.startswith('driven'):
        ctl = d / 'driven'
        ctl.mkdir(exist_ok=True)
        p = subprocess.Popen([sys.executable, '-I', str(HERE / 'lab.py'), 'serve-driven', str(ctl), '--ttl', str(a.ttl)]
                             + [f'--{o}' for o in a.brain.split(':')[1:] if o in ('judge', 'caminus')],
                             stdout=subprocess.PIPE, stderr=open(d / 'driven.log', 'wb'), start_new_session=True)
        port = int(p.stdout.readline())
        s.update(driven=str(ctl), driven_pid=p.pid)
        init = ['--model', 'driven', '--endpoint', f'http://127.0.0.1:{port}/{a.name}/v1', '--context', '65536']
        cfg = {}
    else:
        b = brains_mod.parse(a.brain)
        if b.kind == 'scripted':
            raise SystemExit('a lab session needs a brain that answers freely: driven, openai:..., anthropic:...')
        b.start()
        init, cfg = b.init_args(a.name), b.config()
        s['brain_label'] = b.label()
    r = _agent(s, h, 'init', *init, '--bridge', bridge, '--bus', f'{home}/mail.sqlite3')
    if r.returncode:
        raise SystemExit('init failed: ' + (r.stderr or r.stdout).decode()[-400:])
    cfg.update(arm.config)
    if cfg:
        h.run(['python3', '-I', '-c', runner.MERGE, f'{home}/config.json'], 60, stdin=json.dumps(cfg).encode())
    daemon = [*h.runtime(arm), 'run', 'lab', '--home', home]
    if a.host == 'agent':
        import shlex
        cmd = ' '.join(shlex.quote(x) for x in hosts_mod.bounded(daemon, a.ttl))
        h._ssh(f'nohup {cmd} > {shlex.quote(home)}/daemon.log 2>&1 < /dev/null &', 30)
    else:
        p = subprocess.Popen(hosts_mod.bounded(daemon, a.ttl), stdout=open(Path(home) / 'daemon.log', 'wb'),
                             stderr=subprocess.STDOUT, start_new_session=True)
        s['daemon_pid'] = p.pid
    _save(out, s)
    print(f'lab session {a.name}: {a.host} host, arm {arm.name} ({arm.commit}{" dirty" if arm.dirty else ""}), '
          f'brain {a.brain}; work folder {work}; daemon up for {a.ttl}s')
    return 0


def say(a):
    s = _load(a.out, a.name)
    h = _host(s)
    text = a.text if a.text != '-' else sys.stdin.read()
    goal = a.goal
    if goal:
        # the agent's send takes a full goal id; a short one (as status and the panel print them) is resolved here, and an
        # unknown or ambiguous one stops before anything is sent
        st = runner._json_tail(_agent(s, h, 'status').stdout) or {}
        hits = [g['id'] for g in st.get('goals', []) if str(g.get('id', '')).startswith(goal)]
        if len(hits) != 1:
            raise SystemExit(f'say: goal {goal} matches {len(hits)} goals' + (f': {", ".join(hits)}' if hits else ''))
        goal = hits[0]
    args = ['--goal-id', goal, '--goal', text] if goal else ['--goal', text]
    r = _agent(s, h, 'send', *args)
    sent = runner._json_tail(r.stdout) or {}
    if r.returncode or sent.get('error'):
        raise SystemExit('say: not sent: ' + (sent.get('error') or (r.stderr or r.stdout).decode()[-300:]))
    gid = sent.get('id') or goal
    print(f'sent ({gid})' + (f': {sent["note"]}' if sent.get('note') else ''))
    wait = a.wait if a.wait is not None else (0 if s['brain'].startswith('driven') else 900)
    end = time.time() + wait
    state = None
    while time.time() < end:
        st = runner._json_tail(_agent(s, h, 'status').stdout) or {}
        row = next((g for g in st.get('goals', []) if g.get('id') == gid), None)
        state = row and row.get('state')
        if state in TERMINAL:
            break
        time.sleep(3)
    return _print_new(a, s, h, extra=f'goal {gid}: {state or "running"}')


def _print_new(a, s, h, extra=None):
    evs = _events(s, h, s.get('seen', 0))
    for e in evs:
        print(fmt(e, a.width))
    if evs:
        s['seen'] = max(int(e.get('seq') or 0) for e in evs)
        _save(a.out, s)
    if extra:
        print(extra)
    return 0


def events(a):
    s = _load(a.out, a.name)
    if a.all:
        s['seen'] = 0
    return _print_new(a, s, _host(s))


def status(a):
    s = _load(a.out, a.name)
    st = runner._json_tail(_agent(s, _host(s), 'status').stdout) or {}
    keep = {k: st.get(k) for k in ('model', 'context', 'context_used', 'daemon_alive', 'spend_today')}
    keep['goals'] = [{k: g.get(k) for k in ('id', 'state', 'steps', 'text')} for g in st.get('goals', [])[-5:]]
    print(json.dumps(keep, indent=1, ensure_ascii=False))
    return 0


def _pending(s):
    ctl = Path(s.get('driven') or '')
    if not s.get('driven'):
        raise SystemExit('this session has no driven brain (open it with --brain driven)')
    return sorted(p for p in ctl.glob('*.req.json') if not p.with_name(p.name.replace('.req.', '.rep.')).exists()
                  and not p.with_name(p.name.replace('.req.', '.gone.')).exists())


def _hung_up(s):
    """the newest request, when its caller hung up before it was answered: (number, gone record), else None"""
    reqs = sorted(Path(s.get('driven') or '').glob('*.req.json'))
    if not reqs:
        return None
    gone = reqs[-1].with_name(reqs[-1].name.replace('.req.', '.gone.'))
    return (reqs[-1].name[:6], json.loads(gone.read_text())) if gone.exists() else None


def turn(a):
    s = _load(a.out, a.name)
    end = time.time() + a.wait
    while not (pend := _pending(s)) and time.time() < end:
        time.sleep(0.5)
    if not pend:
        if (h := _hung_up(s)):
            print(f'request {h[0]}: the resident hung up after {h[1].get("waited_s")}s (an Esc or a /cancel closed the call)')
        print('no request waiting (the agent is idle, or between steps)')
        return 1
    body = json.loads(pend[0].read_text())
    msgs = body.get('messages') or []
    system = [m for m in msgs if m.get('role') == 'system']
    tools = [((t.get('function') or t).get('name')) for t in body.get('tools') or []]
    print(f'request {pend[0].name[:6]}: {len(msgs)} messages, system {sum(len(str(m.get("content"))) for m in system)} chars, '
          f'{len(tools)} tools: {", ".join(tools)}')
    if a.system:
        for m in system:
            print('--- system\n' + str(m.get('content')))
    for m in [m for m in msgs if a.system or m.get('role') != 'system'][-a.last:]:
        c = m.get('content')
        c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        calls = ', '.join(f"{(t.get('function') or {}).get('name')}({(t.get('function') or {}).get('arguments', '')[:200]})"
                          for t in m.get('tool_calls') or [])
        print(f"--- {m.get('role')}{(' ' + m['name']) if m.get('name') else ''}: "
              f"{c[:a.width]}{' ...' if len(c) > a.width else ''}{('  -> ' + calls) if calls else ''}")
    print(f'(full request: {pend[0]})')
    return 0


def reply(a):
    s = _load(a.out, a.name)
    pend = _pending(s)
    if not pend:
        raise SystemExit('no request waiting')
    r = dict(text=a.text or '')
    if a.call:
        r.update(call=a.call, args=json.loads(a.args or '{}'))
    if a.timings:
        r['timings'] = json.loads(a.timings)   # (sent only by a driven:caminus brain, as caminus serve sends them)
    rep = pend[0].with_name(pend[0].name.replace('.req.', '.rep.'))
    tmp = rep.with_suffix('.tmp')
    tmp.write_text(json.dumps(r))
    os.replace(tmp, rep)
    time.sleep(a.settle)
    return _print_new(a, s, _host(s))


CLOSE_GRACE = 10   # seconds between TERM and KILL; longer than a terminal worker's own TERM-to-KILL for its command

# Run on the session's host (as runner.MERGE is): argv ROOT GRACE [PID:LABEL ...]. Signals the daemon named in
# ROOT/heartbeat.json, every worker and command in ROOT/jobs/*/state.json and the PIDs given, each as its process group
# when it leads one; KILLs what is left after GRACE seconds; prints what it stopped, killed and could not end. A pid is
# signalled only when its command line or environment names ROOT (the daemon's --home, a worker's job folder, a
# command's WORKSPACE), so a recycled pid in a stale file is never someone else's process.
STOP = r"""
import json, os, signal, sys, time
from pathlib import Path
root, grace = Path(sys.argv[1]), float(sys.argv[2])
mark = str(root).encode()
def gone(pid):
    try:
        st = Path(f'/proc/{pid}/stat').read_text()
        return st[st.rindex(')') + 2] in 'ZX'
    except (OSError, ValueError, IndexError):
        return True
def ours(pid):
    try:
        return any(mark in Path(f'/proc/{pid}/{f}').read_bytes() for f in ('cmdline', 'environ'))
    except OSError:
        return False
found = {}
def add(pid, what):
    if isinstance(pid, int) and pid > 1 and pid != os.getpid() and not gone(pid) and ours(pid):
        found.setdefault(pid, what)
try:
    add(json.loads((root / 'heartbeat.json').read_text()).get('pid'), 'daemon')
except (OSError, ValueError, AttributeError):
    pass
for arg in sys.argv[3:]:
    pid, _, what = arg.partition(':')
    add(int(pid), what)
for f in sorted(root.glob('jobs/*/state.json')):
    try:
        st = json.loads(f.read_text())
    except (OSError, ValueError):
        continue
    if isinstance(st, dict):
        add(st.get('worker'), 'worker ' + f.parent.name[:8])
        add(st.get('pid'), 'command ' + f.parent.name[:8])
def send(pid, sig):
    try:
        os.killpg(pid, sig) if os.getpgid(pid) == pid else os.kill(pid, sig)
    except OSError:
        pass
def settle(secs):
    end = time.monotonic() + secs
    while time.monotonic() < end and not all(gone(p) for p in found):
        time.sleep(.1)
    return [p for p in found if not gone(p)]
for pid in found:
    send(pid, signal.SIGTERM)
killed = settle(grace)
for pid in killed:
    send(pid, signal.SIGKILL)
left = settle(3)
row = lambda pids: [dict(pid=p, what=found[p]) for p in pids]
print(json.dumps(dict(stopped=row(found), killed=row(killed), left=row(left))))
"""


def _stop(run, root, extra=()):
    """STOP on a host (run: the host's run, or a local one); its report, or a report naming why there is none."""
    r = run(['python3', '-I', '-c', STOP, root, str(CLOSE_GRACE), *extra], CLOSE_GRACE + 20)
    return runner._json_tail(r.stdout) or dict(stopped=[], killed=[], left=[dict(pid=0, what='unknown: ' + (
        r.stderr or r.stdout).decode(errors='replace')[-200:].strip())])


def close(a):
    s = _load(a.out, a.name)
    h = _host(s)
    # the local daemon runs under timeout (its group is ours to signal); on the agent host heartbeat.json names it
    reps = [_stop(h.run, s['home'], [f'{s["daemon_pid"]}:daemon timeout wrapper'] if s.get('daemon_pid') else [])]
    if s.get('driven_pid'):
        reps.append(_stop(hosts_mod.Local(a.out).run, s['driven'], [f'{s["driven_pid"]}:driven brain']))
    names = lambda key: ', '.join(f'{x["what"]} ({x["pid"]})' for r in reps for x in r[key])
    stopped, killed, left = names('stopped'), names('killed'), names('left')
    print(f'lab session {a.name}: ' + (f'stopped {stopped}' if stopped else 'nothing was running'))
    if killed:
        print(f'had to SIGKILL after {CLOSE_GRACE} s: {killed}')
    if left:
        print(f'still running after SIGKILL: {left}; the session stays open (close it again, or end them by hand)')
        return 1
    p = _dir(a.out, a.name) / 'session.json'
    p.rename(p.with_name('session.closed.json'))
    print(f'lab session {a.name} closed; its home stays at {s["home"]} ({s["host"]} host)')
    return 0
