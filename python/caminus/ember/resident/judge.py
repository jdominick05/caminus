"""The loop judge: a finish is checked by evidence, not by the agent's word.

When a graded goal calls finish (after define_done and verify), two things run before the goal completes:

1. The gate: the shell command the agent gave define_done as `gate`, if any, run in its home. A non-zero exit refuses
   the finish and its output becomes the agent's next prompt.
2. The judge: a fresh model call with no tools that sees the goal, the done check, the human's steers, the recorded
   evidence and the recent tool results - not the agent's reasoning or its plan. Images the agent read go with it to a
   judge brain with vision; any other judge is told what the agent says an image shows is an observation (a judge that
   could not see refused correct finishes read off a picture). It applies the evidence rules and
   answers met, not_yet or wait. not_yet refuses the finish with what is missing; wait refuses it because the proof
   depends on something still running (a build, CI, a service coming up) and points the agent at its wait tool. The
   judge never declares a goal impossible (the owner's rule, 11-15: it stopped solvable work; on one benchmark task the
   agent quit after two minutes on a task another harness solved): whether to stop is the agent's call, and a blocked
   agent asks its human; an "impossible" from a model is read as not_yet. After JUDGE_LIMIT refusals in a row the
   advisor is asked what the agent should try differently (without one, or past its limits, the agent's own model
   takes a fresh look) and the advice goes back with a fresh set of tries; after ADVISE_ROUNDS rounds the goal pauses
   for a person, as it always did before.
   Waits have their own, looser limit (WAIT_LIMIT), since waiting for a slow job is not a disagreement.

The judge runs on the agent's own model in a fresh context by default, so it costs nothing on a local brain. With config
`judge_model: "advisor"` it runs on the advisor's model instead (an independent reviewer, paid from the advisor's daily
budget; faster than a busy local model), falling back to the agent's own model when that call cannot be made. A reply that is not the JSON asked
for is asked once more, strictly; a second unreadable reply lets the finish through, marked unchecked, so a broken
judge cannot trap a goal. Every verdict records the model that gave it, and unread ones are counted (judged events
with verdict "unread"), so a model that answers loosely shows up in the evals instead of quietly turning the judge off.
"""
import json
import os
import re
import subprocess

GATE_TIMEOUT = 600
JUDGE_LIMIT = 3
WAIT_LIMIT = 6
EVIDENCE_RESULTS = 12
JUDGE_IMAGES = 4          # newest images the agent read that go with the brief to a judge that can see

# advising rounds per goal before a run of refusals pauses it (each round: the advisor's, or a fresh look)
ADVISE_ROUNDS = 3

FRESH_EYES = '''You are a senior engineer looking at another agent's stuck task with fresh eyes. You see the goal, its done
check and evidence from what actually ran, not the agent's reasoning. It keeps trying to stop without the proof. Name
the most likely reason it is stuck, then three concrete next steps it has not tried (commands, tools, angles), most
promising first. No encouragement, no restating the goal.'''

SYSTEM = '''JUDGE. You check whether an agent's goal is done. You see the goal, the done check the agent wrote, its human's
steers, and evidence from what actually ran. You do not see the agent's reasoning; do not guess at it.

Evidence rules:
- Printed text proves only that text was printed. A claim in a summary is not evidence.
- A command's exit code and output, a file's content, a test result: these are evidence.
- An approval or decision by the human needs a real human message in the steers, not the agent's say-so.
- The done check and every steer must be satisfied, not most of them.
- If the work is done but the proof depends on something still running or not yet observable (a build, a test run,
  CI, a service starting, a scheduled job), say wait, and name what to wait for.
- You never decide that a goal cannot be done. If the agent says it cannot be done, or gives up, that is not evidence
  that it is done: say not_yet and name what is still missing.

Answer with one JSON object and nothing else:
{"verdict": "met" | "not_yet" | "wait", "reason": "one or two sentences naming the evidence, what is missing, or what to wait for"}'''


def run_gate(home, gate, timeout=GATE_TIMEOUT):
    """Run the done check. Its output goes to a file, not a pipe: a gate that starts something in the background (a
    server, `cmd &`) left the pipe open, so waiting for the pipe took the whole timeout and failed a gate that had
    passed. The gate is waited for as a process, in its own process group; whatever it left running is ended after it
    exits (and the whole group on a timeout)."""
    import os
    import signal
    import tempfile
    with tempfile.TemporaryFile() as out:
        p = subprocess.Popen(['bash', '-lc', gate], cwd=home, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
        try:
            os.killpg(p.pid, signal.SIGKILL)    # the gate's leftovers (or the gate itself, timed out)
        except OSError:
            pass
        if code is None:
            p.wait()
            return False, f'the gate did not finish in {timeout} s'
        out.seek(0)
        text = out.read().decode('utf-8', errors='replace')[-4000:]
    return code == 0, f'exit {code}\n{text}'.strip()


def viewed(rt, gid):
    """The images the agent read in this goal, newest last (at most JUDGE_IMAGES, each file once), as the transport's
    image references (files.image_ref). `seen` marks one the agent's brain saw (the runtime sets it when read runs on
    a brain with vision); one without it reached the agent as its size and format only."""
    from .files import image_ref
    seen = {}
    for r in rt.store.db.execute("SELECT data FROM events WHERE goal=? AND kind='tool_result' ORDER BY seq", (gid,)):
        d = json.loads(r['data'])
        ref = image_ref(d.get('name'), d.get('result'))
        if ref and 'sha256' in ref:                    # (one past read's caps was never shown to the agent either)
            seen.pop(ref['path'], None)
            seen[ref['path']] = ref
    return list(seen.values())[-JUDGE_IMAGES:]


def evidence(rt, gid, summary, open_work, images=None, sees=False):
    """The judge's brief. images: what the agent read (viewed()); sees: whether the ones the agent saw go with it (a
    judge brain with vision), else the judge is told the agent's account of them is an observation it cannot check
    here. An image the agent's brain could not see is listed as such: what the agent says it shows is no evidence."""
    g = rt.store.goal(gid)
    parts = ['GOAL:\n' + g['prompt'][:4000], 'DONE CHECK:\n' + (g.get('done_check') or '(none)')]
    steers = [s['text'] for s in rt.store.steers(gid)]
    if steers:
        parts.append('HUMAN STEERS (newest last):\n' + '\n'.join('- ' + s[:600] for s in steers[-10:]))
    events = rt.store.db.execute(
        "SELECT kind,data FROM events WHERE goal=? AND kind IN ('verified','tool_result') ORDER BY seq DESC LIMIT ?",
        (gid, EVIDENCE_RESULTS * 2)).fetchall()
    lines = []
    for r in reversed(events):
        d = json.loads(r['data'])
        if r['kind'] == 'verified':
            lines.append('verify (the agent\'s account of its check): ' + str(d.get('evidence', ''))[:600])
        else:
            res = json.dumps(d.get('result'), ensure_ascii=False)
            lines.append(f"{d.get('name')} -> {res[:800]}")
    parts.append('EVIDENCE (oldest first):\n' + ('\n'.join(lines[-EVIDENCE_RESULTS:]) or '(none recorded)'))
    images = viewed(rt, gid) if images is None else images
    listed = lambda imgs: '\n'.join(f"- {os.path.relpath(i['path'], str(rt.home))} ({i.get('width')}x{i.get('height')} "
                                    f"{str(i.get('format', '')).upper()})" for i in imgs)
    saw, blind = [i for i in images if i.get('seen')], [i for i in images if not i.get('seen')]
    if saw:
        parts.append(('IMAGES THE AGENT SAW (its model can see; each follows this brief, as the agent saw it):\n'
                      + listed(saw)) if sees else
                     ('IMAGES THE AGENT SAW (its model can see; you cannot see them here):\n' + listed(saw) +
                      '\nWhat the agent says such an image shows is an observation, as a command\'s output would be: do '
                      'not refuse a finish only because you cannot see the image yourself.'))
    if blind:
        parts.append('IMAGES THE AGENT READ BUT COULD NOT SEE (its model has no vision: read gave it only each file\'s '
                     'size and format):\n' + listed(blind) + '\nWhat the agent says such an image shows is not an '
                     'observation: count it only where a command\'s output above (an OCR or image tool) shows it.')
    # what the harness itself recorded, so a claimed change can be checked against it
    try:
        rec = rt._change_record(gid)
    except Exception:
        rec = []
    home = str(rt.home)
    changed = sorted({os.path.relpath(p, home) for _s, n, p, ok, _e in rec if p and ok and n in rt.CHANGE_TOOLS})
    failed = [f'{n} {os.path.relpath(p, home)}: {e[:160]}' for _s, n, p, ok, e in rec if p and not ok and n in rt.CHANGE_TOOLS]
    parts.append('FILES CHANGED IN THIS GOAL (the harness\'s record of edit / write calls that succeeded; terminal '
                 'commands are not counted, so read their output above):\n' + ('\n'.join('- ' + c for c in changed) or '(none)')
                 + (('\nREFUSED OR FAILED CHANGES:\n' + '\n'.join('- ' + f for f in failed[-10:])) if failed else ''))
    parts.append('THE AGENT SAYS IT IS DONE:\n' + summary[:2000] + (('\nLEFT OPEN: ' + open_work[:1000]) if open_work else ''))
    return '\n\n'.join(parts)


def verdict(text):
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except ValueError:
        return None
    if v.get('verdict') not in ('met', 'not_yet', 'wait', 'impossible'):
        return None
    # (the judge does not declare goals impossible: a model that still says so is read as not yet done)
    return dict(verdict='not_yet' if v['verdict'] == 'impossible' else v['verdict'], reason=str(v.get('reason', ''))[:1000])


def review(rt, gid, result):
    """Gate, then judge. Returns the finish result to record (an error to refuse it; `pause` to pause the goal)."""
    g = rt.store.goal(gid)
    gate = g.get('gate')
    if gate:
        ok, out = run_gate(rt.home, gate)
        rt.store.event('gate', dict(command=gate, passed=ok, output=out[-2000:]), gid)
        if not ok:
            return dict(error='the gate failed, so the goal is not done; fix what it shows, then finish again:\n' + out)
    # refusals since the last advising round (an advised agent gets a fresh set of tries with a new plan)
    since = rt.store.db.execute("SELECT coalesce(max(seq),0) FROM events WHERE goal=? AND kind='judge_advised'",
                                (gid,)).fetchone()[0]
    refusals = rt.store.db.execute(
        "SELECT count(*) FROM events WHERE goal=? AND kind='judged' AND data LIKE '%\"not_yet\"%' AND seq>?",
        (gid, since)).fetchone()[0]
    images = viewed(rt, gid)

    def brief(sees):
        """the judge's messages: a judge that can see gets the images the agent read (the transport sends the files)"""
        return [dict(role='system', content=SYSTEM),
                dict(role='user', content=evidence(rt, gid, result.get('summary', ''), result.get('open', ''), images,
                                                   sees))] + \
               [dict(role='user', content=os.path.basename(i['path']) + ', as the agent saw it:', image=i)
                for i in (images if sees else []) if i.get('seen')]
    def ask(extra):
        if rt.config.get('judge_model') == 'advisor':
            try:
                # the owner chose an independent, paid reviewer (its config names no vision: it gets the text)
                return rt.advisor.raw(gid, brief(False) + extra, 600)
            except Exception:
                pass
        msgs = brief(bool((getattr(rt.model, 'config', None) or {}).get('vision'))) + extra
        reply = rt.model.complete(msgs, tools=None, max_tokens=600)
        try:
            rt.brains.account(gid)
        except Exception:
            pass
        cfg = getattr(rt.model, 'config', None) or {}
        from . import effort
        # the id sent (a short call runs with thinking off: the -nothink variant), not the base name
        name = effort.resolve(cfg, short=True)['model'] or cfg.get('model') or rt.config.get('model', '')
        return (reply.get('content') if isinstance(reply, dict) else '') or '', name
    try:
        text, who = ask([])
        v = verdict(text)
        if v is None:                                    # once more, strictly
            text, who = ask([dict(role='assistant', content=text[:2000]),
                             dict(role='user', content='Answer with only the JSON object: '
                                  '{"verdict": "met" | "not_yet" | "wait", "reason": "..."}')])
            v = verdict(text)
    except Exception as e:                               # the brain is down or refused: do not trap the goal
        rt.store.event('judged', dict(verdict='unread', reason=f'the judge call failed: {type(e).__name__}'), gid)
        return dict(result, judge='the judge could not run; finish accepted unchecked')
    if v is None:
        rt.store.event('judged', dict(verdict='unread', reason='the judge reply could not be read, twice',
                                      model=who, reply=text[:500]), gid)
        return dict(result, judge='the judge reply could not be read; finish accepted unchecked')
    v['model'] = who
    rt.store.event('judged', v, gid)
    if v['verdict'] == 'met':
        return dict(result, judge='met: ' + v['reason'])
    if v['verdict'] == 'wait':
        # the waits in a row since the judge last said anything else (read from the stored JSON, not its spacing)
        waits = 0
        for row in rt.store.db.execute("SELECT data FROM events WHERE goal=? AND kind='judged' ORDER BY rowid DESC", (gid,)):
            try:
                said = json.loads(row[0]).get('verdict')
            except (TypeError, ValueError, AttributeError):
                said = None
            if said != 'wait':
                break
            waits += 1                          # this verdict is already recorded above
        if waits >= WAIT_LIMIT:
            return dict(error=f'the judge has said wait {waits} times (latest: {v["reason"]}). The goal is paused for '
                        'your human.', pause='judge said wait %d times: %s' % (waits, v['reason']))
        return dict(error='not proven yet, something is still running: ' + v['reason'] + ' Use the wait tool (until '
                    'process, port, file, session or time) for it instead of polling, then verify and finish again.',
                    waiting=True)   # not a failure: the repeat-failure watchdog does not count it (WAIT_LIMIT does)
    if refusals + 1 >= JUDGE_LIMIT:
        # An advising round instead of a pause (the owner, 11-15: nobody is there to decide in an unattended run, and
        # an agent that keeps trying to finish without the proof needs a new plan, not a stop): the advisor sees the
        # goal, the done check, the judge's reason and the agent's last summary, and its advice goes back to the agent
        # with a fresh set of tries. The advisor's per-goal call limit and daily budget bound the rounds; without an
        # advisor (or past its limits) the goal pauses for the human as before.
        question = (f'The agent has tried {refusals + 1} times to finish this goal without meeting its done check. '
                    f'The judge\'s latest reason: {v["reason"]} The agent\'s last summary: '
                    f'{(result.get("summary") or "")[:1500]} What should it try differently? Give concrete next steps '
                    'it has not tried, not encouragement.')
        rounds = rt.store.db.execute("SELECT count(*) FROM events WHERE goal=? AND kind='judge_advised'",
                                     (gid,)).fetchone()[0]
        got, why = None, ''
        if rounds < ADVISE_ROUNDS:
            try:
                got = rt.advisor.ask(gid, question, 'the finish judge asked for this advice after repeated refusals')
            except Exception as e:
                why = str(e)
                # no advisor (or past its limits): the agent's own model in a fresh context, with the evidence but
                # not the agent's reasoning, the way the judge itself runs
                try:
                    reply = rt.model.complete([dict(role='system', content=FRESH_EYES), dict(
                        role='user', content=evidence(rt, gid, result.get('summary', ''), result.get('open', '')) +
                        '\n\n' + question)], tools=None, max_tokens=1500)
                    text = (reply.get('content') if isinstance(reply, dict) else '') or ''
                    try:
                        rt.brains.account(gid)
                    except Exception:
                        pass
                    if text.strip():
                        got = dict(advice=text, model='own model, fresh context')
                except Exception as e2:
                    why += f'; a fresh look failed too ({type(e2).__name__})'
        else:
            why = f'{rounds} advising rounds already'
        if got is None:
            return dict(error=f'the judge refused this finish {refusals + 1} times (latest: {v["reason"]}), and no '
                        f'advice could help ({why}). The goal is paused for your human to decide.',
                        pause='judge refused the finish %d times: %s' % (refusals + 1, v['reason']))
        rt.store.event('judge_advised', dict(refusals=refusals + 1, reason=v['reason'][:500],
                                              model=got.get('model'), advice=(got.get('advice') or '')[:2000]), gid)
        return dict(error='not done yet, by the evidence: ' + v['reason'] + '\n\nYou have tried to finish '
                    f'{refusals + 1} times without the proof, so your advisor looked at where you are. Its advice:\n'
                    + (got.get('advice') or '').strip() + '\n\nTake a new approach from this; finish only when the '
                    'done check is proven.')
    return dict(error='not done yet, by the evidence: ' + v['reason'] + ' Do what is missing, run the check, verify, '
                'then finish again. If one approach failed, try another; if only your human can unblock you, ask them.')
