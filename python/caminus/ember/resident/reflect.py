"""/reflect: the post-goal reflection (issue #341).

When a goal completes, one cheap model call asks what harness change (a prompt section, a tool, a gate, a cap)
would have made this goal faster or more reliable, and writes the answer as a SIM record with status
'proposed'. That record is a queue a person reads, never a change: the harness is edited only by a human and a
pull request. The proposal carries the goal it came from and the evidence the call saw (the done check, the
judge's verdict, the verification, the failed calls), so anyone can check it against the goal that produced it.

The call runs after the goal's completion transaction, outside it (a model call inside BEGIN would block every
other writer), and never raises: a reflection that failed is an event on a completed goal, not a failed goal.
It is OFF unless a session turns it on: config 'reflect' defaults to false, because it is a paid call per graded
goal and its only product is a proposal a person reads, so it must be a spend the user chose. The switch is
`caminus ember agent reflect --on` (and `--off`), which writes the same key. It is also skipped when the SIM log
is off (an A/B arm records nothing), in free time (not graded work), and when the runtime was asked to stop. A
goal re-finished after its human reopened it reflects again: that is a new episode with new evidence.

The system message begins with REFLECT so a log reader (and the test fixture) can name the call, as the
judge's begins with JUDGE.
"""
import json
import threading
import time

from . import sim

MAX_TOKENS = 700              # well under model.SHORT_CALL: this call runs with thinking off
TIMEOUT = 60                  # seconds per attempt, whole: the goal is complete, so a hung endpoint must not hold the
                              # loop for a turn's 900 s. The socket timeout passed with the call only bounds each
                              # read, so the total is enforced by closing the socket (_bounded): one attempt
                              # normally, a second only when the first reply could not be read (60-120 s).
SYSTEM = '''REFLECT. You review one finished agent goal and propose at most one change to the harness (the system
prompt, a tool, a gate, a cap) that would have made this goal faster or more reliable. You see the goal, its
done check, how it ended, its tool calls and the failures along the way. Propose only a change the evidence
supports: a recurring problem this goal hit, not a style preference, and not a restatement of the goal's own
work. Nothing you say is applied; a person reads the proposal.

Answer with only a JSON object:
{"problem_group": "the recurring problem, short", "change": "what would change in the harness",
 "rationale": "why the evidence supports it", "predicted": "the outcome, as a test, CI or a reviewer can confirm or contradict"}

Or, when this goal shows no recurring problem worth a change:
{"skip": true, "reason": "why"}'''
STRICT = 'Answer with only the JSON object (or {"skip": true, "reason": "..."}), nothing else.'


def maybe(rt, gid, summary='', open_work=''):
    """One cheap call after a goal completed: a 'proposed' SIM record, or nothing. Never raises."""
    try:
        from .freetime import KINDS as FREE_KINDS
        g = rt.store.goal(gid)
        if not g or g.get('kind', 'work') in FREE_KINDS:
            return None
        if g.get('kind') == 'delegate':
            # a delegate child is capped and reports to its parent (#382), and its parent reflects on the goal it
            # was doing: a second call inside the child's cap would bill the parent twice for one lesson
            return None
        if not g.get('done_check'):
            # graded work only: a goal with no done check has no judge and no standard it was measured against,
            # so a proposal from it would restate the goal rather than name a harness problem. (It also keeps a
            # plain two-turn conversation two model calls, as the session tests count them.)
            return None
        if not sim.enabled(rt) or not rt.config.get('reflect', False):
            # off unless the session asked for it (`agent reflect --on`): a paid call per graded goal is the user's
            # decision, and this one's only product is a proposal a person reads
            return None
        if rt.stopping():
            return None
        if not callable(getattr(getattr(rt, 'model', None), 'complete', None)):
            return None           # a stubbed runtime (some tests drop a plain namespace in for the model)
        if not _may_spend(rt, gid):
            # the brain in force is the fallback and its daily allowance is spent (or it is unpriced): the turn
            # after this one would not go there either, so the reflection does not spend past it (brains._allowed)
            rt.store.event('reflection_skipped', dict(reason="the fallback's allowance is spent"), gid)
            return None
        msgs = [dict(role='system', content=SYSTEM),
                dict(role='user', content=brief(rt, gid, summary, open_work))]
        text = _ask(rt, gid, msgs)
        rec = parse(text)
        if rec is None:
            # A second 60 s call after a daemon got SIGTERM (the first reply was unreadable) is the shutdown wait
            # stopping()'s docstring warns about: the flag is read again before the retry spends another timeout.
            if rt.stopping():
                rt.store.event('reflection_failed', dict(reason='stopping before the retry', reply=text[:500]), gid)
                return None
            text = _ask(rt, gid, msgs + [dict(role='assistant', content=text[:2000]),
                                         dict(role='user', content=STRICT)])
            rec = parse(text)
        if rec is None:
            rt.store.event('reflection_failed', dict(reason='the reply could not be read, twice', reply=text[:500]), gid)
            return None
        if rec.get('skip'):
            rt.store.event('reflection_skipped', dict(reason=rec['reason'][:300]), gid)
            return None
        goal = rt.store.goal(gid) or {}
        rec.update(status='proposed',
                   eval_context='goal %s · %s steps · judge %s' % (
                       gid[:8], goal.get('steps') or 0, _judge(rt, gid, _episode_start(rt, gid))[0] or 'not run'))
        return sim.record(rt, gid, rec)
    except Exception as e:
        try:
            rt.store.event('reflection_failed', dict(reason='%s: %s' % (type(e).__name__, str(e)[:300])), gid)
        except Exception:
            pass                  # the reflection is a proposal, never a reason to lose a finished goal
        return None


def _bounded(rt, fn, seconds=None):
    """Run fn() with a whole-call bound, or as soon as the daemon is asked to stop.

    The socket timeout passed to complete() bounds each read, not the call: a stream that trickles one byte just
    under the timeout holds the caller indefinitely, and a call already in flight when SIGTERM arrives still adds
    its read to the shutdown wait. The call runs on a worker thread and this one waits at most `seconds` (TIMEOUT
    unless a test says otherwise), checking the stop flag on the way, then closes the call's socket with
    Model.abort() so the blocked read ends now. What fn returned is returned; what it raised is raised again.

    Accounting contract for the abandoned path (#383 review P2): the caller's own metering (`rt._meter`,
    `rt.brains.account`) runs on THIS thread after this returns, so a call abandoned here is never priced by the
    caller. The abandoned call keeps running on its daemon thread and may still write model.last_usage/last_call/
    last_speed when it finally returns; that write cannot be priced either, because every metering site pairs with
    a complete() that rewrote them first, and it cannot write the store (those calls stay on the loop thread).
    """
    out = {}

    def run():
        try:
            out['value'] = fn()
        except BaseException as e:      # handed back to the loop thread, so a Cancelled still cancels
            out['error'] = e

    worker = threading.Thread(target=run, name='reflect-call', daemon=True)
    worker.start()
    end = time.monotonic() + (TIMEOUT if seconds is None else seconds)
    while worker.is_alive():
        left = end - time.monotonic()
        if left <= 0 or rt.stopping():
            stop = bool(rt.stopping())
            abort = getattr(getattr(rt, 'model', None), 'abort', None)
            if callable(abort):
                try:
                    abort()             # (the server stops generating too: an abandoned call keeps it busy)
                except Exception:
                    pass
            worker.join(5)              # (a call that ignores the closed socket is left to daemon exit)
            if stop:
                raise TimeoutError('the reflection call did not finish: the daemon is stopping')
            raise TimeoutError('the reflection call did not finish: no reply in %gs' % (seconds or TIMEOUT))
        worker.join(min(0.2, max(0.01, left)))
    if 'error' in out:
        raise out['error']
    return out.get('value')


def _ask(rt, gid, msgs):
    """One short call, priced into the goal's spend like any other turn, on its own timeout: the goal is already
    complete, so a hung endpoint must not hold the loop for a turn's 900 s. It runs on the loop thread (one call per
    completed goal, and the loop is idle between goals), so the bound that matters is TIMEOUT per attempt, whole
    (_bounded): two at most, when the first reply could not be read, and never one past a stop request."""
    reply = _bounded(rt, lambda: rt.model.complete(msgs, tools=None, max_tokens=MAX_TOKENS, timeout=TIMEOUT))
    try:
        rt.brains.account(gid)
    except Exception:
        pass
    try:
        rt._meter(gid)
    except Exception:
        pass
    return (reply.get('content') if isinstance(reply, dict) else '') or ''


def parse(text):
    """The JSON object in a reply, or None. A record needs a change and a prediction (sim.record refuses one
    without them), and {"skip": true} proposes nothing."""
    from .judge import object_in
    d = object_in(text)
    if d is None:
        return None
    if d.get('skip'):
        return dict(skip=True, reason=str(d.get('reason') or 'no reason given')[:500])
    change, predicted = str(d.get('change') or '').strip(), str(d.get('predicted') or '').strip()
    if not (change and predicted):
        return None
    return dict(problem_group=str(d.get('problem_group') or '').strip(),
                change=change, rationale=str(d.get('rationale') or '').strip(), predicted=predicted)


def _episode_start(rt, gid):
    """The seq this episode began after: how the previous episode ended — an earlier `finished` event, or the error
    that stopped the goal before it was resumed. A re-enqueued goal has no earlier `finished`, so counting only
    `finished` left the boundary at 0 and the pre-error episode's evidence entered the new reflection (review P2).
    `paused` is a boundary too: a paused goal is resumed in place (resume_goal), so the pause ends the episode
    before the one that finishes next, and without it that episode's judge and tool calls came along. `cancelled`
    is listed for the same reason even though a cancelled goal cannot be resumed today: if it ever is, the episode
    before the cancel must not leak into the one that finishes after it."""
    row = rt.store.db.execute(
        "SELECT seq FROM events WHERE goal=? AND kind IN "
        "('finished','runtime_error','model_refused','model_gave_up','paused','cancelled') "
        "ORDER BY seq DESC LIMIT 1 OFFSET 1", (gid,)).fetchone()
    return row['seq'] if row else 0


def _may_spend(rt, gid):
    """Whether the brain in force may take this call: the primary is the model the user chose, so it always may; the
    fallback is gated by the runtime's own rule, not a copy of it (brain.Brains._allowed).
    """
    brains = getattr(rt, 'brains', None)
    try:
        if brains is None or not brains.cfg or getattr(rt, 'model', None) is not brains.fallback:
            return True
        # Reuse the runtime's gate rather than restating it. A copy drifted once already: this guard checked only
        # spent_today() < budget_usd, so an unpriced remote fallback (brain._allowed refuses it: "cannot be
        # limited") still served a billed reflection past the cap the runtime enforces on the goal's own turns
        # (review P1). _allowed is the one rule now: unpriced remote is refused, a spent allowance is refused,
        # and a free-time goal follows _free_choice exactly as a work turn does.
        return bool(brains._allowed(gid))
    except Exception:
        # fail closed when the fallback is in force and the gate cannot be evaluated at all: skipping a
        # reflection costs a proposal, a call past an unreadable cap costs money (a stub runtime never reaches
        # here - `not brains.cfg` above is the stub case).
        return False


def _judge(rt, gid, since=0):
    """The newest `judged` event this episode, as (verdict, reason): the query is here once, and `brief` reads the
    reason from the same row it took the verdict from (review P2: it ran the identical SELECT a second time)."""
    row = rt.store.db.execute(
        "SELECT data FROM events WHERE goal=? AND kind='judged' AND seq>? ORDER BY seq DESC LIMIT 1",
        (gid, since)).fetchone()
    try:
        d = json.loads(row['data']) if row else None
    except (ValueError, KeyError, TypeError):
        return None, ''
    if not isinstance(d, dict):
        return None, ''
    return (d.get('verdict') or None), str(d.get('reason') or '')[:600]


def brief(rt, gid, summary, open_work):
    """The evidence the call sees: the goal, the done check, its own plan, how it ended, and where it struggled."""
    g = rt.store.goal(gid) or {}
    since = _episode_start(rt, gid)
    parts = ['GOAL:\n' + (g.get('prompt') or '')[:3000],
             'DONE CHECK:\n' + (g.get('done_check') or '(none)')]
    todo = rt.store.todo(gid)
    if todo.get('items'):
        mark = {'done': '[x]', 'in_progress': '[>]', 'pending': '[ ]'}
        parts.append("THE AGENT'S PLAN (its todo list):\n" + '\n'.join(
            '- ' + mark.get(it['state'], '[ ]') + ' ' + it['text'][:200] for it in todo['items'][:30]))
        if todo.get('blockers'):
            parts.append('WHAT IT SAID BLOCKED IT:\n' + '\n'.join('- ' + str(b)[:200] for b in todo['blockers'][:10]))
    parts.append('HOW IT ENDED:\n' + ((summary or '(no summary)')[:2000]))
    if open_work:
        parts.append('LEFT OPEN (the agent says):\n' + open_work[:1000])
    parts.append('THE GOAL TOOK %s MODEL TURNS, the agent changed approach %s times, and %d steer(s) arrived.' % (
        g.get('steps') or 0, g.get('approach_changes') or 0, len(rt.store.steers(gid))))
    verd, reason = _judge(rt, gid, since)
    if verd:
        parts.append('THE FINISH JUDGE: %s — %s' % (verd, reason))
    row = rt.store.db.execute(
        "SELECT data FROM events WHERE goal=? AND kind='verified' AND seq>? ORDER BY seq DESC LIMIT 1",
        (gid, since)).fetchone()
    if row:
        try:
            parts.append('WHAT THE AGENT VERIFIED:\n' + str(json.loads(row['data']).get('evidence') or '')[:1000])
        except (ValueError, KeyError, TypeError):
            pass
    # this episode only: after a reopened goal finishes again, the previous episode's calls are not this one's
    # evidence (the judge and verification lookups above are scoped the same way: newest-first within the episode)
    rows = rt.store.db.execute(
        "SELECT data FROM events WHERE goal=? AND kind='tool_result' AND seq>? ORDER BY seq DESC LIMIT 200",
        (gid, since)).fetchall()
    names, failed, errors = {}, 0, []
    for r in rows:
        try:
            d = json.loads(r['data'])
        except ValueError:
            continue
        name = d.get('name') or '?'
        names[name] = names.get(name, 0) + 1
        res = d.get('result')
        # A tool's failure shape: the runtime's own tools put the reason under 'error'; a tool that only says
        # {'ok': False} (no 'error') is a failure too, or the reflection never sees it (#383 review P2).
        if isinstance(res, dict) and (res.get('error') or res.get('ok') is False):
            failed += 1
            errors.append('%s: %s' % (name, str(res.get('error') or 'ok=False')[:200]))
    parts.append('ITS TOOL CALLS (newest first, at most 200): ' +
                 (', '.join('%s×%d' % (k, v) for k, v in names.items()) or '(none)'))
    if failed:
        parts.append('%d of those calls failed; the newest:\n' % failed + '\n'.join(errors[:5]))
    return '\n\n'.join(parts)
