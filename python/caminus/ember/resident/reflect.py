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
NAMES_MAX = 40                # distinct tool names the brief lists before it keeps only the ones seen twice: the
                              # failures below them are the evidence that matters, and a churning goal's one-off
                              # names were pushing them out of the prompt's practical window (#383 review P2)
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
# The retry after an unreadable reply gets the same MAX_TOKENS budget, so a reply clipped at 700 tokens comes back
# clipped again and the goal ends with `reflection_failed` (#383 review P2). Asking for short fields is what makes
# the retry able to succeed: four fields under 200 characters is ~250 tokens, well inside the budget, and a fence or
# a paragraph of preamble is the other way a reply is unreadable (parse -> judge.object_in).
STRICT = ('Answer with only the JSON object (or {"skip": true, "reason": "..."}), nothing else: no markdown fence, '
          'no text before or after it, and every field under 200 characters.')


def maybe(rt, gid, summary='', open_work=''):
    """One cheap call after a goal completed: a 'proposed' SIM record, or nothing. Never raises."""
    msgs = []                     # (bound before the try: the TimeoutError arm below prices what was sent)
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
        why = _spend_refusal(rt, gid)
        if why:
            # the brain in force is the fallback and its daily allowance is spent (or it is unpriced, or the gate
            # itself raised): the turn after this one would not go there either, so the reflection does not spend
            # past it (brains._allowed). The reason is the gate's own (see _spend_refusal): a raised gate is a
            # harness defect, and the queue should not read it as a spent budget (#383 review P2).
            rt.store.event('reflection_skipped', dict(reason=why), gid)
            return None
        episode = _episode_start(rt, gid)    # (captured before the call: the reply belongs to THIS episode)
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
            why = _spend_refusal(rt, gid)
            if why:
                # The first call's own cost is priced into spent_today (brains.account in _ask), so a retry can
                # start out over the allowance the runtime just refuses the goal's next turn on: the gate is read
                # again here, and crossing it ends the reflection instead of billing past it (#383 review P1).
                # The gate's own reason travels, with the retry named: a raised gate is not a spent budget.
                rt.store.event('reflection_skipped', dict(reason=why + ' before the retry'), gid)
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
        if _episode_start(rt, gid) != episode:
            # The call took 60-120 s and the goal was continued while it ran (`send --continue` writes
            # `episode_resumed`): this reply answers the OLD episode, and recording it now would stamp an old
            # lesson with the new episode's eval_context (judge "not run") and then bill a second reflection for
            # the same goal at the new episode's own finish. Drop it; the proposal belongs to an episode that has
            # already ended (#383 review P2).
            rt.store.event('reflection_skipped',
                           dict(reason='the goal was continued while the reflection was in flight'), gid)
            return None
        goal = rt.store.goal(gid) or {}
        rec.update(status='proposed',
                   eval_context='goal %s · %s steps · judge %s' % (
                       gid[:8], goal.get('steps') or 0, _judge(rt, gid, episode)[0] or 'not run'))
        return sim.record(rt, gid, rec)
    except Exception as e:
        try:
            if rt.stopping():
                # _bounded ends an in-flight call when the daemon is asked to stop and raises TimeoutError for it.
                # That is a cut-short reflection, not a failing one: a `reflection_failed` row in the event log
                # reads as a defect of the harness to the person scanning it (#383 review P2-1).
                rt.store.event('reflection_skipped', dict(reason='stopping: the reflection was cut short'), gid)
            elif isinstance(e, TimeoutError):
                # The endpoint did not answer within the attempt's own bound. Same kind of nothing as a stop-cut
                # call: the harness was not at fault and the next goal may try again, so it is a skip, not a
                # `reflection_failed` a person would read as a defect (#383 review P2-2).
                # The abandoned call may still be generating, and its own usage cannot be priced here (see
                # _bounded's contract): the event carries an upper bound marked as such, and the same bound is
                # debited to the ledgers now, on this thread, so a total does not read low by a call no metering
                # site ever saw (#383 review P2).
                bound = _unpriced(rt, msgs)
                rt.store.event('reflection_skipped',
                               dict(reason='the endpoint did not answer in time', detail=str(e)[:200],
                                    unpriced=bound), gid)
                if bound:
                    _charge_estimate(rt, gid, bound)
            else:
                rt.store.event('reflection_failed', dict(reason='%s: %s' % (type(e).__name__, str(e)[:300])), gid)
        except Exception:
            pass                  # the reflection is a proposal, never a reason to lose a finished goal
        return None


def _bounded(rt, fn, seconds=None):
    """Run fn() with a whole-call bound, or as soon as the daemon is asked to stop.

    The socket timeout passed to complete() bounds each read, not the call: a stream that trickles one byte just
    under the timeout holds the caller indefinitely, and a call already in flight when SIGTERM arrives still adds
    its read to the shutdown wait. The call runs on a worker thread and this one waits at most `seconds` (TIMEOUT
    unless a test says otherwise), checking the stop flag on the way, then closes the call's own sockets with
    Model.abort(thread=worker) so the blocked read ends now. Only that thread's sockets: a side call that missed its
    deadline must not cut the turn that shares this Model (#383 review P2). A stop request still closes everything
    (abort() with no thread): the daemon is going away and no server should keep generating for it.

    Accounting contract for the abandoned path (#383 review P2): the caller's own metering (`rt._meter`,
    `rt.brains.account`) runs on THIS thread after this returns, so a call abandoned here is never priced by the
    caller, and pricing `rt.model.last_usage` here would charge the *previous* call. What the caller does instead is
    debit the bound `_unpriced` computes, at abandonment, on this same thread: the ledgers keep one writer, and a
    total that would otherwise read low says what part of it is a bound (see `_charge_estimate`).
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
                    if stop:
                        abort()         # (the daemon is going away: nothing should keep a server busy)
                    else:
                        try:
                            abort(thread=worker.ident)   # (only this call's sockets: #383 review P2)
                        except TypeError:
                            abort()     # (a model whose abort takes no argument: close everything, as before)
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


def _charge_estimate(rt, gid, usd):
    """Debit an abandoned call's upper bound to the two ledgers that would have priced it.

    The call was abandoned by `_bounded` and the provider may still bill it, but no metering site pairs with it, so
    both ledgers would read low: spend.json (what the status line shows) and, when the fallback served it, the
    fallback's allowance (brains.account), which gates the next turn. The bound, never a measurement: spend.json
    counts it apart (`est_usd`) so a reader can tell a bound from a priced turn. Runs on the loop thread, where both
    writers live, so this adds no second writer. Never raises: this is a correction to a report, and losing a
    finished goal to it would be far worse than a ledger reading low."""
    for owner, name in ((rt, '_meter_estimate'), (getattr(rt, 'brains', None), 'account_estimate')):
        fn = getattr(owner, name, None)
        if not callable(fn):
            continue                    # (a stubbed runtime in a test, or an older brain object)
        try:
            fn(gid, usd)
        except Exception:
            pass



def _unpriced(rt, msgs):
    """What an abandoned call may have cost, as an upper bound: never a charge.

    A call `_bounded` abandoned is still running on its daemon thread, and its own last_usage lands only when it
    finally returns. Pricing `last_usage` here would charge the *previous* call, or the same call twice when the
    thread lands, which is why the caller does not price it (test_an_abandoned_reflection_call_is_not_priced_by_the
    _caller). What it can do is say how much is missing from the ledger: the input as sent, counted the way the
    model's own fallback counter counts it (UTF-8 bytes plus framing), and the whole output cap, priced at the
    model's rates. An unpriced model says None rather than a guessed dollar figure (#383 review P2)."""
    if not msgs:
        return None
    try:
        from .model import price
        model = getattr(rt, 'model', None)
        config = getattr(model, 'config', None) or {}
        usage = dict(prompt_tokens=len(json.dumps(msgs, ensure_ascii=False).encode()) + 256 * len(msgs),
                     completion_tokens=MAX_TOKENS)
        return price(config.get('model'), usage, (getattr(model, 'last_call', None) or {}).get('served'))
    except Exception:
        return None              # an upper bound is a courtesy in the event, never a reason to lose the skip


def _ask(rt, gid, msgs):
    """One short call, priced into the goal's spend like any other turn, on its own timeout: the goal is already
    complete, so a hung endpoint must not hold the loop for a turn's 900 s. It runs on the loop thread (one call per
    completed goal, and the loop is idle between goals), so the bound that matters is TIMEOUT per attempt, whole
    (_bounded): two at most, when the first reply could not be read, and never one past a stop request.

    The same thread is why the spend reads around this call are sound: `spent_today` is written by `account()` here
    and re-read by the gate in `_spend_refusal`, with no other writer in between. Move this call to a worker thread
    (the fix for the hung-endpoint bound above) and those reads must move with it, re-reading `spent_today` and
    `probe` there rather than trusting a value read on the loop thread."""
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
    """The seq this episode began after: the newest `episode_resumed`, or 0 for a goal that was never resumed.

    The marker is written where a goal starts again (cli.resume_goal, and `send --continue` on a finished goal),
    not at the stop. A list of stop kinds cannot stay complete: the watchdog, the plain-text guard, the emergency
    stop, either oneshot budget (those two wrote no event at all), a model truncation and a refused model each
    stop a goal, and `delegate_row_missing` added another, so a goal stopped by a kind the list did not name kept
    its stopped episode's judge and tool calls in the next reflection (#383 review P2). A new stop kind cannot
    break the boundary now. A goal resumed before this version has no marker and reads its whole history once."""
    row = rt.store.db.execute("SELECT seq FROM events WHERE goal=? AND kind='episode_resumed' "
                              'ORDER BY seq DESC LIMIT 1', (gid,)).fetchone()
    return row['seq'] if row else 0


def _spend_refusal(rt, gid):
    """Why the brain in force may not take this call, or None when it may: the primary is the model the user chose,
    so it always may; the fallback is gated by the runtime's own rule, not a copy of it (brain.Brains._allowed).

    The reason is returned rather than a bool so a skip event says WHAT refused: a spent allowance and a gate that
    raised are different facts about the harness, and the reviewer reading the queue could not tell them apart while
    both wrote "the fallback's allowance is spent" (#383 review P2: a defect in _allowed read as a spent budget).
    """
    brains = getattr(rt, 'brains', None)
    try:
        if brains is None or not brains.cfg or getattr(rt, 'model', None) is not brains.fallback:
            return None
        # Reuse the runtime's gate rather than restating it. A copy drifted once already: this guard checked only
        # spent_today() < budget_usd, so an unpriced remote fallback (brain._allowed refuses it: "cannot be
        # limited") still served a billed reflection past the cap the runtime enforces on the goal's own turns
        # (review P1). _allowed is the one rule now: unpriced remote is refused, a spent allowance is refused,
        # and a free-time goal follows _free_choice exactly as a work turn does.
        if brains._allowed(gid):
            return None
        return "the fallback's allowance is spent"
    except Exception as e:
        # fail closed when the fallback is in force and the gate cannot be evaluated at all: skipping a
        # reflection costs a proposal, a call past an unreadable cap costs money (a stub runtime never reaches
        # here - `not brains.cfg` above is the stub case). The type name travels with the skip: an exception
        # from _allowed itself is a defect, not a budget, and the queue should say which it was.
        return "the fallback's gate raised %s: %s" % (type(e).__name__, e)


def _may_spend(rt, gid):
    """Whether the brain in force may take this call: `_spend_refusal` as the bool its callers had before the
    reason was kept, so a caller that only branches does not have to unpick a sentence."""
    return _spend_refusal(rt, gid) is None


def _judge(rt, gid, since=0):
    """The newest `judged` event this episode, as (verdict, reason): the query is here once, and `brief` reads the
    reason from the same row it took the verdict from (review P2: it ran the identical SELECT a second time).

    The verdict is bounded to the values a `judged` event may carry. The row is ours, but the reflection pastes it
    into a prompt, and `judge.verdict` normalizes a model's reply while a raw event is read verbatim: a foreign or
    hand-edited value would reach the prompt as a verdict no judge can give (#383 review P2)."""
    from .judge import STORED
    row = rt.store.db.execute(
        "SELECT data FROM events WHERE goal=? AND kind='judged' AND seq>? ORDER BY seq DESC LIMIT 1",
        (gid, since)).fetchone()
    try:
        d = json.loads(row['data']) if row else None
    except (ValueError, KeyError, TypeError):
        return None, ''
    if not isinstance(d, dict):
        return None, ''
    said = d.get('verdict')
    return (said if said in STORED else None), str(d.get('reason') or '')[:600]


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
    # Whole-goal on purpose, and said so: `steps` and `approach_changes` are goal-row counters a resume does not
    # reset, and `store.steers` is keyed by goal. Every other block in this brief is bounded by _episode_start, so
    # without the words a reopened goal's "4 steer(s)" read as this episode's when it was two here and two before
    # (#383 review P2). A per-episode count of all three would mean a new high-water field on the resume marker.
    parts.append('ACROSS ITS EPISODES THE GOAL TOOK %s MODEL TURNS, the agent changed approach %s times, and '
                 '%d steer(s) arrived.' % (
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
        # The runtime's one failure classifier, so a command that exited non-zero (a `terminal`/`sandbox` result
        # carries exit_code and output, not 'error') and a tool that only says {'ok': False} both reach the
        # reflection, exactly as they reach the model's repeat-failure watchdog (#383 review P1).
        why = rt._failure_text(res)
        if why:
            failed += 1
            errors.append('%s: %s' % (name, why[:200]))
    # The summary is newest-first over at most 200 calls and keyed by name, so a churning goal's long tail of
    # one-off tool names crowds the failure list out of the prompt's practical window. Past ~40 distinct names the
    # singletons go and the count of them is said: a name seen once carries little, the failures carry the lesson
    # (#383 review P2).
    shown, dropped = names, 0
    if len(names) > NAMES_MAX:
        shown = {k: v for k, v in names.items() if v > 1}
        dropped = len(names) - len(shown)
    summary = ', '.join('%s×%d' % (k, v) for k, v in shown.items()) or '(none)'
    if dropped:
        summary += ' (and %d name%s seen once)' % (dropped, '' if dropped == 1 else 's')
    parts.append('ITS TOOL CALLS (newest first, at most 200): ' + summary)
    if failed:
        parts.append('%d of those calls failed; the newest:\n' % failed + '\n'.join(errors[:5]))
    return '\n\n'.join(parts)
