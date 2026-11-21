"""/delegate (issue #342, MVP): a child goal run through this harness's own goal loop.

The parent states a task and what done looks like; the child is an ordinary goal of the same store - its own
conversation, todo list, done check, gate, verify and spend row - with four limits that make it safe to run
from inside another goal:

- read-only: write, edit, checkpoint, rollback, ask_user, ask_root, github_handoff, send_message, report,
  remember, sim, schedule, lease, browser, propose_plan, a nested delegate and the advisor are refused, and it
  has no terminal and no gate on this machine: a filter cannot make a shell read-only (four review rounds each
  found another spelling that made an allow-listed host command run the child's own file), so its commands run
  only in the sandbox, where its folder is a read-only mount at /work and /box changes nothing outside itself;
- depth 1: a child cannot delegate, so the tree is one level deep by construction;
- caps: a step limit, a time limit and a dollar cap. The cap is config `delegate_usd` (0.50 by default) and a
  call's `cap_usd` may lower it, never raise it; reaching it pauses the child and the parent gets a partial
  report. Only a missing / null cap is no cap: a zero or negative one is refused, because a stored zero used to
  read as no cap at all. Spend is usd + the OpenCode plan's allowance, because inside a Go allowance a turn costs
  $0 of money and real allowance;
- a short structured report back: outcome (completed / capped / failed / incomplete), summary, what it
  verified, what it skipped, steps, spend, allowance and its cap. `capped` covers all three limits: the dollar
  cap, the step limit and the time limit each end a child the same way, and each writes one `delegate_capped`
  event in the pause's own transaction — `capped()` for the dollar cap, the runtime for the step limit and the
  deadline — so the report reads that event and nothing else: never the error text (an unrelated pause whose
  message named a delegate read as capped), and never the clock (a goal the watchdog paused past its deadline
  read as the deadline ending it) (#382 review);

- the count is capped too: start() counts the children this parent still has **running** (`outcome=''`) and
  refuses past `delegate_max_children` (4 by default), so a parent cannot grow an unbounded tree and a finished
  child does free its slot, as the refusal says. A child's row is closed on every exit, a crash included, so a
  run that died cannot wedge its parent at the limit;
- the child's stop watches the parent: a cancelled, paused or failed parent ends the child's loop at its next
  check (the daemon's `_stop` alone did not, and a cancelled parent's child ran on - #382 review), and a child
  stopped that way reports `incomplete` rather than running on its own brain later;
- a finished child is not resumable: `cli.resume_goal` refuses a delegate child (and so does `continue` on one),
  because its brain registration and caps lived in the process that ran it; a resumed child would run on the
  parent's brain with no cap. Delegate again from the parent instead.

MVP scope: one child at a time - the parent waits for its report - and the child runs in this process through
Runtime.run(target=child, stop=parent_gone), so the loop's single-writer serialization keeps holding. Parallel
children, a worktree per writing child and a card per child are the full version (#342, later).
"""
import time

ROLES = ('worker', 'reviewer')
DEFAULTS = dict(cap_usd=0.50, max_steps=12, max_seconds=600, max_children=4)
REFUSED = ('write', 'edit', 'checkpoint', 'rollback', 'ask_user', 'ask_root', 'github_handoff', 'send_message',
           'report', 'remember', 'sim', 'schedule', 'lease', 'browser', 'propose_plan', 'delegate', 'advisor')
CHILD_PROMPT = '''[delegate child of goal {parent}] You are a read-only child goal, run by that goal's delegate tool.

TASK
{task}

WHAT DONE LOOKS LIKE (call define_done with this, then verify and finish)
{done}

Rules that differ from a top-level goal: you cannot write or edit a file, ask your human, message a peer, or
delegate further, and you have no terminal and no gate on this machine - a filter cannot make a shell read-only,
so your commands run only in the sandbox, where your folder is a read-only mount at /work and /box changes
nothing outside itself. Read, search, use the sandbox, the web and machine, then define_done (without a gate),
verify and finish with what you found. Name files by path and line and quote the few lines that matter. Your
parent reads only your structured report (summary, verified, skipped, spend), so the answer belongs in your
finish summary: say plainly what you checked and what you could not.'''


def _limit(rt, key):
    """A config limit: a missing or null key takes the default, as the dollar cap does; 0 or less is refused.

    `delegate_max_steps: null` raised TypeError at every delegate call (#382 review), so a null now takes the
    default. A configured 0 is *not* read as "the default" (the same silent-literal trap the delegates table's
    `DEFAULT 0` was): no child may take zero steps, and a limit nobody can see is not a limit."""
    value = rt.config.get(key)
    default = DEFAULTS[key.replace('delegate_', '')]
    if value is None:
        return default
    value = int(value)
    if value < 1:
        raise ValueError('%s is %s: a limit of 0 is not "no limit" and does not mean the default (%s). Set it to '
                         'at least 1, or remove the key' % (key, value, default))
    return value


def is_child(store, gid):
    """Whether this goal is a delegate child: its own `kind='delegate'` first, its delegates row second.

    The policy (read-only refusals, no terminal, the resume fence, no gate) is keyed on the goal because the goal
    carries it: `_enqueue(kind='delegate')` writes it in the same insert that creates the goal. The row is a fact
    about one run - its brain, cap, spend and report - and is a second marker, so the answer is their union and
    fails closed: a goal whose row was restored without (a hand edit, a partial backup) is still a child, and a
    leftover row makes a goal read-only until it is removed (a false "child" costs a read-only goal; a false
    "not a child" runs uncapped on the wrong brain). Only `start()` writes a row, and it writes it for a
    `kind='delegate'` goal, so the two agree except where a person edited the store. One indexed read, not a
    full goal."""
    r = store.db.execute('SELECT kind FROM goals WHERE id=?', (gid,)).fetchone()
    if r is not None and r['kind'] == 'delegate':
        return True
    return bool(store.delegate(gid))


def refused(name, args):
    """Why a delegate child may not run this call, or None. Checked by the runtime before the tool runs."""
    if name in REFUSED:
        return ('a delegate child is read-only: %s is not available in it. Report what you found; your parent '
                'makes the change.' % name)
    if name == 'terminal':
        # A child has no host terminal at all (#382 review, fourth round). Reading a command is not the same as
        # running it safely: the child's box is a directory in the workspace (boxes/<gid>), so anything it writes
        # there - a git hook, diff.textconv, a binutils @file naming --plugin - could be executed by an
        # allow-listed host command, and four rounds of filter fixes each found another such spelling. The sandbox
        # needs no classifier: its /work mount is read-only and /box dies with the box.
        return ('a delegate child is read-only: it has no terminal on this machine, because a filter cannot make '
                'a shell read-only. Use sandbox instead - its folder is a read-only mount at /work and /box '
                'changes nothing outside itself - or report what you could not check')
    if name == 'define_done' and args.get('gate'):
        # a gate runs in a shell at finish (and after a change turn), on the host: `define_done(gate='touch x && ...')`
        # created the file and still reported completed (#382 review). A child has no host shell at all now, so it
        # has no gate either; it verifies inside the sandbox and says in its finish summary what it checked.
        return ('a delegate child is read-only: it has no gate, because a gate runs in a shell on this machine. '
                'Check what you can in sandbox, then report what you verified there and what you could not')
    if name.startswith('mcp__'):
        return 'a delegate child cannot call an MCP tool: it can change things this harness does not see'
    return None


def _caps(rt, args):
    """(cap_usd, max_steps, max_seconds): the config's defaults, lowered by the call, never raised.

    A cap is a positive number; a missing or null config key takes the default. Zero or less is refused rather
    than read as "unlimited": `cap_usd=0` used to disable a child's cap altogether (#382 review), and a cap
    nobody can reach is not a cap."""
    cfg = rt.config
    cap = cfg.get('delegate_usd', DEFAULTS['cap_usd'])
    cap = DEFAULTS['cap_usd'] if cap is None else float(cap)
    if cap <= 0:
        raise ValueError('delegate_usd is %s: a cap of 0 does not mean unlimited, so no child can run under it. '
                         'Set a positive cap (the default is %s)' % (cfg.get('delegate_usd'), DEFAULTS['cap_usd']))
    if args.get('cap_usd') is not None:
        their = float(args['cap_usd'])
        if their <= 0:
            raise ValueError('cap_usd must be more than 0: it is what the child may spend, and 0 used to mean '
                             '"uncapped". Omit it to use delegate_usd ($%s)' % cap)
        cap = min(cap, their)
    steps = int(_limit(rt, 'delegate_max_steps'))
    if args.get('max_steps') is not None:
        their = int(args['max_steps'])
        if their <= 0:
            raise ValueError('max_steps must be at least 1: a child that may take no step cannot do anything')
        steps = min(steps, their)
    secs = int(_limit(rt, 'delegate_max_seconds'))
    if args.get('max_seconds') is not None:
        their = int(args['max_seconds'])
        if their <= 0:
            raise ValueError('max_seconds must be at least 1: a child with no time cannot do anything')
        secs = min(secs, their)
    return cap, max(1, steps), max(1, secs)


def _brain(rt, args):
    """(Model, {model, provider}) the child runs: a call's model/provider from the catalog, the role's
    configured delegate.<role>_model, or the parent's own brain (the default)."""
    from .model import Model, remember_prices
    from . import catalog
    role = args.get('role') or 'worker'
    name = str(args.get('model') or '').strip()
    provider = str(args.get('provider') or '').strip() or None
    if not name:
        name = str((rt.config.get('delegate') or {}).get(role + '_model') or '').strip()
    if not name:
        return Model(dict(rt.config)), dict(model=rt.config.get('model'),
                                            provider=rt.config.get('provider') or rt.config.get('provider_kind'))
    cat = catalog.load(rt.home)
    entry = catalog.find(cat, name, provider)
    if entry is None:
        entry = catalog.find(catalog.refresh(rt.home, rt.config), name, provider)
    if entry is None:
        raise ValueError(f'no model {name} in the catalog: `models --refresh` lists what the providers serve')
    spec = catalog.providers(rt.config).get(entry['provider'])
    if not spec:
        raise ValueError('the provider of %s is no longer configured' % entry['id'])
    endpoint, api = catalog.transport(spec, entry)
    conf = dict(rt.config)
    conf.update(model=entry['id'], provider=entry['provider'], provider_kind=spec['kind'], endpoint=endpoint, api=api)
    for k in ('relay', 'prices', 'context', 'context_cap', 'thinking', 'model_variants', 'efforts', 'vision'):
        conf.pop(k, None)                 # (the parent's brain's facts, not this one's)
    if spec.get('relay'):
        conf['relay'] = True
    if entry.get('prices'):
        conf['prices'] = entry['prices']
        remember_prices(entry['id'], entry['prices'])
    if entry.get('context'):
        conf['context'] = int(entry['context'])
    if entry.get('vision'):
        conf['vision'] = True
    if entry.get('variants'):
        conf['model_variants'] = entry['variants']
    return Model(conf), dict(model=entry['id'], provider=entry['provider'])


def prepare(rt, args):
    """What start() needs that may wait or reach the network, resolved before the runtime opens its write
    transaction: the caps, the role and the child's brain (a model from the catalog may need a refresh, which is
    a network call, and the store has one writer: #382 review)."""
    task = str(args.get('task') or '').strip()
    done = str(args.get('done_check') or '').strip()
    if not task:
        raise ValueError('delegate needs a task: what the child should find out or check')
    if not done:
        raise ValueError('delegate needs done_check: what done looks like for the child, as something it can '
                         'check (it becomes the child\'s define_done)')
    role = args.get('role') or 'worker'
    if role not in ROLES:
        raise ValueError('role must be one of ' + ', '.join(ROLES))
    cap, steps, secs = _caps(rt, args)
    model, who = _brain(rt, args)
    return dict(task=task, done=done, role=role, cap_usd=cap, max_steps=steps, max_seconds=secs,
                model=model, who=who)


def start(rt, parent, args, prepared=None):
    """Create the child goal and register its brain; the caller runs it after its transaction commits.
    (Creating the goal is part of the parent's action, atomically; running it is not - the child needs turns of
    its own, and a nested loop inside the write transaction would deadlock on BEGIN.) `prepared` is prepare()'s
    result, which the runtime resolves before BEGIN (#382 review)."""
    p = prepared or prepare(rt, args)
    limit = int(_limit(rt, 'delegate_max_children'))
    used = rt.store.db.execute("SELECT count(*) FROM delegates WHERE parent=? AND outcome=''",
                               (parent,)).fetchone()[0]
    if used >= limit:
        raise ValueError('this goal has %d delegate children still running (delegate_max_children): their '
                         'reports arrive first; do the rest yourself, or delegate again once one finishes' % limit)
    child = rt.store._enqueue(CHILD_PROMPT.format(parent=parent[:8], task=p['task'], done=p['done']),
                              max_steps=p['max_steps'], max_seconds=p['max_seconds'], kind='delegate')
    who = p['who']
    rt.store.delegate_add(child, parent, who.get('model') or '', who.get('provider') or '', p['role'], p['cap_usd'])
    rt.delegate_models[child] = p['model']
    rt.delegate_caps[child] = p['cap_usd']
    rt.store.event('delegate_started', dict(child=child, model=who.get('model'), provider=who.get('provider'),
                                           role=p['role'], cap_usd=p['cap_usd'], max_steps=p['max_steps'],
                                           max_seconds=p['max_seconds']), parent)
    return dict(child=child, model=who.get('model'), provider=who.get('provider'), role=p['role'],
                cap_usd=p['cap_usd'], max_steps=p['max_steps'], max_seconds=p['max_seconds'], outcome='running',
                _delegate_run=True, note='the child runs now; this result is replaced by its report')


def _state(rt, gid):
    """A goal's state by a direct read, '' when its row is gone.

    Not store.goal(): that raises on an unknown id, and this runs inside the child's run and its stop check, where
    a hand-restored store can be missing the parent row (the sweep() path was hardened the same way, #382 review).
    A parent with no row is no longer a parent that can cancel its child."""
    r = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (gid,)).fetchone()
    return (r['state'] if r else '') or ''


def _capped(rt, child):
    """True when one of the child's own three caps ended its run: the `delegate_capped` event, and nothing else.

    One fact for all three caps, written in the transaction that pauses the goal: `capped()` for the dollar cap,
    the runtime for the step limit and the deadline. The report used to guess from the error text (any pause
    whose message carried 'delegate' or 'budget' read as capped), then from the clock (`time.time() >= deadline`
    read as the deadline, so a watchdog pause past a passed deadline read as a spent limit). Neither says *why*
    the goal paused; the event does (#382 review)."""
    return bool(rt.store.db.execute("SELECT 1 FROM events WHERE goal=? AND kind='delegate_capped' LIMIT 1",
                                    (child,)).fetchone())


def run_child(rt, parent, job):
    """Run the child through the loop and turn its goal into the short report the parent reads.

    The child's stop also watches the parent: cancelling or pausing the parent ends the child's loop at its next
    check (#382 review: _stop alone is only the daemon's stop, so a cancelled parent's child ran on)."""
    child = job['child']
    daemon_stop = getattr(rt, '_stop', None)

    def parent_gone():
        """True when the daemon is stopping, or the parent was cancelled, paused or failed while the child ran."""
        if daemon_stop and daemon_stop():
            return True
        return _state(rt, parent) in ('cancelled', 'paused', 'error')

    try:
        rt.run(target=child, stop=parent_gone)
        g = rt.store.goal(child)
        state = g['state']
        if _state(rt, parent) == 'cancelled' and state in ('working', 'queued', 'waiting_model'):
            with rt.store.transaction():
                rt.store.update(child, state='cancelled', error='its parent was cancelled while it ran')
            g, state = rt.store.goal(child), 'cancelled'
        if state == 'completed':
            outcome = 'completed'
        elif state == 'paused' and _capped(rt, child):
            outcome = 'capped'       # the dollar cap, the step limit or the deadline ended it
        elif state == 'cancelled':
            outcome = 'incomplete'   # its parent was cancelled: it did not finish its own work
        elif state in ('working', 'queued', 'waiting_model'):
            # it must not be left runnable: the next turn would pick the parent's brain (its model registration
            # is this process's), and no other runtime can serve it either
            with rt.store.transaction():
                rt.store.update(child, state='paused', error='delegate child did not finish: its model was '
                                                            'unreachable or it ran out of its budget')
            g, state, outcome = rt.store.goal(child), 'paused', 'incomplete'
        else:
            outcome = 'failed'
        row = rt.store.delegate(child) or {}
        report = dict(child=child, role=job.get('role'), model=row.get('model') or job.get('model'),
                      provider=row.get('provider') or job.get('provider'), outcome=outcome, state=state,
                      summary=(g.get('summary') or g.get('error') or '')[:2000],
                      done_check=(g.get('done_check') or '')[:1000],
                      verified=state == 'completed' and rt.store.verified_since_steer(child),
                      skipped=[it['text'] for it in rt.store.todo(child)['items'] if it['state'] != 'done'][:20],
                      steps=int(g.get('steps') or 0), spend_usd=round(row.get('spend_usd') or 0.0, 6),
                      allowance_usd=round(row.get('allowance_usd') or 0.0, 6), cap_usd=job.get('cap_usd') or 0.0)
        rt.store.delegate_done(child, outcome, report)
        rt.store.event('delegate_done', {k: v for k, v in report.items() if k not in ('summary', 'skipped')}, parent)
        if outcome != 'completed':
            report['note'] = ('the child did not complete: %s. Use what it found, or do the rest yourself; a new '
                              'delegate gets a fresh child.' % (report['summary'][:300] or state))
        return report
    finally:
        rt._stop = daemon_stop
        rt.delegate_models.pop(child, None)
        rt.delegate_caps.pop(child, None)
        # A run that died before it reported (an exception out of rt.run, a kill) must not leave a row `outcome=''`
        # forever: the children limit counts running rows now, so an unclosed one would wedge the parent at the
        # limit. The normal path closed the row already, so this is a no-op for it.
        def close():
            """The fall-back close, from the top, so a transient store failure can be tried again."""
            row = rt.store.delegate(child)
            if not row or row.get('outcome'):
                return                        # the normal path closed the row already
            # a direct read, not store.goal(): that raises on an unknown id, and the write closing the row must
            # not be skipped because the goal row went missing
            r = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (child,)).fetchone()
            state = (r['state'] if r else '') or ''
            if state in ('queued', 'working', 'waiting_model'):
                # and the goal must not stay runnable: closing the row here takes it out of sweep()'s reach
                # (sweep() looks for `outcome=''`), and a runnable child left behind would be picked up by
                # the next run on whatever brain that process has, with no cap ledger (#382 review, this
                # push). sweep() pauses exactly this for the process-died case; this is the in-process one.
                with rt.store.transaction():
                    rt.store.update(child, state='paused', error='its run ended without a report '
                                                                '(a delegate child run by another goal)')
                state = 'paused'
            # the report's state is the goal's, so it says `paused` where the goal was just paused
            rt.store.delegate_done(child, 'failed', dict(child=child, outcome='failed', state=state or 'error',
                                                         summary='the run of this child ended without a report'))
            rt.store.event('delegate_done', dict(child=child, outcome='failed'), parent)

        # A failed close must never mask the error that ended the run, but it must not vanish either: a row left
        # `outcome=''` keeps its child slot, and the runtime's sweep() runs only at a start, so a long-lived
        # parent would be refused every further delegate until a restart with nothing said (#382 review). One
        # retry (the usual cause is a transient lock), then a line on the parent's events.
        for attempt in (1, 2):
            try:
                close()
                break
            except Exception as e:
                if attempt == 2:
                    try:
                        rt.store.event('delegate_close_failed', dict(child=child, error=str(e)[:300]), parent)
                    except Exception:
                        pass              # (and the trace itself must not mask the error that ended the run)


def capped(rt, gid):
    """Pause a child whose spend (money + plan allowance) reached its cap; True when it was paused."""
    row = rt.store.delegate(gid)
    if not row:
        return False
    cap = row.get('cap_usd')
    if cap is None:
        return False                      # only a missing cap is no cap: a stored 0 is a $0 cap, and it pauses
    cap = float(cap)
    spend, allowance = float(row.get('spend_usd') or 0.0), float(row.get('allowance_usd') or 0.0)
    if spend + allowance < cap:
        return False
    # one transaction: a crash between them would leave a paused child with no delegate_capped line, and the two
    # are one fact - the cap was reached - so they commit together (#382 review).
    with rt.store.transaction():
        rt.store.update(gid, state='paused', error='delegate spend cap reached: $%.4f of $%.4f' % (spend + allowance, cap))
        rt.store.event('delegate_capped', dict(spend_usd=spend, allowance_usd=allowance, cap_usd=cap), gid)
    return True


def open_child(rt, gid):
    """True when gid is a delegate child whose row is still open (it has not reported yet).

    `rt.delegate_caps` is this process's memory of the children it started, not the fact: a runtime that did not
    run start() for the child (a restart, another process's row) has an empty set while the row is still open.
    Keyed on the set alone, the deadline and step-limit pauses wrote no `delegate_capped`, and the report read
    `failed` where the child was capped (#382 review). The row is the fact; the set is only a fast path."""
    row = rt.store.delegate(gid)
    return bool(row) and not row.get('outcome')


def note_spend(rt, gid, usd, allowance):
    """A child's turn: the per-subagent ledger (delegates.spend_usd / allowance_usd), the metric the cap reads."""
    if gid not in (getattr(rt, 'delegate_caps', None) or ()):
        return
    rt.store.delegate_spend(gid, usd or 0.0, allowance or 0.0)


def _sweep_close(rt, r, outcome, state, summary):
    """Close a child's row at sweep in the same shape run_child reports.

    The close is the last thing written about this child, and a report without its spend reads as a free run
    (#382 review), so the row's model, provider, spend and cap always ride along."""
    rt.store.delegate_done(r['child'], outcome, dict(
        child=r['child'], role=r['role'], model=r['model'], provider=r['provider'], outcome=outcome, state=state,
        spend_usd=round(r['spend_usd'] or 0.0, 6), allowance_usd=round(r['allowance_usd'] or 0.0, 6),
        cap_usd=r['cap_usd'] or 0.0, summary=summary))


def sweep(rt):
    """At a runtime start, a child left runnable belongs to a run that ended (the store has one writer, and the
    runtime lock proved no other process holds it): pause it so it cannot run later on the wrong brain.

    A child left open whose goal has already ended is closed too: a process killed between the child's final
    state write and its report leaves that row open forever, and start()'s child limit counts it against the
    parent on every later start (#382 review)."""
    for r in rt.store.db.execute("SELECT child,role,model,provider,cap_usd,spend_usd,allowance_usd,started "
                                 "FROM delegates WHERE outcome=''").fetchall():
        # a direct read, not store.goal(): that raises on an unknown id, and this runs bare in the runtime loop,
        # so one row whose goal is gone (a partial restore) would end the daemon on every pass and leave the row
        # open for good (#382 review). A row without its goal is closed as orphaned - it can never report.
        row = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (r['child'],)).fetchone()
        if row is None:
            _sweep_close(rt, r, 'orphaned', 'error', 'this child\'s goal row is gone')
            continue
        if row['state'] in ('queued', 'working', 'waiting_model'):
            with rt.store.transaction():
                rt.store.update(r['child'], state='paused', error='its runtime ended mid-delegation')
            _sweep_close(rt, r, 'orphaned', 'paused', 'its runtime ended mid-delegation')
            continue
        # The goal already ended (completed / cancelled / error): the crash window run_child's fall-back close
        # cannot reach, because that process is gone. The goal's own state names the outcome.
        outcome = dict(completed='completed', cancelled='incomplete').get(row['state'], 'failed')
        _sweep_close(rt, r, outcome, row['state'], 'this child\'s goal ended without a report')
