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
  report. The cap is **soft**: it is checked before a turn, and one turn can overshoot it by whatever that turn's
  output costs against what was left, because the price of a reply is only known once it arrives. Only a missing /
  null cap is no cap: a zero or negative one is refused, because a stored zero used to read as no cap at all. Spend
  is usd + the OpenCode plan's allowance, because inside a Go allowance a turn costs $0 of money and real allowance;
- a budget for all of a parent's children together: config `delegate_total_usd` (2.00 by default) caps the sum of
  every child row's spend and allowance - closed rows included - so a parent that delegates again and again cannot
  spend without bound one small child at a time (#382 review);
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
import json
import math
import time

_UNSET = object()            # (note_aux's "no snapshot given", distinct from a snapshot of None)
ROLES = ('worker', 'reviewer')
DEFAULTS = dict(cap_usd=0.50, max_steps=12, max_seconds=600, max_children=4, total_usd=2.00)
# the config key -> its default's name (the key says what it caps, the default says it in DEFAULTS' own words)
_MONEY = {'delegate_usd': 'cap_usd', 'delegate_total_usd': 'total_usd'}
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
    # `delegate_max_steps: "12"` (a JSON-quoted number) reached `math.isfinite` and raised `TypeError: must be real
    # number` from every delegate call, and `"abc"` a bare ValueError with no hint (#382 review, rev-glm P2): one
    # typo in the owner's config bricked the tool with a traceback instead of a refusal naming the key. A bool is an
    # int here, so `delegate_max_steps: true` would silently mean one step; it is refused like any other non-number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s is %r: a limit must be a number, not %s. Set it above 0, or remove the key (the '
                         'default is %s)' % (key, value, type(value).__name__, default))
    if not math.isfinite(value):
        raise ValueError('%s is %s: a limit must be a finite number, not something no child can reach (%s by '
                         'default). Set it above 0, or remove the key' % (key, value, default))
    value = int(value)
    if value < 1:
        raise ValueError('%s is %s: a limit of 0 is not "no limit" and does not mean the default (%s). Set it to '
                         'at least 1, or remove the key' % (key, value, default))
    return value


def switched_on(rt):
    """Whether the delegate tool is offered at all: config `delegate` is not `false`.

    The key is the off switch *and* the role-model dict, and it used to be read as neither where it matters:
    `tools()` listed the tool under `false` (a spending tool the prompt says is off), and `true` reached
    `True.get(...)` in _brain - an AttributeError `_action` does not catch, so `tick()` marked the *parent* goal
    `error` (#382 review, Opus 5.5 P1-2)."""
    return rt.config.get('delegate', True) is not False


def role_models(rt):
    """The role models: the `delegate` key when it is a dict, {} when the key is absent or null.

    Anything else is a typo - `true`, a string, a list - and is refused here naming the key, rather than
    crashing later with an AttributeError from a call on a bool (#382 review, Opus 5.5 P1-2)."""
    cfg = rt.config.get('delegate')
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError('delegate is %r: it is either false (no delegate tool at all) or an object naming the '
                         'role models, e.g. {"worker_model": "..."}. Set it to one of those, or remove the key'
                         % (cfg,))
    return cfg


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


def _call_number(args, key):
    """A cap a delegate call passed, or a refusal naming the key.

    The tool schema says number, but a model can send `"0.9x"` or `true` as plain text: `float(...)` and `int(...)`
    then raised a bare ValueError/TypeError naming nothing from every delegate call, where the config side already
    refuses with the key and the fix (#382 review, rev-glm P2). A bool is an int in Python, so `max_steps: true`
    would silently mean one step; it is refused like any other non-number."""
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s is %r: it must be a number, not %s (send a JSON number, or omit it)'
                         % (key, value, type(value).__name__))
    if not math.isfinite(value):
        raise ValueError('%s is %s: it must be a finite number (send a number, or omit it)' % (key, value))
    return value


def _money(rt, key):
    """A config dollar cap: a missing or null key takes the default, and zero, negative or non-finite is refused.

    `delegate_usd` and `delegate_total_usd` are read the same way, so a typo cannot be a refusal on one and a
    silently uncapped budget on the other. A cap of 0 used to disable a child's cap altogether (#382 review), and
    a non-finite one (`spend < inf`) can never trip: neither is a cap, so both are refused naming the key."""
    default = DEFAULTS[_MONEY[key]]
    cap = rt.config.get(key)
    # the same typo as `_limit`'s ("0.5" quoted, or "abc"): `float` raised a bare ValueError or a TypeError from
    # every delegate call instead of a refusal naming the key (#382 review, rev-glm P2)
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))):
        raise ValueError('%s is %r: a cap must be a number of dollars, not %s. Set a positive cap (the default is '
                         '%s)' % (key, cap, type(cap).__name__, default))
    cap = float(default) if cap is None else float(cap)
    # (a NaN passed `cap <= 0` as False and `min(cap, nan)` carried it on, and a stored NaN is an IntegrityError the
    # owner reads as a crash; an infinite cap is stored and `spend < inf` never trips: a non-finite cap is no cap -
    # #382 review, rev-glm P2)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('%s is %s: a cap of 0 does not mean unlimited and a cap must be a finite number, so no '
                         'child can run under it. Set a positive cap (the default is %s)'
                         % (key, rt.config.get(key), default))
    return cap


def _caps(rt, args):
    """(cap_usd, max_steps, max_seconds): the config's defaults, lowered by the call, never raised."""
    cap = _money(rt, 'delegate_usd')
    if args.get('cap_usd') is not None:
        their = _call_number(args, 'cap_usd')
        if their <= 0:
            raise ValueError('cap_usd must be more than 0 and finite: it is what the child may spend, '
                             'and 0 used to mean "uncapped". Omit it to use delegate_usd ($%s)' % cap)
        cap = min(cap, their)
    steps = int(_limit(rt, 'delegate_max_steps'))
    if args.get('max_steps') is not None:
        their = _call_number(args, 'max_steps')
        if their <= 0:
            raise ValueError('max_steps must be at least 1: a child that may take no step cannot do anything')
        steps = min(steps, int(their))
    secs = int(_limit(rt, 'delegate_max_seconds'))
    if args.get('max_seconds') is not None:
        their = _call_number(args, 'max_seconds')
        if their <= 0:
            raise ValueError('max_seconds must be at least 1: a child with no time cannot do anything')
        secs = min(secs, int(their))
    return cap, max(1, steps), max(1, secs)


def _brain(rt, args):
    """(Model, {model, provider}) the child runs: a call's model/provider from the catalog, the role's
    configured delegate.<role>_model, or the parent's own brain (the default)."""
    from .model import Model, price
    from . import catalog
    role = args.get('role') or 'worker'
    name = str(args.get('model') or '').strip()
    provider = str(args.get('provider') or '').strip() or None
    if not name:
        name = str(role_models(rt).get(role + '_model') or '').strip()
    if not name:
        # the parent's own brain. The same priceability refusal a named catalog model gets (below), because a
        # cap that reads a price of None can never trip: the child would be bounded by its steps and deadline
        # only, its dollar cap a number nothing compares to (#382 review, rev-glm P2). Its provider facts are
        # the runtime's own config, so no catalog entry is needed to know whether it bills.
        conf = dict(rt.config)
        brain = Model(conf)
        spec = catalog.providers(conf).get(conf.get('provider')) or {}
        kind = spec.get('kind') or conf.get('provider_kind')
        if brain.remote and kind not in catalog.OPENCODE and price(conf.get('model'), {}) is None:
            raise ValueError('no model %s has a price: the parent\'s own paid brain must be priceable, or a '
                             'child on it could never trip its dollar cap (set `prices` in config.json, pick a '
                             'catalog model or a priced one by name)' % (conf.get('model') or '(unnamed)'))
        return brain, dict(model=conf.get('model'),
                           provider=conf.get('provider') or conf.get('provider_kind'))
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
    # the parent's key variable is not the child's key: on a direct (non-relay) provider it would send the
    # parent's key to another provider's endpoint, and the child, not the owner, picked that provider. A provider
    # that names its own variable in config `providers` keeps it (#382 review, Opus 5.5 P2).
    conf.pop('api_key_env', None)
    if spec.get('api_key_env'):
        conf['api_key_env'] = spec['api_key_env']
    # the entry's own facts, by the same normalization /model uses (an OpenCode plan charges what its reply says, so
    # its price tables are not needed; any other paid brain must be priceable or its cap can never be read)
    catalog.apply_entry(conf, entry, spec)
    brain = Model(conf)
    if (brain.remote and spec['kind'] not in catalog.OPENCODE and price(entry['id'], {}) is None):
        raise ValueError('no model %s has a price: a paid remote child must be priceable, or its dollar cap can '
                         'never trip (the catalog has no prices for it)' % entry['id'])
    return brain, dict(model=entry['id'], provider=entry['provider'])


def prepare(rt, args, parent=None):
    """What start() needs that may wait or reach the network, resolved before the runtime opens its write
    transaction: the caps, the role and the child's brain (a model from the catalog may need a refresh, which is
    a network call, and the store has one writer: #382 review).

    A free-time goal cannot delegate: its child is kind='delegate', so FreeTime.is_free() is false for it and the
    delegated spend never reaches the free-time ledger, while the child's registered brain bypasses
    Brains.choose() and its plan restrictions (#382 review). Refusing here beats propagating a second policy."""
    from .freetime import KINDS as FREE_KINDS
    if not switched_on(rt):
        # the off switch, checked where the call lands as well as in tools(): a model replaying an old action, or a
        # goal whose tool list was built before the owner flipped the key, must not spend through it (#382 review).
        raise ValueError('delegate is off in config (delegate: false): this goal cannot start a child. Ask your '
                         'human to set delegate to a role-model object or remove the key')
    if parent:
        r = rt.store.db.execute('SELECT kind FROM goals WHERE id=?', (parent,)).fetchone()
        kind = (r['kind'] if r else '') or 'work'
        if kind in FREE_KINDS:
            raise ValueError('a %s goal cannot delegate: the child would spend outside the free-time allowance and '
                             'its provider rules (the child is not a %s goal, and its model is registered, not '
                             'chosen); do this work yourself' % (kind, kind))
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
    # A oneshot parent with 60 s left must not block for the child's 600: the child's deadline is the earlier of
    # its own and the parent's, so a parent that is itself out of time does not outlive its own limit through a
    # child (#382 review, Opus 5.5 P2).
    if parent:
        r = rt.store.db.execute('SELECT deadline FROM goals WHERE id=?', (parent,)).fetchone()
        left = (r['deadline'] if r else None)
        if left:
            secs = max(1, min(secs, int(left - time.time())))
    model, who = _brain(rt, args)
    return dict(task=task, done=done, role=role, cap_usd=cap, max_steps=steps, max_seconds=secs,
                model=model, who=who)


def total(rt, parent):
    """Every dollar this parent's children have spent or been allowed, open rows and closed ones alike.

    `start()`'s running count is checked at a moment when a serial parent has no open row at all - `run_child`
    closes the row before the call returns - so `delegate_max_children` never tripped and a parent could grow an
    unbounded series of children, each under its own cap (#382 review, Opus 5.5 P1-1). The sum over the rows is
    the bound that matches that shape: it is a fact in the store, so it survives a restart and a second runtime,
    and a child whose row is closed still counts what it cost."""
    return float(rt.store.db.execute(
        "SELECT coalesce(sum(coalesce(spend_usd,0)+coalesce(allowance_usd,0)),0) FROM delegates WHERE parent=?",
        (parent,)).fetchone()[0] or 0.0)


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
    budget = _money(rt, 'delegate_total_usd')
    spent = total(rt, parent)
    if spent >= budget:
        raise ValueError('this goal\'s delegate children have used $%.4f of delegate_total_usd $%.4f: another '
                         'child could spend past what this goal may delegate in total. Raise delegate_total_usd, '
                         'or do the rest yourself' % (spent, budget))
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
    # the child's brain was put in the runtime for its run (start() takes the registered model); the parent's next
    # turn must not run on it, and single-brain choose() returns rt.model as it stands (#382 review)
    parent_model = getattr(rt, 'model', None)

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
        elif state == 'paused':
            # a pause that is not one of the child's own three caps: the owner's budget watchdog, a manual pause,
            # another goal's question. The work did not fail, it was stopped, and the pause's own words are the
            # summary run_child already writes, so `incomplete` says that where `failed` blamed the child
            # (#382 review).
            outcome = 'incomplete'
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
        if parent_model is not None:
            rt.model = parent_model
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
        # A child that did not complete is not resumable (its brain and caps lived in this process), so its sandbox
        # is garbage: sandbox.sweep keeps a paused goal's box because a paused goal can resume, which a delegate
        # child never does (#382 review). `completed` keeps its box as an ordinary finished goal does (the runtime
        # ends it), so only the other outcomes tear it down here.
        try:
            row = rt.store.delegate(child) or {}
            if row.get('outcome') and row['outcome'] != 'completed':
                from . import sandbox
                sandbox.end(rt.home, child, rt.store)
        except Exception as e:
            try:
                rt.store.event('box_not_removed', dict(child=child, error='%s: %s' % (type(e).__name__, e)[:300]),
                               parent)
            except Exception:
                pass


def capped(rt, gid):
    """Pause a child whose spend (money + plan allowance) reached its cap; True when it was paused."""
    row = rt.store.delegate(gid)
    if not row:
        return False
    cap = row.get('cap_usd')
    if cap is None:
        return False                      # only a missing cap is no cap: a stored 0 is a $0 cap, and it pauses
    cap = float(cap)
    if not math.isfinite(cap):
        # a row from before the fix, or one written around delegate_add: `spend < inf` never trips, so an infinite
        # cap is no cap at all. Fail closed and let the pause carry the reason (#382 review, rev-glm P2).
        cap = 0.0
    spend, allowance = float(row.get('spend_usd') or 0.0), float(row.get('allowance_usd') or 0.0)
    why = 'delegate spend cap reached: $%.4f of $%.4f' % (spend + allowance, cap)
    # The ledger is written only by delegate_spend, which adds each turn: a negative term (a hand edit, a backup
    # restore) would make `spend + allowance < cap` skip the cap for good, and a non-finite one would decide the
    # compare by accident and make the pause's own reason read "$inf". Neither is a spend figure, so both fail
    # closed the way a non-finite cap does: the child pauses and the reason names the term (#382 review, P2).
    bad = ['%s %r' % (k, v) for k, v in (('spend_usd', spend), ('allowance_usd', allowance))
           if not math.isfinite(v) or v < 0]
    if bad:
        cap = spend = allowance = 0.0
        why = 'delegate spend cap reached: the stored ledger is not a spend (%s)' % ', '.join(bad)
    elif spend + allowance < cap:
        return False
    # one transaction: a crash between them would leave a paused child with no delegate_capped line, and the two
    # are one fact - the cap was reached - so they commit together (#382 review).
    with rt.store.transaction():
        rt.store.update(gid, state='paused', error=why)
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
    """A child's turn: the per-subagent ledger (delegates.spend_usd / allowance_usd), the metric the cap reads.

    Keyed on the row, not only on `rt.delegate_caps`: a child whose start() ran in a previous process (a crash and
    the same parent path re-run) is in no set, so its turns went uncharged and its dollar cap could not trip
    (#382 review, rev-glm P2). open_child is the same fast-path-or-fact test the cap sites already use."""
    if gid not in (getattr(rt, 'delegate_caps', None) or ()) and not open_child(rt, gid):
        return
    rt.store.delegate_spend(gid, usd or 0.0, allowance or 0.0)


def note_aux(rt, gid, model=None, since=_UNSET):
    """A child's auxiliary turn - a compaction summary, a graded finish, the judge's second opinion - to its ledger.

    The runtime meters a turn into spend.json after the main generation only, so these calls were priced nowhere:
    each overwrites the model's last usage before the next one reads it, and the child's dollar cap, the one limit
    that stops it, never saw them (#382 review). Charged here right after the call, from the same pricing, so the
    cap and the report count every turn the child's brain ran. A goal that is not a child has no ledger (note_spend
    says so), and its calls stay in spend.json's turn totals as they were.

    `since` is the usage object from before the call: the same object after it means the call raised before it
    reported anything, and the previous turn's usage must not be charged a second time."""
    if since is not _UNSET and getattr(rt.model, 'last_usage', None) is since:
        return
    usd, allowance = rt._call_cost(model)
    note_spend(rt, gid, usd, allowance)


def _sweep_box(rt, child):
    """The box of a child closed at sweep goes too: a child is not resumable, so a kept box is only garbage.

    Wrapped: sweep runs bare in the runtime loop, and one box that will not go must not end the daemon (the same
    reason sandbox.sweep keeps its own try) (#382 review)."""
    try:
        from . import sandbox
        sandbox.end(rt.home, child, rt.store)
    except Exception as e:
        try:
            rt.store.event('box_not_removed', dict(child=child, error='%s: %s' % (type(e).__name__, e)[:300]))
        except Exception:
            pass


def _priced_but_unledgered(rt, child):
    """The money and allowance `_meter` priced for a child that its own ledger may not have seen.

    `_meter` writes spend.json's per-goal totals on the line before `note_spend` writes the child's ledger, so a
    process killed between the two leaves one priced turn in spend.json alone: the close then reported a cheaper
    run than actually happened (#382 review). Both totals are read: an OpenCode plan turn costs $0 of money and
    real allowance, kept in `goal_allowance` beside `goals` for exactly this window (#382 review, rev-muse).
    Read, never invent - a spend.json without this child, or with less than the ledger (an auxiliary call is
    priced into the ledger only), contributes nothing.
    Returns (usd, allowance)."""
    try:
        state = json.loads((rt.home / 'spend.json').read_text())
        total = (state.get('goals') or {}).get(child)
        plan = (state.get('goal_allowance') or {}).get(child)
    except Exception:
        return 0.0, 0.0      # sweep runs bare in the runtime loop: a bad or absent file is not an error here
    return (float(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else 0.0,
            float(plan) if isinstance(plan, (int, float)) and not isinstance(plan, bool) else 0.0)


def _sweep_close(rt, r, outcome, state, summary):
    """Close a child's row at sweep in the same shape run_child reports.

    The close is the last thing written about this child, and a report without its spend reads as a free run
    (#382 review), so the row's model, provider, spend and cap always ride along. Spend and allowance are each the
    larger of the ledger and spend.json's total for the child (see _priced_but_unledgered), and the summary says
    when the difference was real money rather than a rounding."""
    spent = round(r['spend_usd'] or 0.0, 6)
    used = round(r['allowance_usd'] or 0.0, 6)
    priced, priced_allowance = _priced_but_unledgered(rt, r['child'])
    priced, priced_allowance = round(priced, 6), round(priced_allowance, 6)
    if priced > spent:
        summary = f'{summary}; ${priced:.4f} of it was priced but not yet in its ledger'
        spent = priced
    if priced_allowance > used:
        summary = f'{summary}; ${priced_allowance:.4f} of allowance was priced but not yet in its ledger'
        used = priced_allowance
    rt.store.delegate_done(r['child'], outcome, dict(
        child=r['child'], role=r['role'], model=r['model'], provider=r['provider'], outcome=outcome, state=state,
        spend_usd=spent, allowance_usd=used,
        cap_usd=r['cap_usd'] or 0.0, summary=summary))


def orphan(rt, gid):
    """Close a child that reached a turn with no brain of its own in this process: True when it was one to stop.

    `delegate_models` is live memory: only the process that ran start() holds the child's brain. A child that
    reaches a turn without it belongs to a run that ended (the store has one writer), so it cannot run on
    whatever brain the runtime happens to hold. Its goal is paused as orphaned and its row closed here, the same
    close `sweep` makes at a restart, so the fence cuts a race rather than leaving a row open for the parent's
    next start (#382 review, rev-glm P2). A row already closed but whose goal is still runnable is paused all
    the same: nothing else would ever end a goal no brain may serve."""
    row = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (gid,)).fetchone()
    runnable = row is not None and row['state'] in ('queued', 'working', 'waiting_model')
    if runnable:
        with rt.store.transaction():
            rt.store.update(gid, state='paused', error='its runtime ended mid-delegation')
    r = rt.store.delegate(gid)
    if r and not r.get('outcome'):
        _sweep_close(rt, r, 'orphaned', 'paused', 'its runtime ended mid-delegation')
        _sweep_box(rt, gid)
        return True
    return runnable


def sweep(rt):
    """At a runtime start, a child left runnable belongs to a run that ended (the store has one writer, and the
    runtime lock proved no other process holds it): pause it so it cannot run later on the wrong brain.

    A child left open whose goal has already ended is closed too: a process killed between the child's final
    state write and its report leaves that row open forever, and start()'s child limit counts it against the
    parent on every later start (#382 review).

    The box of every child closed here goes too: a child is not resumable, so `sandbox.sweep` keeping a paused
    goal's box keeps only garbage (#382 review)."""
    for r in rt.store.db.execute("SELECT child,role,model,provider,cap_usd,spend_usd,allowance_usd,started "
                                 "FROM delegates WHERE outcome=''").fetchall():
        # a direct read, not store.goal(): that raises on an unknown id, and this runs bare in the runtime loop,
        # so one row whose goal is gone (a partial restore) would end the daemon on every pass and leave the row
        # open for good (#382 review). A row without its goal is closed as orphaned - it can never report.
        row = rt.store.db.execute('SELECT state, error FROM goals WHERE id=?', (r['child'],)).fetchone()
        if row is None:
            _sweep_close(rt, r, 'orphaned', 'error', 'this child\'s goal row is gone')
            _sweep_box(rt, r['child'])
            continue
        if row['state'] in ('queued', 'working', 'waiting_model'):
            with rt.store.transaction():
                rt.store.update(r['child'], state='paused', error='its runtime ended mid-delegation')
            _sweep_close(rt, r, 'orphaned', 'paused', 'its runtime ended mid-delegation')
            _sweep_box(rt, r['child'])
            continue
        # The goal already ended (completed / cancelled / paused / error): the crash window run_child's fall-back
        # close cannot reach, because that process is gone. The goal's own state names the outcome, and a pause
        # is `capped` only when one of the child's own three caps wrote its event - the same fact `run_child`
        # reads, so a watchdog or manual pause is not reported as the child failing or as a spent limit (#382
        # review).
        state = row['state']
        if state == 'paused':
            outcome = 'capped' if _capped(rt, r['child']) else 'incomplete'
            summary = 'this child\'s goal was paused and never reported' + \
                      (': ' + row['error'] if row['error'] else '')
        else:
            outcome = dict(completed='completed', cancelled='incomplete').get(state, 'failed')
            summary = 'this child\'s goal ended without a report'
        _sweep_close(rt, r, outcome, state, summary)
        if outcome != 'completed':
            _sweep_box(rt, r['child'])
