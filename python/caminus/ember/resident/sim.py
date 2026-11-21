"""/sim: the self-improvement memory (issue #341, arXiv 2611.18856).

One record per harness change, in the shape that paper's harness optimizer reads: the recurring problem
group it belongs to, the change, why, the predicted outcome, the observed outcome after CI and reviews, the
evaluation context, and where the prediction was wrong (prediction_error). Records are append-only events
(kind 'sim', history_text indexes the text fields, so history_search finds them); an observation appends a
new version with the same id and status 'observed', and the list / export collapse each id to its newest
version. The prediction fields are frozen once written: an observation may add only `observed`,
`eval_context`, `prediction_error` and `observed_ts`, and a closed (already observed) record is not written
again. Every record carries the brain that served the turn that wrote it (provider, model, effort), so a
weak-vs-strong run needs no code change; an observation adds `observed_by`, the brain that read the outcome.
The per-session switch is config 'sim' (on unless a session turns it off): a running resident follows it at
its next turn, then hides the tool, the prompt section and the search hits, so an "off" arm cannot read the
log it is being compared against.

Statuses: 'proposed' (a post-goal reflection, no change made), 'recorded' (a prediction written before a
harness change), 'observed' (the outcome after CI and reviews).

The agent's tool is record / observe / list. The JSONL export belongs to the CLI alone (`caminus ember agent
sim NAME --export`), which a human runs: the model cannot name a path, so no export can overwrite a file, in
plan mode or anywhere else.
"""
from contextlib import contextmanager
import datetime
import json
import sqlite3
import time
import uuid

FIELDS = ('problem_group', 'change', 'rationale', 'predicted', 'observed', 'eval_context', 'prediction_error')
TEXT_CAP = 8000               # one field's characters; the whole point is a compact record
STATUSES = ('proposed', 'recorded', 'observed')
# an observation may add only these: the prediction fields are frozen once written, so the bet the record
# measures cannot be restated after the outcome is known
OBSERVED_FIELDS = ('observed', 'eval_context', 'prediction_error')


def enabled(rt):
    """The per-session switch: config 'sim' is on unless a session stored false (an A/B arm turns it off)."""
    return bool(rt.config.get('sim', True))


def serving(rt, cfg=None):
    """The config of the brain answering this turn: the model's own when a runtime set one (a fallback turn),
    else the agent's. Stamping rt.config alone mislabelled every record a fallback wrote as the primary's."""
    if cfg:
        return cfg
    return getattr(getattr(rt, 'model', None), 'config', None) or rt.config


def brain(cfg):
    """The brain a record is written by: its provider, model and the effort level that runs ('' when none)."""
    from . import effort
    level = cfg.get('effort') or effort.default_level(cfg)
    try:
        runs = effort.runs_as(cfg, level) if effort.line(cfg) else ''
    except Exception:
        runs = level
    return dict(provider=cfg.get('provider') or cfg.get('provider_kind') or '',
                model=cfg.get('model') or '', effort=runs or '')


def _clean(value):
    return str(value if value is not None else '').strip()[:TEXT_CAP]


def _event(store, gid, record):
    store.event('sim', record, gid)


@contextmanager
def _atomic(rt):
    """One transaction around a check-then-insert: two writers (the daemon and the CLI) could both pass the
    append-only check in autocommit, and one observation would silently disappear."""
    if rt.store.db.in_transaction:
        yield                         # (the daemon's dispatch already holds the store's transaction)
    else:
        with rt.store.transaction():
            yield


def _get(rt, rid):
    """One record's newest version by id, without decoding the whole log on every write.

    A write only needs the record it appends to; scanning the log for it made every record and every
    observation slower as the log grew. The scan remains the fallback for an SQLite without JSON1.
    """
    try:
        rows = rt.store.db.execute(
            "SELECT seq, ts, goal, data FROM events WHERE kind='sim' AND json_extract(data,'$.id')=? "
            "ORDER BY seq DESC LIMIT 1", (rid,)).fetchall()
    except sqlite3.OperationalError:
        return next((d for d in _load(rt, limit=100000) if d['id'] == rid), None)
    if not rows:
        return None
    r = rows[0]
    d = json.loads(r['data'])
    d.update(goal=d.get('goal') or r['goal'], ts=d.get('ts') or r['ts'], seq=r['seq'])
    return d


def _load(rt, gid=None, limit=200, statuses=None):
    """The newest version of every record, newest first; gid limits to one goal (None: the whole session)."""
    rows = rt.store.db.execute(
        "SELECT seq, ts, goal, data FROM events WHERE kind='sim'" + (" AND goal=?" if gid else "") + " ORDER BY seq",
        ((gid,) if gid else ())).fetchall()
    newest = {}
    for r in rows:
        try:
            d = json.loads(r['data'])
        except ValueError:
            continue
        rid = d.get('id')
        if not rid:
            continue
        d.update(goal=d.get('goal') or r['goal'], ts=d.get('ts') or r['ts'], seq=r['seq'])
        newest[rid] = d
    out = [d for d in newest.values() if not statuses or d.get('status') in statuses]
    out.sort(key=lambda d: (d.get('ts') or 0, d.get('seq') or 0), reverse=True)
    return out[:max(1, int(limit or 200))]


def record(rt, gid, data, cfg=None):
    """A prediction, written before a harness change (or a proposal, when nothing is changed yet)."""
    with _atomic(rt):
        return _record(rt, gid, data, cfg)


def _record(rt, gid, data, cfg=None):
    if not _clean(data.get('change')):
        raise ValueError('a SIM record needs `change`: the harness change it predicts (or would make)')
    if not _clean(data.get('predicted')):
        raise ValueError('a SIM record needs `predicted`: the outcome you expect, as something CI, a test or a '
                         'reviewer can confirm or contradict')
    status = str(data.get('status') or '').strip() or 'recorded'
    if status not in STATUSES:
        raise ValueError('status must be one of ' + ', '.join(STATUSES))
    if status == 'observed' and not _clean(data.get('observed')):
        raise ValueError('an observed record needs `observed`: what CI and the reviews actually showed')
    rid = str(data.get('id') or '').strip()
    if status == 'observed' and not rid:
        # a record born closed carries the outcome before the change ran: no prediction anything can contradict
        raise ValueError('an observation needs the id of the record it completes: write the prediction first '
                         '(action record without status), then observe it')
    prior = None
    if rid:
        prior = _get(rt, rid)
        if prior is None:
            raise ValueError('no SIM record %s in this session' % rid)
        if status != 'observed':
            # append-only: only an observation adds a version. A second `record` with the same id would rewrite
            # the prediction after the fact, which is the one thing this log exists to prevent.
            raise ValueError('SIM record %s already exists: a prediction is append-only, so record a new one or '
                             'append the outcome with action observe' % rid)
        if prior.get('status') == 'observed':
            # and once the outcome is in, the record is closed: a second observation could rewrite the first
            raise ValueError('SIM record %s is already observed: a closed record is not rewritten, record a new '
                             'prediction if the change is made again' % rid)
        for k in FIELDS:
            if k in OBSERVED_FIELDS or not _clean(data.get(k)):
                continue
            if _clean(data.get(k)) != _clean(prior.get(k)):
                raise ValueError('SIM record %s is append-only: `%s` was written with the prediction, so an '
                                 'observation cannot restate or change it' % (rid, k))
        # a version that leaves a field empty keeps the prediction's value: an observation adds the outcome, it
        # never blanks the change or the prediction it is measuring
        data = dict(prior, **{k: v for k, v in data.items() if _clean(v)})
    writer = brain(serving(rt, cfg))
    # ts is always the log's own clock: a caller-supplied one could backdate a prediction and move it in `list`
    rec = dict(id=rid or uuid.uuid4().hex[:8], ts=float((prior or {}).get('ts') or time.time()),
               goal=(prior or {}).get('goal') or gid, status=status,
               **{k: _clean(data.get(k)) for k in FIELDS},
               # the record names the brain that made the prediction; an observation adds the brain that read it
               brain=(prior or {}).get('brain') or writer)
    if prior:
        rec['observed_by'] = writer
    if status == 'observed':
        rec['observed_ts'] = time.time()
    elif prior and prior.get('observed_ts'):
        rec['observed_ts'] = prior['observed_ts']
    if not rec['problem_group']:
        rec['problem_group'] = 'unclassified'
    if prior and prior.get('seq'):
        # history_fts holds one row per event: drop the prior version's so a search names the record once, with
        # its newest text, not once per version
        rt.store.db.execute('DELETE FROM history_fts WHERE rowid=?', (prior['seq'],))
    # an observation lives in the record's goal, whatever goal its observer is in; the record is one record
    _event(rt.store, rec.get('goal') or gid, rec)
    return rec


def observe(rt, gid, rid, observed, eval_context='', prediction_error='', cfg=None):
    """The outcome after CI and the reviews, appended as a new version of a predicted record."""
    if not _clean(observed):
        raise ValueError('say what CI and the reviews showed, not that they passed')
    with _atomic(rt):
        prior = _get(rt, rid)
        if prior is None:
            raise ValueError('no SIM record %s in this session: list them first' % rid)
        data = dict(prior, status='observed', observed=observed, eval_context=eval_context,
                    prediction_error=prediction_error)
        return _record(rt, prior.get('goal') or gid, data, cfg)


def render(rec):
    """One record as the lines a slash command / tool result shows."""
    brain = rec.get('brain') or {}
    head = '%s %s · %s · %s%s' % (rec['id'], rec.get('status', ''), rec.get('problem_group', '').split(' — ')[0][:60],
                                  brain.get('model') or '?', ' ' + brain['effort'] if brain.get('effort') else '')
    lines = [head]
    for k in FIELDS:
        if rec.get(k):
            lines.append('  %s: %s' % (k, rec[k].replace('\n', ' ')[:300]))
    when = datetime.datetime.fromtimestamp(rec.get('ts') or 0).strftime('%Y-%m-%d %H:%M')
    lines.append('  when: %s%s' % (when, ' · goal ' + rec['goal'][:8] if rec.get('goal') else ''))
    if rec.get('observed_ts'):
        lines.append('  observed: ' + datetime.datetime.fromtimestamp(rec['observed_ts']).strftime('%Y-%m-%d %H:%M'))
    return lines


def text(rt, gid=None, limit=20, statuses=None):
    recs = _load(rt, gid, limit, statuses)
    if not recs:
        if statuses:
            return ['no SIM records with status ' + ', '.join(sorted(statuses)) + ': list without a status filter '
                    'to see them all']
        return ['no SIM records yet: sim records a prediction before a harness change, and the observation after CI']
    out = []
    for rec in recs:
        out += render(rec)
    return out


def path_for(home):
    from pathlib import Path
    return Path(home) / 'sim.jsonl'


def export(rt, gid=None, path=None):
    """Write the records as JSONL (one record per line, newest version only) and return (path, count, bytes).

    Reached by the CLI alone (`caminus ember agent sim NAME --export`), which a human runs. The agent's tool has
    no export, so the model never names a file and no export can overwrite one, in plan mode or anywhere else.
    """
    from pathlib import Path
    from .store import atomic_write
    recs = _load(rt, gid, limit=100000)
    p = Path(path).expanduser() if path else path_for(rt.home)
    if not p.is_absolute():
        p = rt.home / p
    data = ''.join(json.dumps(rec, ensure_ascii=False) + '\n' for rec in recs).encode()
    atomic_write(p, data)
    return p, len(recs), len(data)


def tool(rt, gid, args):
    """/sim as the agent's tool: record / observe / list. The JSONL export is the CLI's (no model-chosen path)."""
    if not enabled(rt):
        # an off arm records nothing and reads nothing: the A/B compares the prompt, and a log the off arm could
        # read would measure both arms against the same lessons
        raise ValueError('the SIM log is off in this session (config sim=false): it records and shows nothing. Turn '
                         'it on with `caminus ember agent sim NAME --on`; a record it left open stays open until then')
    action = args.get('action')
    if action == 'record':
        rec = record(rt, gid, args)
        return dict(record=rec, note='SIM record written: add the observation after CI and the reviews with '
                                     'action observe')
    if action == 'observe':
        rec = observe(rt, gid, str(args.get('id') or ''), args.get('observed'), args.get('eval_context', ''),
                      args.get('prediction_error', ''))
        return dict(record=rec)
    if action == 'list':
        # status: the proposal queue is `list` with status proposed; _load keeps each id's newest version first,
        # so one filter on it names a record once, at its current status. Without a goal (gid is None when the
        # caller passes no goal) that queue is every goal of the session on purpose: a person reads what the
        # harness proposes across all of them; pass the goal to see only its own (#383 review P2).
        status = str(args.get('status') or '').strip()
        if status and status not in STATUSES:
            raise ValueError('status must be one of ' + ', '.join(STATUSES))
        statuses = (status,) if status else None
        recs = _load(rt, args.get('goal') and gid or None, args.get('limit', 20), statuses)
        return dict(records=recs, lines=text(rt, args.get('goal') and gid or None, args.get('limit', 20), statuses))
    raise ValueError('action must be record, observe or list; the JSONL export is the CLI\'s: '
                     '`caminus ember agent sim NAME --export`')
