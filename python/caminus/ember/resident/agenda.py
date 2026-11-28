"""report, budget and schedule: talking to the human, knowing what is left, and coming back later.

- report(level, text): fyi lands in the human's digest; need_you asks for attention. Free time may send a few fyi a
  day (its message quota) and never need_you.
- budget(): what this goal has used and has left: context, steps, time, and today's spend against each allowance.
- schedule(after_minutes, note): park this goal until later (or until a message arrives for it) and let other work
  run meanwhile; the note comes back as a nudge when it wakes. Unlike wait, nothing blocks.
"""
import json
import time

SCHEDULE_MAX_MIN = 24 * 60


def report(rt, gid, level, text):
    text = str(text or '').strip()
    if level not in ('fyi', 'need_you'):
        raise ValueError('level is fyi (for the digest) or need_you (asks for your human\'s attention)')
    if not text:
        raise ValueError('a report needs text')
    if rt.free.is_free(gid):
        if level == 'need_you':
            raise ValueError('free time never asks for attention: send it as fyi, or note it in your journal')
        if not rt.free.ask_allowed(gid):
            raise ValueError("today's free-time messages are used up; write it in your journal for your human")
    rt.store.event('report', dict(level=level, text=text[:4000]), gid)
    return dict(delivered=level, note='in the digest' if level == 'fyi' else 'your human is notified')


def budget(rt, gid):
    g = rt.store.goal(gid)
    out = dict(goal=dict(steps=g['steps'], step_limit=g['step_limit'],
                         elapsed_min=round((time.time() - g['created']) / 60, 1),
                         deadline_in_min=round((g['deadline'] - time.time()) / 60, 1) if g['deadline'] else None))
    try:
        note = json.loads((rt.home / 'context.json').read_text())
        if note.get('goal') == gid:
            out['context'] = dict(tokens=note['tokens'], window=note['size'],
                                  used_percent=round(100 * note['tokens'] / note['size'], 1))
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        pass
    spend = {}
    if getattr(rt.brains, 'fallback', None) is not None:
        spend['fallback_brain'] = dict(usd_today=round(rt.brains.spent_today(), 4),
                                       usd_per_day=float(rt.brains.cfg['budget_usd']))
    if rt.free.is_free(gid):
        spend['free_time'] = dict(usd_today=round(rt.free.spent_today(), 4),
                                  usd_per_day=float(rt.free.cfg['budget_usd']))
    if (adv := rt.advisor.cfg):
        spend['advisor'] = dict(usd_today=round(rt.advisor.spent_today(), 4), usd_per_day=float(adv['budget_usd']))
    if (eyes := getattr(rt, 'eyes', None)) and eyes.cfg and eyes.spent_today():
        spend['eyes'] = dict(usd_today=round(eyes.spent_today(), 4), usd_per_day=float(eyes.cfg['budget_usd']))
    out['spend'] = spend or 'no metered brain: the local model costs nothing per token'
    return out


def schedule(rt, gid, after_minutes, note):
    try:
        minutes = float(after_minutes)
    except (TypeError, ValueError):
        raise ValueError('after_minutes is a number of minutes')
    if not 1 <= minutes <= SCHEDULE_MAX_MIN:
        raise ValueError(f'schedule between 1 minute and {SCHEDULE_MAX_MIN // 60} hours ahead')
    note = str(note or '').strip()
    if not note:
        raise ValueError('say what to do when you wake (note): it comes back to you then')
    wake = time.time() + minutes * 60
    rt.store.update(gid, wake_at=wake, wake_note=note[:1000])
    rt.store.event('scheduled', dict(wake_at=wake, note=note[:1000]), gid)
    return dict(sleeping_until=time.strftime('%H:%M', time.localtime(wake)),
                note='this goal is parked until then (a message for it wakes it sooner); end your turn now')


def wake_due(rt, gid, g):
    """Called at the start of each turn: True while the goal sleeps. On waking, the note comes back as a nudge."""
    wake = g.get('wake_at')
    if not wake:
        return False
    early = rt.store.db.execute('SELECT 1 FROM inbox WHERE goal=? AND consumed=0 LIMIT 1', (gid,)).fetchone()
    if time.time() < wake and not early:
        return True
    with rt.store.transaction():
        text = g.get('wake_note') or ''
        rt.store.update(gid, wake_at=None, wake_note=None)
        if not early:
            # A harness note, not the human speaking: `_answering` must not read it as their answer (the plain-text
            # pause resets otherwise, #390 review P1); the note wakes the goal, it does not reply to it.
            rt.store.db.execute('INSERT INTO inbox(goal,text,kind,source) VALUES(?,?,?,?)',
                                (gid, '[scheduled wake-up] ' + text, 'nudge', 'harness'))
        rt.store.event('woke', dict(early=bool(early), note=text), gid)
    return False
