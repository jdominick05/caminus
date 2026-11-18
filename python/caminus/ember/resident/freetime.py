"""Free time: the resident's own hours.

When its human has no work queued, the resident is not idle: it lives in its account. Each stretch of free time
opens with an invitation instead of a task (research what it is curious about, write, make something, tinker, or
rest), keeps a journal and a studio in its home, and ends when it says so. At night one stretch is a dream pass
that consolidates its notes. A goal from its human always wins: it ends the running stretch at once, and the next
stretch picks up from the journal.

Free time costs nothing on a local model. On a remote provider every turn is priced against a separate daily
allowance (config free_time.budget_usd), so free time never spends the work budget; a provider whose price is not
known is not used for free time at all.

Configuration, in config.json under "free_time" (all optional):
    enabled            false     the `heartbeat on` command sets it
    rest_seconds       1800      rest between two stretches
    stretch_steps      0         model steps one stretch may take; 0 = no step limit (its time ends it)
    stretch_seconds    1800      wall time one stretch may take
    budget_usd         0.50      daily allowance on a priced remote model
    afk_seconds        600       a question waits this long for an answer, then the stretch carries on
    messages_per_day   3         questions free time may put to its human in a day
    dream_hour         3         local hour from which the night's dream pass may run (for three hours)
    context_tokens     0         a stretch's own context window; 0 = the resident's full window (owner 11-15)
    on_plan            false     free time may run on an OpenCode plan (catalog.OPENCODE; `heartbeat --on-plan yes`).
                                 Off: a stretch runs on the other brain when that is not a plan, else free time rests
                                 with a note (brain.Brains.free_brain). The plan is for the work its owner asks for, and
                                 its traffic is watched for abuse.

Free time works in the resident's full context window, and must still not cost its human the conversation they come
back to. The local server keeps that conversation by snapshot: a short stretch that fits the prompt cache's side slot
(98K positions on the Coder) runs there and never touches the main slot; a stretch that outgrows it takes the main slot,
and before a request overwrites 4,096 or more of the live conversation's tokens the cache parks that conversation to
its disk tier (prompt.AUTO_PARK_MIN). The human's next request finds it there and unparks it whole: restored, not
re-read. When the human is back, recap() is the "while you were away" summary, built from each stretch's own closing
line (no model call); a UI appends it after the restored conversation, never before it, so the restored prefix stays
valid. context_tokens can still cap a stretch (e.g. to keep it in the side slot on a server without a disk tier).
"""
import datetime
import json
import re
import time

from .model import price
from .store import atomic_json

DEFAULTS = dict(enabled=False, rest_seconds=1800, stretch_steps=0, stretch_seconds=1800, budget_usd=0.50,
                afk_seconds=600, messages_per_day=3, dream_hour=3, context_tokens=0, on_plan=False)
KINDS = ('free', 'dream')
NOW_MARKER = 'free_time_now'
PLAN_REST = ("free time stays off an OpenCode plan (it is for the work you ask for, and its traffic is watched for "
             "abuse), and no other brain can take it: resting. To let free time use the plan: "
             "caminus ember agent heartbeat {name} --on-plan yes")   # `heartbeat now` writes it; the next pulse starts a stretch and removes it

INVITATION = '''Free time. Nobody needs anything from you right now; this time is yours.

Settle in first: see what changed on this machine since you last looked, and read your journal ({journal}) and your notes. Then follow whatever interests you, and once you know what this stretch is for, write it down as a short todo list (todo): it stays in view across a long stretch and your human can see what you are up to. Nothing grades it and finish never asks for it to be complete. Research something you are curious about, write, make something in your studio ({studio}), tinker with an idea, learn a tool, or simply rest. There is no task and nobody is grading this.

House rules for free time:
- Stay inside your own account. No root requests, no GPU work, and nothing that touches your human's files, servers or other residents.
- Read the web freely, but post, send or publish nothing off this machine.
- Leave nothing running when you stop.
- If you make something your human might enjoy, put it in {for_you} with a short note. You may ask your human something with ask_user (at most {messages} times a day); if nobody answers within {afk} minutes, carry on and note the question in your journal.

What to do is yours to choose. Lean toward what is new and your own: rotate among exploring (research, reading), making (build or create something that did not exist), tending (improve one of your projects) and reflecting (journal, notes). Re-deriving or copying something you already made is not new.

Big things are welcome. Keep your projects in {projects}, one line each: `- [active] NAME: what it is for (last: YYYY-MM-DD)`, with [paused] or [done] when that is true. Keep each project's next steps in a note of its own (remember, or a file beside the project), so the next stretch can pick it up where you left it. Keep at most 3 active at once; finish or park one before starting another. Between stretches nothing of yours keeps running.

Before you stop, add a dated entry to {journal}: what you did, what you noticed, what you want to pick up next time. To stop, call finish with one line about the stretch (name the project if it was one), or simply reply in plain text to rest.{past}{notes}{resume}'''

DREAM = '''Night: a dream pass. Nothing is asked of you tonight except to consolidate.

Read your journal entries since the last dream ({journal}) and your notes. Merge what you learned into your notes and keep them short, link ideas that belong together, and drop what no longer matters. Then write a short dream entry in {journal}: what the day meant, and what you want to explore next. Start no new projects tonight. The free-time house rules apply. Call finish with one line when you are done.'''


def plan_note(config):
    """PLAN_REST for a resident whose free time has no brain but an OpenCode plan it may not use, by its config (the
    owner's commands: heartbeat status / now), else ''."""
    from .catalog import OPENCODE
    fb = config.get('fallback') if isinstance(config.get('fallback'), dict) else {}
    if settings(config)['on_plan'] or config.get('provider_kind') not in OPENCODE:
        return ''                     # (a plan fallback serves only while the primary is down: Brains.free_brain)
    if fb.get('model') and fb.get('endpoint') and fb.get('provider_kind') not in OPENCODE:
        return ''                     # the fallback takes free time
    return PLAN_REST.format(name=config.get('name') or 'NAME')


def settings(config):
    raw = config.get('free_time') or {}
    if isinstance(raw, bool):
        raw = dict(enabled=raw)
    return {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}


class FreeTime:
    def __init__(self, runtime):
        self.rt = runtime
        self.state_path = runtime.home / 'free_time.json'

    # ---- state ---------------------------------------------------------------------------------------------

    @property
    def cfg(self):
        """The free-time settings, re-read when config.json changes: `heartbeat on` reaches a running daemon."""
        path = self.rt.home / 'config.json'
        try:
            stamp = path.stat().st_mtime_ns
            if stamp != getattr(self, '_stamp', None):
                self._stamp, self._live = stamp, json.loads(path.read_text()).get('free_time')
        except (OSError, ValueError):
            self._live = self.rt.config.get('free_time')
        return settings(dict(free_time=self._live))

    def _state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, state):
        atomic_json(self.state_path, state)

    @staticmethod
    def _today():
        return datetime.date.today().isoformat()

    def spent_today(self):
        return float(self._state().get('days', {}).get(self._today(), {}).get('usd', 0.0))

    def messages_today(self):
        return int(self._state().get('days', {}).get(self._today(), {}).get('messages', 0))

    def _day(self, state):
        days = state.setdefault('days', {})
        for old in sorted(days)[:-7]:        # a week of history is plenty
            days.pop(old)
        return days.setdefault(self._today(), dict(usd=0.0, turns=0, messages=0))

    def is_free(self, gid):
        try:
            return self.rt.store.goal(gid).get('kind') in KINDS
        except ValueError:
            return False

    # ---- the pulse -----------------------------------------------------------------------------------------

    def pulse(self):
        """Called by the daemon loop once a pass. Ends free time when its human's work arrives, and starts a
        stretch when nothing is active, the rest interval has passed and the budget allows. Returns the id of a
        stretch it started, or None."""
        store = self.rt.store
        # A stretch that used up its steps or time is paused by the loop's budget checks; for free time that is
        # simply the end of the stretch.
        # Paused for anything else, it keeps its reason: a person's pause or the emergency stop stays paused (theirs to
        # resume), and any other stop (a watchdog, the judge) ends the stretch with that reason, not as "used its time".
        for r in list(store.db.execute("SELECT id, error FROM goals WHERE kind IN ('free','dream') AND state='paused'")):
            why = r['error'] or ''
            if why == 'paused by your human' or why.startswith('emergency stop'):
                continue
            budget = why in ('oneshot time budget reached', 'oneshot model-step budget reached')
            with store.transaction():
                store.update(r['id'], state='completed',
                             summary='the stretch used its time' if budget else 'the stretch stopped: ' + why[:300])
                store.event('free_time_end', dict(reason='stretch budget' if budget else why[:300]), r['id'])
            self.ended(r['id'])
        active = [store.goal(g) for g in store.next_goal()]
        work = [g for g in active if g.get('kind', 'work') not in KINDS]
        free = [g for g in active if g.get('kind') in KINDS]
        if work and free:
            for g in free:
                self._end(g['id'], 'preempted: your human queued work; pick this up from your journal next time')
            return None
        if free and self.rt.brains.free_brain() is None:
            # the brain became an OpenCode plan mid-stretch (/model), and free time stays off it
            for g in free:
                self._end(g['id'], PLAN_REST.format(name=self.rt.config.get('name') or 'NAME'))
            return None
        cfg = self.cfg
        # `heartbeat now` (the panel's /freetime): a stretch at the next pass, past the rest interval and even with
        # free time switched off; work still comes first (the request waits for it) and the budget still holds.
        asked = (self.rt.home / NOW_MARKER).exists()
        if (not cfg['enabled'] and not asked) or work or free:
            return None
        state = self._state()
        now = time.time()
        # the rest counts from the last start too: a stretch that failed records no end, and the next one started at
        # once (a relay at its daily cap refused 1,184 stretches in a row)
        last = max(float(state.get('last_end', 0)), float(state.get('last_start', 0)))
        # ...and from its human's last work: a goal that finished (or changed) counts, so the rest begins when the work
        # ends (it counted from the last stretch only, and free time started the moment a goal finished)
        worked = store.db.execute("SELECT MAX(updated) FROM goals WHERE COALESCE(kind,'work') NOT IN ('free','dream')").fetchone()[0]
        last = max(last, float(worked or 0))
        if not asked and now - last < float(cfg['rest_seconds']):
            return None
        wait = self._failure_wait(state)
        if wait and now < float(state.get('last_start', 0)) + wait:
            return None
        if self.rt.brains.free_brain() is None:
            return self._note_once(state, 'plan', PLAN_REST.format(name=self.rt.config.get('name') or 'NAME'))
        if self.rt.brains.next_is_remote():
            if self._price_unknown():
                return self._note_once(state, 'unpriced', 'free time needs a known price on a remote model')
            if self.spent_today() >= float(cfg['budget_usd']):
                return self._note_once(state, 'budget', 'the free-time allowance for today is spent; resting '
                                       'until tomorrow or until a local model is configured')
        dream = self._dream_due(state) and not asked
        (self.rt.home / NOW_MARKER).unlink(missing_ok=True)
        gid = self._start('dream' if dream else 'free', state)
        if asked:
            self.rt.store.event('free_time_asked', {}, gid)
        return gid

    def _failure_wait(self, state):
        """seconds to wait after stretches that failed (state error), counted from the last start: an hour after a
        relay or provider refused for spend (HTTP 402, a cap: raising it resumes within the hour), else the rest
        interval doubled per failure in a row, at most six hours; 0 when the last stretch did not fail"""
        rows = list(self.rt.store.db.execute(
            "SELECT state,error FROM goals WHERE kind IN ('free','dream') ORDER BY created DESC LIMIT 8"))
        fails = 0
        for r in rows:
            if r['state'] != 'error':
                break
            fails += 1
        if not fails:
            return 0
        err = (rows[0]['error'] or '').lower()
        if '402' in err or 'cap' in err or 'billing' in err or 'payment' in err:
            self._note_once(state, 'spend_refused', 'the brain refused for spend (a daily cap): free time waits an '
                            'hour, then tries again')
            return 3600
        return min(6 * 3600, float(self.cfg['rest_seconds']) * 2 ** fails)

    def _price_unknown(self):
        return price(self.rt.brains.next_model_name(), {}) is None

    def _note_once(self, state, reason, text):
        day = self._day(state)
        if day.get('noted_' + reason):
            return None
        day['noted_' + reason] = True
        self._save(state)
        self.rt.store.event('free_time_rest', dict(reason=reason, text=text))
        return None

    def _dream_due(self, state):
        hour = datetime.datetime.now().hour
        start = int(self.cfg['dream_hour'])
        night = start <= hour < start + 3
        return night and state.get('last_dream') != self._today()

    def _start(self, kind, state):
        cfg = self.cfg
        home = self.rt.home
        journal, studio = home / 'journal', home / 'studio'
        for d in (journal, studio, studio / 'for-you'):
            d.mkdir(parents=True, exist_ok=True)
        resume = ''
        if state.get('last_summary'):
            resume = '\n\nLast time: ' + str(state['last_summary'])[:400]
        history = [h for h in state.get('history', []) if h.get('summary')]
        past = ''
        if history[:-1]:
            past = '\n\nYour earlier stretches (oldest first):\n' + '\n'.join(
                '- %s %s' % (time.strftime('%m-%d', time.localtime(h['ts'])), h['summary'][:120]) for h in history[-7:-1])
        notes = ''.join('\n\n' + n for n in self.nudges(history, studio / 'PROJECTS.md'))
        values = dict(journal=str(journal) + '/', studio=str(studio) + '/', for_you=str(studio / 'for-you') + '/',
                      projects=str(studio / 'PROJECTS.md'), past=past, notes=notes,
                      messages=cfg['messages_per_day'], afk=max(1, int(cfg['afk_seconds']) // 60), resume=resume)
        prompt = (DREAM if kind == 'dream' else INVITATION).format(**values)
        # stretch_steps 0 (the default): no step limit; the stretch's time ends it (the owner: "unlimited steps, just
        # time per stretch"; 40 steps lasted two minutes on a fast paid brain). The daily allowance still caps spend.
        steps = int(cfg['stretch_steps'] or 0)
        gid = self.rt.store.enqueue(prompt, max_steps=steps if steps > 0 else None,
                                    max_seconds=float(cfg['stretch_seconds']), kind=kind)
        if kind == 'dream':
            state['last_dream'] = self._today()
        state['last_start'] = time.time()
        self._save(state)
        self.rt.store.event('free_time_start', dict(kind=kind), gid)
        return gid

    @staticmethod
    def active_projects(path):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            m = re.match(r'\s*[-*]\s*\[active\]\s*([^:]+?)\s*(?::|$)', line, re.I)
            if m:
                out.append(m.group(1).strip())
        return out

    def nudges(self, history, projects_path):
        """Notes for the next invitation, from the resident's own record (no model call): its last two stretches look
        alike; three in a row went to the same project; more than three projects are active."""
        out = []
        summaries = [h['summary'] for h in history]
        if len(summaries) >= 2 and similarity(summaries[-1], summaries[-2]) >= 0.5:
            out.append('Note: your last two stretches look alike. Do something different this time.')
        active = self.active_projects(projects_path)
        for name in active:
            if len(summaries) >= 3 and all(name.lower() in x.lower() for x in summaries[-3:]):
                out.append('Note: your last three stretches went to %s. Finish it or park it, or try something new for a '
                           'while.' % name)
                break
        if len(active) > 3:
            out.append('Note: %d projects are active (%s). Finish or park some before starting another.'
                       % (len(active), ', '.join(active)))
        return out

    def _end(self, gid, why):
        with self.rt.store.transaction():
            g = self.rt.store.goal(gid)
            if g['state'] in ('completed', 'cancelled', 'error'):
                return
            self.rt.store.update(gid, state='cancelled', error=why)
            self.rt.store.event('free_time_end', dict(reason=why), gid)
        self.ended(gid, why)

    def ended(self, gid, summary=''):
        """A stretch is over (finished, rested, preempted or out of budget): the rest interval starts now."""
        state = self._state()
        state['last_end'] = time.time()
        if summary:
            state['last_summary'] = summary
            try:
                kind = self.rt.store.goal(gid).get('kind')
            except ValueError:
                kind = 'free'
            state['history'] = (state.get('history', []) + [dict(ts=time.time(), kind=kind, summary=summary[:400])])[-20:]
        self._save(state)

    # ---- per turn ------------------------------------------------------------------------------------------

    def account(self, gid):
        """Price the turn that just ran, when it was free time on a remote model."""
        if not self.is_free(gid) or not getattr(self.rt.model, 'remote', False):
            return
        served = (getattr(self.rt.model, 'last_call', None) or {}).get('served')
        name = (getattr(self.rt.model, 'config', None) or {}).get('model') or self.rt.config.get('model', '')
        usd = price(name, getattr(self.rt.model, 'last_usage', {}) or {}, served) or 0.0
        state = self._state()
        day = self._day(state)
        day['usd'] = round(day['usd'] + usd, 6)
        day['turns'] += 1
        self._save(state)
        if day['usd'] >= float(self.cfg['budget_usd']):
            self._end(gid, 'the free-time allowance for today is spent')

    def ask_allowed(self, gid):
        """Free time may put a few questions a day to its human; count this one if it is allowed."""
        state = self._state()
        day = self._day(state)
        if day['messages'] >= int(self.cfg['messages_per_day']):
            return False
        day['messages'] += 1
        self._save(state)
        return True

    def recap(self, since=None):
        return recap(self.rt.store, since)

    def window(self, gid, size):
        """the context window a goal works in: the full one, unless context_tokens caps free time"""
        cap = int(self.cfg.get('context_tokens') or 0)
        return min(size, cap) if cap and self.is_free(gid) else size

    @property
    def afk_seconds(self):
        return float(self.cfg['afk_seconds'])


def recap(store, since=None):
    """The "while you were away" summary: the stretches that ended after `since` (default: the newest goal from the
    human), each with its kind, when it ran and its own closing line. Plain text, no model call."""
    if since is None:
        r = store.db.execute("SELECT MAX(created) AS t FROM goals WHERE kind='work'").fetchone()
        since = r['t'] or 0
    rows = list(store.db.execute(
        "SELECT id,kind,state,summary,error,created,updated FROM goals WHERE kind IN ('free','dream') AND created>? "
        "ORDER BY created", (since,)))
    if not rows:
        return ''
    lines = ['While you were away (%d stretch%s):' % (len(rows), '' if len(rows) == 1 else 'es')]
    # stretches in a row with the same outcome are one line with a count and their time span (a refused relay made
    # 1,184 identical error lines)
    runs = []
    for r in rows:
        kind = 'dream' if r['kind'] == 'dream' else 'free time'
        what = first_sentences(r['summary'] or r['error'] or r['state'], 300)
        if runs and runs[-1][1] == kind and runs[-1][2] == what:
            runs[-1][3] = r['created']
            runs[-1][4] += 1
        else:
            runs.append([r['created'], kind, what, r['created'], 1])
    for start, kind, what, end, n in runs:
        when = time.strftime('%H:%M', time.localtime(start))
        if n > 1:
            when += '-' + time.strftime('%H:%M', time.localtime(end))
            kind = '%s ×%d' % (kind, n)
        lines.append('- %s %s: %s' % (when, kind, what))
    unanswered = [e for e in store.events() if e['kind'] == 'question_unanswered' and e['ts'] > since]
    if unanswered:
        lines.append('Questions it left for you: ' + '; '.join(str(e['data'].get('text', ''))[:120] for e in unanswered))
    return '\n'.join(lines)


def first_sentences(text, limit):
    """the text's opening up to `limit` characters, cut at a sentence end where there is one"""
    t = ' '.join((text or '').split())
    if len(t) <= limit:
        return t
    cut = max(t.rfind('. ', 0, limit), t.rfind('; ', 0, limit))
    return t[:cut + 1] if cut > limit // 3 else t[:limit].rsplit(' ', 1)[0] + ' ...'


def similarity(a, b):
    """word-set overlap of two closing lines (0..1): cheap, and enough to notice a stretch repeating the last one"""
    words = lambda x: {w for w in re.findall(r'[a-z0-9]+', (x or '').lower()) if len(w) > 3}
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0
