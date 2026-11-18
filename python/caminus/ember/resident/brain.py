"""Two brains: the primary model (normally the local server) and an optional fallback (normally a remote API).

The owner's machine runs the local model most of the time, but the server goes down for GPU work. A resident with a
fallback keeps going on it while the primary is unreachable and returns to the primary as soon as it answers again:

- a connection failure of the primary (no HTTP status, or 502 / 503 / 504) marks it down for `down_seconds` and the
  turn is retried at once on the fallback; afterwards a short probe decides when to go back
- the fallback spends money, so it has allowances: free-time turns count against the free-time allowance
  (freetime.py), work turns against `budget_usd` per day here; with an allowance spent the goal waits for the primary
- every switch is an event (`brain`), and failures of either brain go to failures.jsonl like any other

Configuration, in config.json under "fallback" (absent: one brain, as before): the same keys as the resident's own
model settings (model, endpoint, api, context, max_tokens, api_key_env, effort, ...) plus
    budget_usd     1.00   dollars per day the fallback may spend on work turns
    down_seconds   60     how long a failed primary is left alone before it is probed again

An OpenCode plan (catalog.OPENCODE) is for the work its owner asks for, and its traffic is watched for abuse: free time
stays off a brain on one unless free_time.on_plan is true (`heartbeat NAME --on-plan yes`). A free-time turn then runs
on the other brain when that one is not a plan (and its free-time allowance holds), else free time rests with a note
(freetime.py).
"""
import datetime
import json
import time

from . import catalog
from .model import Model, ModelContextError, ModelTruncated, price
from .store import atomic_json

DEFAULTS = dict(budget_usd=1.0, down_seconds=60)


class Brains:
    def __init__(self, runtime, primary):
        self.rt = runtime
        self.primary = primary
        self.path = runtime.home / 'fallback.json'
        self.down_until = 0.0
        self.last = {}                      # goal -> 'primary' | 'fallback': the brain its last turn used
        raw = runtime.config.get('fallback')
        self.cfg = {**DEFAULTS, **raw} if raw and raw.get('model') and raw.get('endpoint') else None
        self.fallback = None
        if self.cfg:
            conf = {k: v for k, v in self.cfg.items() if k not in DEFAULTS}
            conf.setdefault('name', runtime.config.get('name'))
            if runtime.config.get('max_tokens') is not None:   # (else the brain's own default, effort.reply_limit)
                conf.setdefault('max_tokens', runtime.config['max_tokens'])
            self.fallback = getattr(runtime, 'fallback_model', None) or Model(conf)

    # ---- spend ---------------------------------------------------------------------------------------------

    def _state(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def spent_today(self):
        return float(self._state().get(datetime.date.today().isoformat(), {}).get('usd', 0.0))

    def _allowed(self, gid):
        if self.fallback is None or self.plan_barred(self.fallback, gid):
            return False
        if price(self.fallback.config['model'], {}) is None and getattr(self.fallback, 'remote', False):
            return False                    # an unpriced remote fallback cannot be limited
        if self.rt.free.is_free(gid):
            return self.rt.free.spent_today() < float(self.rt.free.cfg['budget_usd'])
        return self.spent_today() < float(self.cfg['budget_usd'])

    def account(self, gid):
        """price a work turn the fallback served (free-time turns are priced by freetime.py)"""
        if self.rt.model is not self.fallback or self.rt.free.is_free(gid):
            return
        usd = price(self.fallback.config['model'], self.fallback.last_usage,
                    (getattr(self.fallback, 'last_call', None) or {}).get('served')) or 0.0
        state = self._state()
        day = state.setdefault(datetime.date.today().isoformat(), dict(usd=0.0, turns=0))
        day['usd'] = round(day['usd'] + usd, 6)
        day['turns'] += 1
        atomic_json(self.path, state)

    # ---- choosing ------------------------------------------------------------------------------------------

    def _primary_up(self):
        if time.time() < self.down_until:
            return False
        if self.down_until and not getattr(self.primary, 'remote', False):
            # it was down: a short probe before handing it a turn, so a still-dead server costs 3 s, not a timeout
            if not self.primary.probe():
                self.down_until = time.time() + float(self.cfg['down_seconds'])
                return False
        self.down_until = 0.0
        return True

    # ---- OpenCode plans ------------------------------------------------------------------------------------

    def plan_barred(self, model, gid=None):
        """Whether a free-time turn (of goal gid, or any when None) may not run on this model: an OpenCode plan
        while free_time.on_plan is off."""
        if gid is not None and not self.rt.free.is_free(gid):
            return False
        kind = (getattr(model, 'config', None) or {}).get('provider_kind')
        return kind in catalog.OPENCODE and not self.rt.free.cfg.get('on_plan')

    def free_brain(self):
        """The brain a free-time stretch would run on now: the usual one, or the other when the usual one is a plan it
        may not use; None when neither may serve it (free time then rests)."""
        if self.fallback is None:
            return None if self.plan_barred(self.rt.model) else self.rt.model
        usual = self.fallback if time.time() < self.down_until else self.primary
        other = self.primary if usual is self.fallback else self.fallback
        for m in (usual, other):
            if not self.plan_barred(m) and (m is self.primary or self.rt.free.spent_today() < float(self.rt.free.cfg['budget_usd'])):
                return m
        return None

    def choose(self, gid):
        """the model for this goal's next turn"""
        if self.fallback is None:
            return self.rt.model            # one brain: whatever the runtime holds, untouched
        use = 'primary' if self._primary_up() or not self._allowed(gid) else 'fallback'
        if use == 'primary' and self.plan_barred(self.primary, gid) and self._allowed(gid):
            use = 'fallback'                # free time off an OpenCode plan, onto the other brain
        if self.last.get(gid) != use:
            if gid in self.last or use == 'fallback':
                model = (self.primary if use == 'primary' else self.fallback).config.get('model')
                self.rt.store.event('brain', dict(brain=use, model=model), gid)
            self.last[gid] = use
        return self.primary if use == 'primary' else self.fallback

    def next_is_remote(self, gid=None):
        """whether a new goal would run on a remote (priced) model now; free time asks before it starts a stretch"""
        if self.fallback is None:
            return bool(getattr(self.rt.model, 'remote', False))
        if getattr(self.primary, 'remote', False):
            return True
        return time.time() < self.down_until and bool(getattr(self.fallback, 'remote', False))

    def next_model_name(self):
        if self.fallback is None:
            return (getattr(self.rt.model, 'config', None) or {}).get('model') or self.rt.config.get('model', '')
        if time.time() < self.down_until:
            return self.fallback.config.get('model', '')
        return self.primary.config.get('model', '') if hasattr(self.primary, 'config') else self.rt.config.get('model', '')

    def on_failure(self, gid, e):
        """A primary that is unreachable: mark it down and say the turn should be retried at once on the fallback.
        Returns True when the caller should retry immediately."""
        if self.fallback is None or self.rt.model is not self.primary:
            return False
        if isinstance(e, ModelContextError) or (isinstance(e, ModelTruncated) and not getattr(e, 'retryable', False)):
            return False                    # the request itself, not the server: another brain would not help
        status = getattr(e, 'status', None)
        unreachable = status in (None, 0, 502, 503, 504)
        if not unreachable or not self._allowed(gid):
            return False
        self.down_until = time.time() + float(self.cfg['down_seconds'])
        return True
