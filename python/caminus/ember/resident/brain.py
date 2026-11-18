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
from .model import Model, ModelContextError, ModelTruncated, is_remote, price
from .store import atomic_json

DEFAULTS = dict(budget_usd=1.0, down_seconds=60)


def plan_allows(cfg, on_plan):
    """Whether background work (free time) may call cfg's model: never an OpenCode plan unless free_time.on_plan. The
    plan half of free_choice; a side call made for a free-time goal (the advisor's) asks it too."""
    return cfg.get('provider_kind') not in catalog.OPENCODE or bool(on_plan)


def free_choice(primary, fallback, on_plan, free_left, primary_down=False):
    """The one rule for the brain a free-time turn may use: 'primary', 'fallback' or None. The daemon (Brains) and the
    owner's status (freetime.plan_note) both ask it. primary / fallback: the brains' configs (None: no such brain);
    on_plan: free_time.on_plan; free_left: today's free-time allowance holds. Never an OpenCode plan unless on_plan;
    the fallback also needs a known price when it bills, and the allowance. The primary comes first unless it is down
    (then it is still named when the fallback may not serve: the turn waits for it)."""
    def ok(cfg, fallback_side):
        if cfg is None or not plan_allows(cfg, on_plan):
            return False
        if not fallback_side:
            return True
        if price(cfg.get('model'), {}) is None and is_remote(cfg):
            return False                    # an unpriced paid fallback cannot be limited
        return free_left
    order = (('fallback', fallback), ('primary', primary)) if primary_down else (('primary', primary), ('fallback', fallback))
    return next((name for name, cfg in order if ok(cfg, name == 'fallback')), None)


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
        """whether the fallback may take this goal's turn: a free-time turn by free_choice (never a plan it may not use);
        a work turn while the fallback is priced (or free to run) and its daily budget holds"""
        if self.fallback is None:
            return False
        if self.rt.free.is_free(gid):
            return self._free_choice(primary=None) == 'fallback'
        if price(self.fallback.config['model'], {}) is None and getattr(self.fallback, 'remote', False):
            return False                    # an unpriced remote fallback cannot be limited
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

    def _free_choice(self, primary_down=False, primary=True):
        """free_choice for this resident now: 'primary', 'fallback' or None (primary=None: the fallback alone)"""
        if primary is True:
            primary = getattr(self.primary if self.fallback is not None else self.rt.model, 'config', None) or {}
        ft = self.rt.free.cfg
        return free_choice(primary, getattr(self.fallback, 'config', None) if self.fallback is not None else None,
                           bool(ft.get('on_plan')), self.rt.free.spent_today() < float(ft['budget_usd']), primary_down)

    def free_brain(self):
        """The brain a free-time stretch would run on now (free_choice), or None: free time then rests."""
        name = self._free_choice(primary_down=self.fallback is not None and time.time() < self.down_until)
        if self.fallback is None:
            return self.rt.model if name else None
        return {'primary': self.primary, 'fallback': self.fallback}.get(name)

    def choose(self, gid):
        """the model for this goal's next turn; None for a free-time stretch no brain may serve (free_choice: it is
        ended, and the turn is not run)"""
        if self.rt.free.is_free(gid):
            name = self._free_choice(primary_down=self.fallback is not None and not self._primary_up())
            if name is None:
                self.rt.free.end_off_plan(gid)
                return None
            if self.fallback is None:
                return self.rt.model
            use = name
        elif self.fallback is None:
            return self.rt.model            # one brain: whatever the runtime holds, untouched
        else:
            use = 'primary' if self._primary_up() or not self._allowed(gid) else 'fallback'
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
