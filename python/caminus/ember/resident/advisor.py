"""The advisor: a stronger model the resident may consult, on its human's key and within its human's limits.

The resident calls the `advisor` tool with a question when it is stuck, before a risky or irreversible step, or
before declaring a hard goal done. The runtime sends the advisor a focused brief (the goal, its done check, the
ledger, the newest part of the transcript) and the question, with no tools; the advice is the tool result. The
resident knows the advisor exists and what it has left: the tool's description says when to use it, and the
per-turn context carries the calls and dollars remaining.

Configuration, in config.json under "advisor" (absent: no advisor; the tool answers that none is configured):
    model          the advisor model, e.g. claude-opus-5-5 or claude-sonnet-5-5
    endpoint       the Messages API root; point it at the owner's own key relay (its own key file, model allowlist
                   and daily cap) so the advisor's key and budget stay separate from the resident's
    api_key_env    the key variable, when the endpoint is not a local relay that adds the key itself
    budget_usd     1.00   dollars per day the runtime lets the advisor spend (priced per call)
    calls_per_goal 5      advice calls one goal may make
    effort         medium Messages API effort for the advisor (its thinking stays adaptive: the current Opus and
                          Sonnet do not accept thinking off)
    max_tokens     4096   the advice length budget, thinking included
    fallbacks      default Anthropic's server-side refusal fallback: a safety classifier can decline legitimate work on
                          one's own machine (the cyber category trips easily), and the server then re-runs the request
                          on a fallback model in the same call; false/null turns it off

Every failure is recorded in failures.jsonl (who 'advisor'): an error, a hard refusal (stop_reason refusal, with its
category), a soft refusal (advice that declines in words: refusal.py, the Qwen Code advisor's detector), and a call that
a server-side fallback rescued. Refusal records keep the brief, so the call can be rerun against other models.

The eyes (Eyes, below) are the same kind of helper for a brain that can't see: when it reads an image, the runtime sends
that one image, with the agent's question, to a vision model on the owner's relay and returns the description as the
read's result. They keep their own ledger (eyes.json) and daily budget, priced per call as the advisor's are.
"""
import datetime
import json
import os
import urllib.request

from . import catalog
from .model import Model, ModelError, ModelRefused, price, remember_prices
from .refusal import refusal_kind
from .store import atomic_json, log_failure

DEFAULTS = dict(budget_usd=1.0, calls_per_goal=5, effort='medium', max_tokens=4096, fallbacks='default')

BRIEF = ('You are the advisor to an autonomous agent that works in its own Unix account on its human\'s Linux '
         'machine. It consults you when it is stuck, before a risky step, or before declaring a hard goal done. You '
         'see a brief of its goal and recent work, and its question. Give concrete, prioritized advice: what to do '
         'next, what to check, what to avoid, and why. Point out an assumption it has not verified. Be brief; it pays '
         'for every token. You cannot run anything yourself.')


class Advisor:
    LEDGER = 'advisor.json'

    def __init__(self, runtime):
        self.rt = runtime
        self.path = runtime.home / self.LEDGER

    @property
    def cfg(self):
        raw = self.rt.config.get('advisor')
        if not raw or not raw.get('model') or not raw.get('endpoint'):
            return None
        return {**DEFAULTS, **raw}

    def _state(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def spent_today(self):
        return float(self._state().get(datetime.date.today().isoformat(), {}).get('usd', 0.0))

    def calls(self, gid):
        # refused and failed calls count too: each may be billed, and a refusal must not invite endless retries
        # (counted in the database: events() pages from the first 200, so past them no call was counted)
        return self.rt.store.db.execute(
            "SELECT count(*) FROM events WHERE goal=? AND kind IN ('advice','advice_refused','advice_failed')",
            (gid,)).fetchone()[0]

    def status_line(self, gid):
        """the line the per-turn context carries, so the resident knows what it can still ask"""
        cfg = self.cfg
        if cfg is None:
            return ''
        left_calls = max(0, int(cfg['calls_per_goal']) - self.calls(gid)) if gid else int(cfg['calls_per_goal'])
        left_usd = max(0.0, float(cfg['budget_usd']) - self.spent_today())
        return ('ADVISOR: %s is available through the advisor tool: %d call%s left for this goal, $%.2f left today.'
                % (cfg['model'], left_calls, '' if left_calls == 1 else 's', left_usd))

    def ask(self, gid, question, context=''):
        cfg = self.cfg
        if cfg is None:
            raise ValueError('no advisor is configured for this resident; your human can add one '
                             '(caminus ember agent advisor NAME --model ... --endpoint ...)')
        question = (question or '').strip()
        if not question:
            raise ValueError('ask the advisor a specific question')
        if self.calls(gid) >= int(cfg['calls_per_goal']):
            raise ValueError('this goal has used its %d advisor calls; decide with what you have' % int(cfg['calls_per_goal']))
        if self.spent_today() >= float(cfg['budget_usd']):
            raise ValueError("today's advisor budget is spent; decide with what you have")
        if price(cfg['model'], {}) is None:
            raise ValueError('the advisor model %s has no known price, so its spend cannot be limited' % cfg['model'])
        model = Model(dict(model=cfg['model'], endpoint=cfg['endpoint'], api='anthropic',
                           context=int(cfg.get('context', 200000)), max_tokens=int(cfg['max_tokens']),
                           api_key_env=cfg.get('api_key_env', 'CAMINUS_ADVISOR_API_KEY'), effort=cfg['effort'],
                           thinking='adaptive', timeout=float(cfg.get('timeout', 300)),
                           fallbacks=cfg.get('fallbacks') or None, cache=False),
                      opener=getattr(self.rt, 'advisor_opener', None) or urllib.request.urlopen)
        model.counter = False                     # the brief is small; no count_tokens round trip per question
        brief = self._brief(gid, context) + '\n\nQUESTION FROM THE AGENT:\n' + question
        messages = [dict(role='system', content=BRIEF), dict(role='user', content=brief)]
        try:
            reply = model.complete(messages, max_tokens=int(cfg['max_tokens']))
        except ModelError as e:
            refused = isinstance(e, ModelRefused)
            self._spend(cfg, model)            # a refused or failed call can still have been billed
            log_failure(self.rt.home, dict(goal=gid, who='advisor', model=cfg['model'], error_type=type(e).__name__,
                                           error=str(e)[:2048], status=getattr(e, 'status', None),
                                           category=getattr(e, 'category', None), request_id=getattr(e, 'request_id', None),
                                           call=model.last_call, question=question[:1000],
                                           brief=brief if refused else None))
            self.rt.store.event('advice_refused' if refused else 'advice_failed',
                                dict(error=str(e)[:500], category=getattr(e, 'category', None)), gid)
            raise ValueError(('the advisor declined this question%s; decide with what you have, or ask it differently'
                              % (' (%s)' % e.category if getattr(e, 'category', None) else '')) if refused
                             else 'the advisor did not answer: %s' % e)
        usd = self._spend(cfg, model)
        call = model.last_call or {}
        soft = refusal_kind(reply['content'])
        if soft or call.get('fallback_ran'):
            log_failure(self.rt.home, dict(goal=gid, who='advisor', model=cfg['model'],
                                           error_type='soft refusal' if soft else 'fallback',
                                           error=('the advice declines (%s)' % soft) if soft else
                                           'declined by %s; answered by %s (server-side fallback)'
                                           % (', '.join(call.get('declined_by') or ['?']), call.get('served')),
                                           call=call, question=question[:1000], brief=brief,
                                           advice=reply['content'][:2000]))
        if soft:
            self.rt.store.event('advice_refused', dict(error='soft refusal: ' + soft, model=call.get('served')), gid)
            raise ValueError('the advisor declined this question (in its reply); decide with what you have, '
                             'or ask it differently')
        self.rt.store.event('advice', dict(question=question[:500], model=cfg['model'], served=call.get('served'),
                                           fallback=bool(call.get('fallback_ran')), usd=round(usd, 6)), gid)
        return dict(advice=reply['content'], model=call.get('served') or cfg['model'], usd=round(usd, 5),
                    calls_left=max(0, int(cfg['calls_per_goal']) - self.calls(gid)))

    def raw(self, gid, messages, max_tokens, who='judge'):
        """One call on the advisor's model for the harness itself (the loop judge): the advisor's key, daily budget,
        pricing and failure log, but not the agent's per-goal question count. Returns (text, model) or raises
        ValueError when there is no advisor, the budget is spent, or the call fails."""
        cfg = self.cfg
        if cfg is None or self.spent_today() >= float(cfg['budget_usd']) or price(cfg['model'], {}) is None:
            raise ValueError('no advisor budget')
        model = Model(dict(model=cfg['model'], endpoint=cfg['endpoint'], api='anthropic',
                           context=int(cfg.get('context', 200000)), max_tokens=max_tokens,
                           api_key_env=cfg.get('api_key_env', 'CAMINUS_ADVISOR_API_KEY'), effort='low',
                           thinking='adaptive', timeout=float(cfg.get('timeout', 300)),
                           fallbacks=cfg.get('fallbacks') or None, cache=False),
                      opener=getattr(self.rt, 'advisor_opener', None) or urllib.request.urlopen)
        model.counter = False
        try:
            reply = model.complete(messages, max_tokens=max_tokens)
        except ModelError as e:
            self._spend(cfg, model)
            log_failure(self.rt.home, dict(goal=gid, who=who, model=cfg['model'], error_type=type(e).__name__,
                                           error=str(e)[:2048], status=getattr(e, 'status', None),
                                           category=getattr(e, 'category', None), call=model.last_call))
            raise ValueError('the advisor model did not answer: %s' % e)
        self._spend(cfg, model)
        return reply.get('content') or '', (model.last_call or {}).get('served') or cfg['model']

    def _spend(self, cfg, model):
        usd = price(cfg['model'], model.last_usage, (model.last_call or {}).get('served')) or 0.0
        state = self._state()
        day = state.setdefault(datetime.date.today().isoformat(), dict(usd=0.0, calls=0))
        day['usd'] = round(day['usd'] + usd, 6)
        day['calls'] += 1
        atomic_json(self.path, state)
        return usd

    def _brief(self, gid, context):
        g = self.rt.store.goal(gid)
        parts = ['GOAL:\n' + g['prompt'][:4000]]
        if g.get('done_check'):
            parts.append('DONE CHECK:\n' + g['done_check'][:1000])
        if g.get('ledger'):
            parts.append('TODO:\n' + str(g['ledger'])[:2000])      # the todo list lives in the ledger column
        recent = []
        for m in g['messages'][-14:]:
            text = m.get('content') or ''
            for c in m.get('tool_calls') or []:
                text += '\n[call %s %s]' % (c['function']['name'], c['function']['arguments'][:400])
            recent.append('%s: %s' % (m['role'], text[:1500]))
        parts.append('RECENT WORK (newest last):\n' + '\n---\n'.join(recent))
        if context:
            parts.append('AGENT NOTE:\n' + context[:2000])
        return '\n\n'.join(parts)


# ---- eyes ---------------------------------------------------------------------------------------------------------

# Where no eyes block says otherwise, the first of these whose provider kind is configured through the owner's relay
# (catalog providers with relay true) describes the images: chosen by a lab comparison on screenshots and diagrams
# (docs/ember.md, "Eyes").
EYES_AUTO = (('mistral', 'ministral-8b-latest'), ('anthropic', 'claude-haiku-5-5'))
EYES_DEFAULTS = dict(budget_usd=0.25, max_tokens=1024, timeout=120)
EYES_QUESTION = 'Describe what matters in this image for the task.'
EYES_BRIEF = ('You are the eyes of an agent whose model cannot see images. It read an image file and asks you about it. '
              'Answer its question from what the image shows. Quote text in the image exactly, give numbers, labels and '
              'colours as shown, and say plainly when something is unreadable or not in the image; never guess. Be '
              'concise: no preamble.')
EYES_OFF = ('No eyes model is set up to describe images for you. Your human can turn eyes on: caminus ember agent eyes '
            'NAME --provider mistral --model ministral-8b-latest (a provider on their key relay; see docs/ember.md).')


class Eyes(Advisor):
    """The borrowed eyes: one vision call per image a non-vision brain reads (the advisor's ledger code, its own file).

    Configuration, in config.json under "eyes": false turns them off; {provider, model} picks a configured provider
    (`providers`, normally the owner's relay) and a vision model it serves; budget_usd (0.25 a day), max_tokens (the
    description's length, 1024) and timeout (120 s) are optional. No block: EYES_AUTO's first match, else off."""
    LEDGER = 'eyes.json'

    @property
    def cfg(self):
        raw = self.rt.config.get('eyes')
        if raw is False:
            return None
        provs = catalog.providers(self.rt.config)
        if isinstance(raw, dict) and raw.get('provider') and raw.get('model'):
            spec, picked = provs.get(raw['provider']), dict(raw)
        elif raw is None:
            spec = picked = None
            for kind, model in EYES_AUTO:
                name = next((n for n, s in provs.items() if s.get('kind') == kind and s.get('relay')), None)
                if name:
                    spec, picked = provs[name], dict(provider=name, model=model)
                    break
        else:
            return None
        if not spec or not spec.get('endpoint'):
            return None
        return {**EYES_DEFAULTS, **picked, 'kind': spec.get('kind'), 'endpoint': spec['endpoint'],
                'relay': bool(spec.get('relay'))}

    def see(self, gid, result, question=None):
        """read's result for an image that the brain can't see: with the eyes model's description and
        image.described_by, or eyes_error saying why there is none (the read itself still answers), or, eyes off, a
        note saying how to turn them on."""
        cfg = self.cfg
        if cfg is None:
            return dict(result, note=EYES_OFF)
        try:
            text, who = self._describe(gid, cfg, result, (question or '').strip() or EYES_QUESTION)
        except ValueError as e:
            return dict(result, eyes_error=str(e))
        return dict(result, image=dict(result['image'], described_by=who), description=text)

    def _entry(self, cfg):
        """the eyes model's catalog entry (its price where the provider lists one, and whether it can see)"""
        cat = catalog.load(self.rt.home)
        entry = catalog.find(cat, cfg['model'], cfg['provider'])
        if entry is None:
            cat = catalog.refresh(self.rt.home, self.rt.config, only=cfg['provider'],
                                  opener=getattr(self.rt, 'eyes_opener', None) or urllib.request.urlopen)
            entry = catalog.find(cat, cfg['model'], cfg['provider'])
        return entry

    def _describe(self, gid, cfg, result, question):
        if self.spent_today() >= float(cfg['budget_usd']):
            raise ValueError("today's eyes budget ($%.2f) is spent; ask your human, or use the image tools in the "
                             'terminal' % float(cfg['budget_usd']))
        entry = self._entry(cfg)
        if entry is None:
            raise ValueError('the eyes model %s is not in the %s catalog' % (cfg['model'], cfg['provider']))
        if not entry.get('vision'):
            raise ValueError("the eyes model %s can't see images (its provider says so): your human can pick one that "
                             'can: caminus ember agent eyes NAME --provider P --model M' % cfg['model'])
        if entry.get('prices'):
            remember_prices(cfg['model'], entry['prices'])
        if price(cfg['model'], {}) is None:
            raise ValueError('the eyes model %s has no known price, so its spend cannot be limited' % cfg['model'])
        model = Model(dict(model=cfg['model'], endpoint=cfg['endpoint'], provider_kind=cfg['kind'],
                           api='anthropic' if cfg['kind'] == 'anthropic' else 'openai', relay=cfg['relay'],
                           context=int(entry.get('context') or 100000), max_tokens=int(cfg['max_tokens']), vision=True,
                           api_key_env=cfg.get('api_key_env', 'CAMINUS_EYES_API_KEY'), timeout=float(cfg['timeout']),
                           cache=False, name=self.rt.config.get('name')),
                      opener=getattr(self.rt, 'eyes_opener', None) or urllib.request.urlopen)
        model.counter = False                     # one image and a question: no count_tokens round trip
        img = dict(result['image'], path=result['path'])
        task = ((self.rt.store.goal(gid) or {}).get('prompt') or '').strip() if gid else ''
        ask = '%s%s is a %dx%d %s. The agent asks: %s' % (
            'The agent\'s task: %s\n\n' % task[:600] if task else '', os.path.basename(result['path']), img['width'],
            img['height'], str(img['format']).upper(), question[:2000])
        try:
            reply = model.complete([dict(role='system', content=EYES_BRIEF), dict(role='user', content=ask, image=img)],
                                   max_tokens=int(cfg['max_tokens']))
        except ModelError as e:
            self._spend(cfg, model)               # a failed call can still have been billed
            log_failure(self.rt.home, dict(goal=gid, who='eyes', model=cfg['model'], error_type=type(e).__name__,
                                           error=str(e)[:2048], status=getattr(e, 'status', None), call=model.last_call))
            why = eyes_refusal(cfg, e)
            self.rt.store.event('eyes_failed', dict(path=result['path'], model=cfg['model'], error=why[:500]), gid)
            raise ValueError(why)
        usd = self._spend(cfg, model)
        who = (model.last_call or {}).get('served') or cfg['model']
        self.rt.store.event('eyes', dict(path=result['path'], model=who, question=question[:500], usd=round(usd, 6)), gid)
        return reply.get('content') or '', who


def eyes_refusal(cfg, e):
    """What the agent hears when the eyes call fails, with the exact fix where a relay refuses the model. The Anthropic
    relay's model list is its --model flags, which install replaces (a relay of this version names the whole command;
    an older one only the model); the Mistral relay's is its price file, re-read on change (it takes no --model)."""
    text, status, model = str(e), getattr(e, 'status', None), cfg['model']
    if status == 402:
        return ("the %s relay's daily cap is reached, so no description today: ask your human, or use the image tools "
                'in the terminal' % cfg['provider'])
    if status == 403 and 'model not allowed' in text:
        cmd = text.split('to allow it: ', 1)[1].split(')', 1)[0].strip() if 'to allow it: ' in text else (
            'caminus ember relay install --provider anthropic --cap-usd <its daily cap> --model <each model it allows '
            'now> --model %s' % model)
        return ("your human's relay does not allow the eyes model %s. They can allow it (install replaces the relay's "
                'model list): %s' % (model, cmd))
    if status == 403 and 'no price for' in text:
        return ("your human's relay has no price for the eyes model %s, so it refuses it. They can add it to the relay's "
                'price file (~/.config/caminus/mistral-prices.json unless its --prices names another) as "%s": [input, '
                'output] in USD per million tokens; the relay re-reads the file, no restart' % (model, model))
    return 'the eyes model %s did not answer: %s' % (model, text[:500])
