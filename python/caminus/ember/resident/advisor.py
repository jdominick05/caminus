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
    kind           absent (Anthropic), or opencode-go: a Messages-native model of the owner's OpenCode Go plan through
                   its relay (endpoint: the plan's root, or its /anthropic address); each call then names a session of
                   its own and Ember (catalog.OPENCODE), with relay true and the model's list price (prices) and
                   levels (efforts) from the plan's model list, stored by `agent advisor --kind opencode-go`
    api_key_env    the key variable, when the endpoint is not a local relay that adds the key itself
    budget_usd     1.00   dollars per day the runtime lets the advisor spend (priced per call; a plan call at no less
                          than its list price, also inside the plan's allowance)
    calls_per_goal 5      advice calls one goal may make
    effort         medium Messages API effort for the advisor (its thinking stays adaptive: the current Opus and
                          Sonnet do not accept thinking off)
    max_tokens     4096   the advice length budget, thinking included
    fallbacks      default Anthropic's server-side refusal fallback: a safety classifier can decline legitimate work on
                          one's own machine (the cyber category trips easily), and the server then re-runs the request
                          on a fallback model in the same call; false/null turns it off

A call that cannot reach the advisor model at all (the owner's key relay does not allow the model, is not running or
has spent its daily cap; a direct endpoint without its key variable) raises AdvisorUnavailable: one line with the reason
and, where it is known, the command the owner runs (.fix: a command or nothing, never prose). Nothing was billed, so it
does not count against the goal's calls. The owner's side (agent_host) rebuilds the command from the installed relay
and shows only what it built: every word of this one comes from the agent's account.

Every failure is recorded in failures.jsonl (who 'advisor'): an error, a hard refusal (stop_reason refusal, with its
category), a soft refusal (advice that declines in words: refusal.py, the Qwen Code advisor's detector), and a call that
a server-side fallback rescued. Refusal records keep the brief, so the call can be rerun against other models.

The eyes (Eyes, below) are the same kind of helper for a brain that can't see: when it reads an image, the runtime sends
that one image, with the agent's question, to a vision model on the owner's relay and returns the description as the
read's result. They keep their own ledger (eyes.json) and daily budget, priced per call as the advisor's are.
"""
import datetime
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from . import catalog
from .model import Model, ModelError, ModelRefused, price, remember_prices, valid_prices
from .refusal import refusal_kind
from .store import atomic_json, log_failure

DEFAULTS = dict(budget_usd=1.0, calls_per_goal=5, effort='medium', max_tokens=4096, fallbacks='default')

BRIEF = ('You are the advisor to an autonomous agent that works in its own Unix account on its human\'s Linux '
         'machine. It consults you when it is stuck, before a risky step, or before declaring a hard goal done. You '
         'see a brief of its goal and recent work, and its question. Give concrete, prioritized advice: what to do '
         'next, what to check, what to avoid, and why. Point out an assumption it has not verified. Be brief; it pays '
         'for every token. You cannot run anything yourself.')


class AdvisorUnavailable(ValueError):
    """The advisor cannot be called until the owner changes the setup. str(): the reason and the fix in one line, for
    the agent's tool result and the CLI; .reason and .fix (the command, or '') apart, for the panel."""
    def __init__(self, reason, fix='', kind='', model='', endpoint=''):
        super().__init__('the advisor cannot be called: %s%s' % (reason, ('. Fix it on the owner\'s side: ' + fix) if fix else ''))
        self.reason, self.fix, self.kind, self.model, self.endpoint = reason, fix, kind, model, endpoint


# a relay command as the relay itself prints it: flags and plain values, nothing a shell would read as another command
SAFE_COMMAND = re.compile(r'caminus ember relay install(?: --[a-z-]+ [\w.:/@-]+)+')


def unavailable(e, cfg, name='NAME'):
    """The AdvisorUnavailable for a failure that never reached the model, or None (an outage, a refusal by the model,
    anything the provider answered). Recognises the relay's refusals in the wording of every relay version."""
    model, endpoint = cfg['model'], cfg['endpoint']
    text, status = str(e), getattr(e, 'status', None)
    host = urllib.parse.urlsplit(endpoint).hostname
    local = host in ('127.0.0.1', 'localhost', '::1')
    if status == 403 and 'relay: model not allowed' in text:
        m = SAFE_COMMAND.search(text.split('to allow it too: ', 1)[1]) if 'to allow it too: ' in text else None
        fix = m.group(0) if m else ''
        why = '' if fix else (' (allow it with caminus ember relay install, naming every --model it should serve, '
                              'its current ones too; caminus ember relay status lists them)')
        return AdvisorUnavailable('the key relay at %s does not allow %s%s' % (endpoint, model, why), fix,
                                  'relay_model', model, endpoint)
    if status == 402 and 'relay:' in text:
        return AdvisorUnavailable('the key relay at %s has spent its daily cap; it resets at midnight, or the owner raises '
                                  'it with caminus ember relay install --provider %s --cap-usd%s' % (
                                      (endpoint, 'opencode-go', '') if cfg.get('kind') in catalog.OPENCODE else
                                      (endpoint, 'anthropic', ' (naming every --model again)')),
                                  '', 'relay_cap', model, endpoint)
    # the OpenCode Go relay's refusals (and its gateway's, reached without the relay): a call without a session id (an
    # advisor not set up as a plan model), a model Ember does not offer from the plan, a path it does not serve
    if status == 400 and ('relay: no x-opencode-session' in text or 'MissingSessionID' in text):
        return AdvisorUnavailable('%s is an OpenCode Go plan, which serves only calls that name a session: set the advisor '
                                  'up as a plan model (--kind opencode-go)' % endpoint,
                                  'caminus ember agent advisor %s --kind opencode-go' % name, 'opencode_session', model, endpoint)
    if status == 403 and 'relay: no price for' in text and 'do not train on your prompts' in text:
        return AdvisorUnavailable('the OpenCode Go relay at %s does not offer %s (Ember allows the plan\'s chat models that '
                                  'call tools and do not train on your prompts; caminus ember agent models NAME --refresh '
                                  'lists them)' % (endpoint, model), '', 'opencode_model', model, endpoint)
    if status == 404 and 'relay: path not allowed' in text:
        return AdvisorUnavailable('the key relay at %s does not serve the Messages API there: the advisor\'s endpoint is '
                                  'the relay\'s Messages root (http://127.0.0.1:8765/v1 for Anthropic; an OpenCode Go plan\'s '
                                  'root with --kind opencode-go)' % endpoint, '', 'relay_path', model, endpoint)
    reason = getattr(e, '__cause__', None)
    reason = getattr(reason, 'reason', reason)
    if local and (isinstance(reason, ConnectionRefusedError) or 'Connection refused' in text):
        return AdvisorUnavailable('nothing answers at %s: the owner\'s key relay is not running there (caminus ember '
                                  'relay status shows the installed relays)' % endpoint, '', 'relay_down', model, endpoint)
    if status == 401 and 'is not set in the agent environment' in text:
        return AdvisorUnavailable('%s is not a local key relay and the agent has no key for it' % endpoint,
                                  'caminus ember agent advisor %s --endpoint http://127.0.0.1:8765/v1' % name,
                                  'no_key', model, endpoint)
    return None


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
            # (advice_unavailable is not counted: the call never left the machine, and five of them would hide the fix
            #  behind 'this goal has used its advisor calls')
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
        self._off_plan(cfg, gid)
        if self.calls(gid) >= int(cfg['calls_per_goal']):
            raise ValueError('this goal has used its %d advisor calls; decide with what you have' % int(cfg['calls_per_goal']))
        if self.spent_today() >= float(cfg['budget_usd']):
            raise ValueError("today's advisor budget is spent; decide with what you have")
        if price(cfg['model'], {}) is None:
            raise ValueError('the advisor model %s has no known price, so its spend cannot be limited' % cfg['model'])
        model = self._model(cfg, int(cfg['max_tokens']), cfg['effort'])
        brief = self._brief(gid, context) + '\n\nQUESTION FROM THE AGENT:\n' + question
        messages = [dict(role='system', content=BRIEF), dict(role='user', content=brief)]
        try:
            reply = model.complete(messages, max_tokens=int(cfg['max_tokens']))
        except ModelError as e:
            refused = isinstance(e, ModelRefused)
            setup = None if refused else unavailable(e, cfg, self.rt.config.get('name') or 'NAME')
            if setup is not None:
                # refused before the model (by the relay) or never sent: nothing billed, the owner has a fix to run
                log_failure(self.rt.home, dict(goal=gid, who='advisor', model=cfg['model'], error_type='unavailable',
                                               error=str(e)[:2048], status=getattr(e, 'status', None),
                                               fix=setup.fix, question=question[:1000]))
                self.rt.store.event('advice_unavailable', dict(error=setup.reason[:500], fix=setup.fix[:500],
                                                               kind=setup.kind), gid)
                raise setup
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
        self._off_plan(cfg, gid)
        model = self._model(cfg, max_tokens, 'low')
        try:
            reply = model.complete(messages, max_tokens=max_tokens)
        except ModelError as e:
            setup = unavailable(e, cfg, self.rt.config.get('name') or 'NAME')
            if setup is not None:
                log_failure(self.rt.home, dict(goal=gid, who=who, model=cfg['model'], error_type='unavailable',
                                               error=str(e)[:2048], status=getattr(e, 'status', None), fix=setup.fix))
                raise setup
            self._spend(cfg, model)
            log_failure(self.rt.home, dict(goal=gid, who=who, model=cfg['model'], error_type=type(e).__name__,
                                           error=str(e)[:2048], status=getattr(e, 'status', None),
                                           category=getattr(e, 'category', None), call=model.last_call))
            raise ValueError('the advisor model did not answer: %s' % e)
        self._spend(cfg, model)
        return reply.get('content') or '', (model.last_call or {}).get('served') or cfg['model']

    def _off_plan(self, cfg, gid):
        """Refuse a call for a free-time goal while the advisor is on an OpenCode plan that free time may not use
        (brain.plan_allows, free_choice's rule), before anything is counted or sent."""
        from .brain import plan_allows
        if gid and self.rt.free.is_free(gid) and \
                not plan_allows(dict(provider_kind=cfg.get('kind')), self.rt.free.cfg.get('on_plan')):
            raise ValueError('free time stays off the OpenCode plan the advisor is on; decide with what you have. Your '
                             'human can let free time use the plan: caminus ember agent heartbeat %s --on-plan yes'
                             % (self.rt.config.get('name') or 'NAME'))

    def _model(self, cfg, max_tokens, effort):
        """The advisor model's transport, on the Messages API: through an Anthropic relay, or a Messages-native model of
        an OpenCode Go plan (kind opencode-go). It is sent no session: each call is a conversation of its own, so the
        transport names a fresh one (Model.headers)."""
        plan = cfg.get('kind') in catalog.OPENCODE
        levels = dict(efforts=cfg['efforts']) if 'efforts' in cfg else {}
        model = Model(dict(model=cfg['model'], **side_transport(dict(cfg, kind=cfg.get('kind') or 'anthropic'),
                                                                dict(protocol='anthropic', **levels)),
                           context=int(cfg.get('context', 200000)), max_tokens=max_tokens,
                           api_key_env=cfg.get('api_key_env', 'CAMINUS_ADVISOR_API_KEY'), effort=effort,
                           thinking='adaptive', timeout=float(cfg.get('timeout', 300)),
                           # (Anthropic's server-side fallback is its own API's: a plan's gateway is not sent it)
                           fallbacks=None if plan else cfg.get('fallbacks') or None, cache=False),
                      opener=getattr(self.rt, 'advisor_opener', None) or urllib.request.urlopen)
        model.counter = False                     # the brief is small; no count_tokens round trip per question
        return model

    def _spend(self, cfg, model):
        usage, served = model.last_usage or {}, (model.last_call or {}).get('served')
        plan = cfg.get('kind') in catalog.OPENCODE
        # (a plan has no Anthropic fallback, so a reply naming another model is still the requested one: priced by its
        #  own list price, never at price()'s highest-known rates for an unknown serving model)
        usd = price(cfg['model'], usage, None if plan else served) or 0.0
        if plan:
            # a plan call counts against the advisor's budget at no less than its list price: inside the Go allowance
            # the plan bills $0, and the owner's daily limit still holds (usage without opencode: the list price)
            usd = max(usd, price(cfg['model'], {k: v for k, v in usage.items() if k != 'opencode'}) or 0.0)
        usd = usd if math.isfinite(usd) and usd > 0 else 0.0     # (the ledger never goes NaN or below zero)
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

# What `agent eyes --auto` picks: the first of these whose provider kind is configured through the owner's relay
# (catalog providers with relay true); chosen by a lab comparison on screenshots and diagrams (docs/ember.md, "Eyes").
EYES_AUTO = (('mistral', 'ministral-8b-latest'), ('anthropic', 'claude-haiku-5-5'))
EYES_DEFAULTS = dict(budget_usd=0.25, max_tokens=1024, timeout=120)
EYES_QUESTION = 'Describe what matters in this image for the task.'
EYES_BRIEF = ('You are the eyes of an agent whose model cannot see images. It read an image file and asks you about it. '
              'Answer its question from what the image shows. Quote text in the image exactly, give numbers, labels and '
              'colours as shown, and say plainly when something is unreadable or not in the image; never guess. Be '
              'concise: no preamble.')
EYES_OFF = ('No eyes model is set up to describe images for you. Your human can turn eyes on: caminus ember agent eyes '
            'NAME --provider P --model M (a vision model on their key relay), or --auto (ministral-8b-latest on a '
            'Mistral relay, else Claude Haiku 5.5); see docs/ember.md, "Eyes".')


class Eyes(Advisor):
    """The borrowed eyes: one vision call per image a non-vision brain reads (the advisor's ledger code, its own file).

    Opt-in only. Configuration, in config.json under "eyes": {provider, model} picks a configured provider
    (`providers`, normally the owner's relay) and a vision model it serves; budget_usd (0.25 a day), max_tokens (the
    description's length, 1024) and timeout (120 s) are optional. No block, or false: off."""
    LEDGER = 'eyes.json'

    @property
    def cfg(self):
        raw = self.rt.config.get('eyes')
        if not (isinstance(raw, dict) and raw.get('provider') and raw.get('model')):
            return None                           # opt-in only: no block (or false) is off
        spec = catalog.providers(self.rt.config).get(raw['provider'])
        if not spec or not spec.get('endpoint'):
            return None
        return {**EYES_DEFAULTS, **raw, 'kind': spec.get('kind'), 'endpoint': spec['endpoint'],
                'relay': bool(spec.get('relay'))}

    @staticmethod
    def auto_pick(config):
        """`agent eyes --auto`: {provider, model} for the first of EYES_AUTO whose provider kind is configured through
        the relay, or None. Only the human's command writes it; a session never switches eyes on by itself."""
        provs = catalog.providers(config)
        for kind, model in EYES_AUTO:
            name = next((n for n, s in provs.items() if s.get('kind') == kind and s.get('relay')), None)
            if name:
                return dict(provider=name, model=model)
        return None

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
        model = Model(dict(model=cfg['model'], **side_transport(cfg, entry),
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


def plan_fields(model, endpoint, opener=urllib.request.urlopen):
    """`agent advisor --kind opencode-go`: what the advisor's config keeps for a model of the owner's OpenCode Go plan,
    from the plan's model list (asked at endpoint, normally the owner's relay): its Messages address, relay true, its list
    price (the spend meter's, where a reply does not say what it charged) and its levels. ValueError for a model the plan
    does not offer to Ember, or one it serves on Chat Completions: the advisor speaks the Messages API only."""
    spec = dict(kind=catalog.OPENCODE[0], endpoint=catalog.plan_root(endpoint))
    try:
        entry = next((m for m in catalog.discover('advisor', spec, opener) if m['id'] == model), None)
    except (OSError, ValueError, KeyError) as e:
        raise ValueError("cannot read the plan's model list at %s/v1/models (%s): is its relay running? caminus ember "
                         'relay status shows the relays' % (spec['endpoint'], str(e)[:200]))
    if entry is None:
        raise ValueError('the OpenCode Go plan at %s does not offer %s (Ember allows its chat models that call tools and '
                         'do not train on your prompts)' % (spec['endpoint'], model))
    if entry.get('protocol') != 'anthropic':
        raise ValueError('%s is served on Chat Completions, and the advisor speaks the Messages API: pick a Messages-native '
                         'plan model, e.g. claude-haiku-5-5' % model)
    if entry.get('prices') and not valid_prices(entry['prices']):
        raise ValueError("the plan lists %s at a price that is not a finite amount of at least 0 (%s), so its spend "
                         'cannot be limited' % (model, json.dumps(entry['prices'])[:200]))
    return dict(endpoint=catalog.transport(spec, entry)[0], relay=True, efforts=list(entry.get('efforts') or []),
                **(dict(prices=entry['prices']) if entry.get('prices') else {}))


def side_transport(cfg, entry):
    """A side call's transport fields for its provider (cfg: kind, endpoint, relay): the endpoint and api for the
    model's catalog entry (catalog.transport: an OpenCode model's own API under the plan's root), the kind, which an
    OpenCode plan's session id and user agent follow, and its levels (an OpenCode model's own, so one that lists none is
    sent none)."""
    endpoint, api = catalog.transport(dict(kind=cfg['kind'], endpoint=cfg['endpoint']), entry)
    out = dict(endpoint=endpoint, api=api, provider_kind=cfg['kind'], relay=bool(cfg.get('relay')))
    if cfg['kind'] in catalog.OPENCODE and 'efforts' in entry:
        out['efforts'] = list(entry['efforts'] or [])
    return out


def eyes_refusal(cfg, e):
    """What the agent hears when the eyes call fails, with the exact fix where a relay refuses the model. The Anthropic
    relay's refusals (model not allowed, cap spent, not running) are read by unavailable(), as the advisor's are: the
    relay's own command where it names one, checked against SAFE_COMMAND. The Mistral relay's model list is its price
    file, re-read on change (it takes no --model)."""
    text, status, model = str(e), getattr(e, 'status', None), cfg['model']
    u = unavailable(e, dict(model=model, endpoint=cfg.get('endpoint', '')))
    if u is not None and u.kind in ('relay_model', 'relay_cap', 'relay_down'):
        return ('the eyes model %s cannot be called: %s%s; until then, ask your human or use the image tools in the '
                'terminal' % (model, u.reason, ('. Your human can fix it: ' + u.fix) if u.fix else ''))
    if status == 403 and 'no price for' in text:
        return ("your human's relay has no price for the eyes model %s, so it refuses it. They can add it to the relay's "
                'price file (~/.config/caminus/mistral-prices.json unless its --prices names another) as "%s": [input, '
                'output] in USD per million tokens; the relay re-reads the file, no restart' % (model, model))
    return 'the eyes model %s did not answer: %s' % (model, text[:500])
