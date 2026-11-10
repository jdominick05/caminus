"""The outside advisor: one request to Anthropic, Mistral or a local OpenAI-compatible server, plain-text advice back. Never raises: every failure is a
result with ok=False (the hook then adds nothing). The SDKs are imported only here, when a call is made."""
import json
import os
import re
import time
import urllib.error
import urllib.request

from . import config, refusal

SYSTEM = """You advise a smaller coding agent that runs locally in Qwen Code (a terminal coding harness with shell, file \
read/edit and search tools). You see the user's latest request and the agent's recent steps. You cannot run tools and the \
agent cannot reply to you.

Reply in plain text, at most 150 words, addressed to the agent:
- what is going wrong, in one or two sentences, citing the step that shows it;
- the next one to three concrete steps (commands, files, checks), most useful first.
Keep code to a few lines. Do not restate the request. If the agent is actually on track, reply with exactly ON_TRACK.

The user owns this session and has authorised the work; your job is to help the agent do what the user asked, not to \
judge whether the task should be done. Do not add legal or policy caveats or tell the agent to stop or switch tasks on \
those grounds. If you are not willing to help with this task, reply with exactly DECLINE (another reviewer will be asked).
If a standing goal is shown, the latest message may be a side request inside it: weigh both, and the goal's status."""

REVIEW = """You check a smaller coding agent's work before it tells the user it is done. You see the user's latest \
request and the agent's recent steps. You cannot run tools or read files: decide from what is shown, now. If the request is fully handled (the changes made and checked as asked), reply with \
exactly DONE. Otherwise reply in plain text, at most 100 words, addressed to the agent: what is missing or unverified, and \
the one or two steps that would close it. The user has authorised the work: judge completeness only, no policy \
caveats. If you are not willing to review this task, reply with exactly DECLINE."""

ON_TRACK_RE = re.compile(r'^[\s*_`>#\-]*ON[_ ]TRACK\b', re.I)
DONE_RE = re.compile(r'^[\s*_`>#\-]*DONE\b', re.I)

# server-side refusal fallbacks ("default" form): the models that accept it on the Claude API
ANTHROPIC_FALLBACK_MODELS = ('claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5', 'claude-fable-5-1')


def _result(provider, model, t0, **kw):
    return dict(dict(ok=False, text='', provider=provider, model=model, input_tokens=None, output_tokens=None,
                     latency_s=round(time.monotonic() - t0, 2), error=None, verdict=None), **kw)


def _anthropic(cfg, system, brief, t0):
    pc = cfg['anthropic']
    model = pc.get('model') or 'claude-opus-5-5'
    key = config.read_key(pc.get('key'))
    if not key:
        return _result('anthropic', model, t0, error='no API key (config anthropic.key)')
    import anthropic
    client = anthropic.Anthropic(api_key=key, timeout=float(cfg['timeout_s']), max_retries=0)
    params = dict(model=model, max_tokens=8000, system=system, messages=[{'role': 'user', 'content': brief}],
                  thinking={'type': 'adaptive'})
    if pc.get('effort'):
        params['output_config'] = {'effort': pc['effort']}
    # an older model may not take adaptive thinking or effort (a 400): one retry without them
    plain = {k: v for k, v in params.items() if k not in ('thinking', 'output_config')}
    r = None
    for attempt in (params, plain):
        try:
            if model in ANTHROPIC_FALLBACK_MODELS and pc.get('server_fallback', True):
                r = client.beta.messages.create(betas=['server-side-fallback-2026-07-01'], fallbacks='default', **attempt)
            else:
                r = client.messages.create(**attempt)
            break
        except anthropic.BadRequestError as e:
            if attempt is params:
                continue
            return _result('anthropic', model, t0, error=f'HTTP 400: {str(getattr(e, "message", e))[:200]}')
        except anthropic.APITimeoutError:
            return _result('anthropic', model, t0, error='timeout')
        except anthropic.RateLimitError:
            return _result('anthropic', model, t0, error='rate limited')
        except anthropic.APIStatusError as e:
            return _result('anthropic', model, t0, error=f'HTTP {e.status_code}: {str(getattr(e, "message", e))[:200]}')
        except anthropic.APIConnectionError:
            return _result('anthropic', model, t0, error='connection error')
    u = getattr(r, 'usage', None)
    tok = dict(input_tokens=getattr(u, 'input_tokens', None), output_tokens=getattr(u, 'output_tokens', None))
    # a server-side fallback that ran: a fallback_message entry in usage.iterations; a fallback block per model that declined
    tok['fallback_ran'] = any(getattr(e, 'type', None) == 'fallback_message' for e in (getattr(u, 'iterations', None) or []))
    tok['declined_by'] = [getattr(getattr(b, 'from_', None), 'model', None) for b in (r.content or []) if getattr(b, 'type', None) == 'fallback']
    if r.stop_reason == 'refusal':
        cat = getattr(getattr(r, 'stop_details', None), 'category', None)
        return _result('anthropic', getattr(r, 'model', model), t0, error=f'refusal ({cat})', stop_category=cat, **tok)
    text = '\n'.join(b.text for b in r.content if getattr(b, 'type', None) == 'text').strip()
    return _result('anthropic', getattr(r, 'model', model), t0, ok=bool(text), text=text,
                   error=None if text else f'empty reply (stop_reason {r.stop_reason})', **tok)


def _mistral(cfg, system, brief, t0):
    pc = cfg['mistral']
    model = pc.get('model') or ''
    if not model:
        return _result('mistral', model, t0, error='no model (config mistral.model)')
    key = config.read_key(pc.get('key'))
    if not key:
        return _result('mistral', model, t0, error='no API key (config mistral.key)')
    from mistralai.client import Mistral
    client = Mistral(api_key=key, timeout_ms=int(float(cfg['timeout_s']) * 1000))
    kw = dict(model=model, max_tokens=6000, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': brief}])
    if pc.get('effort'):
        kw['reasoning_effort'] = pc['effort']
    try:
        r = client.chat.complete(**kw)
    except Exception as e:   # (the SDK's error classes vary by version: any failure is a failed result)
        # a model without reasoning_effort answers 400 / 422: one retry without it
        if 'reasoning_effort' in kw and getattr(e, 'status_code', None) in (400, 422):
            kw.pop('reasoning_effort')
            try:
                r = client.chat.complete(**kw)
            except Exception as e2:
                return _result('mistral', model, t0, error=f'{type(e2).__name__}: {str(e2)[:200]}')
        else:
            return _result('mistral', model, t0, error=f'{type(e).__name__}: {str(e)[:200]}')
    c = r.choices[0].message.content if r and r.choices else ''
    if isinstance(c, list):   # text and thinking chunks: keep the text
        c = '\n'.join(getattr(ch, 'text', '') for ch in c if getattr(ch, 'type', '') == 'text')
    text = (c or '').strip()
    u = getattr(r, 'usage', None)
    return _result('mistral', model, t0, ok=bool(text), text=text, error=None if text else 'empty reply',
                   input_tokens=getattr(u, 'prompt_tokens', None), output_tokens=getattr(u, 'completion_tokens', None))


THINK_RE = re.compile(r'<think>.*?(</think>|$)', re.S)


def _local(cfg, system, brief, t0):
    """an OpenAI-compatible /chat/completions endpoint (standard library only); the reply's reasoning is dropped"""
    pc = cfg['local']
    url = (pc.get('url') or '').rstrip('/')
    model = pc.get('model') or ''
    if not url:
        return _result('local', model, t0, error='no URL (config local.url)')
    body = dict(messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': brief}],
                max_tokens=int(pc.get('max_tokens') or 8000), stream=False)
    if model:
        body['model'] = model
    headers = {'Content-Type': 'application/json'}
    key = config.read_key(pc.get('key'))
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(url + '/chat/completions', data=json.dumps(body).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=float(cfg['timeout_s'])) as f:
            r = json.loads(f.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as e:
        with e:
            return _result('local', model, t0, error=f'HTTP {e.code}: {e.read()[:200].decode("utf-8", "replace")}')
    except (TimeoutError, OSError) as e:   # (URLError, a refused connection, a timeout)
        return _result('local', model, t0, error='timeout' if 'timed out' in str(e) else f'connection error: {str(e)[:160]}')
    except ValueError:
        return _result('local', model, t0, error='not a JSON reply')
    ch = (r.get('choices') or [{}])[0]
    msg = ch.get('message') or {}
    text = msg.get('content') or ''
    if '</think>' in text:   # (a template that opens the reasoning in the prompt sends only its closing tag)
        text = text.rsplit('</think>', 1)[1]
    text = THINK_RE.sub('', text).strip()
    u = r.get('usage') or {}
    err = None
    if not text:
        err = 'empty reply' + (' (out of tokens while thinking: raise local.max_tokens)' if ch.get('finish_reason') == 'length' else '')
    return _result('local', r.get('model') or model, t0, ok=bool(text), text=text, error=err,
                   input_tokens=u.get('prompt_tokens'), output_tokens=u.get('completion_tokens'))


def _fake(cfg, system, brief, t0):
    """QWEN_ADVISOR_FAKE=<text>: a canned reply, no network (tests); 'REFUSE:<category>' is a hard refusal"""
    items = os.environ['QWEN_ADVISOR_FAKE'].split('||')   # one reply per attempt ('a||b': the first, then the next)
    t = items[min(int(cfg.get('_fake_index') or 0), len(items) - 1)]
    if t.startswith('REFUSE:'):
        return _result('fake', 'fake', t0, error=f'refusal ({t[7:]})', stop_category=t[7:])
    return _result('fake', 'fake', t0, ok=True, text=t, input_tokens=len(brief) // 4, output_tokens=20)


def _tried(t):
    return dict(provider=t['provider'], model=t.get('model'), error=t['error'], category=t.get('stop_category'),
                declined_by=t.get('declined_by'), soft_text=t.get('soft_text'))


FNS = dict(anthropic=_anthropic, mistral=_mistral, local=_local, fake=_fake)


def attempts(cfg):
    """the advisors to try, in order: the provider, the fallback provider, then each cfg['chain'] entry
    'provider:model[@effort]'; as (provider, model or None, effort or None, fake index)"""
    if os.environ.get('QWEN_ADVISOR_FAKE'):
        return [('fake', None, None, i) for i in range(len(os.environ['QWEN_ADVISOR_FAKE'].split('||')))]
    p = cfg.get('provider') or 'anthropic'
    out = [(p, None, None, 0)]
    fp = cfg.get('fallback_provider')
    if fp and fp != p:
        out.append((fp, None, None, 0))
    for spec in cfg.get('chain') or []:
        prov, _, rest = str(spec).strip().partition(':')
        model, _, effort = rest.partition('@')
        if prov and model:
            out.append((prov, model, effort or None, 0))
    return out


def ask(cfg, brief, review=False):
    """advice for this brief. Tries the advisors in attempts() order until one answers without refusing: an error, a
    timeout, a hard refusal (stop_reason) or a soft one (the reply declines) moves on to the next, within
    chain_budget_s overall. verdict 'advice', 'on_track' / 'done'; on total failure ok=False and the last error."""
    system = REVIEW if review else SYSTEM
    start = time.monotonic()
    deadline = start + float(cfg.get('chain_budget_s') or cfg['timeout_s'])
    tried, skipped = [], 0
    plan = attempts(cfg)
    for n, (prov, model, effort, idx) in enumerate(plan):
        t0 = time.monotonic()
        left = deadline - t0
        if tried and left < 8:
            skipped = len(plan) - n
            break
        own = (cfg.get(prov) or {}).get('timeout_s') if prov in config.PROVIDERS else None   # (a provider's own limit)
        c = dict(cfg, timeout_s=max(5.0, min(float(own or cfg['timeout_s']), left)), _fake_index=idx)
        if prov in config.PROVIDERS:
            c[prov] = dict(cfg.get(prov) or {})
            if model:
                c[prov]['model'] = model
            if effort is not None:
                c[prov]['effort'] = '' if effort == 'default' else effort
        fn = FNS.get(prov)
        if fn is None:
            tried.append(_result(prov, model or '', t0, error='unknown provider'))
            continue
        try:
            r = fn(c, system, brief, t0)
        except Exception as e:   # (an SDK missing or broken: never let it reach the hook)
            r = _result(prov, model or '', t0, error=f'{type(e).__name__}: {str(e)[:200]}')
        if r['ok'] and cfg.get('retry_on_soft_refusal', True) and refusal.soft_refusal(r['text']):
            r = dict(r, ok=False, error='soft refusal', soft_text=r['text'][:600], text='')
        tried.append(r)
        if r['ok']:
            # a reply that starts with the verdict word counts ("On track: the reruns are progress..."): the model judged
            # the agent fine, so nothing is injected
            r['verdict'] = ('done' if review else 'on_track') if (DONE_RE if review else ON_TRACK_RE).match(r['text']) else 'advice'
            r['full_chars'] = len(r['text'])
            r['text'] = r['text'][:int(cfg['max_advice_chars'])]
            r['tried'] = [_tried(t) for t in tried[:-1]]
            r['total_s'] = round(time.monotonic() - start, 2)
            return r
    last = dict(tried[-1])
    last['tried'] = [_tried(t) for t in tried[:-1]]
    last['total_s'] = round(time.monotonic() - start, 2)
    last['skipped'] = skipped
    return last
