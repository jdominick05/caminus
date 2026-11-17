"""The model catalog: every model the agent's providers serve, with what each can do, found by asking them.

A provider is one endpoint the resident can reach (directly, or through the owner's key relay, which adds the key):

- local      the Caminus server (OpenAI-compatible): GET /v1/models, the window and vision from GET /props (n_ctx,
             modalities.vision)
- anthropic  the Messages API: GET /v1/models gives max_input_tokens (the window), max_tokens (the output cap) and a
             capabilities tree (thinking types, effort levels). Prices are not in it: the transport's price table.
- deepinfra  OpenAI-compatible at https://api.deepinfra.com/v1/openai: GET {base}/models gives metadata.context_length,
             max_tokens, pricing ($ per 1M input / output / cache-read tokens) and tags (chat, vision, reasoning,
             reasoning_effort, prompt_cache). Reasoning is set with reasoning_effort (none, low, medium, high).
- mistral    Mistral AI Studio, OpenAI-compatible at https://api.mistral.ai/v1 (through the owner's relay): GET
             {base}/models gives capabilities (completion_chat, function_calling, vision), max_context_length and
             aliases, and no prices: the owner's relay adds each priced model's `pricing` from its price file (the
             agent's spend meter needs them). No output cap.
- openai     any other OpenAI-compatible server: GET {base}/models, ids only.

Every model is normalized to one entry: id, provider, context, max_output, efforts (the levels /effort offers, empty
when there is no control), thinking ('adaptive', 'toggle' or ''), vision, prices ($/1M: input, output, cache_read) or
None. The catalog is cached in the agent home (models.json) and refreshed on demand; a provider that cannot be reached
keeps its last entries, marked stale. What a provider does not say stays unknown (None), never guessed.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

KINDS = ('local', 'anthropic', 'deepinfra', 'mistral', 'openai')
TIMEOUT = 20
ANTHROPIC_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
DEEPINFRA_EFFORTS = ('none', 'low', 'medium', 'high')
# DeepInfra's metadata.max_tokens repeats the context length (checked 11-15: 1048576 on 1M models); its docs give a hard
# output cap of 16384 tokens for most models, so that is the cap where the metadata says more.
DEEPINFRA_OUTPUT_CAP = 16384


def _get(url, headers, opener, timeout=TIMEOUT, limit=8 * 1024 * 1024):
    req = urllib.request.Request(url, None, {'Accept': 'application/json', **headers})
    with opener(req, timeout=timeout) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'reply larger than {limit // 1024} KiB')
    return json.loads(raw)


def _root(endpoint):
    e = endpoint.rstrip('/')
    return e[:-3] if e.endswith('/v1') else e


def anthropic_entries(data, provider):
    out = []
    for m in data.get('data') or []:
        caps = m.get('capabilities') or {}
        effort = caps.get('effort') or {}
        efforts = [lv for lv in ANTHROPIC_EFFORTS if (effort.get(lv) or {}).get('supported')] \
            if effort.get('supported') else []
        types = (caps.get('thinking') or {}).get('types') or {}
        thinking = 'adaptive' if (types.get('adaptive') or {}).get('supported') else \
            'toggle' if (types.get('enabled') or {}).get('supported') else ''
        out.append(dict(id=m['id'], name=m.get('display_name') or m['id'], provider=provider,
                        context=m.get('max_input_tokens'), max_output=m.get('max_tokens'), efforts=efforts,
                        thinking=thinking, vision=bool((caps.get('image_input') or {}).get('supported')),
                        prices=None))
    return out


def deepinfra_entries(data, provider):
    out = []
    for m in data.get('data') or []:
        meta = m.get('metadata') or {}
        tags = set(meta.get('tags') or [])
        if meta and 'chat' not in tags and 'vlm' not in tags:
            continue                                     # embeddings, images, speech: not a brain
        p = meta.get('pricing') or {}
        r = lambda v: None if v is None else round(float(v), 6)
        prices = dict(input=r(p['input_tokens']), output=r(p['output_tokens']),
                      cache_read=r(p.get('cache_read_tokens'))) if 'input_tokens' in p and 'output_tokens' in p else None
        cap = meta.get('max_tokens')
        cap = min(cap, DEEPINFRA_OUTPUT_CAP) if isinstance(cap, int) else DEEPINFRA_OUTPUT_CAP
        out.append(dict(id=m['id'], name=m['id'], provider=provider, context=meta.get('context_length'),
                        max_output=cap,
                        efforts=list(DEEPINFRA_EFFORTS) if 'reasoning_effort' in tags else [],
                        thinking='toggle' if 'reasoning' in tags or 'reasoning_effort' in tags else '',
                        vision='vision' in tags, prices=prices))
    return out


# The reasoning levels each Mistral model takes (capabilities.reasoning only says whether it reasons at all): the cards
# and docs.mistral.ai/capabilities/reasoning, 2026-11. A reasoning model not listed takes low / medium / high.
MISTRAL_EFFORTS = (
    (r'^mistral-large-(4|latest)', ['low', 'medium', 'high']),
    (r'^(mistral-medium-|magistral-medium-)', ['none', 'high']),
    (r'^zai-glm-5-3', ['low', 'high', 'max']),
)


def mistral_efforts(model_id, caps):
    if not caps.get('reasoning'):
        return []
    import re
    for pat, levels in MISTRAL_EFFORTS:
        if re.search(pat, model_id):
            return list(levels)
    return ['low', 'medium', 'high']


def mistral_entries(data, provider):
    """Chat models that call tools (the agent's loop needs both); an alias of another listed id is left out, kept in
    its entry's aliases (find takes it: ministral-8b-latest is listed as ministral-8b-2512)."""
    out, seen = [], set()
    for m in data.get('data') or []:
        caps = m.get('capabilities') or {}
        if not (caps.get('completion_chat') and caps.get('function_calling')) or m.get('deprecation'):
            continue
        if m['id'] in seen:
            continue
        seen.update([m['id'], *(m.get('aliases') or [])])
        p = m.get('pricing') or {}          # (added by the owner's relay from its price file; Mistral lists none)
        prices = dict(input=float(p['input']), output=float(p['output']),
                      cache_read=float(p.get('cache_read', p['input']))) if 'input' in p and 'output' in p else None
        out.append(dict(id=m['id'], name=m.get('name') or m['id'], provider=provider, context=m.get('max_context_length'),
                        max_output=None, efforts=mistral_efforts(m['id'], caps), thinking='', vision=bool(caps.get('vision')),
                        prices=prices, aliases=[a for a in m.get('aliases') or [] if a != m['id']]))
    return out


def openai_entries(data, provider):
    return [dict(id=m['id'], name=m['id'], provider=provider, context=None, max_output=None, efforts=[],
                 thinking='', vision=False, prices=None) for m in data.get('data') or []]


def local_entries(data, provider, context=None, vision=False):
    """A local Caminus server's models, one entry per base model: its -low / -xhigh / -nothink ids are that model's
    effort levels (effort.py), not models of their own. efforts lists the scale (every level runs as the nearest the
    model has); variants maps a level to the id that runs it; max_output is the card's (the reply limit: up to 64K).
    vision is /props' modalities.vision (llama-server's key; caminus serve reports false: K3 has no vision path)."""
    from . import effort
    ids = [m['id'] for m in data.get('data') or []]
    out = []
    for base, variants in effort.group_local(ids).items():
        cfg = dict(provider_kind='local', model=base, model_variants=variants)
        line = effort.line(cfg)
        out.append(dict(id=base, name=base, provider=provider, context=context,
                        max_output=effort.card(base).get('max_output'), efforts=list(effort.LEVELS) if line else [],
                        variants=variants, thinking='', vision=vision, prices=None))
    return out


def discover(name, spec, opener=urllib.request.urlopen, key=None):
    """One provider's entries. spec: {kind, endpoint}. key: an API key to send (none through a relay)."""
    kind, base = spec.get('kind', 'openai'), spec['endpoint'].rstrip('/')
    if kind not in KINDS + ('auto',):
        raise ValueError(f'unknown provider kind {kind!r}')
    if kind == 'anthropic':
        headers = {'anthropic-version': '2023-06-01', **({'x-api-key': key} if key else {})}
        data = _get(_root(base) + '/v1/models?limit=1000', headers, opener)
        return anthropic_entries(data, name)
    headers = {'Authorization': 'Bearer ' + key} if key else {}
    data = _get(base + '/models', headers, opener)
    if kind == 'auto':
        # (agent init, before any provider is configured: model_vision) the list's own shape says whose it is
        rows = [m for m in data.get('data') or [] if isinstance(m, dict)]
        kind = ('deepinfra' if any(isinstance(m.get('metadata'), dict) for m in rows) else
                'mistral' if any(isinstance(m.get('capabilities'), dict) for m in rows) else 'local')
    if kind == 'deepinfra':
        return deepinfra_entries(data, name)
    if kind == 'mistral':
        return mistral_entries(data, name)
    if kind == 'local':
        try:
            props = _get(_root(base) + '/props', headers, opener)
        except (OSError, ValueError, urllib.error.HTTPError):
            props = {}
        return local_entries(data, name, props.get('n_ctx'), bool((props.get('modalities') or {}).get('vision')))
    return openai_entries(data, name)


def providers(config):
    """The configured providers: config `providers` {name: {kind, endpoint}}, plus the agent's own endpoint."""
    out = dict(config.get('providers') or {})
    own = config.get('provider')
    if own and own not in out and config.get('endpoint'):
        out[own] = dict(kind=config.get('provider_kind') or ('anthropic' if config.get('api') == 'anthropic'
                                                             else 'openai'), endpoint=config['endpoint'])
    return out


def local_candidate(config):
    """Whether the agent's own brain may be a local Caminus server: provider_kind local, or a config from before
    providers (agent init, create --model) on a loopback endpoint that is neither marked relayed nor the Messages API.
    A candidate without provider_kind is local only when its /props says so (local_brain): the key relay listens on
    loopback too, and init writes no provider_kind or relay for it."""
    kind = config.get('provider_kind')
    if kind:
        return kind == 'local'
    if config.get('relay') or config.get('api') == 'anthropic' or not config.get('endpoint'):
        return False
    from .model import LOOPBACK
    return urllib.parse.urlsplit(config['endpoint']).hostname in LOOPBACK


LOCAL_READ = 512 * 1024   # a local server's /models (a few ids) and /props (its chat template among them): at most this


def caminus_props(endpoint, opener=urllib.request.urlopen, timeout=3):
    """A local server's /props when it names the engine as `caminus serve` does (engine: caminus), else None. Raises
    when the server cannot be reached or answers garbage, so a caller can tell "not Caminus" from "not asked"."""
    props = _get(_root(endpoint) + '/props', {}, opener, timeout, LOCAL_READ)
    return props if isinstance(props, dict) and props.get('engine') == 'caminus' else None


STATUS_VERSION = 1   # the /caminus/status layout read here (caminus serve's stats.STATUS_VERSION; the panel reads the same)


def decode_tps(status):
    """A Caminus server's decode speed from its /caminus/status: 1000 / decode.ms_per_token (the engine's mean over its
    recent passes), rounded to 0.1 tok/s; None for another engine, another status layout or no passes yet. The one
    reading of that figure: `caminus ps` / the servers list (cmd/manage.py) and Ember's live speed use it."""
    if not isinstance(status, dict) or status.get('engine') != 'caminus' or status.get('version') != STATUS_VERSION:
        return None
    ms = (status.get('decode') or {}).get('ms_per_token')
    if isinstance(ms, bool) or not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return round(1000.0 / ms, 1)


def local_brain(config, opener=urllib.request.urlopen, timeout=3):
    """Ask the agent's local server which model it serves and that model's effort ids: (base id, {level: id}), or None
    when it cannot be reached, is not a Caminus server (its /props engine; asked unless provider_kind is local) or
    names no such model. The configured id when the server lists it; an effort id (k3-coder-low) its base; else its
    only model (with several, its default: /props model_alias). The session switcher stores the catalog name (coder),
    and the server runs an id it does not list as its default model."""
    from . import effort
    base = config['endpoint'].rstrip('/')
    props = None
    try:
        if config.get('provider_kind') != 'local':
            props = caminus_props(base, opener, timeout)
            if props is None:
                return None
        groups = local_entries(_get(base + '/models', {}, opener, timeout, LOCAL_READ), '')
        groups = {m['id']: m['variants'] for m in groups}
        model = config.get('model') or ''
        if model not in groups:
            bases = [model[:-len(s)] for s in effort.SUFFIX.values() if model.endswith(s) and model[:-len(s)] in groups]
            if bases:
                model = bases[0]
            elif len(groups) == 1:
                model = next(iter(groups))
            else:
                props = props or _get(_root(base) + '/props', {}, opener, timeout, LOCAL_READ)
                model = props.get('model_alias') if props.get('model_alias') in groups else None
    except (OSError, ValueError, AttributeError, KeyError, TypeError, urllib.error.HTTPError):
        return None
    return (model, groups[model]) if model else None


def path(home):
    return Path(home) / 'models.json'


def load(home):
    try:
        return json.loads(path(home).read_text())
    except (OSError, ValueError):
        return dict(updated=None, providers={}, models=[])


def refresh(home, config, opener=urllib.request.urlopen, only=None):
    """Ask every provider (or `only` one) for its models; keep a failed provider's last entries, marked stale."""
    cat = load(home)
    old = {}
    for m in cat.get('models', []):
        old.setdefault(m['provider'], []).append(m)
    status, models = dict(cat.get('providers') or {}), []
    for name, spec in providers(config).items():
        if only and name != only:
            models += old.get(name, [])
            continue
        try:
            got = discover(name, spec, opener)
            models += got
            status[name] = dict(kind=spec.get('kind'), ok=True, models=len(got), at=time.time())
        except (OSError, ValueError, KeyError, urllib.error.HTTPError) as e:
            models += [dict(m, stale=True) for m in old.get(name, [])]
            status[name] = dict(kind=spec.get('kind'), ok=False, error=f'{type(e).__name__}: {str(e)[:200]}',
                                at=time.time())
    cat = dict(updated=time.time(), providers=status, models=models)
    from .store import atomic_json
    atomic_json(path(home), cat)   # (a temp file per writer: two windows refreshing at once both wrote models.tmp)
    return cat


def find(cat, model, provider=None):
    """The entry for a model id, or for one of its aliases when no id matches (a provider/id form picks the provider),
    or None."""
    if provider is None and '/' in model:
        head, rest = model.split('/', 1)
        if head in (cat.get('providers') or {}):
            provider, model = head, rest
    ours = [m for m in cat.get('models', []) if provider is None or m['provider'] == provider]
    hits = [m for m in ours if m['id'] == model] or [m for m in ours if model in (m.get('aliases') or ())]
    return hits[0] if len(hits) == 1 else None if not hits else \
        (_ for _ in ()).throw(ValueError(f'{model} is served by {", ".join(h["provider"] for h in hits)}: '
                                         'name the provider (provider/model)'))


def model_vision(home, cfg, ask=True):
    """Whether cfg's model takes images, by the catalog: home's models.json, else (ask) one short question to cfg's own
    endpoint, since a new home has no catalog yet (agent init). Only a loopback endpoint is asked: the local server or
    the owner's key relay, which every paid brain goes through. Unknown is no."""
    import urllib.parse
    from .model import LOOPBACK
    try:
        entry = find(load(home), cfg['model'], cfg.get('provider'))
    except ValueError:
        entry = None
    if entry is None and ask and urllib.parse.urlsplit(cfg.get('endpoint') or '').hostname in LOOPBACK:
        kind = cfg.get('provider_kind') or ('anthropic' if cfg.get('api') == 'anthropic' else 'auto')
        try:
            listed = discover('init', dict(kind=kind, endpoint=cfg['endpoint']),
                              lambda req, timeout=None: urllib.request.urlopen(req, timeout=3))
            entry = next((m for m in listed if m['id'] == cfg['model']), None)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, urllib.error.HTTPError):
            entry = None
    return bool(entry and entry.get('vision'))
