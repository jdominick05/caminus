# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""One effort scale on every brain, and each model's generation settings from its lab.

The scale: none, low, medium, high, xhigh, max (/effort LEVEL; 'no' means none). Each model runs a level as its
nearest real one:
- a local Caminus model with an effort line (the Qwen3.8 line: Coder, Swift 1.5, ThinkingCap, Qwen3.8-27B): its
  template knows low, medium and xhigh, each a model id of the server (BASE-low, BASE, BASE-xhigh; BASE-nothink for
  none). high and max run as xhigh. The server pins the effort per id (a per-request effort would rewrite the
  system message and void the prompt cache), so the level is chosen by the id sent.
- a local model with only thinking on or off (Qwen3-30B-A3B): none is BASE-nothink, every other level is BASE.
- Anthropic: low .. max natively; none is thinking disabled, which the API takes only at high effort or below, so
  it is sent with effort high.
- an OpenAI-compatible provider with reasoning_effort (DeepInfra, Mistral): none, low, medium, high natively;
  xhigh and max run as high.

With no level set, the lab's recommendation for agentic work applies (model cards, 2026-11): the Qwen3.8 line xhigh,
Qwen3-30B-A3B thinking on, Anthropic medium, Mistral Large 4 / Medium 3.5 high.

Sampling: a local brain gets none (the server applies each model's own defaults from its GGUF); a paid brain gets
its card's values when the model is known, none otherwise; Anthropic never gets temperature, top_p or top_k (the
current API version refuses them).
"""
import re

LEVELS = ('none', 'low', 'medium', 'high', 'xhigh', 'max')
SUFFIX = {'none': '-nothink', 'low': '-low', 'xhigh': '-xhigh'}
LOCAL_CAP = 65536          # the owner's cap on one local reply (a runaway think held the server for an hour)
PAID_CAP = 65536           # a paid brain's output at most (its catalog maximum below that)

# known models, matched on the id (lower case): the effort line, the lab's default level, the card's maximum output
# and, for paid models, the card's sampling
CARDS = (
    (r'qwen3-30b|q3moe|qwen3moe', dict(local=True, default='medium', max_output=32768)),
    (r'k3-coder|coder|swift|thinkingcap|qwen3\.8|qwen38', dict(local=True, default='xhigh', max_output=393216)),
    (r'deepseek-v4\.1-flash', dict(sampling=dict(temperature=1.0, top_p=1.0))),
    (r'mistral-large-4|mistral-large-latest', dict(default='high', sampling=dict(temperature=1.0, top_p=0.95))),
    (r'mistral-medium-3\.5|mistral-medium-latest', dict(default='high', sampling=dict(temperature=0.7, top_p=0.95))),
)


def card(model):
    m = (model or '').lower()
    for pat, facts in CARDS:
        if re.search(pat, m):
            return facts
    return {}


def normalize(level):
    level = (level or '').strip().lower()
    return 'none' if level in ('no', 'off') else level


def group_local(ids):
    """A local server's ids grouped by base model: {base: {level: id}} for the ids it lists (BASE-nothink,
    BASE-low, BASE-xhigh next to BASE)."""
    ids = list(ids)
    have = set(ids)
    out = {}
    for i in ids:
        if any(i.endswith(s) and i[:-len(s)] in have for s in SUFFIX.values()):
            continue                                   # a variant: it belongs to its base
        out[i] = {lvl: i + s for lvl, s in SUFFIX.items() if i + s in have}
    return out


def line(cfg):
    """How the brain takes effort: 'scale' (local ids low/medium/xhigh), 'onoff' (local, thinking on or off),
    'anthropic', 'reasoning' (reasoning_effort) or '' (no control)."""
    kind = cfg.get('provider_kind') or ('anthropic' if cfg.get('api') == 'anthropic' else '')
    if kind == 'anthropic' or cfg.get('api') == 'anthropic':
        return 'anthropic'
    variants = cfg.get('model_variants') or {}
    if kind == 'local' or variants:
        if 'low' in variants or 'xhigh' in variants:
            return 'scale'
        return 'onoff' if 'none' in variants else ''
    if 'efforts' in cfg:        # what the catalog says this model takes (stored by /model)
        return 'reasoning' if cfg['efforts'] else ''
    return 'reasoning' if cfg.get('effort') else ''   # (an older config with an effort set: sent as before)


def default_level(cfg):
    """The lab's recommended level for this brain, or '' (the model's own default)."""
    ln = line(cfg)
    if ln == 'anthropic':
        return 'medium'
    c = card(cfg.get('model'))
    if ln in ('scale', 'onoff'):
        return (c.get('default') if c.get('local') else '') or ('xhigh' if ln == 'scale' else 'medium')
    if ln == 'reasoning' and not c.get('local'):
        return c.get('default', '')   # (a local card's default never applies to a paid model of the same family)
    return ''


# How each Claude model turns thinking off (Anthropic's error-code docs): 'disabled' at effort high or below (Haiku 5.5,
# Opus 5; a 400 at xhigh / max); 'between_tools' at high or below (Sonnet 5.5, where 'disabled' is a 400); None: it
# cannot be turned off (Opus 5.5, Fable: 'disabled' is a 400 at any effort), so thinking is omitted and effort low is
# the nearest. An unknown Claude model gets None: omitting thinking at effort low never fails.
THINKING_OFF = {
    'claude-haiku-5-5': 'disabled',
    'claude-opus-5': 'disabled',
    'claude-sonnet-5-5': 'between_tools',
    'claude-opus-5-5': None,
    'claude-fable-5': None, 'claude-fable-5-1': None, 'claude-fable-5-2': None,
}


def thinking_off_mode(model):
    return THINKING_OFF.get(model)


def anthropic_thinking_off(body):
    """Turn a Messages API body's thinking off the way its model takes it (THINKING_OFF); the effort follows."""
    mode = thinking_off_mode(body.get('model'))
    oc = body.setdefault('output_config', {})
    if mode:
        body['thinking'] = dict(type=mode)
        if oc.get('effort') not in ('low', 'medium', 'high'):
            oc['effort'] = 'high'                      # (accepted up to high: the nearest level that runs)
    else:
        body.pop('thinking', None)
        oc['effort'] = 'low'


def always_thinks(cfg):
    """An Anthropic model whose thinking cannot be turned off: none runs as low there."""
    return line(cfg) == 'anthropic' and thinking_off_mode(cfg.get('model')) is None


def template_line(cfg, level):
    """The effort line the local template writes first in the system message for a level: 'xhigh' and 'low' write
    their own, medium and none write none (the Qwen3.8 template). A change of it rewrites the prompt from its
    first token, so the server reads the whole conversation again; a model with thinking on or off only has none."""
    if line(cfg) != 'scale':
        return ''
    real = runs_as(cfg, level)
    return real if real in ('xhigh', 'low') else ''


def runs_as(cfg, level):
    """The level the model really runs (the nearest it has)."""
    ln, level = line(cfg), normalize(level)
    if level == 'none' and always_thinks(cfg):
        return 'low'
    if ln == 'scale':
        return {'high': 'xhigh', 'max': 'xhigh'}.get(level, level)
    if ln == 'onoff':
        return 'none' if level == 'none' else 'on'
    if ln == 'reasoning':
        return {'xhigh': 'high', 'max': 'high'}.get(level, level)
    return level


def resolve(cfg, short=False):
    """What the transport sends for the effort in force: dict(model=ID or None, reasoning_effort=... or None,
    anthropic_effort=... or None, thinking_off=bool). short: a side call (the judge) runs with thinking off."""
    ln = line(cfg)
    level = normalize(cfg.get('effort')) or default_level(cfg)
    if short and (level or ln in ('scale', 'onoff')):
        level = 'none'      # (a provider whose model shows no effort control gets no reasoning field at all)
    out = dict(model=None, reasoning_effort=None, anthropic_effort=None, thinking_off=False)
    if ln in ('scale', 'onoff'):
        variants = cfg.get('model_variants') or {}
        real = runs_as(cfg, level)
        out['model'] = variants.get(real) if real in variants else cfg.get('model')
        out['thinking_off'] = real == 'none' and 'none' not in variants
    elif ln == 'anthropic':
        if level == 'none' and always_thinks(cfg):
            out['anthropic_effort'] = 'low'
        elif level == 'none':
            out.update(anthropic_effort='high', thinking_off=True)   # (thinking off is accepted up to high)
        else:
            out['anthropic_effort'] = level or 'medium'
    elif ln == 'reasoning' and level:
        out['reasoning_effort'] = runs_as(cfg, level)
    return out


def note(cfg):
    """One line for /effort: how this model runs the scale."""
    ln = line(cfg)
    if ln == 'scale':
        return ('on this model high and max run as xhigh. Pick the level at the start of a chat; on the Qwen3.8 '
                'models a change re-reads the conversation once')
    if ln == 'onoff':
        return 'this model only has thinking on or off: none turns it off, every other level is on'
    if ln == 'reasoning':
        return 'on this model xhigh and max run as high'
    if ln == 'anthropic' and always_thinks(cfg):
        return 'this model always thinks: none runs as low'
    if ln == 'anthropic' and thinking_off_mode(cfg.get('model')) == 'between_tools':
        return 'none thinks only between tools (sent with effort high)'
    if ln == 'anthropic':
        return 'none turns thinking off (sent with effort high)'
    return 'this model has no effort control'


def sampling(cfg, remote):
    """The sampling fields to send: the config's own when set; else a paid brain's card values; else none (a local
    server applies its model's defaults)."""
    out = {k: cfg[k] for k in ('temperature', 'top_k', 'top_p', 'min_p', 'presence_penalty') if k in cfg}
    if out or not remote:
        return out
    return dict(card(cfg.get('model')).get('sampling') or {})


def reply_limit(cfg, remote):
    """A reply's token limit: config max_tokens when set; a local brain the card's maximum output, at most 64K (the
    owner's cap); a paid brain 4096 (its catalog maximum, up to 64K, is set as max_tokens when it is picked)."""
    if cfg.get('max_tokens') is not None and not _legacy_limit(cfg, remote):
        return int(cfg['max_tokens'])
    if remote:
        return 4096
    return min(int(card(cfg.get('model')).get('max_output') or LOCAL_CAP), LOCAL_CAP)


def _legacy_limit(cfg, remote):
    """agent init wrote max_tokens 4096 into every config before the per-model limits: on a local brain that value is
    the old default, not the owner's choice, so it gives way to the card's limit."""
    return not remote and cfg.get('max_tokens') == 4096 and not cfg.get('max_tokens_auto')
