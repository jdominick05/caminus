"""Paths and settings. The config file is TOML (~/.config/qwen-advisor/config.toml, or $QWEN_ADVISOR_CONFIG); every key
has a default here, so a missing or partial file still works. API keys are never stored in the config: it names where
each key lives (a file path, or env:VARNAME)."""
import os
import tomllib

HOME = os.path.expanduser('~')
CONFIG_DIR = os.path.join(os.environ.get('XDG_CONFIG_HOME') or os.path.join(HOME, '.config'), 'qwen-advisor')
STATE_DIR = os.path.join(os.environ.get('XDG_STATE_HOME') or os.path.join(HOME, '.local', 'state'), 'qwen-advisor')
SHARE_DIR = os.path.join(os.environ.get('XDG_DATA_HOME') or os.path.join(HOME, '.local', 'share'), 'qwen-advisor')
CONFIG_PATH = os.environ.get('QWEN_ADVISOR_CONFIG') or os.path.join(CONFIG_DIR, 'config.toml')

PROVIDERS = ('anthropic', 'mistral', 'local')
EFFORTS = {
    'anthropic': ('low', 'medium', 'high', 'xhigh', 'max'),
    'mistral': ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'),
    'local': (),   # (an OpenAI-compatible server: no effort setting is sent)
}

DEFAULTS = {
    'enabled': True,
    'provider': 'anthropic',            # the advisor asked first
    'fallback_provider': '',            # tried when the first fails (an error, a timeout, a refusal); '' = none
    'timeout_s': 55,                    # per advisor request
    'chain': [],                        # more advisors after those two, 'provider:model[@effort]', tried in order
    'chain_budget_s': 150,              # all attempts together; the hooks' own timeouts are this + 15 s
    'retry_on_soft_refusal': True,      # a reply that declines counts as a failure: the next advisor is asked
    'max_calls_per_session': 8,         # advisor calls (any outcome) per Qwen Code session
    'min_interval_s': 600,              # automatic hints (stuck detection, the done-check) at most this often per session;
                                        # in between the local model keeps working on its own. /advise is never limited
    'cooldown_batches': 6,              # tool batches between two automatic calls
    'review_on_stop': False,            # ask the advisor once per user message whether the task is done (Stop hook)
    'review_min_batches': 4,            # ... only after at least this many tool batches in that turn
    'max_advice_chars': 1400,
    'log_refusals': True,               # refusals.jsonl in the state directory: every call that met a refusal
    'log_refusal_briefs': True,         # ... with the (redacted) brief that was sent, so it can be replayed
    # server_fallback: on Opus / Sonnet 5.x and Fable 5.1, a refused request is retried server-side on another model
    'anthropic': {'model': 'claude-opus-5-5', 'effort': 'medium', 'key': '~/.config/anthropic/api_key', 'server_fallback': True},
    'mistral': {'model': '', 'effort': '', 'key': '~/.config/qwen-advisor/keys/mistral'},
    # local: any OpenAI-compatible server (Caminus or llama.cpp on another card, vLLM, Ollama ...); key optional; model ''
    # = whatever the server serves. A thinking model's reasoning counts against max_tokens: keep it generous.
    # Its own reply limit (timeout_s): a thinking model on one card can take a minute; a late reply adds nothing. The
    # hooks wait chain_budget_s at most (their timeouts in settings.json are written by setup: rerun it after a change).
    'local': {'url': 'http://127.0.0.1:8093/v1', 'model': '', 'effort': '', 'key': '', 'max_tokens': 8000, 'timeout_s': 120},
    'triggers': {
        'repeat': 3,          # the same call (tool + arguments) this many times among the last 8 calls
        'fail_streak': 3,     # this many failed calls in a row
        'edit_churn': 4,      # the same file edited this many times since the user's last message, a failure after the last edit
        'long_run': 40,       # this many tool batches since the user's last message without advice (a check-in)
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def load(path=None):
    """the settings: DEFAULTS overlaid with the config file (a missing file is DEFAULTS; a broken one raises)"""
    p = path or CONFIG_PATH
    if not os.path.exists(p):
        return _merge(DEFAULTS, {})
    with open(p, 'rb') as f:
        return _merge(DEFAULTS, tomllib.load(f))


def read_key(spec):
    """an API key from 'env:NAME' or a file path ('~' expanded); None when absent or empty"""
    if not spec:
        return None
    if spec.startswith('env:'):
        v = os.environ.get(spec[4:], '').strip()
        return v or None
    p = os.path.expanduser(spec)
    try:
        with open(p) as f:
            v = f.read().strip()
    except OSError:
        return None
    return v or None
