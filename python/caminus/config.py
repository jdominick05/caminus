"""Caminus's paths in one place (docs/dev/packaging.md): the environment first, then the config file, then the XDG defaults.
No machine's own path is a default.

  config file   $CAMINUS_CONFIG, else $XDG_CONFIG_HOME/caminus/config.toml (~/.config/caminus/config.toml); optional; string
                values only, `~` expanded; an unknown key is refused (ConfigError)

  key          environment                    default
  model_dir    CAMINUS_MODEL_DIR              $XDG_DATA_HOME/caminus/models (~/.local/share/caminus/models; caminus download)
  coder        CAMINUS_CODER (old: K3_GGUF)   the Coder's first shard under model_dir (caminus.models: <repo>/<path>)
  dflash       CAMINUS_DFLASH                 the DFlash2 Q8_0 GGUF under model_dir
  mtp          CAMINUS_MTP                    none (the MTP sidecar is not distributed)
  kv_dir       CAMINUS_KV_DIR                 $XDG_CACHE_HOME/caminus/kv (~/.cache/caminus/kv: the prompt cache's disk tier)
  flight_dir   CAMINUS_FLIGHT_DIR             $XDG_CACHE_HOME/caminus/flight (the flight recorder)
  state_dir    CAMINUS_STATE_DIR              $XDG_STATE_HOME/caminus (~/.local/state/caminus: the guard's log of caminus serve)
  kernel_dir   CAMINUS_KERNEL_DIR             none: caminus.device.kernels finds the code objects (a checkout's build/, else the
                                              package's own)
  gguf_py      CAMINUS_GGUF_PY (old: K3_GGUF_PY)   the vendored gguf-py reader (caminus/_vendor; tools/release/vendor_gguf.py)
  model_search CAMINUS_MODEL_SEARCH           none: more directories (':'-separated) where caminus models / start look for
                                              the model files, three levels deep, by name and size (files kept elsewhere)

Command-line flags (caminus serve --model / --drafter / --cache-dir / --flight-dir, --config FILE) override all of these.
Tests and dev tools keep their own environment-driven paths (CLAUDE.md)."""
import os
import sys

KEYS = {
    'model_dir': ('CAMINUS_MODEL_DIR',),
    'coder': ('CAMINUS_CODER', 'K3_GGUF'),
    'dflash': ('CAMINUS_DFLASH',),
    'mtp': ('CAMINUS_MTP',),
    'kv_dir': ('CAMINUS_KV_DIR',),
    'flight_dir': ('CAMINUS_FLIGHT_DIR',),
    'state_dir': ('CAMINUS_STATE_DIR',),
    'kernel_dir': ('CAMINUS_KERNEL_DIR',),
    'gguf_py': ('CAMINUS_GGUF_PY', 'K3_GGUF_PY'),
    'model_search': ('CAMINUS_MODEL_SEARCH',),
}
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_vendor')


class ConfigError(ValueError):
    """a config file that cannot be used (unreadable TOML, an unknown key, a non-string value)"""


def xdg(var, rel):
    """an XDG base directory: $var when it is an absolute path (the spec ignores relative ones), else ~/rel"""
    v = os.environ.get(var)
    return v if v and os.path.isabs(v) else os.path.join(os.path.expanduser('~'), rel)


def config_path():
    return os.environ.get('CAMINUS_CONFIG') or os.path.join(xdg('XDG_CONFIG_HOME', '.config'), 'caminus', 'config.toml')


_FILE = {}


def file_values(path=None):
    """the config file's values ({} when there is none); cached per path"""
    p = path or config_path()
    if p in _FILE:
        return _FILE[p]
    vals = {}
    if os.path.isfile(p):
        import tomllib
        try:
            with open(p, 'rb') as f:
                raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError(f'{p}: {e}') from None
        for k, v in raw.items():
            if k not in KEYS:
                raise ConfigError(f'{p}: unknown key {k!r} (known: {", ".join(KEYS)})')
            if not isinstance(v, str):
                raise ConfigError(f'{p}: {k} must be a string (a path), not {type(v).__name__}')
            vals[k] = os.path.expanduser(v)
    _FILE[p] = vals
    return vals


def lookup(key):
    """(value, where) of an explicitly set key: where 'env NAME' or 'file PATH'; (None, None) when unset"""
    for n in KEYS[key]:
        v = os.environ.get(n)
        if v:
            return os.path.expanduser(v), f'env {n}'
    fv = file_values()
    if key in fv:
        return fv[key], f'file {config_path()}'
    return None, None


def _default(key):
    from . import models
    if key == 'model_dir':
        return os.path.join(xdg('XDG_DATA_HOME', os.path.join('.local', 'share')), 'caminus', 'models')
    if key == 'coder':
        return os.path.join(model_dir(), models.local_rel(models.BY_KEY['coder-1']))
    if key == 'dflash':
        return os.path.join(model_dir(), models.local_rel(models.BY_KEY['dflash']))
    if key == 'kv_dir':
        return os.path.join(xdg('XDG_CACHE_HOME', '.cache'), 'caminus', 'kv')
    if key == 'flight_dir':
        return os.path.join(xdg('XDG_CACHE_HOME', '.cache'), 'caminus', 'flight')
    if key == 'state_dir':
        return os.path.join(xdg('XDG_STATE_HOME', os.path.join('.local', 'state')), 'caminus')
    if key == 'gguf_py':
        return VENDOR
    return None                                     # mtp, kernel_dir: no default


def resolve(key):
    """(value, where): the explicit setting, else the default (where 'default'; value None when the key has none)"""
    v, w = lookup(key)
    return (v, w) if v is not None else (_default(key), 'default')


def get(key):
    return resolve(key)[0]


def model_dir(): return get('model_dir')
def coder(): return get('coder')
def dflash(): return get('dflash')
def mtp(): return get('mtp')
def kv_dir(): return get('kv_dir')
def flight_dir(): return get('flight_dir')
def state_dir(): return get('state_dir')
def gguf_py(): return get('gguf_py')


def ensure_gguf():
    """put the gguf-py reader in front of sys.path as `gguf` (the vendored copy unless CAMINUS_GGUF_PY / K3_GGUF_PY names
    another tree); returns its directory"""
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def summary():
    """{key: (value, where)} for every key (caminus doctor)"""
    return {k: resolve(k) for k in KEYS}
