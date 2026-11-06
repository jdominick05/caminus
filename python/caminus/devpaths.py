"""Paths the tests and the dev tools read (tests/, tools/). Nothing the server or the console commands run imports this module:
their paths come from caminus.config. Every value comes from the environment; no machine's own path is a default.

  CODER              the Coder's first shard (...-00001-of-00002.gguf); unset: caminus.config's coder (CAMINUS_CODER / K3_GGUF,
                     config.toml, else where `caminus download` puts it)
  DFLASH             the DFlash2 Q8_0 GGUF; unset: caminus.config's dflash
  MTP                the MTP sidecar GGUF (not distributed); unset: CAMINUS_MTP
  REAP               the REAP-308 GGUF (the regression model; not distributed)
  ARTIFACTS          the directory of reference dumps, stored results and big outputs (not distributed)
  HF_TOKENIZER_JSON  the base model's tokenizer.json (two optional tokenizer cross-checks skip without it)

An unset variable with no fallback gives the literal placeholder '$NAME' (joined with the rest of the path): reading it fails with
an error that names the variable to set. Where a missing value would create a directory or file (a scratch or output default),
callers use need(), which exits with a message instead."""
import os
import sys


def env(name, *rel):
    """$name joined with rel when set, else the placeholder '$name/rel' (a path that does not exist)"""
    v = os.environ.get(name)
    return os.path.join(os.path.expanduser(v) if v else f'${name}', *rel)


def need(name, *rel):
    """$name joined with rel; exits with a message when the variable is unset (for paths a caller would create)"""
    v = os.environ.get(name)
    if not v:
        sys.exit(f'set {name} (see CLAUDE.md, "Models and data"), or pass the path explicitly')
    return os.path.join(os.path.expanduser(v), *rel)


def coder():
    v = os.environ.get('CODER')
    if v:
        return os.path.expanduser(v)
    from . import config
    return config.coder()


def dflash():
    v = os.environ.get('DFLASH')
    if v:
        return os.path.expanduser(v)
    from . import config
    return config.dflash()


def mtp():
    v = os.environ.get('MTP')
    if v:
        return os.path.expanduser(v)
    from . import config
    return config.get('mtp') or '$MTP'


def reap():
    return env('REAP')


def artifacts(*rel):
    return env('ARTIFACTS', *rel)
