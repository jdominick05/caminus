"""Server jobs in the agent environment (docs/specs/spec-ember.md section 12): `caminus ember serve NAME MODEL
[--placement auto|0|1|2]` resolves placement from the catalog and the running servers (a model that needs two
cards gets no one-card option; pair rules like "the Coder needs a 24 GB card 0" are refused with their reason),
allocates the next free port, gives the server its own home under the agent user (~/homes/NAME/...), and starts
it as a named tmux session running a generated launcher — so each agent job has its own session, port, state dir,
Qwen home and logs. Placement math and the serve argv reuse caminus.cmd.manage (pick_cards / serve_argv); the
server process itself runs as the agent user over ssh.

The launcher runs the server with CAMINUS_ENV_PYTHON (default python3) and CAMINUS_ENV_PKG (PYTHONPATH to a
caminus package the agent user can read — see the bind-mount line in `caminus ember setup`)."""
import os
import shlex
import socket

from .. import models
from ..cmd import manage

PORT0 = manage.PORT0
PLACEMENTS = ('auto', '0', '1', '2')


class ServeError(Exception):
    pass


def resolve(model_name, placement='auto', ctx=None):
    """(model, entry, devs, cards_n): placement resolved against the catalog and the running servers.
    Raises ServeError with the refusal reason (never silently downgrades a placement)."""
    if placement not in PLACEMENTS:
        raise ServeError(f'placement {placement!r}: one of {", ".join(PLACEMENTS)}')
    paused = models.paused(model_name)
    if paused:
        raise ServeError(paused)
    m = models.BY_NAME.get(model_name)
    if m is None:
        raise ServeError(f'unknown model {model_name!r} (caminus models: {", ".join(models.BY_NAME)})')
    e = manage.model_entry(m)
    if not e['present']:
        raise ServeError(f'{m.name} is not downloaded (caminus pull {m.name})')
    running = [s for s in manage.all_servers() if manage.alive(s.get('guard_pid'))]
    held = {d for s in running for d in s.get('devices', [])}
    if placement in ('0', '1') and m.cards >= 2:
        raise ServeError(f'{m.name} needs {m.cards} cards: no one-card option for it')
    card, nc = None, m.cards
    if placement in ('0', '1'):
        card = placement
    elif placement == '2':
        nc = 2
        try:
            from .. import arch
            arch.lookup(m.arch).variant(nc)
        except ValueError as ex:
            raise ServeError(f'{m.name}: {ex}') from None
    try:
        devs = manage.pick_cards(m, e, card, held, ctx=ctx, cards=nc)
    except manage.ManageError as ex:
        raise ServeError(str(ex)) from None
    busy = manage.busy_elsewhere(devs)
    if busy:
        raise ServeError(f'VRAM in use on {", ".join(busy)} by a process Caminus did not start: stop it first')
    return m, e, devs, nc


def alloc_port(port0=PORT0, taken=()):
    """the first free port from port0 up (bind-probed, skipping taken)"""
    p = port0
    while p in taken:
        p += 1
    while True:
        with socket.socket() as s:
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                p += 1


def home_env(name):
    """the per-server environment: every stateful path under the agent user's ~/homes/NAME; the model dir moves
    to the models bind-mount when CAMINUS_ENV_MODELS is set"""
    h = f'~/homes/{name}'
    env = {
        'CAMINUS_STATE_DIR': f'{h}/state',
        'CAMINUS_KV_DIR': f'{h}/kv',
        'CAMINUS_FLIGHT_DIR': f'{h}/flight',
        'CAMINUS_QWEN_HOME': f'{h}/qwen-home',
        'CAMINUS_CONFIG': f'{h}/config/none.toml',
        'XDG_CONFIG_HOME': f'{h}/config',
        'XDG_DATA_HOME': f'{h}/data',
        'XDG_CACHE_HOME': f'{h}/cache',
        'XDG_STATE_HOME': f'{h}/state',
    }
    models_mount = os.environ.get('CAMINUS_ENV_MODELS')
    if models_mount:
        env['CAMINUS_MODEL_DIR'] = models_mount
    return env


def hip_visible(devs):
    """HIP_VISIBLE_DEVICES for card positions devs (the iGPU counted), like caminus start"""
    view = {c['index']: c for c in manage.card_view()}
    return ','.join(str(view[i]['hip']) for i in devs)


def remap_paths(serve_args, model_dir, models_mount):
    """serve_argv bakes the invoking user's absolute paths into --model/--drafter/--cache-dir/--flight-dir; the
    launcher runs as the agent user, so model paths move under the models bind-mount (CAMINUS_ENV_MODELS) and
    cache/flight paths under the server's own home (rewritten by the caller's home_env anyway)."""
    out = list(serve_args)
    for i, a in enumerate(out):
        if i + 1 < len(out) and a in ('--model', '--drafter') and models_mount and out[i + 1].startswith(model_dir):
            out[i + 1] = models_mount + out[i + 1][len(model_dir):]
    return out


def _q(v):
    """shlex.quote, except a leading ~/ stays the agent's home: quoted, '~/homes/N/state' reached the server as a
    literal relative path"""
    v = str(v)
    return '"$HOME"/' + shlex.quote(v[2:]) if v.startswith('~/') else shlex.quote(v)


def launcher(name, serve_args, devs, runtime=None, pkg=None, guard=None):
    """the run.sh written into the agent user's home: per-name env, HIP placement, the guard, then serve"""
    runtime = runtime or os.environ.get('CAMINUS_ENV_PYTHON', 'python3 -I')
    pkg = pkg if pkg is not None else os.environ.get('CAMINUS_ENV_PKG', '')
    guard = guard or os.environ.get('CAMINUS_ENV_GUARD', '')
    env_lines = ' '.join(f'{k}={_q(v)}' for k, v in home_env(name).items())
    lines = ['#!/bin/bash', 'set -e', f'cd ~/homes/{shlex.quote(name)}',
             f'export {env_lines}',
             f'export HIP_VISIBLE_DEVICES={hip_visible(devs)}']
    if pkg:
        lines.append(f'export PYTHONPATH={shlex.quote(pkg)}${{PYTHONPATH:+:$PYTHONPATH}}')
    cmd = ' '.join(_q(a) for a in serve_args)
    # -I ignores PYTHONPATH, so a shared package is put on sys.path explicitly rather than through the
    # environment: otherwise `python3 -I -m caminus` cannot import the package the operator configured.
    if pkg:
        boot = ('import sys;sys.path.insert(0,' + repr(pkg) +
                ');from caminus.__main__ import main;raise SystemExit(main())')
        entry = f'{runtime} -c {shlex.quote(boot)} serve {cmd}'
    else:
        entry = f'{runtime} -m caminus serve {cmd}'
    if guard:
        lines.append(f'exec bash {shlex.quote(guard)} ~/homes/{shlex.quote(name)}/serve.log 0 {entry}')
    else:
        lines.append(f'exec {entry}')
    return '\n'.join(lines) + '\n'
