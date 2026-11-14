"""caminus models / pull / start / stop / restart / ps / logs / top / panel: run and manage Caminus servers, Ollama-style.

  caminus models [--json]                       the named models (caminus.models.CATALOG): on disk or not, cards, speeds
  caminus pull NAME [--json]                    download a named model's files (caminus download --model NAME)
  caminus start NAME [--name N] [--card auto|0|1] [--port P] [--ctx auto|N] [--no-wait] [--json] [-- serve flags...]
  caminus start --team brain-hands [--json]     AREX-2 on the 24 GB card (brain, :8092) + Qwen3.6 on the other (hands, :8093)
  caminus stop NAME | --all [--json]            SIGTERM: the server drains, parks its conversation and exits
  caminus restart NAME [--json]                 stop, then start again with the same command line
  caminus ps [--json]                           the servers this user started, their state, and the cards
  caminus logs NAME [-f]                        the server's log (less; -f follows it)
  caminus top [NAME]                            the monitor (caminus-panel top, the former llm-top) on that server
  caminus panel [-advanced]                     the control panel (caminus-panel)

Each server runs `caminus serve` under the guard (caminus.cmd.guarded) in the background, in its own session, with its own
state directory ($state_dir/servers/NAME: server.json, serve.log, guard.txt), prompt-cache disk tier ($kv_dir/NAME) and
flight recorder ($flight_dir/NAME): two servers never share a disk tier (it takes an owner lock). A one-card model goes on
the card given (--card: its position in `caminus ps`), else on the smallest free card it fits at 32K positions (the
registry's projection, caminus.arch.Arch.fits); a two-card model on the pair caminus.device.host picks (the two biggest free
cards on adequate links, the bigger first). Cards are told apart by PCI address; HIP_VISIBLE_DEVICES is written from their
HIP indices (D-292). Ports: from 8092 up, the first one free. Nothing here touches a GPU itself: caminus serve does, under the
guard, and its own capacity checks refuse a context that does not fit (max_ctx auto picks the largest that does).

--json (for caminus panel): one JSON document on stdout; actions print {"ok": true, "note": ...} or {"ok": false,
"error": ...}; progress goes to stderr."""
import argparse
import contextlib
import glob
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from .. import config, models
from ..device import host, machine
from . import guarded

PORT0 = 8092
ROOT = '/'                # the sysfs root the cards are read from (tests: a fixture tree)
TEAMS = {'brain-hands': (('brain', 'arex-2', 'big', 8092), ('hands', 'qwen3.6', 'small', 8093))}


class ManageError(Exception):
    pass


# ---------------------------------------------------------------- models

def search_dirs():
    """model_dir, then model_search (CAMINUS_MODEL_SEARCH / config: directories separated by ':', searched three levels
    deep for a model file of the manifest's name and size)"""
    out = [config.model_dir()]
    for d in (config.get('model_search') or '').split(':'):
        if d.strip():
            out.append(os.path.expanduser(d.strip()))
    return out


def locate_item(it):
    """the path of a manifest item on this machine (size must match the manifest: a partial file is not the model), or None"""
    cands = []
    if it.key == 'coder-1' and config.coder():
        cands.append(config.coder())
    if it.key == 'dflash' and config.dflash():
        cands.append(config.dflash())
    md = config.model_dir()
    cands.append(os.path.join(md, models.local_rel(it)))
    base = os.path.basename(it.path)
    for d in search_dirs():
        for depth in ('', '*', '*/*', '*/*/*'):
            cands += sorted(glob.glob(os.path.join(d, depth, base)))
    if it.key == 'coder-2' and config.coder():
        c1 = config.coder()
        cands.insert(0, os.path.join(os.path.dirname(c1), os.path.basename(c1).replace('-00001-of-', '-00002-of-')))
    for p in cands:
        try:
            if os.path.getsize(p) == it.size:
                return p
        except OSError:
            continue
    return None


def serving_modes(m):
    """The model picker's card counts and drafters, from the same registry serve uses (no model load or torch).

    The picker also offers auto: it omits --draft, leaving Arch.default_draft to choose from the served measurements.
    A two-card variant advertises drafting only once its registry entry supports it.
    """
    from .. import arch
    A = arch.lookup(m.arch)
    out = []
    for nc in (1, 2):
        try:
            a = A.variant(nc)
        except ValueError:
            continue
        ds = {d for d in (a.drafter, *getattr(a, 'drafters', ())) if d in ('mtp', 'dflash')}
        out.append(dict(cards=nc, drafts=['off'] + [d for d in ('mtp', 'dflash') if d in ds]))
    return out


def model_entry(m):
    paths = {k: locate_item(models.BY_KEY[k]) for k in m.items}
    dk = models.drafter_item(m)
    need = [k for k in m.items if k != dk]     # a model serves without its drafter too (the Coder)
    size = sum(models.BY_KEY[k].size for k in m.items)
    need_b = sum(models.BY_KEY[k].size for k in need)   # (what a card must hold without the optional drafter: the fit rule's size)
    return dict(name=m.name, title=m.title, arch=m.arch, cards=m.cards, size_gb=round(size / 1e9, 1), need_gb=round(need_b / 1e9, 1),
                present=all(paths[k] for k in need), path=paths[m.items[0]], paths=paths, alias=m.alias,
                speed=m.speed, notes=m.notes, serve_modes=serving_modes(m),
                missing_drafter='dflash' if dk and not paths[dk] else None,
                missing_drafter_gb=round(models.BY_KEY[dk].size / 1e9, 1) if dk and not paths[dk] else 0)


def cmd_models(argv):
    ap = argparse.ArgumentParser(prog='caminus models')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    rows = [model_entry(m) for m in models.CATALOG]
    if a.json:
        print(json.dumps({'models': rows}))
        return 0
    for r in rows:
        mark = 'on disk' if r['present'] else 'not downloaded (caminus pull ' + r['name'] + ')'
        print(f"{r['name']:<18} {r['title']:<52} {r['cards']} card{'s' if r['cards'] > 1 else ' '}  {r['size_gb']:>5.1f} GB  {mark}")
        print(f"{'':<18} {r['speed']}")
    return 0


def cmd_pull(argv):
    ap = argparse.ArgumentParser(prog='caminus pull')
    ap.add_argument('name', choices=[m.name for m in models.CATALOG] + [n for n in models.PAUSED if models.paused(n)])
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if models.paused(a.name):                    # (a paused model's name: refused with the reason, not "invalid choice")
        print(f'caminus pull: {models.paused(a.name)}', file=sys.stderr)
        return 2
    from . import download
    args = ['--model', a.name] + (['--dry-run'] if a.dry_run else [])
    if a.json:
        with contextlib.redirect_stdout(sys.stderr):
            rc = download.main(args)
        return result(rc == 0, f'{a.name} downloaded' if rc == 0 else None, None if rc == 0 else f'download failed (rc {rc}): see the lines above')
    return download.main(args)


# ---------------------------------------------------------------- servers

def servers_root():
    return os.path.join(config.state_dir(), 'servers')


def server_dir(name):
    return os.path.join(servers_root(), name)


def load_server(name):
    try:
        with open(os.path.join(server_dir(name), 'server.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def all_servers():
    out = []
    for p in sorted(glob.glob(os.path.join(servers_root(), '*', 'server.json'))):
        s = load_server(os.path.basename(os.path.dirname(p)))
        if s:
            out.append(s)
    return out


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:
        with open(f'/proc/{int(pid)}/cmdline', 'rb') as f:
            return b'guard.sh' in f.read()           # (a reused pid is not ours)
    except OSError:
        return False


def server_pid(s):
    """the caminus serve process under the guard (its 'pid=N' line in the log)"""
    try:
        with open(s['log'], errors='replace') as f:
            pids = [l.split('pid=')[1].split()[0] for l in f if 'pid=' in l and 'caminus serve:' in l]
        return int(pids[-1]) if pids else None
    except (OSError, ValueError, IndexError):
        return None


def http_json(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        return e.code, None
    except (OSError, ValueError):
        return None, None


def find_key(d, key):
    """the first value of key anywhere in a nested JSON document"""
    if isinstance(d, dict):
        if key in d and d[key] is not None:
            return d[key]
        vals = d.values()
    elif isinstance(d, list):
        vals = d
    else:
        return None
    for v in vals:
        r = find_key(v, key)
        if r is not None:
            return r
    return None


def log_error(path):
    """the last error-looking line of a server log ('' when none)"""
    try:
        with open(path, errors='replace') as f:
            lines = f.read().splitlines()[-200:]
    except OSError:
        return ''
    for l in reversed(lines):
        if any(w in l for w in ('Error', 'error:', 'refusing', 'Traceback', 'GPU FAULT', 'GUARD REFUSED', 'killed')):
            return l.strip()[-300:]
    return ''


def server_view(s):
    """a server.json plus its live state: ready (/health 200), loading (alive, not answering yet or 503), failed (the guard
    exited without the clean-stop line), stopped"""
    v = dict(name=s['name'], model=s['model'], url=f"http://127.0.0.1:{s['port']}", port=s['port'], cards=s.get('card_names', []),
             pid=s.get('guard_pid'), log=s['log'], uptime_s=max(0.0, time.time() - s.get('started', time.time())),
             max_ctx=None, decode_tps=None, alias=s.get('alias'), busy=False, error='')
    if alive(s.get('guard_pid')):
        code, _ = http_json(v['url'] + '/health')
        v['state'] = 'ready' if code == 200 else 'loading'
        if code == 200:
            _, props = http_json(v['url'] + '/props')
            if isinstance(props, dict):
                v['max_ctx'] = find_key(props, 'n_ctx')
                v['alias'] = props.get('model_alias') or v['alias']
            _, st = http_json(v['url'] + '/caminus/status', timeout=2.0)
            mpt = find_key(st, 'ms_per_token')
            if isinstance(mpt, (int, float)) and mpt > 0:
                v['decode_tps'] = round(1000.0 / mpt, 1)
            ph = find_key(st, 'phase')
            v['busy'] = ph in ('prefill', 'decode')
    else:
        try:
            with open(s['log'], errors='replace') as f:
                clean = 'stopped in' in f.read()[-4000:]
        except OSError:
            clean = False
        v['state'] = 'stopped' if clean else 'failed'
        if not clean:
            v['error'] = log_error(s['log']) or 'the server exited (caminus logs ' + s['name'] + ')'
    return v


def card_view(gs=None):
    """the usable cards (caminus.device.host: supported or untested gfx1100) in DRM order; index = the position `--card`
    and the panel use; pci = the card's identity; hip = its HIP index (HIP_VISIBLE_DEVICES)"""
    out = []
    for i, c in enumerate(host.usable(host.gpus(ROOT) if gs is None else gs)):
        used = machine._read(os.path.join(c['dev'], 'mem_info_vram_used'))
        out.append(dict(index=i, name=c['label'], card=c['card'], pci=c['pci'], hip=c['hip'], status=c['status'],
                        used_mib=(int(used) >> 20) if used else 0, line_mib=c['line_mib'], total_mib=c['total_mib'],
                        display=c['display'], link=host.link_text(c['link']), link_ok=c['link_ok']))
    return out


def gpu_view(gs):
    """every GPU of the host for the panel: what it is and whether Caminus uses it"""
    return [dict(card=g['card'], pci=g['pci'], name=g['label'], vendor=g['vendor'], status=g['status'], why=g['why'],
                 total_mib=g['total_mib'], link=host.link_text(g['link']), link_ok=g['link_ok'], display=g['display'])
            for g in gs]


def problem_view(gs):
    return [dict(level=lv, card=c, text=x) for lv, c, x in host.problems(gs)]


def cmd_ps(argv):
    ap = argparse.ArgumentParser(prog='caminus ps')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    views = [server_view(s) for s in all_servers()]
    gs = host.gpus(ROOT)
    cards = card_view(gs)
    probs = problem_view(gs)
    if a.json:
        print(json.dumps({'servers': views, 'cards': cards, 'gpus': gpu_view(gs), 'problems': probs}))
        return 0
    if not views:
        print('no servers (caminus start NAME; caminus models lists the names)')
    for v in views:
        extra = (f"  {v['max_ctx']:,} ctx" if v['max_ctx'] else '') + (f"  {v['decode_tps']} tok/s decode" if v['decode_tps'] else '')
        print(f"{v['name']:<12} {v['model']:<18} {v['state']:<8} {v['url']:<24} cards {'+'.join(v['cards'])}{extra}")
        if v['error']:
            print(f"{'':<12} {v['error']}")
    for c in cards:
        print(f"card {c['index']} {c['name']} ({c['pci']}, {c['link']}): {c['used_mib']:,} / {c['line_mib']:,} MiB (the guard's line)")
    for x in probs:
        print(f"{x['level']}: {x['text']}")
    return 0


def port_free(p):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1', p))
            return True
        except OSError:
            return False


def pick_port(want, taken):
    if want:
        if want in taken or not port_free(want):
            raise ManageError(f'port {want} is in use')
        return want
    for p in range(PORT0, PORT0 + 100):
        if p not in taken and port_free(p):
            return p
    raise ManageError(f'no free port in {PORT0}..{PORT0 + 99}')


def fits_fn(entry, ctx=None):
    """fits(card) for a one-card model file: the registry's projection at 32K positions (or --ctx) on that card now
    (caminus.arch.Arch.fits, sysfs only); without torch (no projection here) a size rule: the file + 4 GiB within the line"""
    size_mib = entry.get('need_gb', entry['size_gb']) * 1e9 / 2 ** 20   # (the files it serves with: an optional drafter is not loaded by default)
    rule = lambda g: size_mib * 1.1 + 4096 <= g['line_mib'] - (machine.DISPLAY_RESERVE_MIB if g['display'] else 256)
    try:
        from .. import arch
        A = arch.of(entry['path'])
        opt = dict(model=entry['path'], chunk=2048, max_ctx=ctx or 'auto', auto_cache_slots=4)
        A.fits(opt, [])                          # (the projection loads here: a failure takes the size rule)
    except Exception:                            # (no torch, or a header the projection cannot read: the server's own check decides)
        return rule
    return lambda g: A.fits(opt, [g])


def pair_fits(m, entry, ctx=None, cards=None):
    """why(cards) for a two-card catalog model: None when the registry's projection at 32K positions (or --ctx) fits those
    cards now, with its drafter when that file is here (caminus.arch.Arch.fits, sysfs only; the projection caminus serve
    places itself with), else the reason. None when it can't run here (the model file missing, no torch, a header it can't
    read): the server decides then"""
    try:
        from .. import arch
        A = arch.of(entry['path']).variant(cards)   # (cards 2 for a one-card model: its two-card form)
    except Exception:
        return None
    dk = models.drafter_item(m)
    dpath = (entry.get('paths') or {}).get(dk) if dk else None
    opt = dict(model=entry['path'], drafter=dpath, no_drafter=not dpath, chunk=2048, max_ctx=ctx or 'auto', auto_cache_slots=4)

    def fit(cards):
        try:
            if A.fits(opt, cards):
                return None
        except host.PlacementError as e:
            return str(e)
        except Exception as e:                   # (the projection could not run here: the server's own placement decides)
            print(f'caminus: {m.name}: the memory projection did not run ({type(e).__name__}: {e}); the server checks at start',
                  file=sys.stderr)
            return None
        return (f"doesn't fit {' + '.join(host.describe(c) for c in cards)} at {int(ctx or 32768):,} positions with a working "
                f"prompt cache (what the cards hold now counts: caminus ps)")
    return fit


def pick_cards(m, entry, card, held, ctx=None, cards=None):
    """the card positions (caminus ps's index) a model runs on, in engine order. held: the positions other servers hold.
    cards: how many (default: the model's own; 2 for a one-card model with a two-card form: caminus start --cards 2)"""
    gs = host.gpus(ROOT)
    cs = card_view(gs)
    if not cs:
        raise ManageError('no supported GPU (RX 7900 XTX / XT; caminus doctor)')
    by_pci = {c['pci']: c['index'] for c in cs}
    held_pci = {c['pci'] for c in cs if c['index'] in held}
    free = [c for c in cs if c['index'] not in held]
    if (cards or m.cards) >= 2:
        try:
            pr = host.pair(gs, held=held_pci)
        except host.PlacementError as e:
            raise ManageError(f'{m.name}: {e}' + (' (caminus ps)' if held else '')) from None
        try:
            from .. import arch
            why = arch.lookup(m.arch).variant(cards).pair_refusal(pr)
        except Exception:                        # (no registry here: the server's own placement still refuses)
            why = None
        if why:
            raise ManageError(f'{m.name}: {why}')
        fit = pair_fits(m, entry, ctx, cards=cards)
        why = fit(pr) if fit is not None else None
        if why:
            raise ManageError(f'{m.name}: {why}')
        return [by_pci[g['pci']] for g in pr]
    if card not in (None, '', 'auto'):
        i = int(card)
        if i not in [c['index'] for c in cs]:
            raise ManageError(f'no card {i} (cards: {", ".join(str(c["index"]) for c in cs)})')
        if i in held:
            raise ManageError(f'card {i} is held by another Caminus server (caminus ps)')
        return [i]
    if not free:
        raise ManageError('every card is held by a Caminus server (caminus ps; caminus stop NAME)')
    try:
        return [by_pci[host.one(gs, fits_fn(entry, ctx), held=held_pci)['pci']]]
    except host.PlacementError as e:
        raise ManageError(f'{m.name}: {e}') from None


def busy_elsewhere(devs):
    """cards of devs with more than 2 GiB in use (another GPU process: llama-server, a test, another engine)"""
    return [f"{c['name']} {c['used_mib']:,} MiB" for c in card_view() if c['index'] in devs and c['used_mib'] > 2048]


def _arch_drafter(m):
    from .. import arch as _arch
    return _arch.lookup(m.arch).drafter


def _serve_arg(extra, flag):
    """The last exact flag's value, in either spelling. Malformed options remain serve's argparse error to report."""
    value = None
    for i, arg in enumerate(extra):
        if arg.startswith(flag + '='):
            value = arg[len(flag) + 1:]
        elif arg == flag:
            value = extra[i + 1] if i + 1 < len(extra) and not extra[i + 1].startswith('--') else None
    return value.strip() if value else value


def serve_argv(m, e, name, port, ctx=None, extra=()):
    """caminus serve's flags for catalog model m (model_entry e) as server `name`: the model's first file, the port, a cache
    and a flight directory of its own; its drafter's file when its architecture loads one (caminus.arch; --no-drafter when
    that file is not on disk). No --model-alias: the server finds its catalog id itself (arch.default_alias)"""
    serve = ['--model', e['path'], '--port', str(port),
             '--cache-dir', os.path.join(config.kv_dir(), name), '--flight-dir', os.path.join(config.flight_dir(), name)]
    dk = models.drafter_item(m)
    if dk is not None:
        custom = _serve_arg(extra, '--drafter')
        if custom:
            pass                           # (the supplied file is validated by serve, not replaced by catalog fallback)
        elif e['paths'].get(dk): serve += ['--drafter', e['paths'][dk]]
        elif _arch_drafter(m) == 'dflash' and _serve_arg(extra, '--draft') != 'dflash': serve += ['--no-drafter']
        # (an optional drafter's missing file adds nothing: --no-drafter would turn the server's --draft off, and a model
        #  whose own drafter is another kind (swift-1.5's MTP block) must keep it; an explicit --draft dflash must either
        #  load its file or report it missing, never silently become off)
    if ctx:
        serve += ['--max-ctx', str(ctx)]
    return serve + list(extra)


def start_one(model_name, name=None, card=None, port=None, ctx=None, extra=(), wait=True, log=print, argv_override=None, devs_override=None,
              cards=None):
    if models.paused(model_name):
        raise ManageError(models.paused(model_name))
    if model_name not in models.BY_NAME:
        raise ManageError(f'unknown model {model_name!r} (caminus models: {", ".join(models.BY_NAME)})')
    m = models.BY_NAME[model_name]
    e = model_entry(m)
    if not e['present']:
        raise ManageError(f'{m.name} is not downloaded (caminus pull {m.name})')
    name = name or m.name
    running = [s for s in all_servers() if alive(s.get('guard_pid'))]
    if any(s['name'] == name for s in running):
        raise ManageError(f'{name} is already running (caminus ps)')
    held = {d for s in running for d in s.get('devices', [])}
    nc = int(cards) if cards else m.cards
    if nc != m.cards:                            # (--cards 2: the model's two-card form, where its architecture has one)
        try:
            from .. import arch
            arch.lookup(m.arch).variant(nc)
        except ValueError as ex:
            raise ManageError(f'{m.name}: {ex}') from None
        extra = list(extra) + ['--cards', str(nc)]
    devs = devs_override or pick_cards(m, e, card, held, ctx=ctx, cards=nc)
    busy = busy_elsewhere(devs)
    if busy:
        raise ManageError(f'VRAM in use on {", ".join(busy)} by a process Caminus did not start: stop it first')
    port = pick_port(port, {s['port'] for s in running})
    d = server_dir(name)
    os.makedirs(d, mode=0o700, exist_ok=True)
    logp = os.path.join(d, 'serve.log')
    if argv_override is not None:
        serve = list(argv_override)
        for i, x in enumerate(serve):
            if x == '--port' and i + 1 < len(serve):
                serve[i + 1] = str(port)
    else:
        serve = serve_argv(m, e, name, port, ctx, extra)
    cmd = guarded.command_reloading('serve', serve, logp, 0)   # (exit 75, a broken engine: reloaded once)
    view = {c['index']: c for c in card_view()}
    env = dict(os.environ, HIP_VISIBLE_DEVICES=','.join(str(view[i]['hip']) for i in devs))   # (HIP's indices, the iGPU counted)
    env.pop('CAMINUS_GUARD', None)
    cs = {i: c['name'] for i, c in view.items()}
    with open(os.path.join(d, 'guard.txt'), 'ab') as g:
        g.write(f"== start {time.strftime('%F %T')} {shlex.join(cmd)}\n".encode())
        g.flush()
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=g, stderr=subprocess.STDOUT, start_new_session=True)
    s = dict(name=name, model=m.name, path=e['path'], port=port, devices=devs, card_names=[cs.get(i, str(i)) for i in devs],
             pcis=[view[i]['pci'] for i in devs],
             argv=serve, guard_pid=p.pid, log=logp, started=time.time(), alias=m.alias)
    with open(os.path.join(d, 'server.json'), 'w') as f:
        json.dump(s, f, indent=1)
    log(f"{name}: starting {m.name} on {'+'.join(s['card_names'])} at http://127.0.0.1:{port} (log: caminus logs {name})")
    if wait:
        t0 = time.time()
        while True:
            v = server_view(s)
            if v['state'] == 'ready':
                log(f"{name}: ready in {time.time() - t0:.0f} s: http://127.0.0.1:{port}/v1 (model id {m.alias})")
                break
            if v['state'] in ('failed', 'stopped'):
                raise ManageError(f"{name} exited during startup: {v['error'] or 'caminus logs ' + name}")
            if time.time() - t0 > 900:
                raise ManageError(f'{name} not ready after 900 s (still running: caminus logs {name})')
            time.sleep(1.0)
    return s


def stop_one(name, log=print, timeout=170):
    s = load_server(name)
    if s is None:
        raise ManageError(f'no server named {name!r} (caminus ps)')
    g = s.get('guard_pid')
    if alive(g):
        pid = server_pid(s)
        try:
            os.kill(pid or int(g), signal.SIGTERM)
        except OSError:
            pass
        t0 = time.time()
        while alive(g):
            if time.time() - t0 > timeout:
                raise ManageError(f'{name} still running after {timeout} s (guard pid {g}): caminus logs {name}')
            time.sleep(0.5)
    with contextlib.suppress(OSError):
        os.remove(os.path.join(server_dir(name), 'server.json'))
    log(f'{name}: stopped')


def result(ok, note=None, error=None):
    print(json.dumps({'ok': bool(ok), 'note': note or '', 'error': error or ''}))
    return 0 if ok else 1


def run_action(a, fn):
    """an action's output: --json prints one result document (progress to stderr), else plain lines"""
    say = (lambda s: print(s, file=sys.stderr, flush=True)) if a.json else (lambda s: print(s, flush=True))
    try:
        note = fn(say)
    except (ManageError, config.ConfigError) as e:
        if a.json:
            return result(False, error=str(e))
        print(f'caminus: {e}', file=sys.stderr)
        return 1
    if a.json:
        return result(True, note)
    return 0


def cmd_start(argv):
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog='caminus start')
    ap.add_argument('model', nargs='?')
    ap.add_argument('--name')
    ap.add_argument('--card', help='auto (default), or a card index')
    ap.add_argument('--cards', type=int, choices=(1, 2), help='how many cards (default: the model\'s own; 2: swift-1.5 / '
                    'thinkingcap on two cards, tensor parallel)')
    ap.add_argument('--port', type=int)
    ap.add_argument('--ctx', help='auto (default: the largest context that fits) or a number of positions')
    ap.add_argument('--team', choices=sorted(TEAMS))
    ap.add_argument('--no-wait', action='store_true', help='return once the server is launched (caminus ps shows it loading)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if bool(a.team) == bool(a.model):
        ap.error('a model name or --team')

    def go(say):
        if a.team:
            why = next((models.paused(n) for _, n, _, _ in TEAMS[a.team] if models.paused(n)), None)
            if why:                               # (before either server starts: a half-started team holds a card)
                raise ManageError(f'the {a.team} team is paused with its model: {why}')
            cs = sorted(card_view(), key=lambda c: (-c['total_mib'], c['pci'] or c['card']))   # (the biggest card is the brain)
            if len(cs) < 2:
                raise ManageError('the brain / hands team needs two supported cards')
            notes = []
            for role, model_name, which, port in TEAMS[a.team]:
                dev = cs[0]['index'] if which == 'big' else cs[1]['index']
                s = start_one(model_name, name=role, card=str(dev), port=port, wait=not a.no_wait, log=say)
                notes.append(f"{role}: {model_name} on {'+'.join(s['card_names'])} :{s['port']}")
            return '; '.join(notes)
        s = start_one(a.model, name=a.name, card=a.card, port=a.port, ctx=a.ctx, extra=extra, wait=not a.no_wait, log=say,
                      cards=a.cards)
        return f"{s['name']}: {s['model']} on {'+'.join(s['card_names'])} at http://127.0.0.1:{s['port']}"
    return run_action(a, go)


def cmd_stop(argv):
    ap = argparse.ArgumentParser(prog='caminus stop')
    ap.add_argument('name', nargs='?')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if bool(a.all) == bool(a.name):
        ap.error('a server name or --all')

    def go(say):
        names = [s['name'] for s in all_servers()] if a.all else [a.name]
        for n in names:
            stop_one(n, log=say)
        return f"stopped {', '.join(names)}" if names else 'no servers'
    return run_action(a, go)


def cmd_restart(argv):
    ap = argparse.ArgumentParser(prog='caminus restart')
    ap.add_argument('name')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)

    def go(say):
        s = load_server(a.name)
        if s is None:
            raise ManageError(f'no server named {a.name!r} (caminus ps)')
        stop_one(a.name, log=say)
        n = start_one(s['model'], name=s['name'], port=s['port'], argv_override=s['argv'], devs_override=s['devices'], log=say)
        return f"{n['name']} restarted at http://127.0.0.1:{n['port']}"
    return run_action(a, go)


def cmd_logs(argv):
    ap = argparse.ArgumentParser(prog='caminus logs')
    ap.add_argument('name')
    ap.add_argument('-f', '--follow', action='store_true')
    a = ap.parse_args(argv)
    s = load_server(a.name)
    p = s['log'] if s else os.path.join(server_dir(a.name), 'serve.log')
    if not os.path.exists(p):
        print(f'caminus logs: no log for {a.name!r} ({p})', file=sys.stderr)
        return 1
    if a.follow:
        os.execvp('tail', ['tail', '-n', '200', '-F', p])
    if sys.stdout.isatty() and shutil.which('less'):
        os.execvp('less', ['less', '+G', p])
    with open(p, errors='replace') as f:
        sys.stdout.write(f.read())
    return 0


def panel_binary():
    """caminus-panel: CAMINUS_PANEL, beside this package (caminus/share/caminus-panel), a checkout's panel/ build, PATH"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    packaged = os.path.join(here, 'share', 'caminus-panel')
    if os.path.isfile(packaged) and not os.access(packaged, os.X_OK):   # an installer that dropped the wheel's exec bit
        try: os.chmod(packaged, 0o755)
        except OSError: pass
    for p in (os.environ.get('CAMINUS_PANEL'), packaged,
              os.path.join(os.path.dirname(os.path.dirname(here)), 'panel', 'caminus-panel'), shutil.which('caminus-panel')):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def cli_argv():
    """how caminus-panel runs this caminus (the same package copy: an installed one or a checkout's python/)"""
    boot = f'import sys; sys.path.insert(0, {guarded.PARENT!r}); from caminus.__main__ import main; sys.exit(main())'
    return [sys.executable, '-I', '-c', boot]


def exec_panel(args):
    b = panel_binary()
    if b is None:
        print('caminus: caminus-panel is not installed (build it: cd panel && go build -o caminus-panel .; or set CAMINUS_PANEL)',
              file=sys.stderr)
        return 1
    os.execve(b, [b, *args], dict(os.environ, CAMINUS_CLI=json.dumps(cli_argv())))


def cmd_panel(argv):
    return exec_panel(list(argv))


def top_config(s):
    """Merge monitor preferences into a server's generated config; leave the preferences file untouched."""
    from . import qwen_code
    source = os.path.abspath(os.path.expanduser(os.environ.get('LLMTOP_CONFIG') or
                              os.path.join(config.xdg('XDG_CONFIG_HOME', '.config'), 'caminus', 'top.json')))
    prefs = {}
    try:
        with open(source, encoding='utf-8-sig') as f:
            prefs = json.load(f)
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError) as err:
        raise ManageError(f'cannot read monitor settings {source}: {err}') from None
    if not isinstance(prefs, dict):
        raise ManageError(f'monitor settings {source}: expected a JSON object')
    # Go resolves paths against the generated file. Keep a preference's relative hold file beside its source.
    hold = prefs.get('hold_file')
    if isinstance(hold, str) and hold:
        hold = os.path.expanduser(hold)
        prefs['hold_file'] = hold if os.path.isabs(hold) else os.path.join(os.path.dirname(source), hold)
    prefs.update(server_url=f"http://127.0.0.1:{s['port']}", backend='caminus', server_process='caminus',
                 sessions_dir=os.path.join(qwen_code.home_for(s), 'projects'))
    d = os.path.join(config.state_dir(), 'servers', s['name'])
    p = os.path.join(d, 'top.json')
    os.makedirs(d, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.top-', suffix='.json', dir=d)  # 0600: preferences can include a web token
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(prefs, f, indent=1)
        os.replace(tmp, p)  # concurrent monitor launches never read a partial config
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    return p


def cmd_top(argv):
    ap = argparse.ArgumentParser(prog='caminus top')
    ap.add_argument('name', nargs='?')
    ap.add_argument('-config', '--config', dest='monitor_config', metavar='FILE', help='use a monitor config directly')
    a, rest = ap.parse_known_args(argv)
    ss = [s for s in all_servers() if alive(s.get('guard_pid'))]
    if a.name:
        ss = [s for s in ss if s['name'] == a.name]
    if not ss:
        print('caminus top: no running server' + (f' named {a.name!r}' if a.name else '') + ' (caminus ps)', file=sys.stderr)
        return 1
    try:
        p = os.path.expanduser(a.monitor_config) if a.monitor_config else top_config(ss[0])
    except (ManageError, OSError) as err:
        print(f'caminus top: {err}', file=sys.stderr)
        return 1
    return exec_panel(['top', '-config', p, *rest])
