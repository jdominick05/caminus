"""`caminus serve`: the OpenAI / Anthropic server on the K3 engine (M7).

    caminus serve [--config FILE] [--model GGUF] [--drafter GGUF | --no-drafter]
        [--host 127.0.0.1] [--port 8092] [--max-ctx 262144] [--chunk 2048] [--cache-dir ~/.cache/caminus/kv] [--no-disk-cache]
        [--log-dir DIR] [--no-stats] [--hw-root /] [--idle-park-s 28800] [--max-queue 16] [--model-alias k3-coder]
        [--drain-s 60] [--park-s 90] [--no-side-route]

(Installed, `caminus serve` runs itself under the package's guard (caminus/share/guard.sh) unless it already runs under one.
From a checkout: `python -I tools/caminus_serve.py ...` under tools/guard.sh, which puts the repo's python/ on sys.path;
tools/k3_serve.py is the mock launcher. Paths default through caminus.config: --model the Coder's first shard, --drafter the
DFlash GGUF, --cache-dir / --flight-dir under ~/.cache/caminus.) A config file (JSON, or TOML by its .toml suffix) gives the same keys with underscores (model, drafter, port,
max_ctx, cache_dir, ...); flags override it.

Order: the process names itself `caminus` (before any thread starts: llm-top finds it by /proc comm), the HTTP surface comes
up answering 503 'Loading model' (llama-server's behaviour; /health too), then the engine is built in the serving order
(FlashNextTP T = 4 at max_ctx with the DFlash features and the prefill counted in its VRAM check, the DFlash drafter, the
batched prefill, the prompt cache last: its VRAM tier is sized from the headroom left), and the server switches to the
K3Session. The model profiles are template.MODEL_PROFILES (`k3-coder`: medium reasoning, `k3-coder-nothink`: thinking off,
`k3-coder-xhigh`, `k3-coder-low`). Every GPU process runs under tools/guard.sh (CLAUDE.md); tests use --port 0, a scratch
--cache-dir and a scratch --flight-dir.
The per-request log is the flight recorder (--flight-dir: one JSONL line per request); the old access log (--log-dir DIR, one
more JSONL row per request) is off by default (the lead's ruling of 11-05: two rows per request were redundant) and kept as an
option. Side-slot routing is on (K3Session / PromptCache.route: short requests unrelated to the conversation, such as harness
title calls, run in the prompt cache's side slot and never displace it); --no-side-route sends every request to the main slot.

Then (M7 ops): the hardware reader on the engine's own cards and the clock check (logged, in /caminus/status; read only).
Idle: the live conversation is parked in the background after --idle-park-s (8 h) without a request, when it gained >= 4096
positions since its last park. Shutdown (SIGTERM / SIGINT / systemctl --user stop caminus): stop accepting, cancel the
running job at its next pass boundary, wait for the engine thread (--drain-s), park the live conversation when it changed
(--park-s), release the cache (shutdown()). packaging/systemd/caminus.service runs this under tools/guard.sh."""
import argparse
import json
import os
import sys
import threading
import time

from .. import config as _config

# the model paths and cache directories from caminus.config (the environment, ~/.config/caminus/config.toml, then the XDG
# defaults: the files `caminus download` puts under the model directory, ~/.cache/caminus/{kv,flight})
DEFAULT_GGUF = _config.coder()
DEFAULT_DRAFTER = _config.dflash()
DEFAULTS = dict(model=DEFAULT_GGUF, drafter=DEFAULT_DRAFTER, no_drafter=False, host='127.0.0.1',
                port=8092, max_ctx=262144, chunk=2048, cache_dir=_config.kv_dir(), no_disk_cache=False,
                log_dir='', no_stats=False, no_side_route=False, hw_root='/', idle_park_s=28800.0, max_queue=16,
                model_alias='k3-coder', ping=10.0, incremental_tool_args=False, drain_s=60.0, park_s=90.0,
                flight_dir=_config.flight_dir(), flight_cap_gb=10.0)


def load_config(path):
    """the config file's keys (dashes or underscores) -> a dict of DEFAULTS' names; unknown keys refused"""
    if path.endswith('.toml'):
        import tomllib
        with open(path, 'rb') as f:
            cfg = tomllib.load(f)
    else:
        with open(path) as f:
            cfg = json.load(f)
    out = {}
    for k, v in cfg.items():
        k2 = k.replace('-', '_')
        if k2 not in DEFAULTS:
            raise SystemExit(f'caminus serve: unknown config key {k!r} in {path}')
        out[k2] = v
    return out


def parse(argv):
    ap = argparse.ArgumentParser(prog='caminus serve', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='a JSON / TOML file with the options below (flags override it)')
    ap.add_argument('--model', help='the Coder GGUF (its first shard; default: caminus.config coder)')
    ap.add_argument('--drafter', help='the DFlash GGUF (default: caminus.config dflash)')
    ap.add_argument('--no-drafter', action='store_true', default=None, help='no DFlash drafter (plain decode)')
    ap.add_argument('--host')
    ap.add_argument('--port', type=int, help='0: any free port (printed)')
    ap.add_argument('--max-ctx', type=int, help='the engine\'s KV positions (default 262,144: the model\'s cap)')
    ap.add_argument('--chunk', type=int, help='the batched prefill\'s chunk (tokens)')
    ap.add_argument('--cache-dir', help='the prompt cache\'s disk tier (default ~/.cache/caminus/kv)')
    ap.add_argument('--no-disk-cache', action='store_true', default=None, help='no disk tier (VRAM checkpoints only)')
    ap.add_argument('--log-dir', help="an access log: one more JSONL row per request in DIR (default '' = off: the flight "
                                      "recorder's line is the per-request log)")
    ap.add_argument('--no-side-route', action='store_true', default=None,
                    help='every request to the prompt cache\'s main slot (default: short requests unrelated to the conversation '
                         'go to its side slot: PromptCache.route)')
    ap.add_argument('--no-stats', action='store_true', default=None, help='the metrics hook off')
    ap.add_argument('--hw-root', help="a sysfs root for the hardware readings (default '/': the engine's own cards, by PCI slot; "
                                      "'' = no hardware readings)")
    ap.add_argument('--idle-park-s', type=float, help='park the live conversation in the background after this many idle seconds '
                                                     '(default 28800: 8 hours; 0 = never); only when it gained >= 4096 positions '
                                                     'since its last park. A clean shutdown parks it too (--park-s)')
    ap.add_argument('--max-queue', type=int)
    ap.add_argument('--model-alias')
    ap.add_argument('--ping', type=float)
    ap.add_argument('--incremental-tool-args', action='store_true', default=None)
    ap.add_argument('--flight-dir', help="the flight recorder: one JSONL line per request and replay bundles (default "
                                         "~/.cache/caminus/flight, 0700 / 0600; '' = off; tests: a scratch directory)")
    ap.add_argument('--flight-cap-gb', type=float, help='the flight directory\'s size cap (default 10 GB; the oldest bundles go first)')
    ap.add_argument('--drain-s', type=float, help='shutdown: seconds to wait for the running job to stop (default 60)')
    ap.add_argument('--park-s', type=float, help='shutdown: seconds the park of the live conversation may take (default 90; 0 = no park)')
    a = ap.parse_args(argv)
    opt = dict(DEFAULTS)
    if a.config:
        opt.update(load_config(a.config))
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None:
            opt[k] = v
    return opt


class LoadingSession:
    """the session the server holds while the engine loads (every route answers 503 then: nothing calls it)"""
    ntok, n_vocab, stats, wants_marks, drafter, idle_s = 1, 0, None, False, None, None

    def __init__(self, max_ctx, eos_ids=(248046, 248044)):
        self.max_ctx, self.eos_ids = max_ctx, tuple(eos_ids)

    def __getattr__(self, name):
        raise RuntimeError(f'the engine is loading ({name})')


def build_engine(opt, stats, log=print):
    """the serving configuration in its order; returns (K3Session, parts dict)"""
    import torch
    from .. import flashnext                     # noqa: F401  (puts gguf-py on the path)
    from gguf import GGUFReader
    from .. import dflash_k3 as dk, prefill as P, promptcache as PC
    from ..flashnext_tp import FlashNextTP
    from .k3_session import K3Session
    t0 = time.time()
    T = 4
    if opt['no_drafter']:
        m = FlashNextTP(opt['model'], ntok=T, max_ctx=int(opt['max_ctx']), vram_prefill=int(opt['chunk']))
        df = dr = None
    else:
        rd = GGUFReader(opt['drafter'], 'r'); f = {x.name: x for x in rd.fields.values()}
        layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
        dg = dk.geom(opt['drafter'])
        m = FlashNextTP(opt['model'], ntok=T, max_ctx=int(opt['max_ctx']), feature_layers=layers, feature_card=0,
                        vram_drafters=(('dflash', dg, None),), vram_prefill=int(opt['chunk']))
        df = dk.DFlashK3(m, opt['drafter'])
        dr = dk.DFlashK3Drafter(df)
    pf = P.Prefill(m, chunk=int(opt['chunk']))
    for d in m.devs: torch.cuda.synchronize(d)
    disk = not opt['no_disk_cache']
    pc = PC.PromptCache(m, pf, dr, cache_dir=opt['cache_dir'] if disk else None, disk=disk)
    sess = K3Session(m, pf, dr, pc, stats=stats, idle_s=(float(opt['idle_park_s']) or None), side_route=not opt['no_side_route'])
    log(f'caminus serve: engine built in {time.time() - t0:.1f} s: max_ctx {m.max_ctx} (usable {sess.max_ctx}), qsa {m.qsa}, '
        f'drafter {"DFlash (device accept)" if dr is not None else "none"}, prefill chunk {pf.chunk} (T* {pf.crossover}), '
        f'cache {pc.n_slots} VRAM slots ({pc.tier_mib[0]:.0f} / {pc.tier_mib[1]:.0f} MiB), disk {"on: " + pc.store.root if pc.store else "off"}, '
        f'side slot {pc.side_rows} positions (routing {"on" if sess.side_route else "off"})',
        flush=True)
    return sess, dict(m=m, df=df, dr=dr, pf=pf, pc=pc)


def main(argv=None):
    opt = parse(sys.argv[1:] if argv is None else argv)
    from ..torchcheck import absent
    msg = absent('caminus serve')             # (no torch import here: set_process_name below runs before any thread starts; the
                                              # console command checks the build itself before the guard, caminus.hip at its first call)
    if msg:
        print(msg, file=sys.stderr)
        return 2
    for k, what, env in (('model', 'the Coder GGUF (its first shard)', 'CAMINUS_CODER'), ('drafter', 'the DFlash GGUF', 'CAMINUS_DFLASH')):
        if (k == 'model' or not opt['no_drafter']) and not os.path.isfile(opt[k]):
            print(f'caminus serve: {what} {opt[k]} does not exist: run `caminus download`, set {env} (or {"coder" if k == "model" else "dflash"} '
                  f'in {_config.config_path()}), or pass --{k}' + ('' if k == 'model' else ' (--no-drafter: plain decoding)'), file=sys.stderr)
            return 2
    from .server import K3Server, set_process_name
    set_process_name('caminus')                  # on the main thread, before any thread starts
    from .hw import HwReader
    from .stats import EngineStats
    from .tokenizer import QwenTokenizer
    tok = QwenTokenizer.from_gguf(opt['model'])
    stats = EngineStats(enabled=not opt['no_stats'])
    hw = None
    flight = None
    if opt['flight_dir']:
        from .flight import FlightRecorder
        flight = FlightRecorder(opt['flight_dir'], cap_gb=float(opt['flight_cap_gb']))
    srv = K3Server(LoadingSession(int(opt['max_ctx']) - 2), tok, host=opt['host'], port=int(opt['port']),
                   log_dir=opt['log_dir'] or None, ping_interval=float(opt['ping']), max_queue=int(opt['max_queue']),
                   incremental_tool_args=bool(opt['incremental_tool_args']), model_path=opt['model'], hw=hw, stats=stats,
                   loading=True, model_alias=opt['model_alias'], flight=flight)
    srv.start(background=True)
    print(f'caminus serve: http://{opt["host"]}:{srv.port} (loading)  pid={os.getpid()}', flush=True)
    from .. import kernels
    print(f'caminus serve: {kernels.describe()}', flush=True)
    try:
        sess, parts = build_engine(opt, stats)
    except BaseException:
        srv.stop()
        raise
    srv.set_session(sess, sess.drafter)
    if opt['hw_root']:
        # the hardware reader on the engine's own cards (by PCI slot, engine order), its 1 Hz sampler thread (VRAM, power,
        # clocks, energy by phase, KFD evictions, page migrations); then the clock check (read only)
        from .k3_session import engine_pci
        hw = HwReader(root=opt['hw_root'], cards=engine_pci(parts['m']), phase=lambda: stats.phase, on_host=stats.host_sample)
        srv.hw = hw
        hw.start(srv.hw_interval)
        chk = hw.clock_check()
    else:
        from .hw import clock_check
        chk = clock_check([])
    srv.clock_check = chk
    print(f'caminus serve: {chk["summary"]}', flush=True)
    for c in chk['cards']:
        print(f"caminus serve: clock {c['name']} ({c['pci']}): level {c['perf_level']}, runtime PM {c['runtime_pm']} / "
              f"{c['runtime_status']}, sclk {c['sclk_mhz']} MHz{'' if c['ok'] else ': ' + '; '.join(c['notes'])}", flush=True)
    srv.set_loading(False)
    print(f'caminus serve: ready on http://{opt["host"]}:{srv.port}  max_ctx={sess.max_ctx}', flush=True)
    stop = threading.Event()
    try:
        import signal
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
    except (ValueError, OSError):
        pass
    try:
        while not stop.wait(3600):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(srv, sess, parts['pc'], drain_s=float(opt['drain_s']), park_s=float(opt['park_s']))
    return 0


def shutdown(srv, sess, pc, drain_s=60.0, park_s=90.0, log=print):
    """a clean shutdown (SIGTERM / SIGINT / a systemd stop), each part bounded: the HTTP server stops accepting and the
    running job is cancelled at its next pass (or prefill chunk) boundary, the engine thread exits (at most drain_s); then
    the live conversation is parked when it changed since its last park (K3Session.shutdown_park, at most park_s: a 262K
    conversation writes ~7 GiB in a few seconds; past the bound the process exits anyway and the next start reaps the
    temp file); then the cache's spill files are dropped and its directory released (pc.close). The systemd unit's
    TimeoutStopSec covers drain_s + park_s. Returns the park's result"""
    t0 = time.time()
    res = None
    try:
        ok = srv.drain(drain_s)
    except Exception as e:                       # noqa: BLE001 - shutting down
        ok = False
        log(f'caminus serve: shutdown: drain failed: {type(e).__name__}: {e}', flush=True)
    if not ok:
        log(f'caminus serve: shutdown: the engine thread did not stop within {drain_s:.0f} s: no park (the state is not known)', flush=True)
    elif getattr(sess, 'shutdown_park', None) is not None and park_s > 0:
        box = {}

        def run():
            try: box['r'] = sess.shutdown_park()
            except Exception as e:               # noqa: BLE001
                box['e'] = e
        t = threading.Thread(target=run, name='caminus-shutdown-park', daemon=True)
        tp = time.time(); t.start(); t.join(park_s)
        if t.is_alive():
            log(f'caminus serve: shutdown: the park did not finish within {park_s:.0f} s: exiting without it', flush=True)
            return None
        res = box.get('r')
        if 'e' in box:
            log(f'caminus serve: shutdown: park failed: {type(box["e"]).__name__}: {box["e"]}', flush=True)
        else:
            lp = getattr(pc, 'last_park', None) or {}
            log(f'caminus serve: shutdown: park {res!r} in {time.time() - tp:.2f} s'
                + (f' ({lp.get("bytes", 0) / 2**30:.2f} GiB)' if res not in (None, 'unchanged') else '')
                + (f'; {pc.park_error}' if res is None and getattr(pc, 'park_error', None) else ''), flush=True)
    try:
        srv.stop()
    except Exception:                            # noqa: BLE001 - shutting down
        pass
    try:
        pc.close()
    except Exception:                            # noqa: BLE001 - shutting down
        pass
    fl = getattr(srv, 'flight', None)
    if fl is not None:
        fl.close()
    log(f'caminus serve: stopped in {time.time() - t0:.2f} s', flush=True)
    return res


if __name__ == '__main__':
    sys.exit(main())
