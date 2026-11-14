"""`caminus serve`: the OpenAI / Anthropic server on the K3 engine (M7).

    caminus serve [--config FILE] [--model GGUF] [--drafter GGUF | --no-drafter]
        [--host 127.0.0.1] [--port 8092] [--max-ctx auto|N] [--chunk 2048] [--cache-dir ~/.cache/caminus/kv] [--no-disk-cache] [--cache-cap-gb N]
        [--log-dir DIR] [--no-stats] [--hw-root /] [--idle-park-s 28800] [--max-queue 16] [--model-alias k3-coder]
        [--drain-s 60] [--park-s 90] [--no-side-route]

(Installed, `caminus serve` runs itself under the package's guard (caminus/share/guard.sh) unless it already runs under one.
From a checkout: `python -I tools/caminus_serve.py ...` under tools/guard.sh, which puts the repo's python/ on sys.path;
tools/k3_serve.py is the mock launcher. Paths default through caminus.config: --model the Coder's first shard, --drafter the
DFlash GGUF, --cache-dir / --flight-dir under ~/.cache/caminus.) A config file (JSON, or TOML by its .toml suffix) gives the same keys with underscores (model, drafter, port,
max_ctx, cache_dir, ...); flags override it.

What a model is served with (its engine, prompt path, drafter, cards, max_ctx auto, model ids) is the architecture
registry's (caminus.arch, by the GGUF's architecture); this module reads it once per start (arch.of) and never branches on
the architecture itself.

Order: the process names itself `caminus` (before any thread starts: llm-top finds it by /proc comm), the HTTP surface comes
up answering 503 'Loading model' (llama-server's behaviour; /health too), then the engine is built in the serving order
(Arch.build; the Coder: FlashNextTP T = 4 at max_ctx with the DFlash features and the prefill counted in its VRAM check, the
DFlash drafter, the batched prefill), the prompt cache last (its VRAM tier is sized from the headroom left), and the server
switches to the K3Session. The model ids: the Coder's are template.MODEL_PROFILES (`k3-coder`: medium reasoning,
`k3-coder-nothink`: thinking off, `k3-coder-xhigh`, `k3-coder-low`); any other model answers to its own id
(arch.default_alias: the catalog's, e.g. `qwen3.6`, `arex-2`) and that id's -nothink. Every GPU process runs under
tools/guard.sh (CLAUDE.md); tests use --port 0, a scratch --cache-dir and a scratch --flight-dir.
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

from .. import arch as _arch
from .. import config as _config

# the model paths and cache directories from caminus.config (the environment, ~/.config/caminus/config.toml, then the XDG
# defaults: the files `caminus download` puts under the model directory, ~/.cache/caminus/{kv,flight})
DEFAULT_GGUF = _config.coder()
DEFAULT_DRAFTER = _config.dflash()
DEFAULTS = dict(model=DEFAULT_GGUF, drafter=DEFAULT_DRAFTER, no_drafter=False, draft=None, host='127.0.0.1',
                port=8092, max_ctx='auto', auto_cache_slots=4, chunk=2048, cache_dir=_config.kv_dir(), no_disk_cache=False, cache_cap_gb=None, kv_host=False,
                log_dir='', no_stats=False, no_side_route=False, no_loop_guard=False, hw_root='/', idle_park_s=28800.0, max_queue=16,
                model_alias='', ping=10.0, incremental_tool_args=False, drain_s=60.0, park_s=90.0,
                flight_dir=_config.flight_dir(), flight_cap_gb=10.0, cards=None)


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


# the exit code after a terminal numerical failure (non-finite logits or a refused KV encoding), distinct from the guard's
# 99 / 98 (GPU faults, never restarted): the systemd unit (RestartForceExitStatus) and caminus start reload once on it
EXIT_BROKEN = 75


def options_json():
    """every option caminus serve accepts, for caminus panel's advanced layout (and scripts): flag, dest, type (int / float /
    str / bool / ctx), the default caminus serve uses, the help text; read from the parser itself, so it never drifts"""
    out = []
    for act in parser()._actions:
        if not act.option_strings or act.dest == 'help':
            continue
        flag = max(act.option_strings, key=len)
        if act.nargs == 0:
            typ = 'bool'
        elif act.type is ctx_arg:
            typ = 'ctx'
        elif act.type in (int, float):
            typ = act.type.__name__
        else:
            typ = 'str'
        out.append(dict(flag=flag, dest=act.dest, type=typ, default=DEFAULTS.get(act.dest), help=act.help or '',
                        choices=list(act.choices) if act.choices else None))
    return out


def ctx_arg(v):
    """--max-ctx: a positive int or 'auto'"""
    if str(v).strip().lower() == 'auto': return 'auto'
    n = int(v)
    if n < 1: raise argparse.ArgumentTypeError(f'max_ctx {v!r}: a positive int or auto')
    return n


def parser():
    ap = argparse.ArgumentParser(prog='caminus serve', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='a JSON / TOML file with the options below (flags override it)')
    ap.add_argument("--model", help="the GGUF (default: caminus.config coder); its architecture picks how it is served "
                                    "(caminus.arch): the Coder (its first shard; two cards), a one-card model: qwen35moe "
                                    "(Qwen3.6-35B-A3B) / qwen35 (dense: AREX-2), or qwen3next (Qwen3-Coder-Next; two cards). "
                                    "The cards are picked from the host (caminus.device.host: the bigger card first for a "
                                    "two-card model, the smallest that fits for a one-card one); HIP_VISIBLE_DEVICES limits the choice")
    ap.add_argument('--cards', type=int, choices=(1, 2), help='the cards the model runs on (default: its own count; 2: the '
                    'dense own-code models swift-1.5 / thinkingcap on two cards, tensor parallel)')
    ap.add_argument('--drafter', help='the Coder\'s DFlash GGUF (default: caminus.config dflash)')
    ap.add_argument('--no-drafter', action='store_true', default=None, help='no DFlash drafter (plain decode)')
    ap.add_argument('--draft', choices=('off', 'mtp', 'dflash'), default=None, help='speculative decoding: off (plain '
                    'decode), dflash (the DFlash2 drafter: the Coder\'s sidecar, needs --drafter; swift-1.5: the 27B base\'s '
                    'DFlash2 file, one card), mtp (the model\'s own MTP block, one '
                    'card: swift-1.5; default: the architecture\'s drafter when its file is present, else off)')
    ap.add_argument('--host')
    ap.add_argument('--port', type=int, help='0: any free port (printed)')
    ap.add_argument('--max-ctx', type=ctx_arg, help='the engine\'s KV positions, or auto (the default): the largest context '
                                                   'that fits the card(s) now with a working prompt cache (Arch.auto_max_ctx; the '
                                                   'Coder: 262,144)')
    ap.add_argument('--auto-cache-slots', type=int, help='max_ctx auto: VRAM checkpoint slots the context must leave room for '
                                                         '(default 4; more: a smaller context, more cached restore points)')
    ap.add_argument('--chunk', type=int, help='the batched prefill\'s chunk (tokens)')
    ap.add_argument('--cache-dir', help='the prompt cache\'s disk tier (default ~/.cache/caminus/kv)')
    ap.add_argument('--no-disk-cache', action='store_true', default=None, help='no disk tier (VRAM checkpoints only)')
    ap.add_argument('--cache-cap-gb', type=float, help='the disk tier\'s size cap in GB (default 30; 64 in the long-context mode)')
    ap.add_argument('--kv-host', action='store_true', default=None, help='the Coder: its attention KV cache in pinned host memory '
                                                                      '(both cards read it directly; frees VRAM; same results). A '
                                                                      '--max-ctx past the model\'s context_length implies it (the '
                                                                      'long-context mode: also NTK RoPE scaling, no disk tier)')
    ap.add_argument('--log-dir', help="an access log: one more JSONL row per request in DIR (default '' = off: the flight "
                                      "recorder's line is the per-request log)")
    ap.add_argument('--no-side-route', action='store_true', default=None,
                    help='every request to the prompt cache\'s main slot (default: short requests unrelated to the conversation '
                         'go to its side slot: PromptCache.route)')
    ap.add_argument('--no-stats', action='store_true', default=None, help='the metrics hook off')
    ap.add_argument('--no-loop-guard', action='store_true', default=None,
                    help='allow sustained identical prose lines in answers (default: stop after eight; code and tool payloads excluded)')
    ap.add_argument('--hw-root', help="a sysfs root for the hardware readings (default '/': the engine's own cards, by PCI slot; "
                                      "'' = no hardware readings)")
    ap.add_argument('--idle-park-s', type=float, help='park the live conversation in the background after this many idle seconds '
                                                     '(default 28800: 8 hours; 0 = never); only when it gained >= 4096 positions '
                                                     'since its last park. A clean shutdown parks it too (--park-s)')
    ap.add_argument('--max-queue', type=int)
    ap.add_argument('--model-alias', help="the model id the server answers to (default '': the model's own: the catalog's for a "
                                          "catalog file, else its architecture's: k3-coder, qwen3.6, qwen3.5, qwen3-coder-next)")
    ap.add_argument('--ping', type=float)
    ap.add_argument('--incremental-tool-args', action='store_true', default=None)
    ap.add_argument('--flight-dir', help="the flight recorder: one JSONL line per request and replay bundles (default "
                                         "~/.cache/caminus/flight, 0700 / 0600; '' = off; tests: a scratch directory)")
    ap.add_argument('--flight-cap-gb', type=float, help='the flight directory\'s size cap (default 10 GB; the oldest bundles go first)')
    ap.add_argument('--drain-s', type=float, help='shutdown: seconds to wait for the running job to stop (default 60)')
    ap.add_argument('--park-s', type=float, help='shutdown: seconds the park of the live conversation may take (default 90; 0 = no park)')
    return ap


def parse(argv):
    a = parser().parse_args(argv)
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


def eos_ids_of(tok):
    """the ids that end a generation, from the model's own vocabulary: its eos (<|im_end|>) and <|endoftext|> (the Coder and
    Qwen3.6 / AREX-2: 248046, 248044; Qwen3-Coder-Next's 151,936-token vocabulary: 151645, 151643)"""
    return tuple(dict.fromkeys(i for i in (tok.eos_id, tok.special.get('<|endoftext|>')) if i is not None and i >= 0))


def build_engine(opt, stats, log=print):
    """the serving configuration in its order; returns (K3Session, parts dict)"""
    import torch
    from ..cache import prompt as PC
    from .k3_session import K3Session
    pl = opt.get('placement')
    if pl and torch.cuda.device_count() != len(pl['cards']):   # (HIP read HIP_VISIBLE_DEVICES before main() set it: the cards
        raise RuntimeError(f'caminus serve: HIP sees {torch.cuda.device_count()} devices, the placement chose {len(pl["cards"])} '
                           f'({", ".join(pl["names"])}): HIP started before the cards were chosen')   # would not be the ones chosen)
    t0 = time.time()
    A = _arch.ARCHS[opt['arch']] if opt.get('arch') else _arch.of(opt['model']).variant(opt.get('cards'))
    from ..errors import NumericalFailure
    try:
        m, pf, df, dr = A.build(opt, log)
        for d in m.devs: torch.cuda.synchronize(d)
        if callable(getattr(m, 'selfcheck', None)):
            # (fresh-state fixed tokens: non-finite logits or an invalid KV encoding must stop before a cache exists)
            m.selfcheck()
    except NumericalFailure as e:
        log(f'caminus serve: the startup self-check failed: {e}')
        raise SystemExit(EXIT_BROKEN)
    disk = not opt['no_disk_cache']
    pc = PC.PromptCache(m, pf, dr, cache_dir=opt['cache_dir'] if disk else None, disk=disk,
                        cap_gb=float(opt['cache_cap_gb']) if opt.get('cache_cap_gb') else PC.DEFAULT_CAP_GB)
    sess = K3Session(m, pf, dr, pc, stats=stats, idle_s=(float(opt['idle_park_s']) or None), side_route=not opt['no_side_route'],
                     eos_ids=opt.get('eos_ids') or (248046, 248044))
    log(f'caminus serve: engine built in {time.time() - t0:.1f} s: max_ctx {m.max_ctx} (usable {sess.max_ctx}), qsa {m.qsa}, '
        f'drafter {"DFlash (device accept)" if dr is not None and getattr(dr, "kind", "dflash") == "dflash" else (f"{getattr(dr, "kind", "mtp")} (host accept)" if dr is not None else "none")}, '
        f'prefill chunk {pf.chunk} (T* {pf.crossover}), '
        f'cache {pc.n_slots} VRAM slots ({" / ".join(f"{v:.0f}" for v in pc.tier_mib)} MiB), disk {"on: " + pc.store.root if pc.store else "off"}, '
        f'side slot {pc.side_rows} positions (routing {"on" if sess.side_route else "off"})',
        flush=True)
    return sess, dict(m=m, df=df, dr=dr, pf=pf, pc=pc)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if '--options-json' in argv:                 # (caminus panel's advanced layout: nothing is loaded, no GPU)
        import json
        print(json.dumps(options_json()))
        return 0
    opt = parse(argv)
    from ..device.torchcheck import absent, prepare_runtime
    prepare_runtime()                           # tools/caminus_serve.py also enters here directly under an outer guard
    msg = absent('caminus serve')             # (no torch import here: set_process_name below runs before any thread starts; the
                                              # console command checks the build itself before the guard, caminus.device.hip at its first call)
    if msg:
        print(msg, file=sys.stderr)
        return 2
    if not opt['model'] or not os.path.isfile(opt['model']):
        print(f'caminus serve: the model GGUF {opt["model"]} does not exist: run `caminus download` (or caminus pull NAME), set '
              f'CAMINUS_CODER (or coder in {_config.config_path()}), or pass --model', file=sys.stderr)
        return 2
    A = _arch.of(opt['model'])                   # (the GGUF's header, read once: every later question asks the registry)
    why = _arch.paused(A)
    if why:
        print(f'caminus serve: {why}', file=sys.stderr)
        return 2
    try:
        A = A.variant(opt.get('cards'))           # (--cards 2: a one-card model's two-card form, where it has one)
    except ValueError as e:
        print(f'caminus serve: --cards {opt["cards"]}: {e}', file=sys.stderr)
        return 2
    opt['arch'] = A.name
    # the drafter kind this serve runs (--draft; the default: the architecture's own when it needs no extra file, else off)
    draft = opt.get('draft')
    if opt.get('no_drafter'): draft = 'off'
    if draft is None:
        # (the architecture's measured default: the Coder drafts - the missing DFlash file falls into the error below, as
        # before; the others by their served A/B, arch.Dense35.DRAFT_DEFAULT)
        draft = A.default_draft(opt)
        if draft != 'off':
            print(f'caminus serve: --draft {draft} (the default for {A.name} on {A.cards} card(s); --draft off: plain decoding)',
                  flush=True)
    own_dflash = draft == 'dflash' and A.drafter != 'dflash' and 'dflash' in getattr(A, 'drafters', ())
    if own_dflash:
        # (the dense 27B line's drafter: its own catalog file, not the Coder's; --drafter overrides it)
        if opt['drafter'] == DEFAULT_DRAFTER:
            from .. import models as _models
            opt['drafter'] = os.path.join(_config.model_dir(), _models.local_rel(_models.BY_KEY['dflash2-27b']))
        if A.cards != 1:
            print('caminus serve: --draft dflash on swift-1.5 runs on one card only: --cards 1, or --draft off', file=sys.stderr)
            return 2
        if not os.path.isfile(opt['drafter']):
            print(f'caminus serve: the DFlash2 GGUF {opt["drafter"]} does not exist (pass --drafter)', file=sys.stderr)
            return 2
    elif draft == 'dflash' and A.drafter != 'dflash':
        print(f'caminus serve: --draft dflash: no DFlash drafter for {A.name} (its drafter: {A.drafter or "none"})', file=sys.stderr)
        return 2
    if draft == 'mtp' and A.drafter != 'mtp':
        print(f'caminus serve: --draft mtp: no MTP drafter for {A.name} in K3 yet (--draft off: plain decoding)', file=sys.stderr)
        return 2
    if draft == 'mtp' and A.cards != 1:
        print(f'caminus serve: --draft mtp runs on one card only (the two-card drafter is not built yet): --cards 1, or --draft off',
              file=sys.stderr)
        return 2
    if draft == 'dflash' and not own_dflash:
        if not (opt['drafter'] and os.path.isfile(opt['drafter'])):
            print(f'caminus serve: the DFlash GGUF {opt["drafter"]} does not exist: run `caminus download`, set CAMINUS_DFLASH '
                  f'(or dflash in {_config.config_path()}), or pass --drafter (--no-drafter: plain decoding)', file=sys.stderr)
            return 2
    elif draft != 'dflash' and opt['drafter'] != DEFAULT_DRAFTER and not opt['no_drafter'] and 'dflash' not in getattr(A, 'drafters', ()):
        # (the dense 27B line: caminus start passes its optional drafter's file on every start; it loads with --draft dflash)
        print(f'caminus serve: --drafter not used: --draft {draft} (the DFlash file loads with --draft dflash)', flush=True)
    opt['draft'] = draft
    if A.drafter == 'dflash':
        opt['no_drafter'] = (draft == 'off')   # (the Coder loader reads no_drafter: --draft off must not load DFlash)
    # the cards, chosen before the GPU runtime starts (D-292; caminus.device.host): by PCI address, the bigger card first for a
    # two-card model; HIP_VISIBLE_DEVICES, when given, limits the choice and is rewritten into the engine's order
    from ..device import host as _host
    try:
        pl = _host.place(A.cards, fits=lambda cs: A.fits(opt, cs), split_of=A.split)
    except _host.PlacementError as e:
        print(f'caminus serve: {e}', file=sys.stderr)
        return 2
    os.environ['HIP_VISIBLE_DEVICES'] = pl['hip_visible']
    opt['placement'] = dict(cards=[c['pci'] for c in pl['cards']], names=[c['label'] for c in pl['cards']], split=pl['split'])
    print(f"caminus serve: cards {pl['note']} (HIP_VISIBLE_DEVICES={pl['hip_visible']})", flush=True)
    from .server import K3Server, set_process_name
    set_process_name('caminus')                  # on the main thread, before any thread starts
    from .hw import HwReader
    from .stats import EngineStats
    from .tokenizer import QwenTokenizer
    tok = QwenTokenizer.from_gguf(opt['model'], engine=A.tokenizer)   # (the architecture's BPE: 'own' for qwen3moe)
    # the ids that end a generation, from the model's own vocabulary: its eos (<|im_end|>) and <|endoftext|> (the Coder and
    # Qwen3.6 / AREX-2: 248046, 248044; Qwen3-Coder-Next's 151,936-token vocabulary: 151645, 151643)
    opt['eos_ids'] = eos_ids_of(tok)
    opt['max_ctx'] = ctx_arg(opt['max_ctx'])     # (a config file's value too)
    if opt['max_ctx'] == 'auto':                 # the default: as big a context as the card(s) hold (user, 11-07)
        opt['max_ctx'] = A.auto_max_ctx(opt)
    long_note = A.long_context(opt)              # (an explicit --max-ctx past the model's context_length: the long-context mode)
    if long_note: print(f'caminus serve: {long_note}', flush=True)
    opt['model_alias'] = opt['model_alias'] or _arch.default_alias(opt['model'])
    A.set_ids(opt['model_alias'], tok)
    stats = EngineStats(enabled=not opt['no_stats'], physics=A.physics(0))
    hw = None
    flight = None
    if opt['flight_dir']:
        from .flight import FlightRecorder
        flight = FlightRecorder(opt['flight_dir'], cap_gb=float(opt['flight_cap_gb']))
    srv = K3Server(LoadingSession(_arch.usable_ctx(opt['max_ctx'], A.T), eos_ids=opt['eos_ids']), tok, host=opt['host'], port=int(opt['port']),
                   log_dir=opt['log_dir'] or None, ping_interval=float(opt['ping']), max_queue=int(opt['max_queue']),
                   incremental_tool_args=bool(opt['incremental_tool_args']), model_path=opt['model'], hw=hw, stats=stats,
                   loading=True, model_alias=opt['model_alias'], flight=flight, loop_guard=not opt['no_loop_guard'])
    srv.start(background=True)
    print(f'caminus serve: http://{opt["host"]}:{srv.port} (loading)  pid={os.getpid()}', flush=True)
    from ..device import kernels
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
        hw = HwReader(root=opt['hw_root'], cards=engine_pci(parts['m']), names=tuple(stats.physics.card_names), phase=lambda: stats.phase,
                      on_host=stats.host_sample)
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
    srv.worker.on_broken = lambda why: stop.set()   # (terminal numerical state: exit EXIT_BROKEN; the launcher reloads once)
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
    if srv.worker.broken:
        print(f'caminus serve: exiting {EXIT_BROKEN} (engine broken: {srv.worker.broken})', flush=True)
        return EXIT_BROKEN
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
    elif getattr(getattr(srv, 'worker', None), 'broken', None):
        log('caminus serve: shutdown: the engine is broken (terminal numerical failure): no park (its state must not reach the disk cache)',
            flush=True)
    elif park_s > 0:
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
