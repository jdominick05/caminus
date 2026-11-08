#!/usr/bin/env python3
"""K3 server launcher (server spec step S4). CPU only for now: the engine is a MockSession (random tokens, so the text
is garbage) until the M7 wiring; --engine k3 is reserved for that.

    python -I tools/k3_serve.py --port 8092 --max-ctx 2051

The tokenizer and chat template come from the GGUF (vocab only; no tensors are read). Logs: one JSONL row per request
in --log-dir DIR (default '': off, as caminus serve: the flight recorder's line is the per-request log). The process names itself `caminus` (/proc/<pid>/comm)
before any thread starts, so llm-top finds it.

Monitoring (docs/specs/spec-metrics.md): llama-server's /health, /props, /slots, /metrics plus the caminus: series
and GET /caminus/status. The mock fills the engine-internal metrics with synthetic values (mock_engine.SynthProfile:
per-card split, batched prefill at 2,900 tok/s, restores, cache tiers; --no-synth turns it off). Hardware readings
need --hw-root (a sysfs root: '/' reads the real cards) or --hw-fixture DIR (writes a synthetic two-card tree there
and reads it); without either, the hardware series are absent. The mock never touches a GPU."""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from caminus.serve.server import set_process_name  # noqa: E402

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
DEFAULT_GGUF = devpaths.coder()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=os.environ.get('K3_GGUF', DEFAULT_GGUF), help='GGUF (tokenizer + template)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8092, help='0: any free port (the bound port is printed)')
    ap.add_argument('--engine', choices=('mock',), default='mock')
    ap.add_argument('--max-ctx', type=int, default=2051, help='the engine\'s correct maximum (2051 until QSA)')
    ap.add_argument('--ntok', type=int, default=4, help='mock: tokens per pass')
    ap.add_argument('--pass-ms', type=float, default=10.0, help='mock: sleep per pass')
    ap.add_argument('--draft', action='store_true', help='mock: draft with MockDrafter (speculative.types draft-dflash)')
    ap.add_argument('--log-dir', default='', help="an access log: one JSONL row per request in DIR ('' = off, the default)")
    ap.add_argument('--ping', type=float, default=10.0, help='Anthropic ping interval (s)')
    ap.add_argument('--max-queue', type=int, default=16)
    ap.add_argument('--incremental-tool-args', action='store_true',
                    help='stream tool-call arguments as they are parsed (default: each call whole)')
    ap.add_argument('--no-synth', action='store_true', help='mock: no synthetic engine-internal metrics')
    ap.add_argument('--mock-disk', type=int, default=32, help='mock: checkpoints in the stand-in disk tier (0: off)')
    ap.add_argument('--hw-root', default=None, help="a sysfs root for the hardware readings ('/' = the real cards)")
    ap.add_argument('--hw-fixture', default=None, help='write a synthetic two-card sysfs tree into DIR and read it')
    ap.add_argument('--no-stats', action='store_true', help='the stats hook off (the M7 overhead A/B)')
    ap.add_argument('--load-s', type=float, default=0.0, help='answer 503 "Loading model" for this many seconds')
    a = ap.parse_args()

    set_process_name('caminus')         # on the main thread, before any thread starts

    from caminus.serve.engine import MockSession
    from caminus.serve.hw import HwReader, write_fixture
    from caminus.serve.mock_engine import MockDrafter, MockEngine, SynthProfile
    from caminus.serve.server import K3Server
    from caminus.serve.stats import EngineStats
    from caminus.serve.tokenizer import QwenTokenizer

    tok = QwenTokenizer.from_gguf(a.model)
    eng = MockEngine(ntok=a.ntok, vocab=tok.n_vocab, eos_id=tok.eos_id, eos_every=200)
    stats = EngineStats(enabled=not a.no_stats)
    sess = MockSession(eng, max_ctx=a.max_ctx, eos_ids=(248046, 248044), pass_sleep=a.pass_ms / 1e3, stats=stats,
                       synth=None if a.no_synth else SynthProfile(bail_every=5000, evicted_ms=40),
                       disk_slots=a.mock_disk)
    hw = None
    if a.hw_fixture:
        hw = HwReader(root=write_fixture(a.hw_fixture))
    elif a.hw_root:
        hw = HwReader(root=a.hw_root)
    srv = K3Server(sess, tok, host=a.host, port=a.port, log_dir=a.log_dir or None, ping_interval=a.ping,
                   max_queue=a.max_queue, incremental_tool_args=a.incremental_tool_args,
                   drafter=MockDrafter(eng, 0.8) if a.draft else None, model_path=a.model, hw=hw,
                   loading=a.load_s > 0)
    srv.start(background=True)
    if a.load_s > 0:
        threading.Timer(a.load_s, srv.set_loading, (False,)).start()
    print(f'k3-server ({a.engine}) on http://{a.host}:{srv.port}  max_ctx={a.max_ctx}  pid={os.getpid()}  '
          f'hw={"none" if hw is None else f"{len(hw.cards)} cards"}', flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


if __name__ == '__main__':
    main()
