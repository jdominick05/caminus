"""Replay a flight-recorder bundle offline (M7, the flight recorder; docs/specs/spec-metrics.md section 11): rebuild the
serving engine, put it in the request's start state, rerun the turn through the server's own session code with the bundle's
sampling parameters and Philox seed, and diff the generated tokens and the first logits row against the bundle. Bitwise
expected; anything else is a finding.

  caminus replay BUNDLE [--model GGUF] [--drafter GGUF] [--cache-dir DIR] [--json OUT] [--force]
        (installed: runs itself under the package's guard, its output in ~/.local/state/caminus/replay.log, printed at the end)
  tools/guard.sh LOG 1800 python -I tools/replay.py BUNDLE ...      (from a checkout)

BUNDLE: a directory under the flight recorder's bundles/ (bundle.json, first_logits.npy, state/). The start state:
  * a request that started from a cached state (live / VRAM / spill / disk): the bundle's state file (the request's prompt-end
    checkpoint with the KV + indexer rows below it, written by the prompt cache's own writer) is linked into a scratch cache
    directory and loaded explicitly (PromptCache.load_entry; the key, layout, token-hash and crc checks of an unpark);
  * a request that started cold: an empty engine and the same prefill (prepare with the same marks).
The engine key must equal the bundle's (the model, the numerics, T*, the TP layout, the drafter, torch / HIP, every build/*.hsaco
and the package's sources): a bundle saved by another build is refused (RESULT KEY MISMATCH, exit 2) unless --force (then
reported, not bitwise expected). The engine runs at the bundle's max_ctx with the DFlash drafter when the server had one; the
draft policy's pass schedule may differ from the original's (it is timing driven): tokens do not depend on it (greedy:
spec == plain; T > 0: sample-and-match with Philox by position). Logits: the first row (the one the first token was sampled
from); the device-accept passes keep no later logits on the host, so every later position is compared by its token.

--cache-dir: a scratch directory for the replay's prompt cache (default BUNDLE/replay-<pid>, inside the 0700 bundle; refused
under ~/.cache/caminus/kv, the server's); its linked entry is removed at the end (the bundle's own file stays). GPU: both cards,
one engine at a time; run under tools/guard.sh."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time


def key_diff(a, b, path=''):
    """the paths where two engine keys differ"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += key_diff(a.get(k), b.get(k), f'{path}.{k}' if path else str(k))
        return out
    return [] if a == b else [path]


def main(argv=None):
    ap = argparse.ArgumentParser(prog='caminus replay', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bundle')
    ap.add_argument('--model', default=None)
    ap.add_argument('--drafter', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    from ..device.torchcheck import refusal
    msg = refusal('caminus replay')
    if msg:
        print(msg, file=sys.stderr)
        return 2
    import numpy as np
    bdir = os.path.abspath(a.bundle)
    B = json.load(open(os.path.join(bdir, 'bundle.json')))
    if B.get('format') != 1:
        raise SystemExit(f'replay: bundle format {B.get("format")} (want 1)')
    rq, eng, ref = B['request'], B.get('engine'), B.get('cache_ref') or {}
    if eng is None:
        raise SystemExit('replay: the bundle has no engine part (saved by a session without an engine: MockSession)')
    from .. import arch as _arch
    from ..serve import cli
    from ..serve.k3_session import K3Session
    from ..serve.generate import TokenText, run_generation
    from ..serve.sampler import Sampler, SamplingParams
    from ..serve.stats import EngineStats
    from ..serve.tokenizer import QwenTokenizer
    from ..cache import prompt as PC
    import torch
    model = a.model or B.get('model_path') or cli.DEFAULT_GGUF
    A = _arch.of(model)
    if A.cards == 2:                            # (the two-card engines: the Coder, Qwen3-Coder-Next)
        for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
        assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    cdir = os.path.abspath(a.cache_dir or os.path.join(bdir, f'replay-{os.getpid()}'))
    for bad in (os.path.realpath(PC.DEFAULT_DIR),):
        if os.path.realpath(cdir) == bad or os.path.realpath(cdir).startswith(bad + os.sep):
            raise SystemExit(f'replay: --cache-dir {cdir} is the server\'s cache directory: use a scratch one')
    os.makedirs(cdir, mode=0o700, exist_ok=True); os.chmod(cdir, 0o700)
    linked = None
    if ref.get('state'):
        src = os.path.join(bdir, ref['state'])
        linked = os.path.join(cdir, os.path.basename(src))
        if not os.path.exists(linked):
            try: os.link(src, linked)
            except OSError: shutil.copyfile(src, linked); os.chmod(linked, 0o600)
    t0 = time.time()
    # the serving configuration: caminus serve's own build (the architecture registry's Arch.build: the bundle's architecture
    # picks the engine, its prompt path and drafter), the cache in the scratch directory (no retention eviction)
    T = int(eng.get('T', 4))
    opt = dict(model=model, max_ctx=int(eng['max_ctx']), chunk=2048, no_drafter=not eng.get('drafter', True),
               drafter=a.drafter or eng.get('drafter_path') or cli.DEFAULT_DRAFTER)
    m, pf, df, dr = A.build(opt, log=print, T=T)
    for d in m.devs: torch.cuda.synchronize(d)
    pc = PC.PromptCache(m, pf, dr, cache_dir=cdir, retention_days=1e6)
    out = dict(bundle=bdir, request_id=B.get('request_id'), reason=B.get('reason'), source=ref.get('source'), state=ref.get('state'))
    rc = 0
    try:
        kd = key_diff(pc.key, eng.get('key'))
        out['key_equal'] = pc.key_sha == eng.get('key_sha')
        if not out['key_equal']:
            out['key_diff'] = kd
            print(f'replay: the engine key differs from the bundle\'s in {kd[:12]}' + (' (--force: reported, not bitwise expected)' if a.force else ''), flush=True)
            if not a.force:
                print('RESULT KEY MISMATCH'); return 2
        stats = EngineStats()
        tok = QwenTokenizer.from_gguf(model)      # (the session's end ids are the model's own, as caminus serve's)
        sess = K3Session(m, pf, dr, pc, stats=stats, idle_s=None, eos_ids=cli.eos_ids_of(tok))
        ids = [int(t) for t in rq['prompt']]
        pe = ref.get('prompt_end')
        if linked is not None:
            r = pc.load_entry(os.path.basename(linked), at=pe)
            out['loaded'] = dict(st=r['st'], cached=r['cached'])
        prm = SamplingParams(**rq['params'])
        orig = [int(t) for t in rq['generated']]
        max_new = int(rq['max_new']) if rq.get('finish') in ('eos', 'stop', 'length') else len(orig)
        tt = TokenText(tok, rq.get('stops') or [], sess.eos_ids)

        def on_tokens(gids):
            return tt.feed(gids)[1]
        force = None if rq.get('force') is None else (int(rq['force'][0]), [int(t) for t in rq['force'][1]])
        marks = rq.get('marks')
        t1 = time.time()
        res = run_generation(sess, ids, Sampler(prm), max_new, on_tokens, drafter=sess.drafter, force=force,
                             tool_open_id=tok.special.get('<tool_call>'), marks=marks)
        gen = [int(t) for t in res.gen_ids]
        out.update(tokens=len(gen), tokens_bundle=len(orig), finish=res.finish, finish_bundle=rq.get('finish'),
                   source_replay=sess.last_prefill.source, cached_replay=sess.last_prefill.cached_n, replay_s=round(time.time() - t1, 2))
        n = min(len(gen), len(orig))
        out['tokens_equal'] = gen[:len(orig)] == orig if rq.get('finish') in ('cancel', 'error') else gen == orig
        out['first_token_diff'] = next((j for j in range(n) if gen[j] != orig[j]), None if len(gen) == len(orig) else n)
        fl = B.get('first_logits')
        lg = sess.last_first_logits
        if fl and lg is not None:
            ref_lg = np.load(os.path.join(bdir, fl['file']))
            out['first_logits_equal'] = ref_lg.tobytes() == np.ascontiguousarray(lg, dtype=np.float32).tobytes()
            out['first_logits_sha256'] = hashlib.sha256(np.ascontiguousarray(lg, dtype=np.float32).tobytes()).hexdigest()
            out['first_logits_sha256_bundle'] = fl['sha256']
            if not out['first_logits_equal']:
                d = np.abs(ref_lg.astype(np.float64) - np.asarray(lg, dtype=np.float64))
                out['first_logits_max_abs_diff'] = float(d.max())
        else:
            out['first_logits_equal'] = None
        ok = out['tokens_equal'] and out['first_logits_equal'] is not False
        print(f'replay: {B.get("reason")} bundle of request {B.get("request_id")}: start {ref.get("source")} '
              f'({"state file" if linked else "cold prefill"}; replay source {out["source_replay"]}, cached {out["cached_replay"]}); '
              f'{len(gen)} tokens vs {len(orig)} in the bundle, equal {out["tokens_equal"]} (first difference {out["first_token_diff"]}); '
              f'first logits bitwise {out["first_logits_equal"]}; engine built in {t1 - t0:.1f} s, replay {out["replay_s"]} s', flush=True)
        print('RESULT ' + ('BITWISE' if ok else 'DIFF'))
        rc = 0 if ok else 1
    finally:
        if a.json:
            json.dump(out, open(a.json, 'w'), indent=1, default=str)
        try:
            pc.purge(); pc.close()                      # (the linked entry: the bundle's own file stays)
        except Exception:                               # noqa: BLE001
            pass
        try: os.rmdir(cdir)
        except OSError: pass
    return rc


if __name__ == '__main__':
    sys.exit(main())
