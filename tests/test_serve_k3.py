"""M7 gates: the server front end on the real engine (caminus.serve.k3_session.K3Session over FlashNextTP T = 4 + Prefill +
DFlashK3Drafter (device accept) + PromptCache), through the HTTP server in this process (K3Server, port 0, the stats hook on).
Every GPU run under tools/guard.sh, both cards, one engine at a time.

  guard.sh LOG SECS python -I tests/test_serve_k3.py CODER STAGE... --cache-dir <job scratch>/<step>/pc [--max-ctx 33024] [--small]

Stages (each prints PASS / FAIL lines; RESULT PASS at the end when every check passed):
  t1a       greedy through the server == the engine (DFlashK3Drafter.generate with the batched prefill, from cold), bitwise
            tokens, on 20 prompts: raw token prompts (/v1/completions; K3-routed below T*, batched, across 2051, every length
            mod 4) and chat prompts (/v1/chat/completions, k3-coder / k3-coder-nothink, synthetic messages)
  t1d       seeded T > 0: with drafting == without drafting (the policy forced to plain), token for token, 3 texts x 2000
            tokens (T 1.0; one at T 0.7); a short prompt == the host reference (sampler.sampled_generate on full logits rows),
            with and without presence_penalty; the sampler's time per pass (pass_sampler_ms) -> the 0.15 ms gate
  t1dlong   t1d's drafting == no drafting on 3 prompts from the long-context text, up to 2000 tokens each (T 1.0 / 1.0 / 0.7)
  t1e       the device sampler on real rows: each card's top-k' + the penalty ids (K3Session.queue_rows / rows_wait) ==
            sampler.filter_chain on the whole row, Candidates bitwise, over passes x rows x parameter sets
  overhead  decode ms/token, server (HTTP, stats on) vs the direct engine (the same prompt-end state, DFlashK3Drafter.resume),
            interleaved medians of --reps, at a short prompt and at 32K: the <= 3 % gate
  cancel    a client that disconnects during a long cold prompt: the batched prefill stops at the next chunk boundary; the
            engine state == a cold run to that position, bitwise (the drafter's window difference reported); the next request
            continues it (source live) and its tokens == an uncancelled request's
  abort     a client that disconnects mid-generation (streaming): the accepted tokens committed, the VRAM tier kept (the next
            requests: live / vram), the state == a direct run stopped at the same token (teacher-forced continuation bitwise)
  abortx    abort() after a failure at every point of a request (injected host exceptions): after the target's launch in each
            pass form (the device step after its wait, the device sampler, the inject of the host greedy pass and of verify()):
            the pass dropped with commit 0 (it applied the commit it was launched with); a refusal before the launch: the commit
            kept; after a returned pass; in the prompt phase after first_logits launched: the live state dropped (checkpoints
            kept, the next request restores one); a refusal before prepare touched anything: the previous state unchanged. Each:
            HTTP 500, the next request's first logits and 31 greedy tokens == the reference built from cold, bitwise, the
            request's prompt-end checkpoint still in the VRAM tier, no reset. And (final round 4) the engine reset inside end()
            (a wrapper of the cache's commit): the live state dropped, the reset taken in (cold), the next request == a cold run
  widepen   penalty windows wider than the device sampler's pinned candidate buffers (repeat_last_n -1; 256 with top_k 64;
            greedy with -1): the rows go whole; HTTP 200, tokens == the host reference on full rows, drafting == forced plain
  bail      K3_TEST_HOOKS=police-controls + K3_TEST_BAIL=<n>: the n-th target pass of a request bails on every attempt (the
            engine's retries exhausted); the session restores the prompt-end checkpoint and replays: HTTP 200, tokens == no
            bail (greedy DFlash at a decode pass and at the first-logits pass; seeded T > 0)
  policy    draft only when it pays: forced plain / draft schedules leave the tokens bitwise unchanged (greedy and seeded
            T > 0); the policy's view after natural runs
  vram      the XTX reserve: sysfs use + 512 <= 23,800 after every stage (checked after each stage anyway)
  reuse     T1b through the server: turn 2 over the live decoded reply, a VRAM prompt-end restore, an exact hit == the engine
            through the same cache calls, tokens bitwise, prompt lengths 401 .. 404 (every residue mod 4)
  samppen   the sampler's time per pass with presence_penalty 1.5 (reported)
  idle      (bgpark) the idle park (1 s here; 8 h by default): a short conversation skipped; a 5,000-position one parked once in the
            background, not again while unchanged; a switch to another conversation (a 9,001-token one: a short unrelated request is a
            side call since side-slot routing) finds it parked (no write in the request path); the first again from disk (routed to
            main: 'parked'): first logits and tokens == before the park, bitwise
  supersede (final round 1) an idle park at A (a 160-token reply's live state, no checkpoint at A), two more turns, the shutdown
            park at B > A, then a request branching at A: in this process from the line's checkpoint at A, and after a reset of the
            cache from B's entry at A: first logits and 24 tokens == the in-process reference (the branch as a live continuation
            from A), bitwise; A's entry superseded only because B's holds its state
  side      side-slot routing (impl:m7-finish): a multi-turn tool conversation (chat with 20 synthetic tools, a ~4.6K-token root, then
            token-prompt continuations of the live state) interleaved with short unrelated requests (a title chat, a summary chat, a
            raw token prompt, a side call whose prompt phase fails (HTTP 500: drop_live inside the side call), a side call whose decode
            pass bails on every attempt (K3_TEST_BAIL: recover(at, ids) inside the side call)) and a long unrelated request (9,001
            tokens: main, auto-park of the conversation; its next turn routed main 'parked', unparked from disk): every main
            request's first logits and tokens == the same main requests with no side calls, bitwise; every side call's == a cold
            run of it alone; the main line's VRAM checkpoints, log and state unchanged across the side calls; main_lost 0, no reset;
            the decisions in /caminus/status cache.route and caminus:cache_route_total; a request sharing the root goes to main;
            an idle park after a side call ends it and parks the conversation
  cli       (alone: no engine in this process) `python -I tools/caminus_serve.py --port 0` in a child process: /health 503 while
            loading, then 200; one chat request, its second turn and a flight bundle of it (fix:m7: the export launches nothing),
            then a short unrelated one (a side call); SIGTERM: the conversation parked (the side call ended first)
Side-slot routing is on (K3Session's default, as caminus serve) in every stage but abortx, which tests the main slot's abort
contract and runs with K3Session.side_route = False (the side stage covers a failure inside a side call).
Inputs: synthetic chat messages written here, the D-281 texts ($ARTIFACTS/refs/d281-short tokens) and the long-context
probe contexts ($ARTIFACTS/qsa/lc ctx.ids); nothing private; the cache directory is a scratch one (refused under
~/.cache/caminus or $ARTIFACTS) and purged at the end."""
import argparse
import http.client
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates')); sys.path.insert(0, HERE)
import common as C   # noqa: E402  (tools/gates/common.py)

FAILS, NUMBERS = [], {}
SYS = 'You are a careful coding assistant. Answer briefly.'
CHATS = [
    [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'Write a Python function that returns the n-th Fibonacci number iteratively.'}],
    [{'role': 'user', 'content': 'Explain in two sentences what a hash map is.'}],
    [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'List three shell commands to inspect disk usage on Linux.'}],
    [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'What does the following do?\n\nfor i in range(3):\n    print(i * i)'}],
    [{'role': 'user', 'content': 'Give a one-line regular expression that matches an ISO date like 2024-01-31.'}],
]
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file from the workspace',
                                           'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                                                          'required': ['path']}}}]


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def host_note():
    comp = open('/proc/sys/vm/compaction_proactiveness').read().strip()
    mem = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:'))
    return f'compaction_proactiveness {comp}, MemAvailable {mem / 1024:.0f} MiB'


def vram_used():
    return [int(open(f'/sys/class/drm/{c}/device/mem_info_vram_used').read()) / 2**20 for c in ('card1', 'card2')]


def toks_of(name):
    if name.startswith('d') and name[1:].isdigit(): return C.tokens(os.path.join(C.LC, name, 'ctx.ids'))
    return C.tokens(os.path.join(C.D281_SHORT, f'e2e_{name}_decode', 'tokens.txt'))


def safe_dir(d):
    d = os.path.realpath(os.path.expanduser(d))
    for bad in (os.path.realpath(os.path.expanduser('~/.cache/caminus')), os.path.realpath(C.ARTIFACTS)):
        if d == bad or d.startswith(bad + os.sep): raise SystemExit(f'--cache-dir {d}: refused (under {bad}): use a scratch directory')
    return d


# ---------------------------------------------------------------------------------------------------------------- the rig
class Rig:
    def __init__(self, coder, max_ctx, cache_dir, ops=True):
        import torch
        from caminus import flashnext   # noqa: F401
        from gguf import GGUFReader
        from caminus import dflash_k3 as dk, prefill as P, promptcache as PC
        from caminus.flashnext_tp import FlashNextTP
        from caminus.serve.k3_session import K3Session
        from caminus.serve.server import K3Server
        from caminus.serve.stats import EngineStats
        from caminus.serve.tokenizer import QwenTokenizer
        self.torch = torch
        rd = GGUFReader(dk.DRAFTER, 'r'); f = {x.name: x for x in rd.fields.values()}
        layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
        self.idle = vram_used()
        t0 = time.time()
        self.m = FlashNextTP(coder, ntok=4, max_ctx=max_ctx, feature_layers=layers, feature_card=0,
                             vram_drafters=(('dflash', dk.geom(dk.DRAFTER), None),), vram_prefill=2048)
        self.df = dk.DFlashK3(self.m, dk.DRAFTER); self.dr = dk.DFlashK3Drafter(self.df)
        self.pf = P.Prefill(self.m, chunk=2048)
        self.sync()
        self.pc = PC.PromptCache(self.m, self.pf, self.dr, cache_dir=cache_dir, min_free_gb=10)
        self.stats = EngineStats()
        self.sess = K3Session(self.m, self.pf, self.dr, self.pc, stats=self.stats, idle_s=None)
        self.tok = QwenTokenizer.from_gguf(coder)
        # M7 ops, everything on (ops=True): the hardware reader on the engine's own cards (1 Hz sampler: sysfs, KFD evictions,
        # page migrations), the flight recorder in a scratch directory, per-card pass timing (K3Session, stats on)
        self.hw = self.flight = None
        if ops:
            from caminus.serve.flight import FlightRecorder
            from caminus.serve.hw import HwReader
            from caminus.serve.k3_session import engine_pci
            self.hw = HwReader(root='/', cards=engine_pci(self.m), phase=lambda: self.stats.phase, on_host=self.stats.host_sample)
            self.flight = FlightRecorder(os.path.join(os.path.dirname(cache_dir), 'flight'))
        self.srv = K3Server(self.sess, self.tok, port=0, log_dir=None, drafter=self.sess.drafter, stats=self.stats, proc_name=None,
                            hw=self.hw, flight=self.flight, model_path=coder)
        if ops:
            self.srv.clock_check = self.hw.clock_check()
        self.jobs = []
        sub = self.srv.submit
        def submit(spec):
            j = sub(spec); self.jobs.append(j); return j
        self.srv.submit = submit
        self.srv.start(background=True)
        self.url = f'http://127.0.0.1:{self.srv.port}'
        self.sync()
        print(f'rig: built in {time.time() - t0:.1f} s; max_ctx {max_ctx} (usable {self.sess.max_ctx}); cache {self.pc.n_slots} slots; '
              f'VRAM {[round(x) for x in vram_used()]} MiB (idle {[round(x) for x in self.idle]}); server {self.url}; {host_note()}', flush=True)

    def sync(self):
        for d in (0, 1): self.torch.cuda.synchronize(d)

    def post(self, path, body, timeout=1200):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())

    def complete(self, ids, n, **kw):
        """a /v1/completions request with a token prompt; returns (response, the job's generated ids)"""
        body = dict(prompt=list(ids), max_tokens=n, temperature=0.0); body.update(kw)
        st, r = self.post('/v1/completions', body)
        return r, list(self.jobs[-1].result.gen_ids), st

    def chat(self, messages, n, model='k3-coder', **kw):
        body = dict(model=model, messages=messages, max_tokens=n, temperature=0.0); body.update(kw)
        st, r = self.post('/v1/chat/completions', body)
        return r, list(self.jobs[-1].result.gen_ids), list(self.jobs[-1].prompt_ids), st

    def direct_greedy(self, ids, n):
        """the engine on its own: the device-accept loop with the batched prefill from cold (resets the engine: the cache
        drops its slot at its next call, external_reset)"""
        out = self.dr.generate(self.m, list(ids), n, prefill=self.pf)
        return [int(t) for t in out[len(ids):]]

    def wait_idle(self, t=600):
        t0 = time.time()
        while (self.srv.worker.busy or self.srv.worker.q.qsize()) and time.time() - t0 < t: time.sleep(0.02)

    def close(self):
        self.srv.stop()
        self.pc.purge(); self.pc.close()
        if self.flight is not None: self.flight.close()


def host_reference(R, prm, ids, n, eos=(248046, 248044)):
    """the host reference sampler on whole logits rows (sampler.Sampler.sample: filter_chain + Philox by position), one
    token per pass on the engine from cold (longctx.first_logits: m.verify's full rows), no drafts"""
    from caminus import longctx
    from caminus.serve import sampler as S
    smp = S.Sampler(prm)
    R.m.reset(); R.dr.reset()
    toks = list(ids)
    lg, st = longctx.first_logits(R.m, dict(i0=0, commit=0), toks)
    out = []
    while len(out) < n:
        y = smp.sample(lg.numpy(), toks)
        out.append(y); toks.append(y)
        if y in eos or len(out) >= n: break
        lg, st = longctx.first_logits(R.m, st, toks)
    return out


def host_reference_pc(R, prm, ids, n, eos=(248046, 248044)):
    """host_reference from the prompt-end state the server builds (a cold request through the cache: the batched prefill at
    T* and above, then longctx.first_logits), then one teacher-forced pass per token on whole logits rows"""
    from caminus import longctx
    from caminus.serve import sampler as S
    smp = S.Sampler(prm)
    R.pc.reset()
    toks = list(ids)
    o = R.pc.prepare(toks)
    lg, st = longctx.first_logits(R.m, o['st'], toks, draft=R.dr)
    out = []
    while len(out) < n:
        y = smp.sample(lg.numpy(), toks)
        out.append(y); toks.append(y)
        if y in eos or len(out) >= n: break
        lg, st = longctx.first_logits(R.m, st, toks, draft=R.dr)
    R.pc.reset()
    return out


def upto_eos(ref, eos=(248046, 248044)):
    for j, t in enumerate(ref):
        if t in eos: return ref[:j + 1]
    return ref


# ---------------------------------------------------------------------------------------------------------------- stages
def stage_t1a(R, small):
    prose, code, shell, d8 = toks_of('prose'), toks_of('code'), toks_of('shell'), toks_of('d8192')
    raw = [(prose, 30), (code, 31), (shell, 45), (prose, 47), (code, 200), (shell, 201), (prose, 202), (code, 203), (d8, 513),
           (d8, 1026), (d8, 2050), (d8, 2051), (d8, 2052), (d8, 2600), (d8, 4097)]
    if small: raw = raw[:4] + raw[8:10]
    n = 48 if small else 64
    ok_all, rows = True, []
    for src, L in raw:
        ids = [248044] + [int(t) for t in src[:L - 1]] if src[0] != 248044 else src[:L]
        r, got, st = R.complete(ids, n)
        ref = upto_eos(R.direct_greedy(ids, n))
        ok = st == 200 and got == ref
        ok_all &= ok; rows.append((L, len(got), ok, R.jobs[-1].result.cached_n))
    for k, msgs in enumerate(CHATS[:2] if small else CHATS):
        model = 'k3-coder-nothink' if k % 2 else 'k3-coder'
        kw = dict(tools=TOOLS) if k == 2 else {}
        r, got, pids, st = R.chat(msgs, n, model=model, **kw)
        ref = upto_eos(R.direct_greedy(pids, n))
        ok = st == 200 and got == ref and r['choices'][0]['message'] is not None
        ok_all &= ok; rows.append((f'chat{k}:{model}:{len(pids)}', len(got), ok, R.jobs[-1].result.cached_n))
    NUMBERS['t1a'] = rows
    check(ok_all, f't1a: greedy through the server == the engine (device accept + batched prefill, from cold), bitwise, on {len(rows)} prompts '
                  f'(length, tokens, equal, cached): {rows}; session counters {R.sess.counters}')


def stage_t1d(R, small):
    from caminus.serve import sampler as S
    texts = [('prose', 1.0), ('code', 1.0), ('shell', 1.0), ('prose', 0.7)]
    n = 300 if small else 2000
    if small: texts = texts[:2]
    ok_all, rows, smp = True, [], []
    pol = R.sess.policy
    for name, temp in texts:
        src = toks_of(name); ids = src[:200]
        kw = dict(temperature=temp, top_k=20, top_p=0.95, seed=12345)
        pol.force = None
        r1, g1, _ = R.complete(ids, n, **{k: v for k, v in kw.items()})
        res1 = R.jobs[-1].result
        smp += list(R.stats.pass_series['sampler'].window())[-max(1, res1.passes):]
        pol.force = lambda i: 'plain'
        r2, g2, _ = R.complete(ids, n, **kw)
        res2 = R.jobs[-1].result
        pol.force = None
        ok = g1 == g2 and len(g1) > 0
        ok_all &= ok
        rows.append(dict(text=name, T=temp, tokens=len(g1), equal=ok, drafted_passes=res1.passes, accepted=res1.accepted,
                         plain_passes=res2.passes, finish=(res1.finish, res2.finish)))
    check(ok_all, f't1d: seeded T > 0 with drafting == without drafting (forced plain), token for token: {rows}')
    # the host reference on a short (K3-routed) prompt: sampler.sampled_generate on full logits rows
    from caminus.serve import sampler as S
    for pp in (0.0, 1.5):
        prm = S.SamplingParams(temperature=1.0, top_k=20, top_p=0.95, presence_penalty=pp, seed=777).resolve()
        ids = [248044] + toks_of('code')[:40]
        nn = 100 if small else 200
        r, g, _ = R.complete(ids, nn, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=pp, seed=777)
        ref = host_reference(R, prm, ids, nn)
        check(g == ref, f't1d: a {len(ids)}-token prompt at T 1.0 (presence_penalty {pp}) through the server (device sampler, DFlash drafts) == '
                        f'the host reference on full rows (sampler.Sampler on each whole row, one token per pass): {len(g)} vs {len(ref)} tokens, first difference at '
                        f'{next((j for j, (a, b) in enumerate(zip(g, ref)) if a != b), None)}')
    smp = np.asarray(smp)
    NUMBERS['t1d'] = dict(rows=rows, sampler_ms=dict(median=float(np.median(smp)) if len(smp) else None, p95=float(np.percentile(smp, 95)) if len(smp) else None,
                                                     n=int(len(smp))), counters=dict(R.sess.counters))
    stt = np.asarray(list(R.sess.samp_t))
    if len(stt):
        NUMBERS['t1d']['split'] = dict(wait_ms=float(np.median(stt[:, 0])), host_ms=float(np.median(stt[:, 1])), rows=float(np.mean(stt[:, 2])))
        print(f'  sampler split (medians over {len(stt)} sampled passes): the device top-k wait + rows {np.median(stt[:, 0]):.4f} ms, host sample-and-match '
              f'{np.median(stt[:, 1]):.4f} ms, {np.mean(stt[:, 2]):.2f} tokens emitted per pass', flush=True)
    if len(smp):
        check(float(np.median(smp)) <= 0.15, f'sampler: the T > 0 pass\'s sampler part (the device top-k wait + host sampling, pass_sampler_ms) median '
                                             f'{np.median(smp):.4f} ms, p95 {np.percentile(smp, 95):.4f} ms over {len(smp)} passes (gate <= 0.15 ms)')


def stage_t1dlong(R, small):
    """t1d's drafting == no drafting on generations that run to 2000 tokens (prompts from the long-context text, which the model
    continues without an end; ignore_eos is not offered, so a run may still end at EOS: the count is reported)"""
    d = toks_of('d8192')
    n = 400 if small else 2000
    rows, ok_all = [], True
    pol = R.sess.policy
    for off, temp in ((0, 1.0), (3000, 1.0), (6000, 0.7)):
        ids = [248044] + d[off:off + 600]
        kw = dict(temperature=temp, top_k=20, top_p=0.95, seed=4242)
        pol.force = None
        _, g1, _ = R.complete(ids, n, **kw); r1 = R.jobs[-1].result
        pol.force = lambda i: 'plain'
        _, g2, _ = R.complete(ids, n, **kw); r2 = R.jobs[-1].result
        pol.force = None
        ok = g1 == g2 and len(g1) > 0
        ok_all &= ok
        rows.append(dict(offset=off, T=temp, tokens=len(g1), equal=ok, drafted_passes=r1.passes, accepted=r1.accepted, plain_passes=r2.passes,
                         finish=(r1.finish, r2.finish)))
    NUMBERS['t1dlong'] = rows
    check(ok_all, f't1d long: seeded T > 0 with drafting == without (forced plain), token for token, up to {n} tokens: {rows}')


def direct_turn(R, ids, n, eos=(248046, 248044)):
    """one turn on the engine directly, through the same cache calls the session makes (prepare, first_logits, extend, the
    device-accept loop, commit), stopping at EOS as the server does (a second run to the exact length when EOS came early)"""
    from caminus import longctx

    def run(k):
        o = R.pc.prepare(ids)
        lg, st2 = longctx.first_logits(R.m, o['st'], ids, draft=R.dr); R.pc.extend(ids, st2)
        y0 = int(np.argmax(lg))
        if k == 1:
            out, st3 = list(ids) + [y0], st2
        else:
            out, st3 = R.dr.resume(R.m, st2, list(ids) + [y0], k - 1)
        return out, st3, o['source']
    out, st3, src = run(n)
    gen = out[len(ids):]
    cut = next((j + 1 for j, t in enumerate(gen) if t in eos), None)
    if cut is not None and cut < len(gen):
        out, st3, src = run(cut)                 # (the restore of the prompt-end state: the same state as the first run's)
        gen = out[len(ids):]
    R.pc.commit(out, st3)
    return gen, src


def stage_reuse(R, small):
    """T1b through the server: a conversation's turns on the cache (turn 2 continuing the live state over the decode-built
    reply, a request diverging right after the prompt: its VRAM prompt-end checkpoint, an exact hit: the request == the live
    log) == the same request sequence on the engine directly through the same cache calls, tokens bitwise, at prompt lengths of
    every residue mod 4"""
    d8, code = toks_of('d8192'), toks_of('code')
    n = 24
    rows, ok_all = [], True
    for L in (401, 402, 403, 404) if not small else (401, 404):
        P = [248044] + d8[:L - 1]
        seq_srv, seq_dir, srcs = [], [], []
        R.pc.reset()
        _, g1, _ = R.complete(P, n); srcs.append(R.jobs[-1].result.restore); seq_srv.append(g1)
        T2 = P + g1 + code[:16]
        _, g2, _ = R.complete(T2, n); srcs.append((R.jobs[-1].result.restore, R.jobs[-1].result.cached_n)); seq_srv.append(g2)
        D = P + code[100:116]
        _, g3, _ = R.complete(D, n); srcs.append((R.jobs[-1].result.restore, R.jobs[-1].result.cached_n)); seq_srv.append(g3)
        E = D + g3[:-1]
        _, g4, _ = R.complete(E, n); srcs.append((R.jobs[-1].result.restore, R.jobs[-1].result.cached_n, R.sess.last_prefill.exact)); seq_srv.append(g4)
        R.pc.reset()
        dsrc = []
        for ids in (P, P + seq_srv[0] + code[:16], D, D + seq_srv[2][:-1]):
            g, src = direct_turn(R, ids, n); seq_dir.append(g); dsrc.append(src)
        ok = seq_srv == seq_dir and srcs[1][0] == 'live' and srcs[2][0] == 'checkpoint' and srcs[3][0] == 'live' and srcs[3][2]
        ok_all &= ok
        rows.append(dict(L=L, equal=seq_srv == seq_dir, server_sources=srcs, direct_sources=dsrc, tokens=[len(g) for g in seq_srv]))
    NUMBERS['reuse'] = rows
    check(ok_all, f'reuse (T1b through the server): turn 1, turn 2 continuing the live state over the decoded reply, a request diverging after the '
                  f'prompt (its VRAM prompt-end checkpoint), an exact hit (request == the live log): server tokens == the engine through the same '
                  f'cache calls, bitwise, at prompt lengths 401 .. 404: {rows}')


def stage_samppen(R, small):
    """the sampler's time per pass with penalties on (presence_penalty 1.5: the per-row candidate path), reported"""
    ids = [248044] + toks_of('d8192')[:600]
    n = 200 if small else 600
    R.pc.reset()
    R.sess.samp_t.clear()           # (a deque of the last 4096 sampled passes: full after t1d + t1dlong in one process)
    R.complete(ids, n, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5, seed=31)
    res = R.jobs[-1].result
    smp = np.asarray(list(R.stats.pass_series['sampler'].window())[-max(1, res.passes):])
    stt = np.asarray(list(R.sess.samp_t))
    NUMBERS['samppen'] = dict(median=float(np.median(smp)), p95=float(np.percentile(smp, 95)), passes=int(len(smp)),
                              wait=float(np.median(stt[:, 0])), host=float(np.median(stt[:, 1])), tokens_per_pass=float(np.mean(stt[:, 2])))
    print(f'  samppen: presence_penalty 1.5, T 1.0, DFlash: pass_sampler_ms median {np.median(smp):.4f} / p95 {np.percentile(smp, 95):.4f} ms over '
          f'{len(smp)} passes; the top-k wait {np.median(stt[:, 0]):.4f}, host {np.median(stt[:, 1]):.4f} ms; {np.mean(stt[:, 2]):.2f} tokens per pass '
          f'(reported: the 0.15 ms gate is measured with the penalties off, the server default)', flush=True)


def _capture(R):
    """wrap the session's prefill (its returned first logits) and the cache's prepare (its ms / source): returns the lists"""
    lgs, preps = [], []
    sp, pp = R.sess.prefill, R.pc.prepare

    def prefill(*a, **k):
        lg = sp(*a, **k); lgs.append(None if lg is None else np.array(lg, copy=True)); return lg

    def prepare(*a, **k):
        r = pp(*a, **k); preps.append(dict(source=r['source'], cached=r['cached'], ms=dict(r['ms']))); return r
    R.sess.prefill, R.pc.prepare = prefill, prepare

    def undo():
        R.sess.__dict__.pop('prefill', None); R.pc.__dict__.pop('prepare', None)
    return lgs, preps, undo


def stage_idle(R, small):
    """M7 ops item 3, the background park of idle conversations (bgpark): after idle_s without a request the live conversation
    is parked in the background when it gained >= auto_park_min positions since its last park (a short one is skipped, an
    unchanged one is not written again); a switch to another conversation then finds it parked (prepare's auto-park writes
    nothing: ms park ~0, the park counter unchanged); the first conversation's next request is restored from disk: its first
    logits and tokens == the same request served before the park, bitwise"""
    d8, code, prose = toks_of('d8192'), toks_of('code'), toks_of('prose')
    R.pc.reset()
    rows = {}
    # (a) a short conversation (< auto_park_min): the idle timer fires, nothing is written
    S = [248044] + prose[:300]
    parked0, skip0 = R.pc.stats['parked'], R.sess.counters['idle_skipped']
    R.sess.idle_s = R.srv.worker.idle_s = 1.0       # (set before the request: the worker's idle timer starts after a job)
    R.complete(S, 8)
    time.sleep(2.5)
    rows['short'] = dict(parked=R.pc.stats['parked'] - parked0, skipped=R.sess.counters['idle_skipped'] - skip0)
    # (b) a 5,000-position conversation: turn 1, turn 2 (its first logits captured), the idle park in the background
    P = [248044] + d8[:5000]
    _, g1, _ = R.complete(P, 16)
    R1 = P + g1 + code[:12]
    lgs, preps, undo = _capture(R)
    try:
        parked0 = R.pc.stats['parked']
        _, gA, _ = R.complete(R1, 16)
        lgA = lgs[-1]
        t0 = time.time()
        while R.pc.stats['parked'] == parked0 and time.time() - t0 < 20: time.sleep(0.05)
        R.pc.fence()
        rows['park'] = dict(parked=R.pc.stats['parked'] - parked0, error=R.pc.park_error, last=dict(R.pc.last_park or {}),
                            wait_s=round(time.time() - t0, 2), log=len(R.pc.log))
        time.sleep(2.5)                                 # another idle period with nothing new: not written again
        R.pc.fence()
        rows['again'] = R.pc.stats['parked'] - parked0
        R.sess.idle_s = R.srv.worker.idle_s = None
        # (c) the switch: another conversation; prepare's auto-park finds the slot parked already (a 9,001-token one: side-slot
        # routing sends a short unrelated request to the side slot, which parks nothing; this one does not fit it: main)
        p_sw = R.pc.stats['parked']
        R.complete([248044] + toks_of('d32768')[1000:10000], 8)
        sw = preps[-1]
        rows['switch'] = dict(source=sw['source'], park_ms=round(sw['ms'].get('park', 0.0), 3), parked=R.pc.stats['parked'] - p_sw,
                              route=(R.sess.last_route or {}).get('reason'))
        # (d) back to the first conversation: from disk, bitwise
        _, gB, _ = R.complete(R1, 16)
        back = preps[-1]; lgB = lgs[-1]
        rows['back'] = dict(source=back['source'], cached=back['cached'], restore_ms=round(back['ms'].get('restore', 0.0), 1),
                            route=(R.sess.last_route or {}).get('reason'))
    finally:
        undo(); R.sess.idle_s = R.srv.worker.idle_s = None
    same = lgA is not None and lgB is not None and lgA.tobytes() == lgB.tobytes()
    NUMBERS['idle'] = rows
    check(rows['short'] == dict(parked=0, skipped=1), f'idle: a {len(S)}-position conversation (< auto_park_min 4096) is not parked: {rows["short"]}')
    check(rows['park']['parked'] == 1 and rows['park']['error'] is None and rows['again'] == 1,
          f'idle: a 5,000-position conversation parked once in the background after 1.0 s idle (the default is 8 h: caminus serve '
          f'--idle-park-s 28800), not again while unchanged: {rows["park"]}, after another idle period {rows["again"]}')
    check(rows['switch']['parked'] == 0 and rows['switch']['park_ms'] < 50.0,
          f'idle: the switch to another conversation finds it parked (no write in the request path): {rows["switch"]}')
    check(rows['back']['source'] == 'disk' and same and gB == gA,
          f'idle: the first conversation again: restored from {rows["back"]}; first logits bitwise {same}, {len(gB)} tokens == before the park {gB == gA}')
    R.pc.purge()


def stage_supersede(R, small):
    """final round 1 (the lead's ruling of 11-06: park supersede must not lose a deeper restore point; the fix:m7 finding: a branch
    at the idle point fell back to 4,700, G4-bounded only). A 5,000-position conversation, turn 2 with a 160-token reply (the live
    state at A: no checkpoint holds it), the idle park in the background (1 s here), two more turns continuing from A, the
    shutdown park (K3Session.shutdown_park) at B > A, then a request branching at A (the log to A, then other tokens): in this
    process from the line's checkpoint at A (backed by the park entry), and after a reset of the cache (the VRAM tier and the live
    slot dropped, as a restart) from B's entry at A. Both: first logits and 24 greedy tokens == the in-process reference (the same
    two turns, then the branch as a live continuation from A, nothing parked), bitwise; A's entry superseded (removed) since B's
    holds its state. Before the fix B's entry had no state at A and A's was removed: the branch restored turn 2's prompt end and
    re-prefilled the reply (the batched numerics)"""
    d8, code, prose = toks_of('d8192'), toks_of('code'), toks_of('prose')
    P = [248044] + d8[:5000]
    n2 = 96 if small else 160
    rows = {}
    lgs, preps, undo = _capture(R)
    try:
        # the reference (nothing parked: an empty disk tier, the idle park off)
        R.sess.idle_s = R.srv.worker.idle_s = None
        R.pc.reset(); R.pc.purge()
        _, g1, _ = R.complete(P, 16)
        R1 = P + g1 + code[:12]
        _, g2, _ = R.complete(R1, n2)
        R.wait_idle()
        A = int(R.sess.st['i0'])
        k = next(j for j in range(len(prose)) if prose[j] != g2[-1])
        X = R1 + g2[:-1] + prose[k:k + 24]
        _, gX, _ = R.complete(X, 24)
        ref = dict(src=(preps[-1]['source'], preps[-1]['cached']), lg=lgs[-1], toks=gX)
        rows.update(A=A, reply=len(g2), ref=ref['src'])
        # the scenario: the same two turns, the idle park at A, two more turns, the shutdown park at B, the branch at A
        R.pc.reset(); R.pc.purge()
        _, h1, _ = R.complete(P, 16)
        parked0 = R.pc.stats['parked']
        R.sess.idle_s = R.srv.worker.idle_s = 1.0      # (set before the request: the worker's idle timer starts after a job)
        _, h2, _ = R.complete(R1, n2)
        t0 = time.time()
        while R.pc.stats['parked'] == parked0 and time.time() - t0 < 30: time.sleep(0.05)
        R.pc.fence()
        R.sess.idle_s = R.srv.worker.idle_s = None
        nmA = R.pc.parked[1] if R.pc.parked else None
        rows['idle'] = dict(same_turns=(h1 == g1, h2 == g2), parked=R.pc.stats['parked'] - parked0, error=R.pc.park_error,
                            line_at_A=[(ck.kind, ck.pos, ck.slot is None) for ck in R.pc.ckpts if ck.pos == A])
        R2 = R1 + h2 + code[20:40]
        _, h3, _ = R.complete(R2, 16)
        R3 = R2 + h3 + code[40:60]
        _, h4, _ = R.complete(R3, 16)
        R.wait_idle()
        B = int(R.sess.st['i0'])
        nmB = R.sess.shutdown_park()
        hB = R.pc.store.index.get(nmB) or {}
        rows['park'] = dict(B=B, B_entry=nmB is not None and nmB != 'unchanged', A_entry_left=nmA in R.pc.store.index,
                            B_has_A=any(int(x['pos']) == A for x in hB.get('ckpts', [])), superseded=R.pc.stats['superseded'],
                            kept=R.pc.stats['supersede_kept'], B_ckpts=[(x['kind'], int(x['pos'])) for x in hB.get('ckpts', [])])
        # in this process: the line's checkpoint at A (its bytes in B's entry now)
        _, gI, _ = R.complete(X, 24)
        inproc = dict(src=(preps[-1]['source'], preps[-1]['cached']), lg=lgs[-1], toks=gI, ms=round(preps[-1]['ms'].get('restore', 0.0), 1))
        # as after a restart: the VRAM tier and the live slot dropped, B's entry on disk
        R.pc.reset()
        _, gD, _ = R.complete(X, 24)
        disk = dict(src=(preps[-1]['source'], preps[-1]['cached']), lg=lgs[-1], toks=gD, ms=round(preps[-1]['ms'].get('restore', 0.0), 1))
    finally:
        undo(); R.sess.idle_s = R.srv.worker.idle_s = None
    def same(x):
        return x['lg'] is not None and ref['lg'] is not None and x['lg'].tobytes() == ref['lg'].tobytes() and x['toks'] == ref['toks']
    rows['inproc'] = dict(src=inproc['src'], restore_ms=inproc['ms'], logits_equal=same(inproc), tokens=len(inproc['toks']))
    rows['disk'] = dict(src=disk['src'], restore_ms=disk['ms'], logits_equal=same(disk), tokens=len(disk['toks']))
    NUMBERS['supersede'] = rows
    check(ref['src'] == ('live', A) and rows['reply'] >= 64, f'supersede: the reference branch at A = {A} continues the live state '
          f'{ref["src"]} (turn 2 replied {rows["reply"]} tokens: >= 64, so a fallback below A re-prefills batched)')
    check(rows['idle']['same_turns'] == (True, True) and rows['idle']['parked'] == 1 and rows['idle']['error'] is None
          and rows['idle']['line_at_A'] == [('park', A, True)],
          f'supersede: the idle park at A wrote one entry and the line holds A as a checkpoint backed by it: {rows["idle"]}')
    check(rows['park']['B_entry'] and rows['park']['B_has_A'] and not rows['park']['A_entry_left'] and rows['park']['B'] > A,
          f'supersede: the shutdown park at B holds the state at A and superseded A\'s entry: {rows["park"]}')
    check(inproc['src'] == ('spill', A) and same(inproc),
          f'supersede: the branch at A in this process restores A ({rows["inproc"]}): first logits and {len(gX)} tokens == the reference, bitwise')
    check(disk['src'] == ('disk', A) and same(disk),
          f'supersede: the branch at A after a reset of the cache unparks B\'s entry at A ({rows["disk"]}): first logits and tokens == the reference, bitwise')
    R.pc.purge()


def stage_t1e(R, small):
    from caminus.serve import candsample as CS, sampler as S
    m, sess = R.m, R.sess
    src = toks_of('d8192')[:1200]
    R.pc.reset()
    rng = np.random.default_rng(3)
    T = m.ntok; i0 = commit = 0; n_rows = n_cand = 0; bad = []
    passes = 30 if small else 120
    m.reset(); R.dr.reset()
    for p in range(passes):
        seq = src[:i0 + T]
        m.launch_tokens(seq, i0, commit); m.finish_pass(i0, commit)
        for q in range(4):
            prm = S.SamplingParams(temperature=float(rng.choice([0.0, 0.6, 1.0])), top_k=int(rng.choice([1, 20, 40, 64])),
                                   top_p=float(rng.choice([1.0, 0.95])), min_p=float(rng.choice([0.0, 0.05])),
                                   presence_penalty=float(rng.choice([0.0, 1.5])), repeat_penalty=float(rng.choice([1.0, 1.1])),
                                   seed=int(rng.integers(0, 2 ** 62))).resolve()
            hist = list(src[:i0 + 1]); drafts = list(src[i0 + 1:i0 + T])
            pid = CS.penalty_ids(prm, hist, drafts); kp = CS.k_prime(prm, len(pid))
            sess.queue_rows(0, T, pid, kp); rows = sess.rows_wait(T, pid, kp)
            for j in range(T):
                h = hist + drafts[:j]
                full = sess.full_row(j)
                r = rows[j]
                exact = r.cards is not None and not CS.needs_full(r, prm, h)
                n_cand += exact; n_rows += 1
                if not exact: r.full = full
                if prm.greedy:
                    same = CS.sample(r, S.Sampler(prm), h) == S.Sampler(prm).sample(full, h)
                else:
                    a, b = CS.distribution(r, prm, h), S.filter_chain(full, prm, h)
                    same = (np.array_equal(a.ids, b.ids) and a.logits.tobytes() == b.logits.tobytes() and a.probs.tobytes() == b.probs.tobytes())
                if not same: bad.append((p, q, j))
        commit = T; i0 += T
    R.sync()
    R.pc.reset()
    check(not bad and n_cand > n_rows // 2, f't1e: the device sampler on real rows (each card\'s top-k\' + the penalty ids\' raw logits, K3Session.queue_rows / rows_wait) '
                                            f'== filter_chain on the whole row (Candidates / the greedy token bitwise): {n_rows} rows x params, {n_cand} from the '
                                            f'candidates, {n_rows - n_cand} via the full row; mismatches {bad[:5]}')


def stage_overhead(R, small, reps):
    from caminus import longctx
    res = {}
    cases = [('short', [248044] + toks_of('prose')[:199], 128 if small else 256)]
    if not small:
        d = toks_of('d32768')
        cases.append(('32K', [248044] + d[:32000], 256))
    for name, ids, n in cases:
        R.pc.reset()
        r0, g0, _ = R.complete(ids, n)                      # (a cold request first: the prompt-end checkpoint exists afterwards)
        n_gen = len(g0)
        srv_ms, dir_ms = [], []
        for rep in range(reps):
            r, g, _ = R.complete(ids, n)
            t = r['timings']; srv_ms.append(t['predicted_ms'] / t['predicted_n'])
            ok_tok = g == g0
            # the direct engine from the same prompt-end state: prepare (VRAM restore) + first logits, then the device-accept loop
            # (the engine's own calls: card timing off for this arm, on again for the server's; the hardware sampler thread runs
            # in both)
            R.wait_idle(); ct = R.sess.card_timing
            if ct: R.m.enable_card_timing(False)
            o = R.pc.prepare(ids)
            lg, st2 = longctx.first_logits(R.m, o['st'], ids, draft=R.dr); R.pc.extend(ids, st2)
            t0 = time.perf_counter()
            y0 = int(np.argmax(lg))
            out, st3 = R.dr.resume(R.m, st2, list(ids) + [y0], n_gen - 1)
            t1 = time.perf_counter()
            R.pc.commit(out, st3)
            if ct: R.m.enable_card_timing(True)
            dir_ms.append((t1 - t0) * 1e3 / n_gen)
            ok_tok &= out[len(ids):] == g0
            if not ok_tok: check(False, f'overhead {name}: server and direct tokens differ (rep {rep})')
        ms, md = float(np.median(srv_ms)), float(np.median(dir_ms))
        res[name] = dict(tokens=n_gen, server_ms_per_token=srv_ms, direct_ms_per_token=dir_ms, median_server=ms, median_direct=md,
                         overhead_pct=(ms / md - 1) * 100)
        check(ms <= md * 1.03, f'overhead {name} ({len(ids)} prompt tokens, {n_gen} generated, greedy DFlash, stats on): server {ms:.3f} vs direct '
                                f'{md:.3f} ms/token (medians of {reps}, interleaved; server {[round(x, 3) for x in srv_ms]}, direct '
                                f'{[round(x, 3) for x in dir_ms]}): {res[name]["overhead_pct"]:+.2f} % (gate <= 3 %); {host_note()}')
    NUMBERS['overhead'] = res


def _raw_post(R, path, body, close_after_s=None, read_events=None):
    """a POST on a raw socket: closed after close_after_s seconds (before any reply) or after read_events SSE events"""
    s = socket.create_connection(('127.0.0.1', R.srv.port))
    data = json.dumps(body).encode()
    s.sendall(f'POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n'.encode() + data)
    if close_after_s is not None:
        time.sleep(close_after_s); s.close(); return None
    buf = b''; s.settimeout(600)
    while buf.count(b'\n\ndata:') < read_events:
        x = s.recv(65536)
        if not x: break
        buf += x
    s.close()
    return buf


def stage_cancel(R, small):
    import test_prompt_cache as TPC
    from caminus import longctx
    d = toks_of('d32768')
    N = 8000 if small else 16000
    ids = [248044] + d[:N - 1]
    R.pc.reset()
    t0 = time.time()
    _raw_post(R, '/v1/completions', dict(prompt=ids, max_tokens=16, temperature=0.0), close_after_s=2.0 if not small else 1.2)
    R.wait_idle()
    t_cancel = time.time() - t0
    job = R.jobs[-1]; res = job.result
    b = int(R.sess.st['i0']); st = dict(R.sess.st)
    pf = R.sess.last_prefill
    ok1 = res is not None and res.finish == 'cancel' and pf.cancelled and st['commit'] == 0 and 0 < b < N - 1 and R.pc.stats['cancelled'] >= 1
    dig_cut = TPC.edigest(R, b)
    # the next request continues the live state; its tokens == an uncancelled request's (and the engine state at the prompt end)
    plan = R.pc.plan(ids)
    r2, g2, _ = R.complete(ids, 32)
    src2 = (R.jobs[-1].result.restore, R.jobs[-1].result.cached_n)
    dig_after = TPC.edigest(R, R.sess._pe[0], df=False)
    # references: a cold run of [0, b) and an uncancelled request of the whole prompt
    m, dr, pf_ = R.m, R.dr, R.pf
    R.pc.reset()
    m.reset(); dr.reset(); pf_.run(ids, 0, 0, b, dr); R.sync()
    dig_cold = TPC.edigest(R, b)
    R.pc.reset()
    r3, g3, _ = R.complete(ids, 32)
    dig_ref = TPC.edigest(R, R.sess._pe[0], df=False)
    diff = TPC.ddiff(dig_cut, dig_cold)
    eng_diff = [k for k in diff if k not in ('dflash',)]
    gap = [g for g in R.df.gaps] if hasattr(R.df, 'gaps') else None
    check(ok1 and not eng_diff, f'cancel: a client gone after {t_cancel:.1f} s of a cold {N}-token prompt: finish cancel, the batched prefill stopped at {b} '
                                f'(commit 0); the engine state == a cold run of [0, {b}) bitwise (differing keys: {diff or "none"}; the drafter\'s window '
                                f'differs when the cut segment\'s feature export does not reach below {b}: those positions are given up, filler drafts '
                                f'until the window passes)')
    check(plan == dict(cached=b, source='live') and src2 == ('live', b) and g2 == g3 and dig_after == dig_ref,
          f'cancel: the next request continues the live state ({plan}; {src2}); its 32 tokens == an uncancelled request\'s; the engine state at '
          f'its prompt end == the uncancelled one, bitwise')
    NUMBERS['cancel'] = dict(N=N, cut=b, t_cancel_s=t_cancel, diff=diff)


def stage_abort(R, small):
    import test_prompt_cache as TPC
    from caminus import longctx
    ids = [248044] + toks_of('d8192')[:3000]
    R.pc.reset()
    # (another conversation first: checkpoints in the tier. Not a prefix of ids: had its greedy reply continued ids' own text,
    # the request below would reuse decode-built positions, which a cold run batch-prefills: G3c, not bitwise)
    R.complete([248044] + toks_of('prose')[:600], 8)
    n_vram0 = len([c for c in R.pc.ckpts if c.slot is not None])
    er0 = R.pc.stats['external_reset']
    buf = _raw_post(R, '/v1/completions', dict(prompt=ids, max_tokens=400, temperature=0.0, stream=True), read_events=6)
    R.wait_idle()
    res = R.jobs[-1].result
    gen = list(res.gen_ids)
    n_vram1 = len([c for c in R.pc.ckpts if c.slot is not None])
    st_s, toks_s = dict(R.sess.st), list(R.sess.toks)
    parts = (res.finish, len(gen), toks_s == ids + gen[:-1], (n_vram0, n_vram1), R.pc.stats['external_reset'] - er0)
    ok1 = res.finish == 'cancel' and 0 < len(gen) < 400 and toks_s == ids + gen[:-1] and n_vram1 >= n_vram0 and R.pc.stats['external_reset'] == er0
    cont = toks_of('code')[:16]
    dg_s = TPC.edigest(R, st_s['i0'], df=False)
    lg_s, _ = longctx.teacher_forced(R.m, st_s, toks_s + cont, 16)
    R.pc.reset()
    # the direct run that stopped at the same token: prepare + first logits + the device-accept loop for len(gen) - 1 tokens
    o = R.pc.prepare(ids)
    lg, st2 = longctx.first_logits(R.m, o['st'], ids, draft=R.dr); R.pc.extend(ids, st2)
    out, st3 = R.dr.resume(R.m, st2, list(ids) + [int(np.argmax(lg))], len(gen) - 1)
    toks_d = list(out[:len(ids) + len(gen) - 1])            # (the state is at the last returned token: i0 == len(toks_d))
    dg_d = TPC.edigest(R, st3['i0'], df=False)
    lg_d, _ = longctx.teacher_forced(R.m, st3, toks_d + cont, 16)
    same = out[len(ids):] == gen and lg_s.numpy().tobytes() == lg_d.numpy().tobytes()
    parts += (out[len(ids):] == gen, lg_s.numpy().tobytes() == lg_d.numpy().tobytes(), st_s, st3, TPC.ddiff(dg_s, dg_d),
              int((lg_s != lg_d).any(dim=1).nonzero()[0]) if not torch_equal(lg_s, lg_d) else None)
    R.pc.reset()
    # the VRAM tier survives an abort: a request that diverges after the prompt restores a checkpoint, one that continues reuses live
    R.complete([248044] + toks_of('prose')[:600], 8)
    _raw_post(R, '/v1/completions', dict(prompt=ids, max_tokens=400, temperature=0.0, stream=True), read_events=6)
    R.wait_idle()
    gen2 = list(R.jobs[-1].result.gen_ids)
    R.complete(ids + gen2[:-1] + [11], 4); live = (R.jobs[-1].result.restore, R.jobs[-1].result.cached_n)
    R.complete(ids + [13, 17], 4); vram = (R.jobs[-1].result.restore, R.jobs[-1].result.cached_n)
    check(ok1 and same, f'abort: a streaming client gone after a few events: finish cancel after {len(gen)} tokens, the log == prompt + reply[:-1], VRAM '
                        f'checkpoints {n_vram0} -> {n_vram1} (kept), no reset; the state == a direct run stopped at the same token (its tokens equal, 16 '
                        f'teacher-forced logits after it bitwise); (finish, tokens, log ok, VRAM slots, resets, tokens equal, logits equal, state server / direct) {parts}')
    check(live[0] == 'live' and live[1] == len(ids) + len(gen2) - 1 and vram[0] == 'checkpoint' and vram[1] > 0,
          f'abort: after another aborted generation the next request continues the live state {live}; one diverging right after the prompt '
          f'restores its prompt-end VRAM checkpoint {vram}')
    NUMBERS['abort'] = dict(gen=len(gen), vram=(n_vram0, n_vram1), live=live, vram_restore=vram)


def _post_status(R, path, body, timeout=1200):
    """a POST that returns (status, json) for any status (an injected failure answers 500)"""
    c = http.client.HTTPConnection('127.0.0.1', R.srv.port, timeout=timeout)
    c.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    r = c.getresponse(); data = r.read(); c.close()
    try: js = json.loads(data)
    except ValueError: js = {'raw': data[:200].decode('utf-8', 'replace')}
    return r.status, js


def _vram_ckpts(R):
    return sorted((c.pos, c.commit) for c in R.pc.ckpts if c.slot is not None)


def _next_server(R, Q):
    """the next request Q from whatever state the session left, through the same cache calls the session makes (prepare,
    first logits, extend), then 31 greedy tokens (the device-accept loop); returns (logits, tokens, (source, cached))"""
    from caminus import longctx
    o = R.pc.prepare(Q)
    lg, st = longctx.first_logits(R.m, o['st'], Q, draft=R.dr); R.pc.extend(Q, st)
    y, _ = R.dr.resume(R.m, st, list(Q) + [int(np.argmax(lg.numpy()))], 31)
    return lg, [int(t) for t in y[len(Q):]], (o['source'], o['cached'])


def _next_direct(R, P, Q):
    """the reference for Q: P's prompt end from cold through the cache (prepare, first logits, extend), then Q's first logits by
    K3 passes from there (a T-token pass == one-token steps), then 31 greedy tokens"""
    from caminus import longctx
    R.pc.reset()
    o = R.pc.prepare(P)
    lg, st = longctx.first_logits(R.m, o['st'], P, draft=R.dr); R.pc.extend(P, st)
    lg_d, st_d = longctx.first_logits(R.m, st, Q, draft=R.dr)
    y, _ = R.dr.resume(R.m, st_d, list(Q) + [int(np.argmax(lg_d.numpy()))], 31)
    R.pc.reset()
    return lg_d, [int(t) for t in y[len(Q):]]


def _abort_case(R, name, run, expect, base_of, cont, pe_len):
    """one failure injected into a request (run() -> HTTP status); then the next request from the state the abort left (Q =
    the session's log + cont, or base_of's when the log was dropped) == the reference from the base prompt, bitwise first logits
    and 31 greedy tokens; the VRAM tier kept (the failing request's prompt-end checkpoint, pe_len(other) tokens, is still there;
    nothing reset). expect: 'commit0' (a launched pass dropped), 'kept' (no launch: the state as it was), 'pass' (a returned pass
    dropped), 'dropped' (the live state dropped, checkpoints kept), 'intact' (a refusal before anything ran: the previous
    conversation's state and every checkpoint unchanged)"""
    other = [248044] + toks_of('prose')[:600]
    R.pc.reset()
    R.complete(other, 8)
    R.wait_idle()
    st_before, toks_before = dict(R.sess.st), list(R.sess.toks)
    ck0, er0, rs0 = _vram_ckpts(R), R.pc.stats['external_reset'], R.srv.worker.stats['resets']
    c0, gap0 = dict(R.sess.counters), R.pc.stats['inject_gap_positions']
    http_st = run(other)
    R.wait_idle(); R.sync()
    toks_s, st_s = list(R.sess.toks), dict(R.sess.st)
    ck1 = _vram_ckpts(R)
    dc = {k: R.sess.counters[k] - c0[k] for k in ('aborts', 'aborts_commit0', 'dropped_live')}
    gaps = R.pc.stats['inject_gap_positions'] - gap0
    P, Q = base_of(other, toks_s)
    Q = Q + cont
    lg_s, y_s, src = _next_server(R, Q)
    lg_d, y_d = _next_direct(R, P, Q)
    same = lg_s.numpy().tobytes() == lg_d.numpy().tobytes()
    a_, b_ = lg_s.numpy().astype(np.float64), lg_d.numpy().astype(np.float64)
    shape = {'commit0': st_s['commit'] == 0 and dc['aborts_commit0'] == 1 and dc['dropped_live'] == 0 and src == ('live', len(toks_s)),
             'pass': st_s['commit'] == 0 and dc['aborts_commit0'] == 0 and dc['dropped_live'] == 0 and src == ('live', len(toks_s)),
             'kept': dc['aborts_commit0'] == 0 and dc['dropped_live'] == 0 and src == ('live', len(toks_s)),
             'intact': dc['aborts_commit0'] == 0 and dc['dropped_live'] == 0 and src == ('live', len(toks_s)) and toks_s == toks_before
                       and st_s == st_before,
             'dropped': not toks_s and dc['dropped_live'] == 1 and src[0] in ('vram', 'spill') and src[1] > 0}[expect]
    L = pe_len(other)
    kept_ck = any(L - R.m.ntok <= c[0] <= L for c in ck1) and (expect != 'intact' or set(ck0) <= set(ck1))
    ok = (http_st == 500 and dc['aborts'] == 1 and shape and kept_ck and same and y_s == y_d
          and R.pc.stats['external_reset'] == er0 and R.srv.worker.stats['resets'] == rs0)
    row = dict(case=name, http=http_st, state_before=st_before, state_after=st_s, log=len(toks_s), counters=dc, next=src,
               vram_ckpts=(len(ck0), len(ck1)), kept=kept_ck, logits_equal=same, max_abs_diff=float(np.abs(a_ - b_).max()),
               tokens_equal=y_s == y_d, gap_positions=gaps)
    check(ok, f'abortx ({name}): expect {expect}: {row}')
    return row


def stage_abortx(R, small):
    """abort() after a failure at every point of a request (the verifier's defect: a pass launched and not returned kept its
    commit, so the next pass applied it twice): injected host exceptions after the target's launch in each pass form (the
    device step after its wait, the device sampler, the inject of the host greedy pass and of verify()), a refusal before the
    launch (commit kept), a failure after a returned pass, a failure in the prompt phase after first_logits launched (the live
    state dropped, checkpoints kept), a refusal before prepare touched anything (the live state stands)"""
    sess = R.sess
    sess.side_route = False                    # (the main slot's contract: d8192[:600] after prose[:600] would be a side call)
    try:
        _stage_abortx(R, small)
    finally:
        sess.side_route = True


def _stage_abortx(R, small):
    from caminus.prefill import PrefillRefused
    sess, dr, df, pc, pf = R.sess, R.dr, R.df, R.pc, R.pf
    P = [248044] + [int(t) for t in toks_of('d8192')[:600]]
    cont = [int(t) for t in toks_of('code')[:16]]
    rows = []
    cnt = [0]

    def boom(what):
        raise RuntimeError(f'test_serve_k3 abortx: injected {what}')

    def req(kw=None, n=300):
        def run(other):
            body = dict(prompt=P, max_tokens=n, temperature=0.0); body.update(kw or {})
            return _post_status(R, '/v1/completions', body)[0]
        return run

    def on_log(other, toks):                   # the next request continues the session's log, the reference from P
        return (P if toks[:len(P)] == P else other), list(toks)

    def armed(run, arm, disarm):
        def r(other):
            cnt[0] = 0; arm()
            try: return run(other)
            finally: disarm()
        return r

    # (1) the device step: an exception after the step's wait (the target pass launched and finished, not returned)
    def sw(_o=df.step_wait):
        r_ = _o(); cnt[0] += 1
        if cnt[0] == 6: boom('error after the step wait')
        return r_
    rows.append(_abort_case(R, 'device step, after its wait', armed(req(), lambda: setattr(df, 'step_wait', sw), lambda: df.__dict__.pop('step_wait', None)),
                            'commit0', on_log, cont, lambda o: len(P)))
    # (2) the sampled pass: an exception in the device sampler right after the launch
    def qr(*a, _o=sess.queue_rows, **k):
        cnt[0] += 1
        if cnt[0] == 5: boom('error in the device sampler after the launch')
        return _o(*a, **k)
    rows.append(_abort_case(R, 'sampled pass, the device sampler after the launch',
                            armed(req(dict(temperature=1.0, top_k=20, top_p=0.95, seed=3)), lambda: setattr(sess, 'queue_rows', qr),
                                  lambda: sess.__dict__.pop('queue_rows', None)), 'commit0', on_log, cont, lambda o: len(P)))
    # (3) the host greedy pass (plain mode): an exception in the inject after verify_tokens
    def sp3(*a, _o=sess.sample_pass, **k):
        cnt[0] += 1
        if cnt[0] != 5: return _o(*a, **k)
        dr.inject = lambda *x, **y: boom('error in the inject after verify_tokens')
        try: return _o(*a, **k)
        finally: dr.__dict__.pop('inject', None)
    def arm3():
        sess.policy.force = lambda i: 'plain'; sess.sample_pass = sp3
    def dis3():
        sess.policy.force = None; sess.__dict__.pop('sample_pass', None); dr.__dict__.pop('inject', None)
    rows.append(_abort_case(R, 'host greedy pass (plain), the inject after verify_tokens', armed(req(), arm3, dis3), 'commit0', on_log, cont, lambda o: len(P)))
    # (4) verify() (forced tool-call ids beyond one pass): an exception in its inject after verify_tokens; the session driven here
    def run4(other):
        R.wait_idle()
        lg = sess.prefill(P)
        feed = [int(np.argmax(lg))] + [int(t) for t in toks_of('code')[200:200 + sess.T - 1]]
        dr.inject = lambda *x, **y: boom('error in verify()\'s inject after verify_tokens')
        try:
            try: sess.verify(feed)
            except RuntimeError: pass
        finally:
            dr.__dict__.pop('inject', None)
        sess.abort()
        return 500
    rows.append(_abort_case(R, 'verify(), the inject after verify_tokens', run4, 'commit0', on_log, cont, lambda o: len(P)))
    # (5) a refusal before the launch (the device step's host check): the commit is kept
    st5 = dict(n=0, pre=False)
    def ds(*a, _o=dr.device_step, **k):
        st5['n'] += 1; st5['pre'] = st5['n'] == 3
        return _o(*a, **k)
    def sc(*a, _o=df.step_check, **k):
        if st5['pre']:
            st5['pre'] = False; boom('refusal before the target launch (step_check)')
        return _o(*a, **k)
    def arm5():
        st5['n'] = 0; dr.device_step = ds; df.step_check = sc
    def dis5():
        dr.__dict__.pop('device_step', None); df.__dict__.pop('step_check', None)
    rows.append(_abort_case(R, 'device step, refused before the launch', armed(req(), arm5, dis5), 'kept', on_log, cont, lambda o: len(P)))
    # (6) after a pass returned (the documented abort between calls)
    def sp6(*a, _o=sess.sample_pass, **k):
        out = _o(*a, **k); cnt[0] += 1
        if cnt[0] == 5: boom('error after a pass returned')
        return out
    rows.append(_abort_case(R, 'after a pass returned', armed(req(), lambda: setattr(sess, 'sample_pass', sp6), lambda: sess.__dict__.pop('sample_pass', None)),
                            'pass', on_log, cont, lambda o: len(P)))
    # (7) the prompt phase: a request that adds two tokens to the live log (prepare runs no pass), an exception after first_logits
    # launched its pass (in the cache's extend): the device state is the previous request's plus that launch, so the live state is
    # dropped (every checkpoint kept) and the next request restores one
    p7 = {}
    def run7(other):
        _, g, _ = R.complete(P, 8)
        R.wait_idle()
        P2 = P + g + [int(toks_of('code')[250])]
        p7['P2'] = P2
        def ext(*a, _o=pc.extend, **k):
            pc.__dict__.pop('extend', None); boom('error after first_logits launched (the cache\'s extend)')
        pc.extend = ext
        try: return _post_status(R, '/v1/completions', dict(prompt=P2, max_tokens=16, temperature=0.0))[0]
        finally: pc.__dict__.pop('extend', None)
    rows.append(_abort_case(R, 'prompt phase, after first_logits launched', run7, 'dropped', lambda other, toks: (P, p7['P2']), cont,
                            lambda o: len(p7['P2'])))
    # (8) a refusal before prepare touched anything (PrefillRefused from the token check): the previous conversation stands
    def run8(other):
        def ct(*a, **k):
            pf.__dict__.pop('check_tokens', None); raise PrefillRefused('test_serve_k3 abortx: injected refusal in check_tokens')
        pf.check_tokens = ct
        try: return _post_status(R, '/v1/completions', dict(prompt=P, max_tokens=16, temperature=0.0))[0]
        finally: pf.__dict__.pop('check_tokens', None)
    rows.append(_abort_case(R, 'prefill refused before anything ran', run8, 'intact', on_log, cont, lambda o: len(o)))
    # (9) final round 4 (the verifier's Y2): the engine reset inside end() (a scripted wrapper of the cache's commit:
    # FlashNextTP.reset(), then an exception): abort() sees the sequence epoch move and drops the live state, the cache takes in
    # the reset (cold: the KV rows went too, every checkpoint void); the next request == a cold run of it, bitwise. Before,
    # abort() committed the session's state again (no launch since the mark) and the next request continued it 'live'
    rows.append(_abort_reset_case(R, P, cont))
    NUMBERS['abortx'] = rows


def _abort_reset_case(R, P, cont):
    pc, m, sess = R.pc, R.m, R.sess
    pc.reset()
    R.complete([248044] + toks_of('prose')[:600], 8)
    R.wait_idle()
    c0, er0, rs0 = dict(sess.counters), pc.stats['external_reset'], R.srv.worker.stats['resets']

    def cm(*a, **k):
        pc.__dict__.pop('commit', None); m.reset()
        raise RuntimeError('test_serve_k3 abortx: injected after an engine reset in end()')
    pc.commit = cm
    try: http_st = _post_status(R, '/v1/completions', dict(prompt=P, max_tokens=16, temperature=0.0))[0]
    finally: pc.__dict__.pop('commit', None)
    R.wait_idle(); R.sync()
    toks_s, st_s = list(sess.toks), dict(sess.st)
    dc = {k: sess.counters[k] - c0[k] for k in ('aborts', 'aborts_commit0', 'dropped_live')}
    resets = pc.stats['external_reset'] - er0
    Q = P + list(cont)
    lg_s, y_s, src = _next_server(R, Q)
    pc.reset()
    lg_c, y_c, src_c = _next_server(R, Q)
    pc.reset()
    same = lg_s.numpy().tobytes() == lg_c.numpy().tobytes()
    ok = (http_st == 500 and dc == dict(aborts=1, aborts_commit0=0, dropped_live=1) and not toks_s and st_s == dict(i0=0, commit=0)
          and src == ('cold', 0) and src_c == ('cold', 0) and same and y_s == y_c and resets == 1 and R.srv.worker.stats['resets'] == rs0)
    row = dict(case='engine reset inside end()', http=http_st, state_after=st_s, log=len(toks_s), counters=dc, external_reset=resets,
               next=src, cold=src_c, logits_equal=same, tokens_equal=y_s == y_c)
    check(ok, f'abortx (engine reset inside end()): expect the live state dropped, the reset taken in, the next request cold == a cold run: {row}')
    return row


def stage_widepen(R, small):
    """penalty windows wider than the device sampler's pinned candidate buffers (the verifier's HTTP 500: repeat_last_n -1, or
    256 with top_k 64): the rows go whole (the reference path); HTTP 200, tokens == the host reference on full rows (from the
    same prompt-end state: the batched prefill through the cache, then one teacher-forced pass per token), drafting == forced
    plain; the sampler's ms per pass on this path reported. A 601-token prompt: its window holds > 143 distinct ids from the
    first pass on"""
    from caminus.serve import sampler as S
    ids = [248044] + [int(t) for t in toks_of('d8192')[:600]]
    n = 120 if small else 200
    pol = R.sess.policy
    rows = []
    for name, kw in (('repeat_last_n -1, top_k 20', dict(temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5, repeat_last_n=-1, seed=5)),
                     ('repeat_last_n 256, top_k 64', dict(temperature=1.0, top_k=64, top_p=0.95, presence_penalty=1.5, repeat_last_n=256, seed=6)),
                     ('greedy, repeat_last_n -1', dict(temperature=0.0, presence_penalty=1.5, repeat_last_n=-1, seed=7))):
        w0, f0 = R.sess.counters['wide_rows'], R.sess.counters['full_rows']
        st1, js1 = _post_status(R, '/v1/completions', dict(prompt=ids, max_tokens=n, **kw))
        res = R.jobs[-1].result
        g1 = list(res.gen_ids) if res is not None else None
        smp = np.asarray(list(R.stats.pass_series['sampler'].window())[-max(1, res.passes if res is not None else 1):])
        wide, full = R.sess.counters['wide_rows'] - w0, R.sess.counters['full_rows'] - f0
        pol.force = lambda i: 'plain'
        st2, _ = _post_status(R, '/v1/completions', dict(prompt=ids, max_tokens=n, **kw))
        g2 = list(R.jobs[-1].result.gen_ids) if R.jobs[-1].result is not None else None
        pol.force = None
        prm = S.SamplingParams(temperature=kw['temperature'], top_k=kw.get('top_k', 20), top_p=kw.get('top_p', 0.95),
                               presence_penalty=kw['presence_penalty'], penalty_last_n=kw['repeat_last_n'], seed=kw['seed']).resolve()
        ref = host_reference_pc(R, prm, ids, n)
        ok = st1 == 200 and st2 == 200 and g1 == ref and g2 == g1 and wide > 0
        rows.append(dict(case=name, http=(st1, st2), tokens=len(g1 or []), ref=len(ref), equal_ref=g1 == ref, draft_eq_plain=g2 == g1,
                         wide_passes=wide, full_rows=full, sampler_ms_median=float(np.median(smp)) if len(smp) else None,
                         sampler_ms_p95=float(np.percentile(smp, 95)) if len(smp) else None,
                         first_diff=next((j for j, (a, b) in enumerate(zip(g1 or [], ref)) if a != b), None)))
        check(ok, f'widepen ({name}): HTTP {st1} / {st2}; {len(g1 or [])} tokens == the host reference on full rows ({len(ref)}): {g1 == ref}; drafting == forced '
                  f'plain: {g2 == g1}; {wide} passes over the candidate buffers ({full} full rows); sampler {rows[-1]["sampler_ms_median"]} ms per pass (median, reported)')
    NUMBERS['widepen'] = rows


def torch_equal(a, b):
    import torch
    return bool(torch.equal(a, b))


def stage_bail(R, small):
    from caminus.serve.k3_session import K3Session
    ids = [248044] + toks_of('prose')[:150]
    cases = [('greedy DFlash, a decode pass', 6, dict()), ('greedy DFlash, the first-logits pass', 0, dict()),
             ('seeded T 1.0, a decode pass', 9, dict(temperature=1.0, top_k=20, top_p=0.95, seed=99))]
    rows = []
    for name, k, kw in cases:
        R.pc.reset()
        r0, g0, _ = R.complete(ids, 96, **kw)                # the reference: no bail (the server's own run)
        rec0, bails0 = R.pc.stats['recovered'], R.stats.bails_recovered if hasattr(R.stats, 'bails_recovered') else None
        os.environ['K3_TEST_HOOKS'] = 'police-controls'; os.environ['K3_TEST_BAIL'] = str(k)
        try:
            s2 = K3Session(R.m, R.pf, R.dr, R.pc, stats=R.stats, idle_s=None)
        finally:
            os.environ.pop('K3_TEST_HOOKS', None); os.environ.pop('K3_TEST_BAIL', None)
        old = R.sess
        R.srv.loading = True; R.srv.set_session(s2, s2.drafter); R.srv.loading = False
        nb0 = len(R.m.bails)
        rt0, br0 = R.stats.retries, R.stats.bails['recovered']
        R.pc.reset()
        t0 = time.time()
        r1, g1, st = R.complete(ids, 96, **kw)
        dt = time.time() - t0
        R.srv.loading = True; R.srv.set_session(old, old.drafter); R.srv.loading = False
        nb = len(R.m.bails) - nb0
        rt, br = R.stats.retries - rt0, R.stats.bails['recovered'] - br0
        ok = st == 200 and g1 == g0 and s2.counters['recoveries'] == 1 and R.pc.stats['recovered'] == rec0 + 1 and nb >= R.m.max_retries + 1
        # (metrics: every bailed attempt counts in pass_retries_total, the recovery in bails_total{recovered}; before fix:m7-wiring
        # the recovery reset the bail count first and the failed pass's attempts were never counted)
        ok_m = rt == nb and br >= 1
        rows.append((name, k, ok and ok_m, len(g1), s2.counters['recoveries'], nb, rt, br, round(dt, 2)))
        check(ok, f'bail: {name} (K3_TEST_BAIL={k}: every attempt of that target pass bails, {nb} bails): HTTP {st}, recovered from the '
                  f'prompt-end checkpoint ({s2.counters["recoveries"]} recoveries), {len(g1)} tokens == the run without the bail; {dt:.2f} s')
        check(ok_m, f'bail metrics: {name}: pass_retries_total +{rt} == the {nb} bailed attempts; bails_total{{recovered}} +{br}')
    NUMBERS['bail'] = rows


def stage_policy(R, small):
    pol = R.sess.policy
    ids = [248044] + toks_of('shell')[:180]
    n = 96 if small else 200
    out = {}
    for name, force in (('draft', lambda i: 'draft'), ('plain', lambda i: 'plain'), ('alternate', lambda i: 'plain' if i % 3 == 0 else 'draft'),
                        ('natural', None)):
        pol.force = force
        R.pc.reset()
        _, g, _ = R.complete(ids, n)
        _, gs, _ = R.complete(ids, n, temperature=1.0, top_k=20, top_p=0.95, seed=5)
        out[name] = (g, gs)
    pol.force = None
    ok = all(v[0] == out['draft'][0] and v[1] == out['draft'][1] for v in out.values())
    check(ok, f'policy: forced draft / plain / alternating schedules and the natural policy give bitwise the same tokens (greedy {len(out["draft"][0])}, '
              f'seeded T 1.0 {len(out["draft"][1])}); policy counts {pol.counts}; buckets {pol.view()}')
    NUMBERS['policy'] = dict(counts=dict(pol.counts), view=pol.view())


def _get(R, path):
    with urllib.request.urlopen(R.url + path, timeout=60) as r:
        return r.read()


def _prom(text):
    """{series name: [(labels str, value)]} of a Prometheus text"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        k, v = line.rsplit(' ', 1)
        name, lab = (k.split('{', 1) + [''])[:2]
        out.setdefault(name, []).append((lab.rstrip('}'), float(v)))
    return out


def stage_metrics(R, small):
    """M7 ops item 1: every documented series and status field filled by the real engine, a check per group (llama-server's,
    speed vs physics, DFlash, prompt cache, context, reliability, TP / hardware, agent behaviour, server), after a workload of
    greedy / seeded / forced-plain / chat-with-tools / reused requests, a context-length refusal, an injected prefill refusal and
    a cold 12K prompt whose /slots progress is polled while it is processed"""
    from caminus.prefill import PrefillRefused
    d8, code, prose = toks_of('d8192'), toks_of('code'), toks_of('prose')
    R.pc.reset()
    rows = {}
    # live prompt progress: poll /slots while a cold 12K prompt is processed (batched chunks)
    ids = [248044] + toks_of('d32768')[:12000 if not small else 6000]
    seen = []
    th = threading.Thread(target=lambda: R.complete(ids, 4))
    th.start()
    while th.is_alive():
        try:
            sl = json.loads(_get(R, '/slots'))[0]
            if sl.get('is_processing'): seen.append(sl.get('n_prompt_tokens_processed', 0))
        except Exception: pass
        time.sleep(0.05)
    th.join()
    mid = sorted(set(x for x in seen if 0 < x < len(ids)))
    rows['progress'] = dict(prompt=len(ids), intermediate=mid[:12], n=len(mid))
    # the workload
    R.complete([248044] + prose[:200], 64)                                          # greedy, device accept
    R.complete([248044] + code[:200], 64, temperature=1.0, top_k=20, top_p=0.95, seed=11)   # seeded T > 0
    pol = R.sess.policy; pol.force = lambda i: 'plain'
    R.complete([248044] + d8[:300], 32)                                             # forced plain: the plain pass time measured
    pol.force = None
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
    req = urllib.request.Request(R.url + '/v1/chat/completions', data=json.dumps(dict(model='k3-coder', messages=[
        {'role': 'user', 'content': 'Read the file notes.txt with the tool.'}], tools=tools, tool_choice='required', max_tokens=96,
        temperature=0.0)).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'QwenCode/0.0.14 (linux; x64)'})
    urllib.request.urlopen(req, timeout=600).read()
    P = [248044] + d8[:900]
    _, g1, _ = R.complete(P, 16)
    R.complete(P + g1 + code[:12], 16)                                              # reuse: live
    st400, _ = _post_status(R, '/v1/completions', dict(prompt=[248044] * (R.sess.max_ctx + 8), max_tokens=4))   # context refusal (HTTP 400)
    pf = R.pf
    def ct(*a, **k):
        pf.__dict__.pop('check_tokens', None); raise PrefillRefused('test_serve_k3 metrics: injected refusal')
    pf.check_tokens = ct
    try: st_pr, _ = _post_status(R, '/v1/completions', dict(prompt=P + [5, 6, 7], max_tokens=4))
    finally: pf.__dict__.pop('check_tokens', None)
    R.wait_idle(); time.sleep(2.2)                                                   # (two hardware samples)
    text = _get(R, '/metrics').decode(); S = json.loads(_get(R, '/caminus/status'))
    M = _prom(text)
    def has(name, lab=None): return any(lab is None or lab in l for l, _ in M.get(name, []))
    def val(name, lab=''): return next((v for l, v in M.get(name, []) if lab in l), None)
    groups = {}
    # llama-server's series
    groups['llamacpp'] = all(val(f'llamacpp:{n}') for n in ('prompt_tokens_total', 'prompt_tokens_cached_total', 'tokens_predicted_total',
                                                          'n_decode_total', 'spec_decode_num_draft_tokens_total', 'spec_decode_num_accepted_tokens_total'))\
        and has('llamacpp:spec_decode_num_accepted_tokens_per_pos_total', 'position="0"') and rows['progress']['n'] >= 2
    # speed vs physics
    ph = S['physics']
    cb = list(R.stats.physics.card_bytes)
    plaus = all(abs(b / s - 1) <= 0.2 for b, s in zip(cb, (3.60e9, 2.87e9)))
    print(f'  physics: per-card bytes from the engine {[round(b / 1e9, 3) for b in cb]} GB vs the spec\'s 3.60 / 2.87 (+-20 %: {plaus})', flush=True)
    groups['physics'] = plaus and ph['source'] == 'engine' and ph['pass_source'] == 'card_events' and 0 < ph['pass_pct'] <= 100 and ph['token_pct'] > 0\
        and all(S['decode']['split_ms'][k]['count'] > 0 for k in ('k3', 'drafter', 'sampler', 'host', 'total'))\
        and all(S['ttft']['split_ms'][k]['count'] > 0 for k in ('queue', 'restore', 'batched', 'handoff', 'passes', 'host', 'total'))\
        and S['prefill']['tokens_per_s']['count'] > 0
    # DFlash
    dfl = S['dflash']
    groups['dflash'] = dfl['drafts'] > 0 and dfl['per_pos'] and dfl['plain_source'] == 'measured' and dfl['speedup_vs_plain'] is not None\
        and dfl['classes']['prose']['drafts'] > 0 and has('caminus:spec_accepted_per_pos_total', 'class="prose"')
    # prompt cache
    c = S['cache']
    groups['cache'] = c['tokens']['vram'] > 0 and c['sources']['cold'] > 0 and c['sources']['live'] > 0 and c['vram']['slots'] and c['vram']['slots_used']\
        and c['disk']['cap_bytes'] and c['events']['checkpoints'] > 0 and c['lcp']['count'] > 0
    # context: the KV bytes per position from the engine's layout
    L = R.pc.layout
    per = [len(L.kv[k]) * L.pb[k] + L.na * L.rb * 1.25 for k in (0, 1)]
    cx = S['context']
    groups['context'] = cx['position'] > 0 and cx['max_ctx'] == R.sess.max_ctx and cx['kv_bytes'] == [int(round(cx['position'] * b)) for b in per]
    # reliability
    rl = S['reliability']
    groups['reliability'] = st400 == 400 and rl['refusals']['context'] >= 1 and rl['refusals']['prefill'] >= 1 and rl['host_samples'] >= 2\
        and has('caminus:kfd_evicted_ms_total') and has('caminus:pgmigrate_all_total') and rl['xtx_margin_mib'] is not None
    # TP / hardware
    cards = S['cards']
    u = vram_used()
    dec = S['decode']['passes']
    hw_sub = dict(names=[x['name'] for x in cards] == ['XTX', 'XT'], card_pass=[x['pass_ms']['count'] for x in cards] == [dec, dec],
                  waits_absent=all(x['exchange_wait_ms']['count'] == 0 and x['exchange_wait_note'] for x in cards) and not has('caminus:exchange_wait_ms'),
                  link=S['link']['bytes_total'] == dec * R.sess.link_bytes_per_pass,
                  vram=all(abs(x['vram_used_mib'] - u[i]) < 600 for i, x in enumerate(cards)),
                  readings=all(x[k] is not None for x in cards for k in ('power_w', 'sclk_mhz', 'mclk_mhz', 'perf_level', 'runtime_pm', 'temp_junction_c')),
                  clock=S['clock_check'] is not None and [x['perf_level'] for x in S['clock_check']['cards']] == [x['perf_level'] for x in cards])
    rows['hardware_checks'] = dict(hw_sub, card_pass_counts=[x['pass_ms']['count'] for x in cards], decode_passes=dec, link_bytes=S['link']['bytes_total'])
    groups['hardware'] = all(hw_sub.values())
    # agent behaviour, server
    ag = S['agent']
    groups['agent'] = ag['harness'].get('qwen-code', 0) >= 1 and ag['tool_calls'] >= 1 and ag['tokens']['answer'] > 0 and sum(ag['finish'].values()) > 0
    groups['server'] = has('caminus:info') and has('caminus:http_responses_total') and has('caminus:worker_total') and S['server']['requests'] > 0
    rows['groups'] = groups
    rows['values'] = dict(physics={k: ph[k] for k in ('floor_ms', 'source', 'pass_pct', 'pass_source', 'target_pass_p50_ms', 'k3_pass_p50_ms', 'token_pct')},
                          card_pass_p50=[x['pass_ms']['p50'] for x in cards], link_bytes_per_pass=R.sess.link_bytes_per_pass,
                          card_bytes=list(R.stats.physics.card_bytes), kv_bytes_per_token=per, plain=dfl['plain_ms_per_token'],
                          speedup=dfl['speedup_vs_plain'], refusals=rl['refusals'], kfd_evicted_ms=rl['kfd_evicted_ms'],
                          pgmigrate=(rl['pgmigrate'], rl['pgmigrate_all']), clock=S['clock_check']['summary'], vram=[x['vram_used_mib'] for x in cards],
                          sysfs=u, power=[x['power_w'] for x in cards], filler_passes=R.sess.counters['filler_passes'])
    NUMBERS['metrics'] = rows
    for g, ok in groups.items():
        check(bool(ok), f'metrics ({g}): populated by the real engine')
    print(f'  metrics values: {rows["values"]}; progress {rows["progress"]}', flush=True)


SCRAPER = r"""
import signal, sys, time, urllib.request
url, hz, dur, mode = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
n, ms, t_end = 0, [], time.time() + dur
def done(*_):
    ms.sort()
    print(f'{mode}: {n} ticks, median {ms[len(ms) // 2] if ms else 0:.2f} ms per tick, max {ms[-1] if ms else 0:.2f}', flush=True); sys.exit(0)
signal.signal(signal.SIGTERM, done)
while time.time() < t_end:
    t0 = time.perf_counter()
    if mode == 'cpu':                       # the control: the same wake-up rate and ~4 ms of CPU, no request to the server
        x = 0
        while time.perf_counter() - t0 < 0.004: x += 1
    else:
        for p in ('/metrics', '/caminus/status', '/slots'):
            urllib.request.urlopen(url + p, timeout=30).read()
    ms.append((time.perf_counter() - t0) * 1e3); n += 1
    time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t0)))
done()
"""


def stage_scrape(R, small, reps):
    """M7 ops item 1's gate: the decode loop is not delayed by scraping. Decode ms/token of the same greedy request (DFlash,
    stats on, everything on) with and without a 10 Hz scraper of /metrics + /caminus/status + /slots running in a CHILD process
    (as llm-top would; a new connection per request: the worst case), interleaved, 7 x 512 tokens per arm; a difference over
    0.05 ms per token fails. Reported beside it (not gates): a second scraper-free arm (the noise) and a control child with the
    same wake-up rate and ~4 ms of CPU per tick that sends no request (what any periodic process on the host costs the engine,
    so the server's own share is on - control)"""
    ids = [248044] + toks_of('prose')[:199]
    n = 128 if small else 512
    R.pc.reset()
    _, g0, _ = R.complete(ids, n)
    k = 3 if small else 7                    # (fixed before the run: 7 x 512 tokens per arm)
    off, on, off2, cpu, logs = [], [], [], [], []
    for rep in range(k):
        for arm in ('off', 'on', 'cpu', 'off2'):
            proc = None
            if arm in ('on', 'cpu'):
                proc = subprocess.Popen([sys.executable, '-I', '-c', SCRAPER, R.url, '10', '600', 'http' if arm == 'on' else 'cpu'],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                time.sleep(0.5)
            r, g, _ = R.complete(ids, n)
            t = r['timings']
            {'on': on, 'off': off, 'off2': off2, 'cpu': cpu}[arm].append(t['predicted_ms'] / t['predicted_n'])
            if proc is not None:
                proc.terminate()
                try: logs.append(proc.communicate(timeout=10)[0].strip())
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(10)
            if g != g0: check(False, f'scrape: tokens differ ({arm}, rep {rep})')
    mo, mf, m2, mc = float(np.median(on)), float(np.median(off)), float(np.median(off2)), float(np.median(cpu))
    NUMBERS['scrape'] = dict(off=off, on=on, off2=off2, cpu=cpu, median_off=mf, median_on=mo, median_off2=m2, median_cpu=mc, diff=mo - mf,
                             noise_off2_minus_off=m2 - mf, cpu_child_minus_off=mc - mf, server_share=mo - mc, scraper=logs, tokens=len(g0))
    print(f'  scrape controls (reported, not gates): a second scraper-free arm {m2:.4f} ({m2 - mf:+.4f}); a child with the same wake-up rate '
          f'and CPU but no request {mc:.4f} ({mc - mf:+.4f}): the scrape over that control {mo - mc:+.4f} ms/token; {logs[:2]}', flush=True)
    check(mo - mf <= 0.05, f'scrape: decode {mf:.4f} ms/token without vs {mo:.4f} with a 10 Hz scraper of /metrics + /caminus/status + /slots '
                           f'(a child process; medians of {k}, interleaved; off {[round(x, 3) for x in off]}, on {[round(x, 3) for x in on]}): '
                           f'{mo - mf:+.4f} ms (fail > +0.05); scraper {logs[:1]}; {host_note()}')


def stage_flight(R, small):
    """M7 ops item 2 on the real engine: the flight recorder's line per request; replay bundles on demand (a request whose start
    was decode-built: its prompt-end state saved; a cold request: none needed) and automatically at retry exhaustion (a session
    with max_recoveries 0 and the K3_TEST_BAIL hook: the designed bail path) before the reset; files 0600 / directories 0700.
    The bundles are replayed AFTER this process ends (tools/replay.py, one engine on the cards at a time): NUMBERS['flight']"""
    from caminus.serve.k3_session import K3Session
    fl = R.flight
    d8, code = toks_of('d8192'), toks_of('code')
    R.pc.reset()
    out = dict(bundles=[])
    # (a) a turn whose start is decode-built: turn 1 cold, turn 2 continues the live state over the decoded reply (seeded T > 0)
    P = [248044] + d8[:1500]
    _, g1, _ = R.complete(P, 24)
    R1 = P + g1 + code[:20]
    T1 = dict(temperature=1.0, top_k=20, top_p=0.95, seed=2026)
    _, g2, _ = R.complete(R1, 96, **T1)
    R.wait_idle()
    pend_a, ep_a = R.df.pending, R.pc._epochs()
    st, b = _post_status(R, '/caminus/flight/bundle', {})
    R.wait_idle()
    ep_a2 = R.pc._epochs()
    out['bundles'].append(dict(case='decode-built start (live), T 1.0 seeded', http=st, path=b.get('path'), cache_ref=b.get('cache_ref'), tokens=len(g2)))
    # fix:m7 (verify:m7's blocking item): a bundle leaves the conversation working. The export flushed the drafter's pending
    # inject (a turn's max_tokens end leaves one: T > 0 and greedy alike) without re-marking the cache's launch epochs, so the
    # conversation's next request answered 503, the worker reset and the VRAM tier was lost. Here: the next turn after (a)'s
    # bundle (greedy, ending at max_tokens: pending again), a bundle of it (a greedy live start), a third turn; == the same turns
    # with no bundle (tokens and first logits bitwise, the same sources); no reset, the VRAM checkpoints kept
    def turn(ids, n, **kw):
        s_, _ = _post_status(R, '/v1/completions', dict(prompt=list(ids), max_tokens=n, temperature=0.0, **kw)); R.wait_idle()
        res = R.jobs[-1].result
        fl_ = R.sess.last_first_logits
        return dict(http=s_, gen=list(res.gen_ids) if res is not None and s_ == 200 else None, src=getattr(res, 'restore', None),
                    first=None if fl_ is None or s_ != 200 else np.ascontiguousarray(np.asarray(fl_, dtype=np.float32)).tobytes())
    rs0, ck0 = R.srv.worker.stats['resets'], _vram_ckpts(R)
    R3 = R1 + g2 + code[20:40]
    t3 = turn(R3, 32)
    ck3 = _vram_ckpts(R)
    pend_3, ep_3 = R.df.pending, R.pc._epochs()
    st_b2, b2 = _post_status(R, '/caminus/flight/bundle', {})
    R.wait_idle()
    ep_32 = R.pc._epochs()
    bundle_cont = dict(case='decode-built start (live), greedy, after another bundle', http=st_b2, path=b2.get('path'),
                       cache_ref=b2.get('cache_ref'), tokens=len(t3['gen'] or []))
    R4 = R3 + (t3['gen'] or []) + code[60:80]
    t4 = turn(R4, 16)
    rs1, ck4 = R.srv.worker.stats['resets'], _vram_ckpts(R)
    # the same four turns with no bundle
    R.pc.reset()
    _, h1, _ = R.complete(P, 24)
    _, h2, _ = R.complete(P + h1 + code[:20], 96, **T1)
    u3 = turn(P + h1 + code[:20] + h2 + code[20:40], 32)
    u4 = turn(P + h1 + code[:20] + h2 + code[20:40] + (u3['gen'] or []) + code[60:80], 16)
    out['continue'] = dict(pending=(pend_a, pend_3), epochs_a=(ep_a, ep_a2), epochs_3=(ep_3, ep_32), resets=rs1 - rs0,
                           vram_ckpts=(len(ck0), len(ck3), len(ck4)), turns=[(t['http'], t['src'], len(t['gen'] or [])) for t in (t3, t4)],
                           reference=[(t['http'], t['src'], len(t['gen'] or [])) for t in (u3, u4)],
                           equal=dict(turn1=h1 == g1, turn2=h2 == g2, turn3=t3['gen'] == u3['gen'], turn4=t4['gen'] == u4['gen'],
                                      first3=t3['first'] == u3['first'], first4=t4['first'] == u4['first']))
    ok_cont = (pend_a is not None and pend_3 is not None and ep_a == ep_a2 and ep_3 == ep_32 and rs1 == rs0
               and t3['http'] == 200 and t4['http'] == 200 and t3['src'] == u3['src'] == 'live' and t4['src'] == u4['src'] == 'live'
               and set(ck0) <= set(ck3) and set(ck3) <= set(ck4) and all(out['continue']['equal'].values())
               and st_b2 == 200 and (b2.get('cache_ref') or {}).get('source') == 'live' and (b2.get('cache_ref') or {}).get('state'))
    check(ok_cont, f'flight (fix:m7): the conversation continues after a bundle as if none was saved: the export left the epochs and the pending '
                   f'inject as they were, the next turns live, no reset, the VRAM checkpoints kept, tokens and first logits == the same turns '
                   f'with no bundle: {out["continue"]}')
    # (b) a cold greedy request
    R.pc.reset()
    C = [248044] + toks_of('shell')[:300]
    _, g3, _ = R.complete(C, 64)
    st, b = _post_status(R, '/caminus/flight/bundle', {})
    out['bundles'].append(dict(case='cold start, greedy', http=st, path=b.get('path'), cache_ref=b.get('cache_ref'), tokens=len(g3)))
    # (c) retry exhaustion: a warm turn on the serving session, then a continuation (source live) on a session whose 5th target
    # pass (its first_logits is pass 0) bails on every attempt with no recovery allowed -> EngineError -> a bundle saved before
    # the reset, its start state exported after the failed pass
    R.pc.reset()
    Q = [248044] + d8[2000:3200]
    _, gq, _ = R.complete(Q, 16)
    Q2 = Q + gq + code[40:52]
    os.environ['K3_TEST_HOOKS'] = 'police-controls'; os.environ['K3_TEST_BAIL'] = '5'
    try: s2 = K3Session(R.m, R.pf, R.dr, R.pc, stats=R.stats, idle_s=None, max_recoveries=0)
    finally: os.environ.pop('K3_TEST_HOOKS', None); os.environ.pop('K3_TEST_BAIL', None)
    s2.toks, s2.st = list(R.sess.toks), dict(R.sess.st); s2._mark()
    old = R.sess
    R.srv.loading = True; R.srv.set_session(s2, s2.drafter); R.srv.loading = False
    try:
        st_e, _ = _post_status(R, '/v1/completions', dict(prompt=Q2, max_tokens=64, temperature=0.0))
        R.wait_idle()
    finally:
        R.srv.loading = True; R.srv.set_session(old, old.drafter); R.srv.loading = False
    auto = [x for x in R.srv.worker.bundles if x[1] == 'retry_exhausted']
    out['bundles'].append(dict(case='retry exhaustion after a warm turn (K3_TEST_BAIL 5, max_recoveries 0)', http=st_e,
                               path=auto[-1][2] if auto else None,
                               cache_ref=json.load(open(os.path.join(auto[-1][2], 'bundle.json')))['cache_ref'] if auto else None))
    out['bundles'].append(bundle_cont)          # (index 3: fix:m7's greedy live-start bundle; replayed like the others)
    # the reference for (c): the same two turns without the bail (greedy): the replay's tokens must be its prefix
    R.pc.reset()
    _, gq_, _ = R.complete(Q, 16)
    _, gref, _ = R.complete(Q + gq_ + code[40:52], 64)
    out['retry_reference'] = dict(turn1_equal=gq_ == gq, tokens=gref)
    # files and lines
    def mode(p): return stat.S_IMODE(os.stat(p).st_mode)
    bad = []
    for d, dirs, files in os.walk(fl.root):
        if mode(d) != 0o700: bad.append((d, oct(mode(d))))
        for f_ in files:
            if mode(os.path.join(d, f_)) != 0o600: bad.append((os.path.join(d, f_), oct(mode(os.path.join(d, f_)))))
    rows = [json.loads(x) for x in open(fl.day_path()) if x.strip()]
    out['lines'] = len(rows); out['perm_bad'] = bad
    out['bytes'] = fl.total_bytes()
    NUMBERS['flight'] = out
    ok_a = out['bundles'][0]['http'] == 200 and (out['bundles'][0]['cache_ref'] or {}).get('state') and out['bundles'][0]['cache_ref']['source'] == 'live'
    ok_b = out['bundles'][1]['http'] == 200 and (out['bundles'][1]['cache_ref'] or {}).get('source') == 'cold' and not out['bundles'][1]['cache_ref'].get('state')
    ok_c = st_e in (503, 529) and bool(auto) and (out['bundles'][2]['cache_ref'] or {}).get('state') and out['bundles'][2]['cache_ref']['source'] == 'live'
    check(ok_a and ok_b and ok_c, f'flight: bundles saved (replayed after this process by tools/replay.py): {out["bundles"]}')
    check(not bad and rows and all('request_id' in r for r in rows) and all('sampling' in r and r['request_id'] for r in rows if r.get('status') == 200),
          f'flight: {len(rows)} JSONL lines with the plan\'s fields; every file 0600 and directory 0700 under {fl.root} ({bad[:4] or "all"}); {out["bytes"] / 1e6:.1f} MB')


WORDS = ('file path line offset module config value record index range buffer cache entry token stream block header field '
         'table column row key list map set queue stack node graph edge parent child handle socket request reply status '
         'error check limit count size width depth order level scope frame label name type kind mode flag state').split()
SIDE_SYS = 'You are a coding agent working in a synthetic repository. Use the tools to read files, then answer briefly.'


def side_tools(n):
    """n synthetic tools (names tool_00 .., descriptions of words drawn from WORDS by a fixed generator): a ~4.6K-token system +
    tools block at n = 20 under the server's template"""
    rng = np.random.default_rng(7)
    out = []
    for i in range(n):
        desc = ' '.join(rng.choice(WORDS, 90)).capitalize() + '.'
        props = {f'arg_{j}': {'type': 'string', 'description': ' '.join(rng.choice(WORDS, 12))} for j in range(3)}
        out.append({'type': 'function', 'function': {'name': f'tool_{i:02d}', 'description': desc,
                                                     'parameters': {'type': 'object', 'properties': props, 'required': ['arg_0']}}})
    return out


def side_code(k, n):
    return '\n'.join(f'def f_{k}_{j}(x, y):\n    """Return x times {j} plus y."""\n    return x * {j} + y - {k * j}\n' for j in range(n))


def _main_line(R):
    """the main line as the cache holds it (stashed during a side call): (log, state, VRAM checkpoints)"""
    sd = R.pc.side
    log, st, cks = (sd['log'], sd['st'], sd['ckpts']) if sd is not None else (R.pc.log, R.pc.st, R.pc.ckpts)
    return list(log), dict(st), sorted((c.pos, c.commit, c.kind, c.slot) for c in cks if c.slot is not None)


def stage_side(R, small):
    """side-slot routing (the lead's ruling of 11-05, impl:m7-finish): see the module docstring. Three runs on the same server:
    A (the conversation with the side calls), B (the same main requests alone), C (each side call alone, cold). Every run starts
    from an empty cache (reset + purge)"""
    from caminus.serve.k3_session import K3Session
    tools = side_tools(20)
    model = 'k3-coder-nothink'
    m1 = [{'role': 'system', 'content': SIDE_SYS}, {'role': 'user', 'content': 'Read src/mod_0.py and tell me what f_0_3 returns.'}]
    m2 = m1 + [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
                    'name': 'tool_00', 'arguments': json.dumps({'arg_0': 'src/mod_0.py'})}}]},
               {'role': 'tool', 'tool_call_id': 'call_1', 'content': side_code(0, 12)}]
    prose, code, shell = toks_of('prose'), toks_of('code'), toks_of('shell')
    long_req = [248044] + toks_of('d32768')[1000:10000]        # 9,001 tokens: does not fit the side slot
    n_main, n_side = (24, 16) if small else (48, 24)
    sides = {
        'title': ('chat', [{'role': 'system', 'content': 'Generate a short title for the conversation below. Reply with the title only.'},
                           {'role': 'user', 'content': 'Read src/mod_0.py and tell me what f_0_3 returns.'}]),
        'summary': ('chat', [{'role': 'system', 'content': 'Summarize the text in one sentence.'},
                             {'role': 'user', 'content': 'The cache keeps one live conversation and a few checkpoints of it in VRAM; '
                                                         'idle conversations go to disk.'}]),
        'raw': ('raw', [248044] + prose[:250]),
        'next': ('chat', [{'role': 'system', 'content': 'Answer yes or no: should the assistant speak next?'},
                          {'role': 'user', 'content': 'The assistant just read a file.'}]),
    }

    def send_side(name, **kw):
        kind, x = sides[name]
        if kind == 'chat':
            r, g, pids, st = R.chat(x, n_side, model=model, **kw)
        else:
            r, g, st = R.complete(x, n_side, **kw); pids = x
        return st, g, pids

    lgs, preps, undo0 = _capture(R)
    pp = R.pc.prepare
    rec = []

    def prepare(*a, **k):                                   # (also the route, the slot and the state)
        r = pp(*a, **k); rec.append(dict(route=r.get('route'), side=r.get('side'), source=r['source'], cached=r['cached'], st=dict(r['st'])))
        return r
    R.pc.prepare = prepare

    def undo():
        undo0(); R.pc.__dict__.pop('prepare', None)

    def fresh():
        R.wait_idle(); R.pc.reset(); R.pc.purge()

    def main_turns(with_sides):
        """T1 (chat), T2 (chat: + a tool call and its result), T3 (tokens: T2's prompt + its reply + 200 tokens: the live state
        continues), L (the long unrelated request), T4 (tokens: T3 + its reply + 200 tokens: the parked conversation). with_sides:
        the side calls between them. Returns {name: (tokens, first logits, rec)} of every request, and the per-step checks"""
        out, ckq = {}, []

        def one(name, fn):
            k = len(rec); g = fn(); R.wait_idle()
            out[name] = dict(tokens=list(g), first=lgs[-1], rec=rec[k] if len(rec) > k else None)
            return g

        def sides_between(names):
            before = _main_line(R)
            for nm in names:
                if nm == 'abort':
                    # a side call whose prompt phase fails after first_logits launched (the cache's extend raises): HTTP 500;
                    # the session's abort drops only the side call (drop_live during a side call: the main line back, intact)
                    def ext(*a, _o=R.pc.extend, **k):
                        R.pc.__dict__.pop('extend', None); raise RuntimeError('test_serve_k3 side: injected error after first_logits launched')
                    c0 = dict(R.sess.counters); rs0 = R.srv.worker.stats['resets']
                    R.pc.extend = ext
                    try: st_ = _post_status(R, '/v1/chat/completions', dict(model=model, messages=sides['summary'][1][:1] + [
                        {'role': 'user', 'content': 'A different text: the drafter proposes tokens, the target verifies them.'}],
                        max_tokens=n_side, temperature=0.0))[0]
                    finally: R.pc.__dict__.pop('extend', None)
                    R.wait_idle()
                    out['abort'] = dict(http=st_, side_dropped=R.sess.counters['side_dropped'] - c0['side_dropped'],
                                        dropped_live=R.sess.counters['dropped_live'] - c0['dropped_live'], side_after=R.pc.side is not None,
                                        resets=R.srv.worker.stats['resets'] - rs0, rec=rec[-1] if rec else None)
                elif nm == 'bail':
                    # a side call whose 2nd target pass (its first decode pass) bails on every attempt: recover(at, ids) re-enters it
                    os.environ['K3_TEST_HOOKS'] = 'police-controls'; os.environ['K3_TEST_BAIL'] = '1'
                    try: s2 = K3Session(R.m, R.pf, R.dr, R.pc, stats=R.stats, idle_s=None)
                    finally: os.environ.pop('K3_TEST_HOOKS', None); os.environ.pop('K3_TEST_BAIL', None)
                    old = R.sess
                    R.srv.loading = True; R.srv.set_session(s2, s2.drafter); R.srv.loading = False
                    rc0 = R.pc.stats['recovered']
                    try:
                        st_, g, _ = send_side('title')
                    finally:
                        R.wait_idle(); R.srv.loading = True; R.srv.set_session(old, old.drafter); R.srv.loading = False
                    out['bail'] = dict(http=st_, tokens=list(g), first=s2.last_first_logits, recoveries=s2.counters['recoveries'],
                                       side_recoveries=s2.counters['side_recoveries'], recovered=R.pc.stats['recovered'] - rc0,
                                       route=(s2.last_route or {}).get('route'))
                else:
                    k = len(rec)
                    st_, g, _ = send_side(nm)
                    R.wait_idle()
                    out[f'side:{nm}:{len([x for x in out if x.startswith("side:" + nm)])}'] = dict(tokens=list(g), first=lgs[-1],
                                                                                               rec=rec[k] if len(rec) > k else None, http=st_)
            after = _main_line(R)
            ckq.append(dict(names=list(names), same_log=before[0] == after[0], same_state=before[1] == after[1],
                            same_vram=before[2] == after[2], n_vram=len(after[2]), state=after[1]))

        fresh()
        mlost0, er0, rs0 = R.pc.stats['main_lost'], R.pc.stats['external_reset'], R.srv.worker.stats['resets']
        u0 = vram_used()
        g1 = one('T1', lambda: R.chat(m1, n_main, model=model, tools=tools)[1])
        if with_sides: sides_between(['title'])
        g2 = one('T2', lambda: R.chat(m2, n_main, model=model, tools=tools)[1])
        p2 = list(R.jobs[-1].prompt_ids)
        if with_sides: sides_between(['summary', 'raw', 'abort', 'bail'])
        T3 = p2 + g2 + [int(t) for t in code[:200]]
        g3 = one('T3', lambda: R.complete(T3, n_main)[1])
        if with_sides: sides_between(['next', 'title'])
        parked0 = R.pc.stats['parked']
        one('L', lambda: R.complete(long_req, 8)[1])
        out['L']['parked'] = R.pc.stats['parked'] - parked0
        T4 = T3 + g3 + [int(t) for t in shell[:200]]
        one('T4', lambda: R.complete(T4, n_main)[1])
        if with_sides:
            # a request sharing the root (the same system + tools, another question): main, whatever its size
            one('root', lambda: R.chat(m1[:1] + [{'role': 'user', 'content': 'Which tool reads files?'}], 8, model=model, tools=tools)[1])
            sides_between(['raw'])
            # the idle park after a side call: the side call ended, the conversation parked
            p0, se0 = R.pc.stats['parked'], R.sess.counters['side_ended']
            side_before = R.pc.side is not None
            R.sess.idle_s = R.srv.worker.idle_s = 1.0
            try:
                R.complete(sides['raw'][1], 4)                     # (a side call again: the idle timer starts after a job)
                t0 = time.time()
                while R.pc.stats['parked'] == p0 and time.time() - t0 < 20: time.sleep(0.05)
                R.pc.fence()
            finally:
                R.sess.idle_s = R.srv.worker.idle_s = None
            out['idle'] = dict(side_before=side_before, parked=R.pc.stats['parked'] - p0, side_ended=R.sess.counters['side_ended'] - se0,
                               side_after=R.pc.side is not None, error=R.pc.park_error)
        out['_checks'] = ckq
        out['_totals'] = dict(main_lost=R.pc.stats['main_lost'] - mlost0, external_reset=R.pc.stats['external_reset'] - er0,
                              resets=R.srv.worker.stats['resets'] - rs0, vram=(u0, vram_used()))
        return out

    try:
        A = main_turns(True)
        text = _get(R, '/metrics').decode(); S = json.loads(_get(R, '/caminus/status'))
        B = main_turns(False)
        C = {}
        for nm in ('title', 'summary', 'raw', 'next'):
            fresh()
            k = len(rec)
            st_, g, _ = send_side(nm)
            R.wait_idle()
            C[nm] = dict(tokens=list(g), first=lgs[-1], rec=rec[k] if len(rec) > k else None)
        fresh()
    finally:
        undo()
        R.sess.idle_s = R.srv.worker.idle_s = None

    def same(a, b):
        return a['tokens'] == b['tokens'] and a['first'] is not None and b['first'] is not None and a['first'].tobytes() == b['first'].tobytes()

    def rt(x):
        r = (x.get('rec') or {}).get('route') or {}
        return (r.get('route'), r.get('reason'))

    rows = {}
    # the main requests: == no side calls, bitwise; routes and sources
    for nm in ('T1', 'T2', 'T3', 'L', 'T4'):
        a, b = A[nm], B[nm]
        rows[nm] = dict(equal=same(a, b), route=rt(a), source=(a['rec'] or {}).get('source'), cached=(a['rec'] or {}).get('cached'),
                        tokens=len(a['tokens']), ref_source=(b['rec'] or {}).get('source'))
    want = {'T1': ('main', 'no_main'), 'T2': ('main', 'match'), 'T3': ('main', 'match'), 'L': ('main', 'no_fit'), 'T4': ('main', 'parked')}
    ok_main = all(rows[k]['equal'] and rows[k]['route'] == want[k] for k in want) and rows['T3']['source'] == 'live' \
        and rows['T4']['source'] == 'disk' and A['L']['parked'] == 1
    check(ok_main, 'side: the conversation\'s main requests with side calls between them == the same requests with none (first logits and '
                   'tokens bitwise); routes / sources: ' + '; '.join(f'{k} {rows[k]["route"]} {rows[k]["source"]} {rows[k]["cached"]} '
                                                                     f'({rows[k]["tokens"]} tokens, equal {rows[k]["equal"]})' for k in want)
                   + f'; L auto-parked the conversation ({A["L"]["parked"]} park)')
    # the side calls: == each alone, cold
    srows = []
    for k, v in A.items():
        if not k.startswith('side:'): continue
        nm = k.split(':')[1]
        srows.append((k, rt(v), (v['rec'] or {}).get('source'), v['http'], same(v, C[nm]), len(v['tokens'])))
    ok_side = srows and all(r[1] == ('side', 'unrelated') and r[2] == 'cold' and r[3] == 200 and r[4] for r in srows)
    check(ok_side, f'side: every side call ran in the side slot from cold and == a cold run of it alone (first logits and tokens bitwise): '
                   f'(name, route, source, HTTP, equal, tokens) {srows}')
    # failures inside a side call
    ab, bl = A['abort'], A['bail']
    check(ab['http'] == 500 and ab['side_dropped'] == 1 and ab['dropped_live'] == 1 and not ab['side_after'] and ab['resets'] == 0
          and rt(ab) == ('side', 'unrelated'),
          f'side: a side call whose prompt phase fails (HTTP {ab["http"]}): only the side call dropped (side_dropped {ab["side_dropped"]}, the '
          f'main line back: side active after {ab["side_after"]}), no reset ({ab["resets"]}); the next main request (T3) == no side calls above')
    okb = (bl['http'] == 200 and bl['route'] == 'side' and bl['recoveries'] == 1 and bl['side_recoveries'] == 1 and bl['recovered'] == 1
           and bl['tokens'] == C['title']['tokens'] and bl['first'] is not None and C['title']['first'] is not None
           and bl['first'].tobytes() == C['title']['first'].tobytes())
    check(okb, f'side: a side call whose decode pass bails on every attempt (K3_TEST_BAIL 1): HTTP {bl["http"]}, recovered inside the side '
               f'call ({bl["recoveries"]} recovery, recover(at, ids)), {len(bl["tokens"])} tokens and first logits == the cold run; '
               f'the next main request (T3) == no side calls')
    # the main line across the side calls
    ck = A['_checks']
    tot = A['_totals']
    check(all(c['same_log'] and c['same_state'] and c['same_vram'] for c in ck) and tot['main_lost'] == 0 and tot['external_reset'] == 0
          and tot['resets'] == 0,
          f'side: the main line\'s log, state and VRAM checkpoints unchanged across every group of side calls '
          f'{[(c["names"], c["same_log"], c["same_state"], c["same_vram"], c["n_vram"]) for c in ck]}; main_lost {tot["main_lost"]}, '
          f'external resets {tot["external_reset"]}, worker resets {tot["resets"]}; VRAM (sysfs) before / after the run {tot["vram"]}')
    # a request sharing the root goes to main; the decisions in the status and the series
    rr = S['cache']['route']
    n_side = len(srows) + 1                                  # (+ the bail case's side call; the abort case's counts too)
    check(rt(A['root']) == ('main', 'match') and rr['side']['unrelated'] >= n_side and rr['main']['parked'] >= 1 and rr['main']['no_fit'] >= 1
          and 'caminus:cache_route_total{route="side",reason="unrelated"}' in text and 'caminus:cache_side_active' in text,
          f'side: a request sharing the root (the same system + tools) -> {rt(A["root"])}; /caminus/status cache.route {rr}')
    idl = A['idle']
    check(idl['side_before'] and idl['parked'] == 1 and idl['side_ended'] >= 1 and not idl['side_after'] and idl['error'] is None,
          f'side: the idle park after a side call ended it and parked the conversation: {idl}')
    NUMBERS['side'] = dict(main=rows, side=srows, abort={k: v for k, v in ab.items() if k != 'rec'},
                           bail={k: v for k, v in bl.items() if k not in ('tokens', 'first')}, checks=ck, totals=tot, route=rr, idle=idl)


def stage_vram(R):
    u = vram_used()
    NUMBERS.setdefault('vram', []).append(u)
    check(u[0] + 512 <= 23800 and u[1] <= 19850, f'vram: XTX {u[0]:.0f} MiB (+ the 512 reserve {u[0] + 512:.0f} <= 23,800), XT {u[1]:.0f} <= 19,850')


def stage_cli(cache_dir, coder):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'python'))
    py = sys.executable
    p = subprocess.Popen([py, '-I', os.path.join(ROOT, 'tools', 'caminus_serve.py'), '--port', '0', '--max-ctx', '4096', '--model', coder,
                          '--cache-dir', os.path.join(cache_dir, 'cli'), '--log-dir', '', '--idle-park-s', '0',
                          '--flight-dir', os.path.join(cache_dir, 'flight-cli')],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    port, lines, loading503, ok200, reply = None, [], False, False, None
    reply_t2, bundle, reply2 = (None, {}), (None, {}), (None, {})
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            line = p.stdout.readline()
            if not line: break
            lines.append(line.rstrip())
            if port is None and 'http://' in line:
                port = int(line.split('http://')[1].split()[0].rsplit(':', 1)[1].rstrip(')'))
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=5); c.request('GET', '/health'); loading503 = c.getresponse().status == 503
                except OSError: pass
            if 'ready on' in line: break
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=600); c.request('GET', '/health'); h = c.getresponse(); h.read()
        ok200 = h.status == 200
        body = json.dumps(dict(model='k3-coder-nothink', messages=CHATS[1], max_tokens=24, temperature=0.0)).encode()
        c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'}); rr = c.getresponse()
        reply = (rr.status, json.loads(rr.read()))
        # fix:m7: the conversation's second turn (a non-cold start: its bundle exports the start state), a flight bundle of it,
        # then the side call and the shutdown park must still work (before the fix the export left the cache's launch epochs
        # stale: the next request answered 503 and the shutdown park logged None)
        msgs2 = CHATS[1] + [{'role': 'assistant', 'content': reply[1]['choices'][0]['message'].get('content') or ''},
                            {'role': 'user', 'content': 'Now in one sentence.'}]
        body = json.dumps(dict(model='k3-coder-nothink', messages=msgs2, max_tokens=16, temperature=0.0)).encode()
        c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'}); rr = c.getresponse()
        reply_t2 = (rr.status, json.loads(rr.read()))
        c.request('POST', '/caminus/flight/bundle', b'{}', {'Content-Type': 'application/json'}); rr = c.getresponse()
        bundle = (rr.status, json.loads(rr.read()))
        # a short unrelated request (shorter than the conversation, no shared prefix past the template's first tokens): a side call
        body2 = json.dumps(dict(model='k3-coder-nothink', messages=[{'role': 'user', 'content': 'Say ok.'}], max_tokens=8,
                                temperature=0.0)).encode()
        c.request('POST', '/v1/chat/completions', body2, {'Content-Type': 'application/json'}); rr2 = c.getresponse()
        reply2 = (rr2.status, json.loads(rr2.read()))
        c.request('GET', '/caminus/status'); sst = json.loads(c.getresponse().read())
    finally:
        p.terminate()
        try: p.wait(180)
        except subprocess.TimeoutExpired: p.kill(); p.wait(30)
    rest = p.stdout.read() if p.stdout else ''
    lines += [l for l in rest.splitlines() if l.strip()]
    print('\n'.join('  cli| ' + l for l in lines[-12:]), flush=True)
    ck = sst.get('clock_check') if isinstance(sst, dict) else None
    sysfs = [open(f'/sys/class/drm/{n}/device/power_dpm_force_performance_level').read().strip() for n in ('card1', 'card2')]
    check(ck is not None and [x['perf_level'] for x in ck['cards']] == sysfs and any('clock check' in l for l in lines),
          f'cli: the startup clock check in the log and in /caminus/status: {ck and ck["summary"]}; sysfs levels {sysfs}')
    park = [l for l in lines if 'shutdown: park' in l]
    rt = (sst.get('cache') or {}).get('route') if isinstance(sst, dict) else None
    check(reply2[0] == 200 and rt is not None and rt['side']['unrelated'] == 1 and rt['last']['route'] == 'side' and rt['side_active'],
          f'cli: the short unrelated request ran in the side slot (HTTP {reply2[0]}; /caminus/status cache.route {rt})')
    check(p.returncode == 0 and park and "'unchanged'" not in park[0] and 'None' not in park[0],
          f'cli: SIGTERM after a side call -> a clean shutdown with the conversation parked (the side call ended first): {park[:1]}; '
          f'exit {p.returncode}')
    flight_lines = []
    fd = os.path.join(cache_dir, 'flight-cli')
    for f_ in sorted(os.listdir(fd)) if os.path.isdir(fd) else []:
        if f_.endswith('.jsonl'): flight_lines += [json.loads(x) for x in open(os.path.join(fd, f_)) if x.strip()]
    check(len(flight_lines) == 3 and all(x.get('status') == 200 for x in flight_lines)
          and [(x.get('cache') or {}).get('slot') for x in flight_lines] == ['main', 'main', 'side'],
          f'cli: the flight recorder wrote {len(flight_lines)} line(s) in its scratch directory, slots '
          f'{[(x.get("cache") or {}).get("slot") for x in flight_lines]}')
    bref = (bundle[1].get('cache_ref') or {}) if isinstance(bundle[1], dict) else {}
    check(reply_t2[0] == 200 and bundle[0] == 200 and bref.get('source') in ('live', 'vram') and bref.get('state') and reply2[0] == 200,
          f'cli (fix:m7): the second turn (HTTP {reply_t2[0]}), a flight bundle of it (HTTP {bundle[0]}, start {bref.get("source")}, state '
          f'{bref.get("state")}), then the side call (HTTP {reply2[0]}) and the shutdown park (above) still work')
    check(loading503 and ok200 and reply is not None and reply[0] == 200 and reply[1]['choices'][0]['message']['content'] is not None,
          f'cli: python -I tools/caminus_serve.py --port 0: /health 503 while loading, 200 when ready; a chat request -> {reply and reply[0]} '
          f'{reply and repr(reply[1]["choices"][0]["message"].get("content"))[:80]}; SIGTERM -> exit {p.returncode}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('args', nargs='+', help='[CODER] STAGE...')
    ap.add_argument('--cache-dir', required=True); ap.add_argument('--json', default=None)
    ap.add_argument('--small', action='store_true'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-ctx', type=int, default=33024)
    a = ap.parse_args()
    coder = next((x for x in a.args if x.endswith('.gguf')), C.CODER)
    stages = [s for s in a.args if not s.endswith('.gguf')]
    cache_dir = safe_dir(a.cache_dir); os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    print(host_note(), flush=True)
    if stages == ['cli']:
        stage_cli(cache_dir, coder)
    else:
        import torch
        for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
        assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
        R = Rig(coder, a.max_ctx, os.path.join(cache_dir, 'gpu'))
        try:
            for s_ in stages:
                t0 = time.time(); print(f'== stage {s_} ({host_note()})', flush=True)
                {'t1a': lambda: stage_t1a(R, a.small), 't1d': lambda: stage_t1d(R, a.small), 't1e': lambda: stage_t1e(R, a.small),
                 't1dlong': lambda: stage_t1dlong(R, a.small), 'reuse': lambda: stage_reuse(R, a.small),
                 'samppen': lambda: stage_samppen(R, a.small), 'idle': lambda: stage_idle(R, a.small), 'bgpark': lambda: stage_idle(R, a.small),
                 'supersede': lambda: stage_supersede(R, a.small),
                 'overhead': lambda: stage_overhead(R, a.small, a.reps), 'cancel': lambda: stage_cancel(R, a.small),
                 'abort': lambda: stage_abort(R, a.small), 'abortx': lambda: stage_abortx(R, a.small),
                 'widepen': lambda: stage_widepen(R, a.small), 'bail': lambda: stage_bail(R, a.small), 'policy': lambda: stage_policy(R, a.small),
                 'metrics': lambda: stage_metrics(R, a.small), 'scrape': lambda: stage_scrape(R, a.small, a.reps), 'flight': lambda: stage_flight(R, a.small),
                 'side': lambda: stage_side(R, a.small),
                 'vram': lambda: stage_vram(R)}[s_]()
                R.wait_idle(); R.sync(); stage_vram(R)
                print(f'== stage {s_} done in {time.time() - t0:.0f} s; cache stats {dict(R.pc.stats)}; session {R.sess.counters}', flush=True)
                if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
        finally:
            R.close()
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
