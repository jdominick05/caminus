"""The monitoring surface's renderers (docs/agentic/spec-metrics.md): llama-server's /metrics, /slots and /props
shapes, plus the Caminus series and the versioned JSON status. Read side only: every function here runs on an HTTP
thread and reads stats.EngineStats, the server's own counters and the hardware reader's latest reading.

llama-server compatibility (tools/server in the ignite-llama fork: server-task.cpp to_metrics, server-context.cpp
server_slot::to_json and get_res_props, task_params::to_json(only_metrics=true)): the same `llamacpp:` names, HELP
texts, types and order, the per-position series' `position` label, the same JSON keys and types. Differences, by
design: the speculative counters move per pass (llama-server adds them when a request ends), values print exactly
(llama-server prints 6 significant digits), and /slots never carries prompt or generated text (llama-server's default,
slots_debug off)."""
import math
import time

from . import stats as S

# ------------------------------------------------------------------------------------------------ Prometheus text
LLAMA_COUNTERS = (
    ('prompt_tokens_total', 'Number of prompt tokens processed, excluding cached tokens'),
    ('prompt_tokens_cached_total', 'Number of prompt tokens reused from the cache'),
    ('prompt_seconds_total', 'Total time spent processing prompts'),
    ('tokens_predicted_total', 'Number of generation tokens processed'),
    ('tokens_predicted_seconds_total', 'Total time spent generating tokens'),
    ('n_decode_total', 'Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding'),
    ('n_tokens_max', 'Largest observed sequence length (prompt + generation)'),
    ('spec_decode_num_draft_tokens_total', 'Speculative: Total draft tokens generated'),
    ('spec_decode_num_accepted_tokens_total', 'Speculative: Total draft tokens accepted by the target model'),
    ('spec_decode_num_drafts_total', 'Speculative: Total speculative decoding verification steps'),
)
LLAMA_GAUGES = (
    ('prompt_tokens_seconds', 'Average prompt throughput in tokens/s'),
    ('predicted_tokens_seconds', 'Average generation throughput in tokens/s'),
    ('requests_processing', 'Number of requests processing'),
    ('requests_deferred', 'Number of requests deferred'),
    ('n_busy_slots_per_decode', 'Average number of busy slots per llama_decode() call'),
)
PER_POS = 'spec_decode_num_accepted_tokens_per_pos_total'


def fmt(v):
    """a sample value in the text format (None -> NaN)"""
    if v is None:
        return 'NaN'
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if math.isnan(f):
        return 'NaN'
    if math.isinf(f):
        return '+Inf' if f > 0 else '-Inf'
    f = round(f, 6)
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def esc(v):
    return str(v).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def labels(d):
    if not d:
        return ''
    return '{' + ','.join(f'{k}="{esc(v)}"' for k, v in d.items()) + '}'


class Prom:
    def __init__(self):
        self.lines = []

    def metric(self, name, typ, help_, samples):
        """samples: [(labels dict or None, value)]; samples whose value is None are left out (unknown), and a metric
        with none left is not written"""
        samples = [(lb, v) for lb, v in samples if v is not None]
        if not samples:
            return
        self.lines.append(f'# HELP {name} {help_}')
        self.lines.append(f'# TYPE {name} {typ}')
        for lb, v in samples:
            self.lines.append(f'{name}{labels(lb)} {fmt(v)}')

    def summary(self, name, help_, series_by_label, key):
        """a summary over the window: quantile 0.5 / 0.95 per label value, _sum / _count since start"""
        out = []
        for lv, ser in series_by_label:
            if not ser.count:
                continue
            p50, p95 = ser.quantiles()
            out.append((lv, p50, p95, ser.sum, ser.count))
        if not out:
            return
        self.lines.append(f'# HELP {name} {help_}')
        self.lines.append(f'# TYPE {name} summary')
        for lv, p50, p95, sm, n in out:
            base = {key: lv} if key else {}
            self.lines.append(f'{name}{labels({**base, "quantile": "0.5"})} {fmt(p50)}')
            self.lines.append(f'{name}{labels({**base, "quantile": "0.95"})} {fmt(p95)}')
        for lv, p50, p95, sm, n in out:
            base = {key: lv} if key else {}
            self.lines.append(f'{name}_sum{labels(base)} {fmt(sm)}')
            self.lines.append(f'{name}_count{labels(base)} {fmt(n)}')

    def text(self):
        return '\n'.join(self.lines) + '\n'


def llama_block(st, processing, deferred, take_bucket=True):
    """llama-server's lines, in its order (server_task_result_metrics::to_metrics)"""
    prompt_rate, pred_rate = st.take_bucket() if take_bucket else (0.0, 0.0)
    counters = (st.prompt_tokens, st.prompt_cached, st.prompt_us / 1e6, st.predicted_tokens, st.predicted_us / 1e6,
                st.n_decode, st.n_tokens_max, st.spec_draft_tokens, st.spec_accepted, st.spec_drafts)
    gauges = (prompt_rate, pred_rate, processing, deferred, st.n_busy / max(st.n_decode, 1))
    lines = []
    for (name, help_), v in zip(LLAMA_COUNTERS, counters):
        lines += [f'# HELP llamacpp:{name} {help_}', f'# TYPE llamacpp:{name} counter', f'llamacpp:{name} {fmt(v)}']
    for (name, help_), v in zip(LLAMA_GAUGES, gauges):
        lines += [f'# HELP llamacpp:{name} {help_}', f'# TYPE llamacpp:{name} gauge', f'llamacpp:{name} {fmt(v)}']
    app = list(st.accepted_per_pos)
    if app:
        lines += [f'# HELP llamacpp:{PER_POS} Accepted tokens per draft position',
                  f'# TYPE llamacpp:{PER_POS} counter']
        lines += [f'llamacpp:{PER_POS}{{position="{i}"}} {n}' for i, n in enumerate(app)]
    return lines


def prometheus(st, srv, cards, energy):
    """the whole /metrics text: llama-server's block, then the caminus: series. srv: server_view(); cards: the hardware
    reader's latest per-card dicts (or []); energy: its per-card joules by phase (or [])."""
    P = S.PREFIX
    out = Prom()
    out.lines = llama_block(st, srv['requests_processing'], srv['requests_deferred'])
    m = out.metric
    m(P + 'info', 'gauge', 'Caminus engine and build (the value is always 1)',
      [({'engine': S.ENGINE, 'build': srv['build_info'], 'model': srv['model_alias'],
         'status_version': S.STATUS_VERSION}, 1)])
    m(P + 'stats_enabled', 'gauge', 'Whether the engine stats hook records (0: the M7 overhead A/B)',
      [(None, int(st.enabled))])
    # ---- speed vs physics
    ph = st.physics_view()
    m(P + 'physics_floor_ms', 'gauge', 'Bandwidth floor of one decode token: bytes per token / bandwidth per card '
      '(mode max: the slowest card)', [(None, ph['floor_ms'])])
    m(P + 'physics_pass_pct', 'gauge', 'Bandwidth floor / median K3 pass time, percent (a pass streams at least one '
      "token's bytes: at most 100)", [(None, ph['pass_pct'])])
    m(P + 'physics_token_pct', 'gauge', 'Bandwidth floor / decode ms per emitted token over the window, percent '
      '(above 100 when drafts are accepted)', [(None, ph['token_pct'])])
    m(P + 'decode_ms_per_token', 'gauge', 'Decode wall ms per emitted token over the window', [(None, ph['ms_per_token'])])
    out.summary(P + 'decode_pass_ms', 'Decode pass time split: K3 pass, drafter, sampler, host, total (ms)',
                [(p, st.pass_series[p]) for p in S.PASS_PARTS], 'part')
    out.summary(P + 'ttft_ms', 'Time to first token split: queue, restore, batched prefill, hand-off, K3 prompt '
                'passes, host, total (ms)', [(p, st.ttft[p]) for p in S.TTFT_PARTS], 'part')
    out.summary(P + 'prefill_tokens_per_second', 'Prompt tokens processed per second, per request',
                [(None, st.prefill_rate)], None)
    # ---- DFlash
    sv = st.spec_view()
    m(P + 'spec_draft_tokens_total', 'counter', 'Draft tokens verified, per content class',
      [({'class': c}, st.cls_draft_tokens[c]) for c in S.CLASSES])
    m(P + 'spec_accepted_tokens_total', 'counter', 'Draft tokens accepted, per content class',
      [({'class': c}, st.cls_accepted[c]) for c in S.CLASSES])
    m(P + 'spec_drafts_total', 'counter', 'Verification passes with drafts, per content class',
      [({'class': c}, st.cls_drafts[c]) for c in S.CLASSES])
    m(P + 'spec_accepted_per_pos_total', 'counter', 'Accepted draft tokens per content class and draft position',
      [({'class': c, 'position': i}, n) for c in S.CLASSES for i, n in enumerate(list(st.cls_pos[c]))])
    m(P + 'spec_acceptance_ratio', 'gauge', 'Accepted / drafted tokens since start, per content class',
      [({'class': c}, sv['classes'][c]['acceptance']) for c in S.CLASSES] + [({'class': 'all'}, sv['acceptance'])])
    m(P + 'spec_tokens_per_pass', 'gauge', 'Emitted tokens per decode pass over the window', [(None, sv['tokens_per_pass'])])
    m(P + 'plain_ms_per_token', 'gauge', 'A plain (undrafted, one-token) pass, ms (measured EMA, else the configured '
      'value)', [(None, sv['plain_ms_per_token'])])
    m(P + 'spec_speedup_vs_plain', 'gauge', 'Plain ms per token / decode ms per emitted token over the window',
      [(None, sv['speedup_vs_plain'])])
    # ---- prompt cache
    m(P + 'cache_tokens_total', 'counter', 'Prompt tokens reused, by tier (vram: the live state or a VRAM '
      'checkpoint; disk: spilled or parked; exact: exact hits)', [({'tier': t}, v) for t, v in st.cache_tokens.items()])
    m(P + 'cache_requests_total', 'counter', 'Requests by the state they started from (the prompt cache source)',
      [({'source': s}, v) for s, v in st.cache_sources.items()])
    out.summary(P + 'cache_restore_ms', 'Time to restore a cached state (not the live one), ms',
                [(None, st.cache_restore)], None)
    out.summary(P + 'cache_lcp_tokens', 'Longest common prefix of a request with the live token log',
                [(None, st.cache_lcp)], None)
    g = st.cache_gauges
    m(P + 'cache_vram_slots', 'gauge', 'VRAM checkpoint tier: slots in use / slots',
      [({'state': 'used'}, g['slots_used']), ({'state': 'total'}, g['n_slots'])])
    m(P + 'cache_vram_tier_mib', 'gauge', 'VRAM held by the checkpoint tier, MiB', [(None, g['tier_mib'])])
    m(P + 'cache_vram_headroom_mib', 'gauge', 'VRAM headroom the tier was sized from, MiB', [(None, g['headroom_mib'])])
    m(P + 'cache_disk_bytes', 'gauge', 'Disk tier size, bytes', [(None, g['disk_bytes'])])
    m(P + 'cache_disk_cap_bytes', 'gauge', 'Disk tier cap, bytes', [(None, g['disk_cap_bytes'])])
    m(P + 'cache_events_total', 'counter', "The prompt cache's own counters (checkpoints, evictions, spills, parks, "
      'refusals, side calls)', [({'event': k}, v) for k, v in st.cache_counters.items()])
    m(P + 'cache_route_total', 'counter', 'Requests by the slot side-slot routing sent them to (main / side) and why',
      [({'route': r, 'reason': why}, v) for r, d in st.cache_route.items() for why, v in d.items()])
    m(P + 'cache_side_active', 'gauge', '1 while a side call holds the engine (the main line is in the side slot), else 0',
      [(None, None if g['side_active'] is None else int(bool(g['side_active'])))])
    m(P + 'cache_prefill_seconds_saved_total', 'counter', 'Prefill seconds saved by reuse (estimate: reused tokens / '
      'the prefill rate EMA)', [(None, st.saved_s)])
    jpt = _prefill_joules_per_token(st, energy)
    m(P + 'cache_prefill_joules_saved_total', 'counter', 'Prefill energy saved by reuse (estimate: reused tokens x '
      'measured prefill joules per token)', [(None, None if jpt is None else st.saved_tokens * jpt)])
    # ---- context
    cx = context_view(st)
    m(P + 'context_position', 'gauge', 'Tokens in the sequence of the current (or last) request', [(None, cx['position'])])
    m(P + 'context_max', 'gauge', 'The engine\'s max_ctx', [(None, cx['max_ctx'] or None)])
    m(P + 'context_remaining', 'gauge', 'Tokens left to max_ctx', [(None, cx['remaining'])])
    m(P + 'context_qsa', 'gauge', '1 when the position is past the dense range (QSA attention), else 0',
      [(None, int(cx['attention'] == 'qsa'))])
    m(P + 'kv_bytes', 'gauge', 'KV + indexer bytes the sequence holds, per card',
      [({'card': i}, b) for i, b in enumerate(cx['kv_bytes'])])
    # ---- reliability
    m(P + 'bails_total', 'counter', 'Passes that bailed, by outcome (recovered by the bitwise retry, or failed)',
      [({'outcome': k}, v) for k, v in st.bails.items()])
    m(P + 'pass_retries_total', 'counter', 'Pass retries after a bail', [(None, st.retries)])
    m(P + 'kfd_evicted_ms_total', 'counter', 'KFD queue eviction time charged to the engine, ms', [(None, st.evicted_ms)])
    m(P + 'pgmigrate_total', 'counter', 'Host page migrations (/proc/vmstat pgmigrate_success, system-wide) in sampler '
      'intervals while the engine was busy', [(None, st.pgmigrate)])
    m(P + 'pgmigrate_all_total', 'counter', 'Host page migrations (system-wide) since the first sample, idle included',
      [(None, st.pgmigrate_all)])
    m(P + 'refusals_total', 'counter', 'Requests or launches refused on the host, by kind',
      [({'kind': k}, v) for k, v in st.refusals.items()])
    m(P + 'errors_total', 'counter', 'Requests that ended in an error, by kind (engine: retries exhausted; internal: a '
      'bug)', [({'kind': 'engine'}, st.engine_errors), ({'kind': 'internal'}, st.internal_errors)])
    # ---- TP / hardware
    out.summary(P + 'card_pass_ms', 'Per-card busy time of a decode pass, ms',
                [(str(i), s) for i, s in enumerate(st.card_pass)], 'card')
    out.summary(P + 'exchange_wait_ms', 'Per-card wait on the TP exchange in a decode pass, ms',
                [(str(i), s) for i, s in enumerate(st.card_wait)], 'card')
    m(P + 'link_bytes_total', 'counter', 'Bytes moved over the TP link by decode passes', [(None, st.link_bytes)])
    m(P + 'link_bytes_per_second', 'gauge', 'TP link bytes per second of decode pass time over the window',
      [(None, link_rate(st))])
    for key, name, help_ in (('vram_used_mib', 'vram_used_mib', 'VRAM used on the card (sysfs, every process), MiB'),
                             ('vram_total_mib', 'vram_total_mib', 'VRAM on the card, MiB'),
                             ('power_w', 'power_watts', 'Card power (hwmon power1_average), W'),
                             ('sclk_mhz', 'sclk_mhz', 'Shader clock, MHz'), ('mclk_mhz', 'mclk_mhz', 'Memory clock, MHz'),
                             ('busy_pct', 'gpu_busy_percent', 'GPU busy, percent'),
                             ('temp_junction_c', 'temp_junction_celsius', 'Junction temperature, Celsius')):
        m(P + name, 'gauge', help_, [({'card': c['index'], 'name': c['name']}, c[key]) for c in cards])
    m(P + 'clock_forced_high', 'gauge', '1 when power_dpm_force_performance_level is high (gpu-high), else 0',
      [({'card': c['index'], 'name': c['name']}, None if c['forced_high'] is None else int(c['forced_high']))
       for c in cards])
    m(P + 'runtime_suspend_allowed', 'gauge', "1 when the card's runtime PM control is auto (it may suspend: a forced high "
      "level is then not held), 0 when on", [({'card': c['index'], 'name': c['name']}, None if c.get('runtime_pm') is None else
                                              int(c['runtime_pm'] != 'on')) for c in cards])
    ck = srv.get('clock_check')
    m(P + 'clock_check_ok', 'gauge', "The startup clock check: 1 when every engine card ran at DPM high with runtime PM on "
      "(reported, never changed)", [(None, None if not ck or ck.get('ok') is None else int(ck['ok']))])
    m(P + 'vram_guard_margin_mib', 'gauge', "MiB below tools/guard.sh's kill line minus the card's reserve",
      [({'card': c['index'], 'name': c['name']}, guard_margin(st, c)) for c in cards])
    m(P + 'energy_joules_total', 'counter', 'Card energy integrated from hwmon power, by engine phase',
      [({'card': i, 'phase': ph_}, round(j, 3)) for i, e in enumerate(energy) for ph_, j in e.items()])
    tpj = tokens_per_joule(st, energy)
    m(P + 'tokens_per_joule', 'gauge', 'Tokens per joule of both cards: decode (generated tokens / decode-phase energy)'
      ' and prefill (processed prompt tokens / prefill-phase energy)',
      [({'phase': 'decode'}, tpj['decode']), ({'phase': 'prefill'}, tpj['prefill'])])
    # ---- agent behaviour
    with st.lock:
        harness = dict(st.harness)
        tool_calls, fails, lenient = st.tool_calls, dict(st.tool_failures), st.tool_lenient
        loops, repeated = st.loop_flagged, st.loop_repeated
    m(P + 'requests_total', 'counter', 'Generation requests by harness (from the User-Agent, a fixed set)',
      [({'harness': h}, v) for h, v in sorted(harness.items())])
    m(P + 'tool_calls_total', 'counter', 'Tool calls parsed and reported', [(None, tool_calls)])
    m(P + 'tool_call_failures_total', 'counter', 'Tool-call parse failures, by kind',
      [({'kind': k}, v) for k, v in fails.items()])
    m(P + 'tool_call_lenient_total', 'counter', 'Tool-call leniency the parser applied (aliases, coerced values)',
      [(None, lenient)])
    m(P + 'tokens_generated_total', 'counter', 'Generated tokens by kind (reasoning up to </think>, answer after)',
      [({'kind': 'reasoning'}, st.tokens_reasoning), ({'kind': 'answer'}, st.tokens_answer)])
    m(P + 'finish_total', 'counter', 'Requests by finish reason', [({'reason': k}, v) for k, v in st.finish.items()])
    m(P + 'max_tokens_hits_total', 'counter', 'Requests that ended at max_tokens', [(None, st.max_tokens_hits)])
    m(P + 'loop_requests_total', 'counter', 'Requests whose reasoning repeated a line LoopDetector.REPEAT times or '
      'more', [(None, loops)])
    m(P + 'reasoning_repeated_lines_total', 'counter', 'Reasoning line occurrences beyond the first, per request, '
      'summed', [(None, repeated)])
    m(P + 'loop_live_repeats', 'gauge', "The current request's highest repeat count of one reasoning line",
      [(None, st.loop_live)])
    # ---- server
    m(P + 'queue_depth', 'gauge', 'Requests on the engine plus waiting', [(None, srv['queue_depth'])])
    m(P + 'http_responses_total', 'counter', 'HTTP responses by route and status',
      [({'route': r, 'status': s}, v) for (r, s), v in sorted(srv['http'].items(), key=lambda kv: (str(kv[0][0]),
                                                                                                   str(kv[0][1])))])
    m(P + 'worker_total', 'counter', "The engine worker's own counters (jobs, resets, aborts, ...)",
      [({'counter': k}, v) for k, v in sorted(srv['worker'].items())])
    return out.text()


# ------------------------------------------------------------------------------------------------ derived values
def link_rate(st):
    """link bytes over the window / the window's decode pass time (bytes per second); None before any pass"""
    nb = st.link_ring.window()
    tot = st.pass_series['total'].window()
    n = min(len(nb), len(tot))
    if not n:
        return None
    ms = sum(tot[len(tot) - n:])
    return round(sum(nb[len(nb) - n:]) / (ms / 1e3), 1) if ms > 0 else None


def guard_margin(st, card):
    """MiB the card can still take before tools/guard.sh's line, keeping its reserve (the XTX keeps 0.5 GB)"""
    used = card.get('vram_used_mib')
    i = card['index']
    gm, rs = st.physics.guard_mib, st.physics.reserve_mib
    if used is None or i >= len(gm):
        return None
    return round(gm[i] - (rs[i] if i < len(rs) else 0) - used, 1)


def tokens_per_joule(st, energy):
    dec = sum(e.get('decode', 0.0) for e in energy)
    pre = sum(e.get('prefill', 0.0) for e in energy)
    return {'decode': round(st.predicted_tokens / dec, 4) if dec > 0 and st.predicted_tokens else None,
            'prefill': round(st.prompt_tokens / pre, 4) if pre > 0 and st.prompt_tokens else None,
            'decode_joules': round(dec, 3), 'prefill_joules': round(pre, 3)}


def _prefill_joules_per_token(st, energy):
    pre = sum(e.get('prefill', 0.0) for e in energy)
    return pre / st.prompt_tokens if pre > 0 and st.prompt_tokens else None


def context_view(st):
    pos = st.position
    mx = st.max_ctx
    ph = st.physics
    return {'position': pos, 'max_ctx': mx, 'remaining': max(0, mx - pos) if mx else None,
            'attention': 'qsa' if pos >= ph.qsa_from else 'dense', 'qsa_from': ph.qsa_from,
            'kv_bytes': [int(round(pos * b)) for b in ph.kv_bytes_per_token]}


# ------------------------------------------------------------------------------------------------ /slots
def slot_json(st, n_ctx, speculative):
    """llama-server's server_slot::to_json(only_metrics=true) for slot 0: no prompt or generated text"""
    s = st.slot
    out = {'id': 0, 'n_ctx': n_ctx, 'speculative': bool(speculative), 'is_processing': bool(s.is_processing)}
    if s.started:
        out['id_task'] = s.id_task
        out['n_prompt_tokens'] = s.n_prompt
        out['n_prompt_tokens_processed'] = s.n_processed
        out['n_prompt_tokens_cache'] = s.n_cache
        out['params'] = s.params if s.params is not None else {}
        out['next_token'] = [{'has_next_token': bool(s.has_next_token), 'has_new_line': bool(s.has_new_line),
                              'n_remain': s.n_remain, 'n_decoded': s.n_decoded}]
    return out


# llama-server's task_params::to_json(only_metrics=true) keys and their neutral values where Caminus has no such
# sampler (dynatemp, xtc, dry, mirostat, adaptive, top_n_sigma, typical: off)
SAMPLERS = ['penalties', 'top_k', 'top_p', 'min_p', 'temperature']
LLAMA_DEFAULT_SEED = 4294967295


def task_params(p, n_predict=-1, stream=False, spec_types='none'):
    """p: sampler.SamplingParams (or None: the defaults are given by the caller)"""
    seed = p.seed if p is not None and p.seed is not None else LLAMA_DEFAULT_SEED
    return {
        'seed': int(seed), 'temperature': float(p.temperature), 'dynatemp_range': 0.0, 'dynatemp_exponent': 1.0,
        'top_k': int(p.top_k), 'top_p': float(p.top_p), 'min_p': float(p.min_p), 'top_n_sigma': -1.0,
        'xtc_probability': 0.0, 'xtc_threshold': 0.1, 'typical_p': 1.0, 'repeat_last_n': int(p.penalty_last_n),
        'repeat_penalty': float(p.repeat_penalty), 'presence_penalty': float(p.presence_penalty),
        'frequency_penalty': float(p.frequency_penalty), 'dry_multiplier': 0.0, 'dry_base': 1.75,
        'dry_allowed_length': 2, 'dry_penalty_last_n': -1, 'mirostat': 0, 'mirostat_tau': 5.0, 'mirostat_eta': 0.1,
        'adaptive_target': -1.0, 'adaptive_decay': 0.9, 'max_tokens': int(n_predict), 'n_predict': int(n_predict),
        'n_keep': 0, 'n_discard': 0, 'ignore_eos': False, 'stream': bool(stream), 'n_probs': 0,
        'min_keep': int(p.min_keep), 'chat_format': 'Qwen3 Coder', 'reasoning_format': 'deepseek',
        'reasoning_in_content': False, 'generation_prompt': '', 'samplers': list(SAMPLERS),
        'speculative.types': spec_types, 'timings_per_token': False, 'post_sampling_probs': False,
        'backend_sampling': False, 'lora': []}


# ------------------------------------------------------------------------------------------------ JSON status
def status(st, srv, cards, energy, hist=True):
    """GET /caminus/status (version STATUS_VERSION): every group of docs/agentic/spec-metrics.md as plain JSON, the
    per-card data as a list, windows as p50 / p95 / mean / max and (hist) cumulative bucket counts"""
    ph = st.physics_view()
    sv = st.spec_view()
    g = st.cache_gauges
    tpj = tokens_per_joule(st, energy)
    jpt = _prefill_joules_per_token(st, energy)
    with st.lock:
        agent = {'harness': dict(sorted(st.harness.items())), 'tool_calls': st.tool_calls,
                 'tool_call_failures': dict(st.tool_failures), 'tool_call_lenient': st.tool_lenient,
                 'loop': {'requests_flagged': st.loop_flagged, 'repeated_lines': st.loop_repeated,
                          'live_max_repeat': st.loop_live, 'repeat_threshold': _loop_threshold()}}
    agent.update({'tokens': {'reasoning': st.tokens_reasoning, 'answer': st.tokens_answer},
                  'finish': dict(st.finish), 'max_tokens_hits': st.max_tokens_hits})
    cards_out = []
    for c in cards:
        i = c['index']
        cd = dict(c)
        cd['guard_mib'] = st.physics.guard_mib[i] if i < len(st.physics.guard_mib) else None
        cd['margin_mib'] = guard_margin(st, c)
        cd['energy_j'] = {k: round(v, 3) for k, v in energy[i].items()} if i < len(energy) else None
        cd['kv_bytes'] = context_view(st)['kv_bytes'][i] if i < st.n_cards else None
        if i < st.n_cards:
            cd['pass_ms'] = st.card_pass[i].snapshot()
            cd['exchange_wait_ms'] = st.card_wait[i].snapshot()
            cd['exchange_wait_note'] = _wait_note(st, i)
        cards_out.append(cd)
    if not cards:       # no hardware reader: the engine's per-card timings still have a place
        for i in range(st.n_cards):
            cards_out.append({'index': i, 'name': st.physics.card_names[i] if i < len(st.physics.card_names) else
                              f'card{i}', 'pass_ms': st.card_pass[i].snapshot(),
                              'exchange_wait_ms': st.card_wait[i].snapshot(), 'exchange_wait_note': _wait_note(st, i),
                              'kv_bytes': context_view(st)['kv_bytes'][i]})
    return {
        'version': S.STATUS_VERSION, 'engine': S.ENGINE, 'build_info': srv['build_info'], 'time': round(time.time(), 3),
        'uptime_s': round(st.clock() - st.t0, 3), 'stats_enabled': st.enabled,
        'server': {'model_alias': srv['model_alias'], 'model_path': srv['model_path'], 'n_ctx': srv['n_ctx'],
                   'requests_processing': srv['requests_processing'], 'requests_deferred': srv['requests_deferred'],
                   'queue_depth': srv['queue_depth'], 'loading': srv['loading'], 'speculative_types': srv['spec_types'],
                   'requests': st.requests},
        'llamacpp': {'prompt_tokens_total': st.prompt_tokens, 'prompt_tokens_cached_total': st.prompt_cached,
                     'prompt_seconds_total': round(st.prompt_us / 1e6, 6), 'tokens_predicted_total': st.predicted_tokens,
                     'tokens_predicted_seconds_total': round(st.predicted_us / 1e6, 6), 'n_decode_total': st.n_decode,
                     'n_tokens_max': st.n_tokens_max},
        'physics': dict(ph, cards=[{'bytes_per_token': b, 'bandwidth_bytes_s': w}
                                   for b, w in zip(st.physics.card_bytes, st.physics.card_bw)]),
        'decode': {'split_ms': {p: st.pass_series[p].snapshot(hist) for p in S.PASS_PARTS},
                   'tokens_per_pass': st.emitted.snapshot(hist), 'ms_per_token': ph['ms_per_token'],
                   'passes': st.pass_series['total'].count},
        'ttft': {'split_ms': {p: st.ttft[p].snapshot(hist) for p in S.TTFT_PARTS}},
        'prefill': {'tokens_per_s': st.prefill_rate.snapshot(hist), 'rate_ema': _r(st.prefill_rate_ema)},
        'dflash': sv,
        'cache': {'tokens': dict(st.cache_tokens), 'sources': dict(st.cache_sources),
                  'restore_ms': st.cache_restore.snapshot(hist), 'lcp': st.cache_lcp.snapshot(hist),
                  'vram': {'slots_used': g['slots_used'], 'slots': g['n_slots'], 'tier_mib': g['tier_mib'],
                           'headroom_mib': g['headroom_mib']},
                  'disk': {'bytes': g['disk_bytes'], 'cap_bytes': g['disk_cap_bytes'],
                           'gb': None if g['disk_bytes'] is None else round(g['disk_bytes'] / 1e9, 3),
                           'cap_gb': None if g['disk_cap_bytes'] is None else round(g['disk_cap_bytes'] / 1e9, 3)},
                  'events': dict(st.cache_counters),
                  'route': {'main': dict(st.cache_route['main']), 'side': dict(st.cache_route['side']),
                            'last': None if st.cache_route_last is None else dict(st.cache_route_last),
                            'side_active': g['side_active'], 'side_rows': g['side_rows']},
                  'saved': {'prefill_s': round(st.saved_s, 4), 'tokens': st.saved_tokens,
                            'joules': None if jpt is None else round(st.saved_tokens * jpt, 3)}},
        'context': context_view(st),
        'reliability': {'bails': dict(st.bails), 'retries': st.retries, 'kfd_evicted_ms': st.evicted_ms,
                        'pgmigrate': st.pgmigrate, 'pgmigrate_all': st.pgmigrate_all, 'host_samples': st.host_samples,
                        'refusals': dict(st.refusals),
                        'errors': {'engine': st.engine_errors, 'internal': st.internal_errors},
                        'xtx_margin_mib': next((guard_margin(st, c) for c in cards if c['index'] == 0), None)},
        'cards': cards_out,
        'link': {'bytes_total': st.link_bytes, 'bytes_per_s': link_rate(st)},
        'energy': {'tokens_per_joule': {'decode': tpj['decode'], 'prefill': tpj['prefill']},
                   'joules': {'decode': tpj['decode_joules'], 'prefill': tpj['prefill_joules']}},
        'agent': agent,
        'slot': slot_json(st, srv['n_ctx'], srv['spec_types'] not in ('', 'none')),
        'clock_check': srv.get('clock_check'),
    }


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def _wait_note(st, i):
    """why a card's exchange wait has no values (None when it has): the engine's note (stats.exchange_wait_note)"""
    if st.card_wait[i].count:
        return None
    return getattr(st, 'exchange_wait_note', None)


def _loop_threshold():
    from .agentmon import LoopDetector
    return LoopDetector.REPEAT
