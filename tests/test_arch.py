"""CPU tests of the architecture registry (caminus.arch) and the places that read it: the catalog (caminus.models), caminus
serve's options (caminus.serve.cli) and caminus start's server flags (caminus.cmd.manage). Nothing touches a GPU; the GGUFs
are tiny files written here (a header and two F32 tensors).

  python -I tests/test_arch.py --scratch DIR      RESULT PASS

With torch importable it also runs max_ctx auto's search on the two-card TP entry against fake free VRAM (the one-card entry
needs Qwen35One's shapes of a real model: tools/serve_smoke1.py covers it on the GPU)."""
import argparse
import importlib.util
import os
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def gguf_str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def write_gguf(path, arch, kv, tensors):
    """a GGUF v3 file: general.architecture = arch, kv {key: int (u32) | str}, tensors [(name, ne0, ne1)] in F32 (zeros)"""
    kvs = [('general.architecture', arch)] + list(kv.items())
    out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kvs))
    for k, v in kvs:
        out += gguf_str(k) + (struct.pack('<I', 8) + gguf_str(v) if isinstance(v, str) else struct.pack('<II', 4, int(v)))
    off = 0
    for name, ne0, ne1 in tensors:
        out += gguf_str(name) + struct.pack('<IQQIQ', 2, ne0, ne1, 0, off)
        off += (ne0 * ne1 * 4 + 31) // 32 * 32
    out += b'\0' * ((-len(out)) % 32)
    for name, ne0, ne1 in tensors:
        out += b'\0' * ((ne0 * ne1 * 4 + 31) // 32 * 32)
    with open(path, 'wb') as f:
        f.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.scratch)
    if S.startswith('/tmp'):
        raise SystemExit('--scratch must not be under /tmp')
    os.makedirs(S, exist_ok=True)
    from caminus import arch, config, models

    # ---- the registry's entries
    check(sorted(arch.ARCHS) == ['qwen35', 'qwen35moe', 'qwen3next', 'qwen4exp'], f'archs: the four served ({sorted(arch.ARCHS)})')
    shape = {n: (x.kind, x.cards, x.T, x.drafter) for n, x in arch.ARCHS.items()}
    check(shape == {'qwen4exp': ('coder', 2, 4, 'dflash'), 'qwen35moe': ('one-card', 1, 1, None), 'qwen35': ('one-card', 1, 1, None),
                    'qwen3next': ('tp', 2, 1, None)}, f'archs: kind, cards, T, drafter ({shape})')
    check(arch.lookup('llama') is arch.CODER, 'archs: any other architecture takes the Coder\'s path')
    check(arch.usable_ctx(262144, 4) == 262142 and arch.usable_ctx(90112, 1) == 90112, 'usable_ctx: the engine less T - 2')

    # ---- the catalog agrees with it (cards come from it; the drafter file is the one the architecture loads)
    for m in models.CATALOG:
        check(m.arch in arch.ARCHS and m.cards == arch.ARCHS[m.arch].cards, f'catalog: {m.name} is a served architecture, cards {m.cards}')
    check(models.drafter_item(models.BY_NAME['coder']) == 'dflash', 'catalog: the Coder\'s drafter file is dflash')
    check(all(models.drafter_item(m) is None for m in models.CATALOG if m.name != 'coder'), 'catalog: no other model loads a drafter yet')

    # ---- meta: the header read once per file; tp_shape from it
    config.ensure_gguf()
    import gguf
    opens = []
    real = gguf.GGUFReader

    class Counting(real):
        def __init__(self, path, *x, **k):
            opens.append(path); super().__init__(path, *x, **k)
    gguf.GGUFReader = Counting
    try:
        q3n = os.path.join(S, 'tiny-q3n.gguf')
        hp = {'qwen3next.context_length': 65536, 'qwen3next.block_count': 8, 'qwen3next.full_attention_interval': 4,
              'qwen3next.attention.key_length': 256, 'qwen3next.ssm.time_step_rank': 32, 'qwen3next.ssm.state_size': 128}
        write_gguf(q3n, 'qwen3next', hp, [('token_embd.weight', 8, 16), ('blk.0.attn_q.weight', 64, 32)])
        md = arch.meta(q3n)
        check(md['arch'] == 'qwen3next' and md['weights'] == 64 * 32 * 4 and md['hp']['block_count'] == 8,
              f'meta: arch, hp, weights without token_embd ({md["arch"]}, {md["weights"]})')
        A = arch.of(q3n)
        has_torch = importlib.util.find_spec('torch') is not None
        if has_torch:            # (tp_shape is the engine module's projection: caminus.engine.qwen3next_tp imports torch)
            sh = arch.tp_shape(q3n)
            kv_pos = (8 // 4) * (2 * 256 // 32 * 34)
            check(A is arch.ARCHS['qwen3next'] and sh['n_ctx'] == 65536 and sh['kv_pos'] == kv_pos
                  and sh['state'] == (8 - 2) * 32 * 128 * 128 * 4 // 2
                  and abs(sh['need_mib'](4096) - ((0.55 * 64 * 32 * 4 + kv_pos * 4096) / 2 ** 20 + 768.0)) < 1e-9,
                  f'tp_shape: the engine\'s per-card projection from the registry\'s header facts (kv_pos {sh["kv_pos"]})')
            from caminus.engine import qwen3next_tp
            check(qwen3next_tp.PROJECTION_KEYS == arch.HP_KEYS, 'tp_shape: the registry keeps every header key the engine\'s projection reads')
        check(arch.default_alias(q3n) == 'qwen3-coder-next', 'default_alias: a file outside the catalog answers to its architecture\'s id')
        check(len(opens) == 1, f'meta: one header read for of, tp_shape, default_alias ({len(opens)})')
        st = os.stat(q3n); os.utime(q3n, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        arch.of(q3n)
        check(len(opens) == 2, 'meta: a changed file (mtime) is read again')

        # the AREX-2 id: a bare caminus serve of the catalog's file answers arex-2 (as caminus start's), found by name and
        # size without opening it; the same name at another size is not the catalog's file
        it = models.BY_KEY['arex-2']
        cat = os.path.join(S, 'catalog', os.path.basename(it.path)); os.makedirs(os.path.dirname(cat), exist_ok=True)
        with open(cat, 'wb') as f:
            f.truncate(it.size)                       # (sparse)
        n0 = len(opens)
        check(arch.default_alias(cat) == 'arex-2' and len(opens) == n0, 'default_alias: AREX-2\'s catalog file answers arex-2 (not opened)')
        other = os.path.join(S, os.path.basename(it.path))
        write_gguf(other, 'qwen35', {'qwen35.context_length': 4096}, [('blk.0.ffn_up.weight', 32, 32)])
        check(models.model_of(other) is None and arch.default_alias(other) == 'qwen3.5',
              'default_alias: the same name at another size is not the catalog\'s file (the architecture\'s id)')
        llama = os.path.join(S, 'tiny-llama.gguf')
        write_gguf(llama, 'llama', {}, [('blk.0.ffn_up.weight', 32, 32)])
        check(arch.of(llama) is arch.CODER, 'of: an unknown architecture is given the Coder\'s entry')
    finally:
        gguf.GGUFReader = real

    # ---- caminus serve's options: --model-alias '' = the model's own id
    from caminus.serve import cli
    o = cli.parse([])
    check(o['model_alias'] == '' and any(x['dest'] == 'model_alias' and x['default'] == '' for x in cli.options_json()),
          'cli: --model-alias defaults to the model\'s own id (\'\')')
    check(not any(hasattr(cli, n) for n in ('ONE_CARD_ARCHS', 'TP_ARCHS', 'model_arch', 'build_model', 'auto_max_ctx')),
          'cli: no architecture table or builder of its own')

    # ---- the Coder's long-context mode (an explicit --max-ctx past its context_length) and --kv-host
    check(cli.parse([])['kv_host'] is False and cli.parse(['--kv-host'])['kv_host'] is True, 'cli: --kv-host (default off)')
    real_facts = arch.coder_facts
    arch.coder_facts = lambda path: {'geom': {'ctx_train': 262144}}
    try:
        base = dict(model='/m/c1.gguf', max_ctx=262144, no_disk_cache=False, no_drafter=False)
        o = dict(base)
        check(arch.CODER.long_context(o) is None and o == base and arch.CODER._engine_kw(o) == {},
              'long mode: max_ctx <= the context_length changes nothing (the daily mode)')
        o = dict(base, kv_host=True)
        check(arch.CODER.long_context(o) is None and arch.CODER._engine_kw(o) == {'kv_host': True},
              'long mode: --kv-host alone moves the KV only (no RoPE scaling)')
        o = dict(base, max_ctx=1048576); note = arch.CODER.long_context(o)
        check(bool(note) and o['kv_host'] is True and o['long_ntk'] == arch.LONG_NTK == 4.0 and o['no_disk_cache'] is False
              and o['cache_cap_gb'] == arch.LONG_CACHE_CAP_GB == 64.0 and '64 GB' in note
              and arch.CODER._engine_kw(o) == {'kv_host': True, 'ctx_extend': True, 'rope_ntk': 4.0},
              f'long mode: max_ctx 1048576 -> host KV, NTK s = 4, ctx_extend, the disk tier kept with a 64 GB cap ({note})')
        o = dict(base, max_ctx=1048576, cache_cap_gb=120.0); note = arch.CODER.long_context(o)
        check(o['cache_cap_gb'] == 120.0 and '120 GB' in note, f'long mode: an explicit --cache-cap-gb is kept ({note})')
        check(cli.parse([])['cache_cap_gb'] is None and cli.parse(['--cache-cap-gb', '80'])['cache_cap_gb'] == 80.0,
              'cli: --cache-cap-gb (default: the cache\'s own 30 GB)')
        o = dict(base, max_ctx=1048576, no_disk_cache=True, no_drafter=True); note = arch.CODER.long_context(o)
        check('T = 1' in note and 'no disk tier' in note, 'long mode: the note says plain T = 1 decoding without a drafter')
        check(arch.ARCHS['qwen35moe'].long_context(dict(base, max_ctx=1048576)) is None,
              'long mode: the Coder only (the other models\' engines refuse such a max_ctx)')
    finally:
        arch.coder_facts = real_facts

    # ---- caminus start's flags: the drafter by the architecture, no --model-alias
    from caminus.cmd import manage
    cod, q36 = models.BY_NAME['coder'], models.BY_NAME['qwen3.6']
    e = dict(path='/m/c1.gguf', paths={'coder-1': '/m/c1.gguf', 'coder-2': '/m/c2.gguf', 'dflash': '/m/d.gguf'})
    v = manage.serve_argv(cod, e, 'coder', 8092)
    check(v[v.index('--drafter') + 1] == '/m/d.gguf' and '--model-alias' not in v, f'start: the Coder gets its drafter ({v})')
    v = manage.serve_argv(cod, dict(e, paths=dict(e['paths'], dflash=None)), 'coder', 8092)
    check('--no-drafter' in v and '--drafter' not in v, 'start: the Coder without its drafter file runs --no-drafter')
    v = manage.serve_argv(q36, dict(path='/m/q.gguf', paths={'qwen3.6': '/m/q.gguf'}), 'hands', 8093, ctx=90112, extra=('--chunk', '1024'))
    check('--drafter' not in v and '--no-drafter' not in v and '--model-alias' not in v and v[-4:] == ['--max-ctx', '90112', '--chunk', '1024'],
          f'start: a one-card model gets no drafter flag ({v})')

    # ---- max_ctx auto on the TP entry (torch: caminus.device.capacity imports it)
    if importlib.util.find_spec('torch') is None:
        print('SKIP max_ctx auto: no torch', flush=True)
    else:
        names = {n: x.engine_class().__name__ for n, x in arch.ARCHS.items()}
        check(names == {'qwen4exp': 'FlashNextTP', 'qwen35moe': 'Qwen35One', 'qwen35': 'Qwen35One', 'qwen3next': 'Qwen3NextTP'},
              f'engine_class: every entry\'s engine imports ({names})')
        from caminus.device import capacity
        keep = capacity.caps_mib, capacity.cache_reserve_mib
        try:
            capacity.cache_reserve_mib = lambda devs: tuple(512 if d == 0 else 256 for d in devs)
            opt = dict(model=q3n, chunk=2048, auto_cache_slots=4)
            n_ctx, devs, slot, need = A.auto_need(opt, 4)
            # (free VRAM per card just above the need at 16384 and at 40960: the search must stop on those steps exactly)
            mid = [need(c)[0] + 512 + capacity.CACHE_WORK_MIB[0] + arch.AUTO_SLACK_MIB + 0.5 for c in (16384, 40960)]
            for free in [900.0] + mid + [4000.0]:
                capacity.caps_mib = lambda d, free=free: [(free + 100.0, 100.0) for _ in d]
                room = [free - r - capacity.CACHE_WORK_MIB[0] - arch.AUTO_SLACK_MIB for r in (512, 256)]
                want = max((c for c in range(arch.AUTO_STEP, n_ctx + 1, arch.AUTO_STEP) if all(x <= y for x, y in zip(need(c), room))), default=None)
                try:
                    got = A.auto_max_ctx(opt, log=lambda *x, **k: None)
                except capacity.CapacityError:
                    got = None
                check(got == want and (free not in mid or got in (16384, 40960)), f'max_ctx auto: {free:.1f} MiB free per card -> {got} (the largest fit: {want})')
        finally:
            capacity.caps_mib, capacity.cache_reserve_mib = keep
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    t0 = time.time()
    rc = main()
    print(f'({time.time() - t0:.1f} s)')
    sys.exit(rc)
