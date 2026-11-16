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
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
os.environ['CAMINUS_ALLOW_PAUSED'] = '1'   # (the paused models' entries are fixtures here: tests/cmd/test_manage.py checks the pause)

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def gguf_str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def gguf_value(v):
    if isinstance(v, str): return struct.pack('<I', 8) + gguf_str(v)
    if isinstance(v, float): return struct.pack('<If', 6, v)
    return struct.pack('<II', 4, int(v))


def write_gguf(path, arch, kv, tensors):
    """a GGUF v3 file: general.architecture = arch, kv {key: int (u32) | float (f32) | str}, tensors [(name, ne0, ne1)] in
    F32 (zeros)"""
    kvs = [('general.architecture', arch)] + list(kv.items())
    out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kvs))
    for k, v in kvs:
        out += gguf_str(k) + gguf_value(v)
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
    check(sorted(arch.ARCHS) == ['qwen35', 'qwen35-own', 'qwen35-own-tp', 'qwen35moe', 'qwen3moe', 'qwen3moe-tp', 'qwen3next', 'qwen4exp'],
          f'archs: the six served, the dense own line\'s and Qwen3-30B-A3B\'s two-card forms ({sorted(arch.ARCHS)})')
    check(arch.ARCHS['qwen35-own'].variant(2) is arch.ARCHS['qwen35-own-tp'] and arch.ARCHS['qwen35-own'].variant(1) is arch.ARCHS['qwen35-own']
          and arch.ARCHS['qwen35-own'].variant(None) is arch.ARCHS['qwen35-own']
          and arch.TWO_CARD == {'qwen35-own': 'qwen35-own-tp', 'qwen3moe': 'qwen3moe-tp'},
          'variant: --cards 2 names the dense own line\'s two-card entry; 1 / None its own')
    q3, q3tp = arch.ARCHS['qwen3moe'], arch.ARCHS['qwen3moe-tp']
    check(q3.variant(2) is q3tp and q3.variant(1) is q3 and q3.variant(None) is q3 and q3.cards == 1
          and models.BY_NAME['qwen3-30b-a3b'].cards == 1 and models.BY_NAME['qwen3-30b-a3b'].arch == 'qwen3moe',
          'variant: Qwen3-30B-A3B --cards 2 builds the two-card recipe (qwen3moe-tp); one card stays its default (no --cards, '
          'the catalog\'s card count)')
    try:
        arch.ARCHS['qwen35'].variant(2); refused = False
    except ValueError:
        refused = True
    check(refused and arch.ARCHS['qwen35-own-tp'].split([{'label': 'XTX'}, {'label': 'XT'}]) == {'ffn_share': 0.532}
          and arch.ARCHS['qwen35-own-tp'].split([{'label': 'XT'}, {'label': 'XT'}]) == {'ffn_share': 0.5},
          'variant: a model without a two-card form refuses --cards 2; the two-card split weighs the feed-forward rows by bandwidth')
    xtx, xt = dict(label='XTX', total_mib=24560, card='card1', pci='0000:03:00.0'), dict(label='XT', total_mib=20464, card='card2', pci='0000:08:00.0')
    check(q3tp.split([xtx, xt]) == {'exp_rows': 512, 'vocab_share': 0.532} and q3tp.split([xt, dict(xt, pci='x')]) == {'exp_rows': 512, 'vocab_share': 0.5}
          and q3tp.split([xtx, dict(xtx, pci='x')]) == {'exp_rows': 512, 'vocab_share': 0.5} and q3tp.split(None)['exp_rows'] == 512
          and q3tp._split({'placement': {'split': {'exp_rows': 512, 'vocab_share': 0.532}}}) == (512, 0.532),
          'qwen3moe-tp split: card 0 (the bigger card first) takes 512 of 768 expert rows on every pair; the head by bandwidth; '
          'build and max_ctx auto read the placement\'s stored split')
    why = q3tp.pair_refusal([xt, xtx])
    from caminus.device import host as _h
    try:
        q3tp.fits(dict(model='unused'), [xt, xtx]); raised = None
    except _h.PlacementError as exc:
        raised = str(exc)
    check(q3tp.pair_refusal([xtx, xt]) is None and q3tp.pair_refusal([xt, dict(xt, pci='x')]) is None and bool(why)
          and 'bigger card' in why and raised == why,
          f'qwen3moe-tp pair: XTX + XT and equal cards pass; a smaller card 0 is refused, by placement too ({why})')
    shape = {n: (x.kind, x.cards, x.T, x.drafter) for n, x in arch.ARCHS.items()}
    check(shape == {'qwen4exp': ('coder', 2, 4, 'dflash'), 'qwen35moe': ('one-card', 1, 1, None), 'qwen35': ('one-card', 1, 1, None),
                    'qwen3moe': ('one-card', 1, 1, None), 'qwen35-own': ('one-card', 1, 1, 'mtp'), 'qwen3next': ('tp', 2, 1, None),
                    'qwen35-own-tp': ('tp', 2, 1, 'mtp'), 'qwen3moe-tp': ('tp', 2, 1, None)},
          f'archs: kind, cards, T, drafter ({shape})')
    check(arch.ARCHS['qwen35-own'].tokenizer == 'own' and arch.ARCHS['qwen35'].tokenizer == 'hf',
          'archs: the own-code qwen35 entry takes the own tokenizer, AREX-2\'s entry keeps its own')
    check(arch.lookup('llama') is arch.CODER, 'archs: any other architecture takes the Coder\'s path')
    check(arch.usable_ctx(262144, 4) == 262142 and arch.usable_ctx(90112, 1) == 90112, 'usable_ctx: the engine less T - 2')

    # ---- the catalog agrees with it (cards come from it; the drafter file is the one the architecture loads)
    for m in models.CATALOG:
        check(m.arch in arch.ARCHS and m.cards == arch.ARCHS[m.arch].cards, f'catalog: {m.name} is a served architecture, cards {m.cards}')
    check(models.drafter_item(models.BY_NAME['coder']) == 'dflash', 'catalog: the Coder\'s drafter file is dflash')
    q27 = ('swift-1.5', 'thinkingcap', 'qwen3.8-27b')   # (the dense 27B line; qwen3.8-27b only while paused models are offered)
    check(all(models.drafter_item(m) == 'dflash2-27b' for m in models.ALL if m.name in q27)
          and all(models.drafter_item(m) is None for m in models.CATALOG if m.name not in ('coder',) + q27),
          'catalog: the 27B models share the base model\'s DFlash2 file (--draft dflash); no other model loads a drafter')

    # ---- meta: the header read once per file; tp_shape from it (arch.meta reads it with caminus.ggufio, gguf-py only for
    # a file that reader refuses: opens through either count)
    config.ensure_gguf()
    import gguf
    from caminus import ggufio
    opens = []
    real, real_own = gguf.GGUFReader, ggufio.GGUFFile

    class Counting(real):
        def __init__(self, path, *x, **k):
            opens.append(path); super().__init__(path, *x, **k)

    class CountingOwn(real_own):
        def __init__(self, path, *x, **k):
            opens.append(path); super().__init__(path, *x, **k)
    gguf.GGUFReader = Counting; ggufio.GGUFFile = CountingOwn
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
            check(set(qwen3next_tp.PROJECTION_KEYS) <= set(arch.HP_KEYS), 'tp_shape: the registry keeps every header key the engine\'s projection reads')
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
        # Swift 1.5 and AREX-2 are both qwen35 files: the catalog's file takes its model's entry (the own-code engine), found
        # by name and size without opening it; AREX-2's file and any other qwen35 file keep the header's entry
        sw = models.BY_KEY['swift-1.5']
        scat = os.path.join(S, 'catalog', os.path.basename(sw.path))
        with open(scat, 'wb') as f:
            f.truncate(sw.size)                       # (sparse)
        n0 = len(opens)
        check(arch.of(scat) is arch.ARCHS['qwen35-own'] and arch.default_alias(scat) == 'swift-1.5' and len(opens) == n0,
              'of: Swift 1.5\'s catalog file takes the own-code entry and answers swift-1.5 (not opened)')
        tc = models.BY_KEY['thinkingcap']
        tcat = os.path.join(S, 'catalog', os.path.basename(tc.path))
        with open(tcat, 'wb') as f:
            f.truncate(tc.size)                       # (sparse)
        n0 = len(opens)
        check(arch.of(tcat) is arch.ARCHS['qwen35-own'] and arch.default_alias(tcat) == 'thinkingcap' and len(opens) == n0,
              'of: ThinkingCap\'s catalog file takes the own-code entry and answers thinkingcap (not opened)')
        check(arch.of(cat) is arch.ARCHS['qwen35'], 'of: AREX-2\'s catalog file keeps the qwen35 entry (OneCard)')
        check(arch.of(other) is arch.ARCHS['qwen35'], 'of: a qwen35 file outside the catalog keeps the header\'s entry')
        llama = os.path.join(S, 'tiny-llama.gguf')
        write_gguf(llama, 'llama', {}, [('blk.0.ffn_up.weight', 32, 32)])
        check(arch.of(llama) is arch.CODER, 'of: an unknown architecture is given the Coder\'s entry')
    finally:
        gguf.GGUFReader = real; ggufio.GGUFFile = real_own

    # ---- RoPE scaling keys (gguf-py's Keys.Rope.SCALING_*): meta keeps them; every entry refuses a file that asks for
    # scaling (no engine applies it yet: never served unscaled); a file without them, or with type 'none', is served
    check(set(arch.ROPE_KEYS) <= set(arch.HP_KEYS) and all(
        k.format(arch='a') == 'a.' + r for k, r in ((gguf.Keys.Rope.SCALING_TYPE, 'rope.scaling.type'),
                                                    (gguf.Keys.Rope.SCALING_FACTOR, 'rope.scaling.factor'),
                                                    (gguf.Keys.Rope.SCALING_ORIG_CTX_LEN, 'rope.scaling.original_context_length'),
                                                    (gguf.Keys.Rope.SCALING_ATTN_FACTOR, 'rope.scaling.attn_factor'),
                                                    (gguf.Keys.Rope.SCALING_YARN_BETA_FAST, 'rope.scaling.yarn_beta_fast')))
          and all(v.format(arch='a')[2:] in arch.ROPE_KEYS for n, v in vars(gguf.Keys.Rope).items() if n.startswith('SCALING_')),
          'rope keys: meta keeps every rope.scaling.* key the vendored gguf-py names')
    yarn = {'rope.scaling.type': 'yarn', 'rope.scaling.factor': 4.0, 'rope.scaling.original_context_length': 32768}
    by_header = {'qwen4exp': 'qwen4exp', 'qwen35moe': 'qwen35moe', 'qwen35': 'qwen35', 'qwen3moe': 'qwen3moe',
                 'qwen35-own': 'qwen35', 'qwen3next': 'qwen3next', 'qwen35-own-tp': 'qwen35', 'qwen3moe-tp': 'qwen3moe'}
    check(sorted(by_header) == sorted(arch.ARCHS), 'rope keys: every entry has a header architecture below')

    def rope_file(a, kv, tag):
        p = os.path.join(S, f'rope-{tag}-{a}.gguf')
        write_gguf(p, a, dict({f'{a}.context_length': 4096}, **{f'{a}.{k}': v for k, v in kv.items()}),
                   [('blk.0.ffn_up.weight', 32, 32)])
        return p
    for name, ent in arch.ARCHS.items():
        h = by_header[name]
        why = ent.rope_refusal(rope_file(h, yarn, 'yarn'))
        plain = ent.rope_refusal(rope_file(h, {}, 'none'))
        off = ent.rope_refusal(rope_file(h, {'rope.scaling.type': 'none', 'rope.scaling.factor': 4.0}, 'off'))
        check(bool(why) and f'{h}.rope.scaling.type = yarn' in why and f'{h}.rope.scaling.factor = 4.0' in why
              and f'{h}.rope.scaling.original_context_length = 32768' in why and 'unscaled' in why and ent.name in why
              and plain is None and off is None,
              f'rope keys: {name} refuses a {h} file asking for YaRN x4 ({why}); serves one without the keys or with type none')
    unk = rope_file('llama', yarn, 'yarn')
    check(arch.of(unk) is arch.CODER and bool(arch.of(unk).rope_refusal(unk)),
          'rope keys: an unknown architecture (the Coder\'s entry) refuses the keys too')
    bare = rope_file('qwen3moe', {'rope.scaling.factor': 4.0}, 'bare')
    legacy = rope_file('qwen3moe', {'rope.scale_linear': 2.0}, 'legacy')
    q3m = arch.ARCHS['qwen3moe']
    check(bool(q3m.rope_refusal(bare)) and 'qwen3moe.rope.scaling.factor = 4.0' in q3m.rope_refusal(bare)
          and bool(q3m.rope_refusal(legacy)) and 'qwen3moe.rope.scale_linear = 2.0' in q3m.rope_refusal(legacy),
          'rope keys: a bare factor (llama.cpp\'s default type is linear) and the older rope.scale_linear are refused')
    unit = [rope_file('qwen3moe', d, f'unit{i}') for i, d in enumerate((
        {'rope.scaling.factor': 1.0}, {'rope.scale_linear': 1.0}, {'rope.scaling.type': 'yarn'},
        {'rope.scaling.type': 'linear', 'rope.scaling.factor': 1.0}, {'rope.scaling.original_context_length': 32768},
        {'rope.scaling.factor': 0.0, 'rope.scaling.yarn_beta_fast': 32.0}))]
    attn = rope_file('qwen3moe', {'rope.scaling.type': 'yarn', 'rope.scaling.factor': 1.0,
                                  'rope.scaling.attn_factor': 0.9}, 'unit-attn')
    check(all(q3m.rope_refusal(u) is None for u in unit) and bool(q3m.rope_refusal(attn)),
          'rope keys: an effective factor of 1 (factor or scale_linear 1 or 0, a type or yarn key alone) asks for no '
          'scaling and is served; factor 1 with an attention factor other than 1 is refused')
    junk = os.path.join(S, 'rope-junk.gguf')
    with open(junk, 'wb') as f:
        f.write(b'fixture')
    check(q3m.rope_refusal(junk) is None, 'rope keys: a header no reader reads is left to the engine\'s reader')
    # both meta() readers keep the same keys: caminus.ggufio, and gguf-py for a file ggufio refuses
    yq = rope_file('qwen3moe', dict(yarn, **{'rope.scaling.yarn_beta_fast': 32.0}), 'beta')
    own_hp = arch.meta(yq)['hp']

    class Refusing(real_own):
        def __init__(self, *x, **k): raise ggufio.GGUFError('refused here')
    arch._META.clear(); ggufio.GGUFFile = Refusing
    try:
        py_hp = arch.meta(yq)['hp']
    finally:
        ggufio.GGUFFile = real_own; arch._META.clear()
    check(own_hp == py_hp and py_hp.get('rope.scaling.type') == 'yarn' and py_hp.get('rope.scaling.factor') == 4.0
          and py_hp.get('rope.scaling.yarn_beta_fast') == 32.0 and py_hp.get('context_length') == 4096,
          f'rope keys: meta through caminus.ggufio == through gguf-py ({py_hp})')
    # caminus serve's recipe (and caminus fit, which shares it) refuses with the message; the same file without the keys passes
    from caminus.serve import cli as _cli0
    try:
        _cli0.recipe(_cli0.parse(['--model', rope_file('qwen3moe', yarn, 'yarn'), '--draft', 'off']), log=lambda *a, **k: None)
        msg = None
    except ValueError as exc:
        msg = str(exc)
    ok = _cli0.recipe(_cli0.parse(['--model', rope_file('qwen3moe', {}, 'none'), '--draft', 'off']), log=lambda *a, **k: None)
    check(bool(msg) and 'RoPE scaling' in msg and 'qwen3moe.rope.scaling.type = yarn' in msg and ok is q3m,
          f'rope keys: caminus serve refuses the YaRN file before anything loads ({msg}); the plain file is served')
    import contextlib, io, json
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = _cli0.fit_report(['--model', rope_file('qwen3moe', yarn, 'yarn'), '--draft', 'off', '--json'])
    rep = json.loads(out.getvalue())
    check(rc == 1 and rep['ok'] is False and rep['max_ctx'] == {'f16': None, 'q8': None} and 'RoPE scaling' in rep['error'],
          f'rope keys: caminus fit reports the refusal with no maximum ({rep})')

    # ---- the model cards' context facts in the registry (data only: nothing serves from them yet)
    F = {n: ent.rope_facts for n, ent in arch.ARCHS.items()}
    check(F['qwen3moe'] == arch.RopeFacts(32768, 40960, 'yarn', (2.0, 4.0), None, None, 'https://huggingface.co/Qwen/Qwen3-30B-A3B')
          and F['qwen3moe'] is F['qwen3moe-tp'] is arch.QWEN3MOE_ROPE
          and F['qwen35-own'] is F['qwen35-own-tp'] is arch.QWEN35_ROPE
          and (F['qwen35-own'].native_ctx, F['qwen35-own'].card_factors, F['qwen35-own'].shipped) == (262144, (2.0, 4.0), None)
          and (F['qwen4exp'].native_ctx, F['qwen4exp'].card_kind, F['qwen4exp'].shipped, F['qwen4exp'].shipped_factor)
          == (262144, 'yarn', 'ntk', arch.LONG_NTK)
          and all(F[n] is None for n in ('qwen35moe', 'qwen35', 'qwen3next')),
          'rope facts: Qwen3-30B-A3B native 32,768 (YaRN x2 / x4 on its card, not applied), the dense 27B line 262,144, the Coder '
          'ships NTK x4 where its card names YaRN x4; the paused entries have none')

    # ---- caminus serve's options: --model-alias '' = the model's own id
    from caminus.serve import cli
    o = cli.parse([])
    check(o['model_alias'] == '' and any(x['dest'] == 'model_alias' and x['default'] == '' for x in cli.options_json()),
          'cli: --model-alias defaults to the model\'s own id (\'\')')
    check(not any(hasattr(cli, n) for n in ('ONE_CARD_ARCHS', 'TP_ARCHS', 'model_arch', 'build_model', 'auto_max_ctx')),
          'cli: no architecture table or builder of its own')
    # --kv: f16 by default; q8 only where an entry builds it (the qwen3moe pair); the entries without the choice say what
    # their engine always holds: the GGUF engines' own 8-bit q8_0 cache (each engine's q8 default is True and the build
    # passes none), the dense own-code line F16. /props reports the built engine's (arch.kv_cache_of)
    import ast
    def q8_default(rel, cls):
        tree = ast.parse(open(os.path.join(ROOT, 'python', 'caminus', 'engine', rel)).read())
        fn = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls for n in c.body
                  if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        args = fn.args.args[-len(fn.args.defaults):]
        return next(ast.literal_eval(d) for a_, d in zip(args, fn.args.defaults) if a_.arg == 'q8')
    q8s = {n: (a.kv_formats, a.kv_fixed) for n, a in arch.ARCHS.items()}
    check(cli.parse([])['kv'] == 'f16' and cli.parse(['--kv', 'q8'])['kv'] == 'q8'
          and q8s == {'qwen4exp': (('f16',), 'q8_0'), 'qwen35moe': (('f16',), 'q8_0'), 'qwen35': (('f16',), 'q8_0'),
                      'qwen3moe': (('f16', 'q8'), None), 'qwen35-own': (('f16',), None), 'qwen3next': (('f16',), 'q8_0'),
                      'qwen35-own-tp': (('f16',), None), 'qwen3moe-tp': (('f16', 'q8'), None)}
          and q8_default('flashnext_tp.py', 'FlashNextTP') is True and q8_default('qwen35_one.py', 'Qwen35One') is True
          and q8_default('qwen3next_tp.py', 'Qwen3NextTP') is True
          and q8_default('q3moe_one.py', 'Q3MoEOne') is False and q8_default('q3moe_tp.py', 'Q3MoETP') is False
          and arch.kv_cache_of(arch.CODER, object()) == {'type': 'q8', 'format': 'q8_0'}
          and arch.kv_cache_of(arch.ARCHS['qwen3moe'], object()) == {'type': 'f16', 'format': 'f16'}
          and arch.kv_cache_of(arch.ARCHS['qwen3moe'], types.SimpleNamespace(kv_format_id='q8_0-v1:x')) == {'type': 'q8', 'format': 'q8_0-v1:x'}
          and arch.kv_q8_models() == 'qwen3-30b-a3b (one card or two)',
          f'cli: --kv f16 (default) / q8; kv_formats / kv_fixed per entry match the engines\' own q8 defaults ({q8s})')

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
        check(names == {'qwen4exp': 'FlashNextTP', 'qwen35moe': 'Qwen35One', 'qwen35': 'Qwen35One', 'qwen3moe': 'Q3MoEOne', 'qwen35-own': 'Dense35Own',
                        'qwen3next': 'Qwen3NextTP', 'qwen35-own-tp': 'Dense35TP', 'qwen3moe-tp': 'Q3MoETP'},
              f'engine_class: every entry\'s engine imports ({names})')
        # the pass methods K3Session calls on every engine (the device sampler's launch / finish pair among them: Swift's
        # engine served greedy requests without them and failed its first sampled one on the card)
        need = ('step', 'verify', 'verify_tokens', 'step_token', 'launch_tokens', 'finish_pass', 'reset', 'prefetch')
        lack = {n: [a for a in need if not callable(getattr(x.engine_class(), a, None))] for n, x in arch.ARCHS.items()}
        check(not any(lack.values()), f'engine_class: every engine has the session\'s pass methods (missing: {lack})')
        # the dense own line's projection with --draft mtp runs end to end on a synthetic nextn file (regression: the
        # adapter ring's D must come from shapes_of, not from a header key meta() does not keep - KeyError at startup)
        q35 = os.path.join(S, 'tiny-q35own.gguf')
        hp35 = {'qwen35.context_length': 4096, 'qwen35.block_count': 2, 'qwen35.nextn_predict_layers': 1,
                'qwen35.embedding_length': 64, 'qwen35.feed_forward_length': 128, 'qwen35.attention.head_count': 4,
                'qwen35.attention.head_count_kv': 2, 'qwen35.attention.key_length': 16, 'qwen35.rope.dimension_count': 8,
                'qwen35.ssm.state_size': 16, 'qwen35.ssm.group_count': 1, 'qwen35.ssm.time_step_rank': 8,
                'qwen35.ssm.conv_kernel': 4, 'qwen35.ssm.inner_size': 32}
        write_gguf(q35, 'qwen35', hp35, [('token_embd.weight', 64, 997), ('output.weight', 64, 997),
                                         ('blk.1.nextn.eh_proj.weight', 8, 8)])
        own = arch.ARCHS['qwen35-own']
        from caminus.decode.mtp35 import Mtp35Drafter
        nm, _, _, need_m = own.auto_need(dict(model=q35, draft='mtp'), 4)
        _, _, _, need_o = own.auto_need(dict(model=q35, draft='off'), 4)
        ring = Mtp35Drafter.ring_mib(64)
        check(nm == 4096 and need_m(4096)[0] > need_o(4096)[0] and ring > 0,
              f'qwen35-own auto_need --draft mtp on a synthetic nextn file: {need_m(4096)[0]:.1f} MiB vs off '
              f'{need_o(4096)[0]:.1f} (the adapter ring {ring:.4f} MiB included), no header-key error')
        # the default drafter (no --draft): the table's kind when the file has the MTP block, plain without it, on one card
        # and two (the table here: off on two cards); the Coder keeps DFlash
        # (the table is keyed by (decode module, cards); a module it does not list serves plain: the plain 27B's k3_q35_gsq)
        from caminus import ggufio as _gio
        from caminus.engine import q35_dense as _qd
        _F = _gio.GGUFFile(q35); mod = _qd.module_of(_F); _F.close()
        keep_d, keep_mo = arch.Dense35.DRAFT_DEFAULT, _qd.module_of
        try:
            arch.Dense35.DRAFT_DEFAULT = {(mod, 1): 'mtp', (mod, 2): 'off'}
            q35n = os.path.join(S, 'tiny-q35own-nomtp.gguf')
            write_gguf(q35n, 'qwen35', dict(hp35, **{'qwen35.nextn_predict_layers': 0}),
                       [('token_embd.weight', 64, 997), ('output.weight', 64, 997)])
            got = (own.default_draft(dict(model=q35)), own.default_draft(dict(model=q35n)),
                   own.variant(2).default_draft(dict(model=q35)), arch.ARCHS['qwen4exp'].default_draft({}))
            _qd.module_of = lambda F: 'k3_q35_gsq'      # (a file the GSQ module decodes: not in the table)
            got_gsq = own.default_draft(dict(model=q35))
        finally:
            arch.Dense35.DRAFT_DEFAULT, _qd.module_of = keep_d, keep_mo
        check(got == ('mtp', 'off', 'off', 'dflash') and got_gsq == 'off',
              f'default_draft: mtp file / no block / two cards / the Coder: {got}; a k3_q35_gsq file (no drafter yet): {got_gsq}')
        # the default drafter's file is optional: a default --draft dflash without it serves plain; an explicit one refuses
        from caminus.serve import cli as _cli
        keep_d, keep_of = arch.Dense35.DRAFT_DEFAULT, _cli._arch.of
        try:
            arch.Dense35.DRAFT_DEFAULT = {(mod, 1): 'dflash', (mod, 2): 'off'}
            _cli._arch.of = lambda path: own
            gone = os.path.join(S, 'no-such-drafter.gguf')
            o = dict(model=q35, cards=None, draft=None, no_drafter=False, drafter=gone)
            keep_dd, _cli.DEFAULT_DRAFTER = _cli.DEFAULT_DRAFTER, gone
            keep_md = _cli._config.model_dir
            _cli._config.model_dir = lambda: os.path.join(S, 'no-models')   # (an empty model directory: no shared drafter)
            try:
                _cli.recipe(o, log=lambda *a, **k: None)
                got_default = o['draft']
                o2 = dict(model=q35, cards=None, draft='dflash', no_drafter=False, drafter=gone)
                try:
                    _cli.recipe(o2, log=lambda *a, **k: None); refused = False
                except ValueError:
                    refused = True
            finally:
                _cli.DEFAULT_DRAFTER, _cli._config.model_dir = keep_dd, keep_md
        finally:
            arch.Dense35.DRAFT_DEFAULT, _cli._arch.of = keep_d, keep_of
        # a default drafter stays only where MIN_CTX (or the asked max_ctx) still fits with it
        from caminus.device import host as _host
        def _never(fits): raise _host.PlacementError('no cards fit')
        o3 = dict(model=q35, draft='dflash', draft_defaulted=True, max_ctx='auto')
        _cli.keep_default_draft(own, o3, _never, log=lambda *a, **k: None)
        o4 = dict(model=q35, draft='dflash', draft_defaulted=True, max_ctx='auto'); seen = []
        _cli.keep_default_draft(own, o4, lambda fits: seen.append(fits), log=lambda *a, **k: None)
        o5 = dict(model=q35, draft='dflash', draft_defaulted=False, max_ctx='auto')
        _cli.keep_default_draft(own, o5, _never, log=lambda *a, **k: None)
        check(own.min_ctx(o4) == arch.MIN_CTX and own.min_ctx(o5) == own.DFLASH_MIN_CTX and 'draft_floor' not in o3,
              f'serve: a kept default drafter carries the {arch.MIN_CTX} floor into placement ({own.min_ctx(o4)}); an explicit '
              f'--draft dflash keeps the drafter floor ({own.min_ctx(o5)})')
        check(o3['draft'] == 'off' and o4['draft'] == 'dflash' and len(seen) == 1 and o5['draft'] == 'dflash',
              f'serve: the default drafter is dropped when {arch.MIN_CTX} positions do not fit with it ({o3["draft"]}), kept when '
              f'they do ({o4["draft"]}); an explicit --draft dflash is kept ({o5["draft"]})')
        check(got_default == 'off' and refused,
              f'serve: a default --draft dflash without the drafter file serves plain ({got_default}); an explicit one refuses ({refused})')
        dd = arch.Dense35.DRAFT_DEFAULT
        from caminus.engine import k3 as _k3
        tpv = {one for one, tpm in _k3.Q35_TP_OF.items() if tpm in _k3.Q35_TPV_OF}
        check(all(dd[(m, 1)] == 'dflash' for m in ('k3_q35_swift', 'k3_q35_tcap', 'k3_q35_gsq'))
              and all(dd[(m, 2)] == 'dflash' for m in ('k3_q35_swift', 'k3_q35_tcap'))
              and {m for (m, c) in dd if c == 2} == tpv == {'k3_q35_swift', 'k3_q35_tcap'} and ('k3_q35_gsq', 2) not in dd,
              f'default_draft: DFlash2 for the three dense 27B files on one card and for the two with a two-card verify '
              f'object on two cards ({dd}; two-card verify for {sorted(tpv)})')
        tpa = arch.ARCHS['qwen35-own-tp']
        check(tpa.two_card_models == ('swift-1.5', 'thinkingcap') and all(models.BY_NAME[n].arch == 'qwen35-own' for n in tpa.two_card_models)
              and 'qwen3.8-27b' not in tpa.two_card_models and tpa.draft_refusal('mtp') is None and tpa.draft_refusal('dflash') is None,
              f'two cards: the picker\'s catalog models with a two-card form ({tpa.two_card_models}, not the plain 27B); no refusal without a file')
        # Qwen3-30B-A3B on two cards: the registry's projection is the engine's own check (Q3MoETP.projection_mib at the
        # placed split) + the two-card prefill's construction check + the cache's checkpoint slots (a T = 1 tail row of each
        # card's kv heads), capped at the file's context_length; no drafter
        sys.path.insert(0, os.path.join(ROOT, 'tests', 'engine'))
        import tiny_q3m
        from caminus.engine import q3moe_tp as _q3tp
        from caminus.prefill import q3m_tp as _q3pf
        q3f = os.path.join(S, 'tiny-q3m-tp-40k.gguf'); tiny_q3m.write(q3f, tp=True, n_ctx=40960)
        q3tp = arch.ARCHS['qwen3moe-tp']
        pair = [dict(label='XTX', total_mib=24560), dict(label='XT', total_mib=20464)]
        n_ctx, devs, slot, need = q3tp.auto_need(dict(model=q3f), 4, cards=pair)
        pj = _q3tp.Q3MoETP.projection_mib(q3f, 32768, exp_rows=512, vocab_share=0.532)
        pf = _q3pf.need_tp_mib(_q3pf.tp_shapes(q3f, 512), q3tp.PREFILL_CHUNK)
        sl = [v * 1.05 + 1.0 for v in _q3tp.Q3MoETP.kv_pos_mib(q3f)]
        want = [a + b + 6 * c for a, b, c in zip(pj, pf, sl)]
        check(q3tp.EXP_ROWS == _q3tp.EXP_ROWS and n_ctx == 40960 and devs == [0, 1] and slot == max(sl)
              and all(abs(x - y) < 1e-9 for x, y in zip(need(32768), want)) and need(40960)[0] > need(32768)[0]
              and q3tp.two_card_models == ('qwen3-30b-a3b',) and q3tp.default_draft({}) == 'off' and q3tp.drafter is None
              and q3tp.tokenizer == 'own' and q3tp.min_ctx({}) == arch.MIN_CTX,
              f'qwen3moe-tp auto_need: the engine projection + the prefill check + 6 checkpoint slots per card '
              f'({" / ".join(f"{v:.1f}" for v in need(32768))} MiB at 32768), capped at the file\'s 40,960; no drafter')
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
    D35 = arch.ARCHS['qwen35-own']
    check(D35.min_ctx(dict(draft='dflash')) == D35.DFLASH_MIN_CTX < arch.MIN_CTX and D35.min_ctx(dict(draft='mtp')) == arch.MIN_CTX
          and D35.min_ctx({}) == arch.MIN_CTX and arch.ARCHS['qwen3moe'].min_ctx(dict(draft='dflash')) == arch.MIN_CTX
          and 'dflash' in D35.drafters,
          'the dense own line places a --draft dflash serve from its DFlash floor (the drafter\'s VRAM), every other serve from MIN_CTX')
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    t0 = time.time()
    rc = main()
    print(f'({time.time() - t0:.1f} s)')
    sys.exit(rc)
