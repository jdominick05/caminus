"""The architecture registry: every GGUF architecture Caminus serves and how it is served, in one place.

caminus serve (caminus.serve.cli), caminus replay (caminus.cmd.replay) and caminus start (caminus.cmd.manage) ask this module
what a model needs; nothing else branches on a GGUF's general.architecture to build, size or name a server. ARCHS:

  qwen4exp   the Coder (Qwen3.8-Flash-Next GSQ-RCO Coder): FlashNextTP on both cards, T = 4, the DFlash drafter inside K3,
             the batched prefill (caminus.prefill.coder.Prefill). max_ctx auto: 262,144, its context_length (the measured fit
             with the drafter loaded: tools/serve_smoke.py; its own checks still refuse a build that does not fit)
  qwen35moe  Qwen3.6-35B-A3B: Qwen35One on one card (device 0 of the process: HIP_VISIBLE_DEVICES picks it), T = 1, no
             drafter yet, Prefill35
  qwen35     the dense qwen35 line (AREX-2, a Qwen3.8-27B finetune): the same one-card engine and prefill (a dense FFN per
             layer)
  qwen3next  Qwen3-Coder-Next: Qwen3NextTP on both cards, T = 1, no drafter yet, no batched prefill yet (KPassOnly: every
             prompt runs as K3 passes)

Any other architecture takes the Coder's path, as before the registry: FlashNextTP refuses a GGUF it cannot load.

An entry (Arch) answers: kind ('coder' | 'one-card' | 'tp'), cards, T (its tokens per pass when served), drafter (the drafter
it loads: 'dflash', the catalog role of its file, or None), alias (the architecture's default model id), engine (its engine
class); build(opt) -> (engine, prompt path, DFlash engine or None, drafter or None) in the serving order; auto_max_ctx(opt)
(--max-ctx auto); physics() (the stats hook's card model; None: the Coder's two-card default); set_ids(alias, tok) (the
template's model ids). Module functions: of(path) (the entry for a file), default_alias(path) (the id a server answers to:
the catalog's for a catalog file, else the architecture's), usable_ctx(max_ctx, T), meta(path) (the header facts, read once
per file).

Imports nothing heavy at module import: caminus.models and caminus.cmd.manage read ARCHS without torch. Before 11-07 this was
spread over caminus.serve.cli (11 architecture checks, a second table of model ids: AREX-2 answered qwen3.5 under a bare
caminus serve and arex-2 under caminus start), caminus.cmd.manage (the Coder's drafter by model name) and the test rigs, and
a start read the GGUF's header up to 4 times.
"""
import os

from . import config as _config

AUTO_STEP = 4096             # max_ctx auto: a multiple of this (or the model's context_length itself)
MIN_CTX = 32768              # the smallest context a model is placed on a card for (D-292 G4: below it the setup is refused)
AUTO_SLACK_MIB = 384.0       # max_ctx auto: kept on top (the projections' own error, allocator rounding: AREX-2 measured
                             # ~270 MiB over its projection with Prefill35's buffers counted exactly, 11-07)
PREFILL_MODULES_MIB = 64.0   # Prefill35's code objects, streams and its padded alpha / beta copies (a few MiB each)
CARD_BW = {'XTX': 881e9, 'XT': 774e9}           # measured streaming bandwidth (stats.PhysicsConfig's defaults, per card name)
CODER_CTX = 262144           # the Coder's max_ctx auto: its context_length, the measured fit with DFlash

# the header keys meta() keeps, under the architecture's prefix: what the projections below read
HP_KEYS = ('context_length', 'block_count', 'full_attention_interval', 'attention.key_length', 'ssm.time_step_rank',
           'ssm.state_size')
_META = {}


def _file_key(path):
    st = os.stat(path)
    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)


_CODER = {}


def coder_facts(path):
    """the Coder's header facts for its projection, read once per file: geom (capacity.geometry), tensors
    (capacity.split_tensors + the GDN recurrent state's bytes, gdn_state), bytes (capacity.model_bytes)"""
    k = _file_key(path)
    if k not in _CODER:
        _config.ensure_gguf()
        from gguf import GGUFReader
        from .device import capacity as C
        rd = GGUFReader(path, 'r'); g = C.geometry(rd); st = C.split_tensors(rd)
        n_gdn = sum(1 for t in rd.tensors if t.name.endswith('.ssm_a'))
        st['gdn_state'] = n_gdn * g['ssm_heads'] * g['ssm_state'] ** 2 * 4      # fp32 S per value head, one per GDN layer
        _CODER[k] = dict(geom=g, tensors=st, bytes=C.model_bytes(rd)); del rd
    return _CODER[k]


def dflash_geom(path):
    """the DFlash drafter's (layers, kv heads, head dim, window) from its header (caminus.decode.dflash.geom without torch)"""
    _config.ensure_gguf()
    from gguf import GGUFReader
    f = {x.name: x for x in GGUFReader(path, 'r').fields.values()}
    return tuple(int(f[f'dflash.{k}'].contents()) for k in ('block_count', 'attention.head_count_kv', 'attention.key_length',
                                                            'attention.sliding_window'))


def meta(path):
    """the GGUF header facts the registry reads, read once per file (cached by real path, size and mtime): arch
    (general.architecture), hp ({key: value} of HP_KEYS under the architecture's prefix, those present), weights (the bytes
    of every tensor of this file but token_embd, which the engines keep on the host)"""
    k = _file_key(path)
    if k not in _META:
        _config.ensure_gguf()
        from gguf import GGUFReader
        rd = GGUFReader(path, 'r'); f = {x.name: x for x in rd.fields.values()}
        a = f['general.architecture'].contents()
        hp = {h: f[f'{a}.{h}'].contents() for h in HP_KEYS if f'{a}.{h}' in f}
        w = sum(int(t.n_bytes) for t in rd.tensors if t.name != 'token_embd.weight')
        del rd, f
        _META[k] = dict(arch=a, hp=hp, weights=w)
    return _META[k]


def _read_int(p):
    try:
        with open(p) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def usable_ctx(max_ctx, T):
    """the context a server answers with on an engine of max_ctx positions and T tokens per pass: max_ctx less T - 2 (the
    speculative loop keeps prompt + max_new <= max_ctx - (T - 2)); a T = 1 engine keeps its max_ctx (K3Session.max_ctx)"""
    return int(max_ctx) - max(0, int(T) - 2)


def one_card_extra_mib(shape, chunk, max_ctx=None):
    """what the one-card engine's check counts after it (before anything is read): Prefill35's own buffers at chunk
    (caminus.prefill.q35.buffer_need_mib, from the model's shapes: the dense AREX-2 needs twice Qwen3.6's) at max_ctx
    (default the shape's own) + PREFILL_MODULES_MIB"""
    from .prefill import q35 as prefill35
    if max_ctx is not None and int(max_ctx) != shape.max_ctx:
        shape.max_ctx, keep = int(max_ctx), shape.max_ctx
        try:
            return prefill35.buffer_need_mib(shape, int(chunk)) + PREFILL_MODULES_MIB
        finally:
            shape.max_ctx = keep
    return prefill35.buffer_need_mib(shape, int(chunk)) + PREFILL_MODULES_MIB


def tp_shape(path):
    """Qwen3NextTP's host projection per card (caminus.engine.qwen3next_tp.projection, the one its own check reads) from the
    file's header facts (meta: read once): dict(n_ctx, kv_pos, state, need_mib(c))"""
    from .engine.qwen3next_tp import projection
    md = meta(path)
    return projection(md['hp'], md['weights'])


class Arch:
    """one architecture's serving recipe; the kinds below fill it in"""
    kind = None
    cards = None
    T = 1
    drafter = None
    engine = None            # 'module:Class' under caminus

    def __init__(self, name, alias):
        self.name, self.alias = name, alias

    def __repr__(self):
        return f'<arch {self.name}: {self.kind}, {self.cards} card{"s" if self.cards > 1 else ""}, T {self.T}>'

    def engine_class(self):
        """the engine class (imported now: torch)"""
        import importlib
        mod, cls = self.engine.split(':')
        return getattr(importlib.import_module(f'{__package__}.{mod}'), cls)

    def build(self, opt, log=print, T=None):
        """(m, pf, df, dr): the engine, its prompt path, the DFlash engine and the drafter (None without one), built in the
        serving order with every VRAM check before anything is read. opt: caminus serve's options (model, max_ctx, chunk,
        drafter, no_drafter). T: tokens per pass where the engine takes a choice (the Coder; caminus replay passes the
        bundle's), default self.T"""
        raise NotImplementedError

    def auto_need(self, opt, slots, cards=None):
        """max_ctx auto's projection, before anything is read: (the model's context_length, the devices, one checkpoint
        slot's MiB, need(c) -> per device MiB at context c with the prompt cache's side slot and slots + 2 checkpoint slots).
        cards: the caminus.device.host cards it would run on (fits, before the placement), else opt['placement']'s"""
        raise NotImplementedError

    def slack_mib(self, opt, cards=None):
        """max_ctx auto: MiB kept on top of the projection for its own error"""
        return AUTO_SLACK_MIB

    def pair_refusal(self, cards):
        """why this two-card model must not run on these cards (caminus.device.host dicts, engine order), or None"""
        return None

    def split(self, cards):
        """the TP split this two-card model runs at on these cards (engine order): caminus.device.host.split's, unless the
        architecture needs its own"""
        from .device import host
        return host.split(cards)

    def extras_mib(self, opt, devs, cards=None):
        """per card, what the server loads beside the engine, its prefill and the prompt cache: a drafter (DFlash / MTP: its
        weights, its ring, the feature export, the T > 1 verify buffers), so max_ctx auto leaves room for it. The one-card
        models and Qwen3-Coder-Next load no drafter yet (11-07): 0. A drafter port fills this from the same projection the
        engine's own check counts (the Coder: FlashNextTP(vram_drafters=...)), and its gate includes max_ctx auto with it."""
        return [0.0 for _ in devs]

    def auto_max_ctx(self, opt, log=print):
        """--max-ctx auto (the default): the largest context the cards hold now, under the guard's line, together with a
        working prompt cache: its reserve (capacity.cache_reserve_mib + CACHE_WORK_MIB), its side slot's rows, its io / side
        checkpoint slots and --auto-cache-slots checkpoint slots (default 4: more slots, less context), estimated from the
        engine's own host projection (the one its build checks) before anything is read. A multiple of AUTO_STEP, at most
        the model's context_length; extras_mib (a drafter's VRAM) comes off the room first"""
        from .device import capacity
        slots = int(opt.get('auto_cache_slots', 4))
        n_ctx, devs, slot, need = self.auto_need(opt, slots)
        sl = self.slack_mib(opt)
        room = [cap - used - res - capacity.CACHE_WORK_MIB[0] - sl - ex
                for (cap, used), res, ex in zip(capacity.caps_mib(devs), capacity.cache_reserve_mib(devs), self.extras_mib(opt, devs))]
        cands = sorted({min(c, n_ctx) for c in range(AUTO_STEP, n_ctx + AUTO_STEP, AUTO_STEP)}, reverse=True)
        for c in cands:
            n = need(c)
            if all(n[i] <= room[i] for i in range(len(devs))):
                log(f'caminus serve: max_ctx auto -> {c} (the model\'s {n_ctx}; room {" / ".join(f"{r:.0f}" for r in room)} MiB, '
                    f'need {" / ".join(f"{v:.0f}" for v in n)} MiB with {slots} cache slots of ~{slot:.0f} MiB)', flush=True)
                return c
        raise capacity.CapacityError(f'max_ctx auto: not even {AUTO_STEP} positions fit (room {" / ".join(f"{r:.0f}" for r in room)} '
                                     f'MiB, need {" / ".join(f"{v:.0f}" for v in need(AUTO_STEP))} MiB)')

    def fits(self, opt, cards):
        """whether the model holds MIN_CTX positions (or its context_length, if smaller) with a working prompt cache on these
        cards (caminus.device.host dicts, engine order) as they are now, from sysfs alone (no GPU runtime: caminus serve picks
        its cards with this before HIP starts) and the same projection max_ctx auto reads. True when the architecture has no
        host projection here (the Coder: its engine's own check decides)"""
        from .device import capacity
        try:
            n_ctx, devs, slot, need = self.auto_need(opt, int(opt.get('auto_cache_slots', 4)), cards=cards)
        except NotImplementedError:
            return True
        try:
            want = int(opt.get('max_ctx'))              # an explicit --max-ctx: that context; auto: MIN_CTX
        except (TypeError, ValueError):
            want = MIN_CTX
        n = need(min(want, n_ctx))
        for c, v, ex in zip(cards, n, self.extras_mib(opt, devs, cards=cards)):
            used = _read_int(os.path.join(c['dev'], 'mem_info_vram_used'))
            res = capacity.DISPLAY_RESERVE_MIB if c['display'] else capacity.MARGIN_MIB
            room = c['line_mib'] - (used >> 20 if used else 0) - res - capacity.CACHE_WORK_MIB[0] - self.slack_mib(opt, cards) - ex
            if v > room:
                return False
        return True

    def physics(self, d=0):
        """the stats hook's card model (caminus.serve.stats.PhysicsConfig), or None: EngineStats' default, the Coder's cards"""
        return None

    def set_ids(self, alias, tok):
        """the model ids the server lists (caminus.serve.template): alias (thinking on, unless the model's chat template never
        opens a think block: Qwen3-Coder-Next has no thinking mode) and alias-nothink"""
        from .serve.template import set_model_alias
        set_model_alias(alias, thinking='<think>' in (tok.vocab.chat_template or ''))


class Coder(Arch):
    kind, cards, T, drafter, engine = 'coder', 2, 4, 'dflash', 'engine.flashnext_tp:FlashNextTP'

    def build(self, opt, log=print, T=None):
        T = self.T if T is None else int(T)
        from .engine.flashnext_tp import FlashNextTP
        from .prefill import coder as P
        chunk, max_ctx = int(opt['chunk']), int(opt['max_ctx'])
        sp = dict((opt.get('placement') or {}).get('split') or {})     # (caminus.device.host.split of the placed pair; else the default)
        if opt['no_drafter']:
            m = FlashNextTP(opt['model'], ntok=T, max_ctx=max_ctx, vram_prefill=chunk, **sp)
            return m, P.Prefill(m, chunk=chunk), None, None
        from .decode import dflash as dk
        _config.ensure_gguf()
        from gguf import GGUFReader
        rd = GGUFReader(opt['drafter'], 'r'); f = {x.name: x for x in rd.fields.values()}
        layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
        dg = dk.geom(opt['drafter'])
        # (the drafter's VRAM, the DFlash feature export and the prefill counted in the target's own check)
        m = FlashNextTP(opt['model'], ntok=T, max_ctx=max_ctx, feature_layers=layers, feature_card=0,
                        vram_drafters=(('dflash', dg, None),), vram_prefill=chunk, **sp)
        df = dk.DFlashK3(m, opt['drafter'])
        dr = dk.DFlashK3Drafter(df)
        return m, P.Prefill(m, chunk=chunk), df, dr

    # one checkpoint slot per card, measured at the tuned split with DFlash (tests/cache/test_prompt_cache.py vram, 11-07:
    # 77.04 / 86.53 MiB: the recurrent states by value head, T tail rows, on card 1 the drafter's window rows); another
    # split moves the GDN state (by key head) between the cards
    SLOT_MIB = (77.04, 86.53)

    def split(self, cards):
        """the measured split on every pair: its batched prefill runs each card's share of the experts through MMQ, whose
        vendored kernels take whole 128-row tiles only, and the even split (8 / 10) gives each card 320 rows (the shared
        expert's IQ4_XS / Q4_K / Q5_K gate and up in 35 of the 48 layers would be refused); 10 / 12 gives 384 / 256. Card 0
        holds the bigger share, so the pair needs a 24 GB card 0 (fits: XTX + XT, XTX + XTX)"""
        from .device import capacity as C
        return dict(gdn_keys=C.MEASURED_SPLIT[0], moe_blocks=C.MEASURED_SPLIT[1])

    def _split(self, opt, cards):
        sp = self.split(cards) if cards is not None else ((opt.get('placement') or {}).get('split') or self.split(None))
        return int(sp['gdn_keys']), int(sp['moe_blocks'])

    def fits(self, opt, cards):
        ok = super().fits(opt, cards)
        if not ok and len(cards) >= 2 and cards[0]['total_mib'] < 22 * 1024:
            from .device import host
            raise host.PlacementError(f"the Coder needs a 24 GB RX 7900 XTX as card 0 (it holds the bigger share of the split): "
                                      f"{' + '.join(host.describe(c) for c in cards)} is too small")
        return ok

    def auto_need(self, opt, slots, cards=None):
        """the Coder's projection per card at its pair's split (D-292): FlashNextTP's own check (capacity.fixed_mib with the
        split's moved tensors, the context-sized buffers, the QSA tapes, the batched prefill's plan) + the prompt cache's
        side rows and slots + 2 checkpoint slots; the DFlash drafter in extras_mib"""
        from .device import capacity as C
        from .prefill.plan import PrefillPlan
        from .cache.prompt import SIDE_ROWS
        fa = coder_facts(opt['model']); g, st = fa['geom'], fa['tensors']
        sp, T, chunk = self._split(opt, cards), self.T, int(opt['chunk'])
        fx = C.fixed_mib('coder', T, fa['bytes'], split=sp, tensors=st)
        if fx is None: raise C.CapacityError(f'max_ctx auto: no measured fixed VRAM for this model ({fa["bytes"]} tensor bytes; measured {C.MODEL_BYTES})')
        nf = 0 if opt.get('no_drafter') else C.DFLASH_FEATURE_LAYERS
        gs = st['gdn_state'] / C.MIB
        f0 = lambda s_: s_[0] / st['keys']
        slot = [C.SLOT_PAD_MIB + self.SLOT_MIB[k] + gs * ((f0(sp) - f0(C.MEASURED_SPLIT)) if k == 0 else (f0(C.MEASURED_SPLIT) - f0(sp))) for k in (0, 1)]
        # the measured pair (XTX + XT at the tuned split): its parts and its prompt cache were measured together (262,144 with
        # DFlash: 15 checkpoint slots, XTX 23,071 MiB at ready, tools/serve_smoke.py 11-06), and there card 0's context and
        # prefill buffers land in blocks the load freed (~1 GB under their bytes at 262,144), so its projection is the
        # engine's own check + the side rows, the tier taking what is left; any other split adds the checkpoint slots and
        # slack_mib (unmeasured: conservative, so a pair is refused rather than guessed)
        measured = sp == C.MEASURED_SPLIT
        def need(c):
            ctx = C.tp_ctx_bytes(g, c, True)
            if c > C.QSA_EXACT: ctx.update(C.tp_qsa_bytes(g, c, T))
            side = sum(C.tp_ctx_bytes(g, min(SIDE_ROWS, c), True).values()) / C.MIB
            pf = PrefillPlan.of(g, chunk, gdn_keys=sp[0], moe_blocks=sp[1], T_pass=T, n_feat=nf, feature_card=0,
                                qsa_ctx=c if c > C.QSA_EXACT else 0).need_mib()
            return [fx[k] + sum(ctx.values()) / C.MIB + pf[k] + side + (0 if measured else (slots + 2) * slot[k]) for k in (0, 1)]
        return min(int(g['ctx_train']), CODER_CTX), [0, 1], max(slot), need

    def slack_mib(self, opt, cards=None):
        """none at the measured split (its parts are measured, auto_need), AUTO_SLACK_MIB at any other"""
        from .device import capacity as C
        return 0.0 if self._split(opt, cards) == C.MEASURED_SPLIT else AUTO_SLACK_MIB

    def extras_mib(self, opt, devs, cards=None):
        """the DFlash drafter (unless --no-drafter): its fixed part and its ring, as FlashNextTP(vram_drafters=) counts them"""
        if opt.get('no_drafter'): return [0.0, 0.0]
        from .device import capacity as C
        return list(C.drafter_need_mib('dflash', coder_facts(opt['model'])['geom'], CODER_CTX, self.T, dflash_geom=dflash_geom(opt['drafter'])))

    def set_ids(self, alias, tok):
        """the Coder keeps template.MODEL_PROFILES' ids (k3-coder and its reasoning-effort profiles: its template has the
        effort line); a --model-alias renames only /props' default id"""


class OneCard(Arch):
    kind, cards, T, engine = 'one-card', 1, 1, 'engine.qwen35_one:Qwen35One'

    def __init__(self, name, alias):
        super().__init__(name, alias)
        self._shapes = {}

    def shape(self, path):
        """Qwen35One's shapes and host projection for a file (shape_only: nothing on the card), made once per file: max_ctx
        auto and the build's own check use the same one (build releases it, and its reader, before the engine loads)"""
        k = _file_key(path)
        if k not in self._shapes:
            from .engine.qwen35_one import Qwen35One
            self._shapes[k] = Qwen35One(path, device=0, max_ctx=AUTO_STEP, ntok=1, vram_check=False, shape_only=True)
        return self._shapes[k]

    def auto_need(self, opt, slots, cards=None):
        from .cache.prompt import SIDE_ROWS
        sh = self.shape(opt['model']); p = sh.projection_parts(); chunk = int(opt['chunk'])
        slot = (p['gdn'] + p['kv_pos']) / 2 ** 20 * 1.05 + 1.0          # a checkpoint: the recurrent state + T = 1 tail rows
        need = lambda c: [sh.projection_mib(c) + one_card_extra_mib(sh, chunk, c) + min(SIDE_ROWS, c) * p['kv_pos'] / 2 ** 20
                          + (slots + 2) * slot]
        return sh.n_ctx_train or 262144, [0], slot, need

    def build(self, opt, log=print, T=None):
        from .engine.qwen35_one import Qwen35One
        from .prefill.q35 import Prefill35
        chunk, max_ctx = int(opt['chunk']), int(opt['max_ctx'])
        # (Prefill35's buffers, counted in the engine's check before anything is read: its own formula on the model's shapes)
        sh = self.shape(opt['model'])
        extra = one_card_extra_mib(sh, chunk, max_ctx)
        self._shapes.pop(_file_key(opt['model']), None); del sh
        m = Qwen35One(opt['model'], device=0, max_ctx=max_ctx, ntok=1, vram_extra_mib=extra)
        return m, Prefill35(m, chunk=chunk), None, None

    def physics(self, d=0):
        """the one card on torch device d: named by its label (caminus.device.machine: XTX / XT, else its DRM node), its guard
        line, bandwidth and the display card's reserve; card bytes and KV bytes from the engine later (K3Session); no QSA
        (dense attention at every depth)"""
        from .device import capacity
        from .serve.stats import PhysicsConfig
        c = capacity._machine.by_card(capacity.drm_card(d))
        name = c['label'] if c else f'card{d}'
        return PhysicsConfig(card_bytes=(1.0,), card_bw=(CARD_BW.get(name, CARD_BW['XT']),), mode='max', kv_bytes_per_token=(0,),
                             qsa_from=1 << 40, guard_mib=(int(capacity.guard_line_mib(d)),),
                             reserve_mib=(capacity.DISPLAY_RESERVE_MIB if c and c['display'] else 0,), card_names=(name,))


class TwoCardTP(Arch):
    kind, cards, T, engine = 'tp', 2, 1, 'engine.qwen3next_tp:Qwen3NextTP'

    def auto_need(self, opt, slots, cards=None):
        from .cache.prompt import SIDE_ROWS
        sh = tp_shape(opt['model'])
        slot = (sh['state'] + sh['kv_pos']) / 2 ** 20 * 1.05 + 1.0
        need = lambda c: [sh['need_mib'](c) + min(SIDE_ROWS, c) * sh['kv_pos'] / 2 ** 20 + (slots + 2) * slot] * 2
        return sh['n_ctx'], [0, 1], slot, need

    def build(self, opt, log=print, T=None):
        from .engine.qwen3next_tp import Qwen3NextTP
        from .prefill.kpass import KPassOnly
        # (its own check, before anything is read: qwen3next_tp.projection, the one auto_need reads)
        m = Qwen3NextTP(opt['model'], max_ctx=int(opt['max_ctx']), ntok=1, vram_check=True)
        return m, KPassOnly(m), None, None


ARCHS = {a.name: a for a in (Coder('qwen4exp', 'k3-coder'), OneCard('qwen35moe', 'qwen3.6'), OneCard('qwen35', 'qwen3.5'),
                             TwoCardTP('qwen3next', 'qwen3-coder-next'))}
CODER = ARCHS['qwen4exp']


def lookup(name):
    """the entry for an architecture name (any other: the Coder's, whose engine refuses what it cannot load)"""
    return ARCHS.get(name, CODER)


def of(path):
    """the entry for a GGUF, by its header's general.architecture (meta: read once per file)"""
    return lookup(meta(path)['arch'])


def default_alias(path):
    """the model id a server answers to for this GGUF when --model-alias is not given: the catalog's alias when the file is a
    catalog model's (caminus.models.model_of: its first file, same name and size; so a bare caminus serve and caminus start
    give AREX-2 the same id), else its architecture's"""
    from . import models
    m = models.model_of(path)
    return m.alias if m is not None else of(path).alias
