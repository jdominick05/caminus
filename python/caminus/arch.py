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
  qwen3moe   Qwen3-30B-A3B: Q3MoEOne on one card (own code from the GGUF to the logits), T = 1, no drafter, its own batched
             prefill (caminus.prefill.q3m.Q3MPrefill); its two-card form qwen3moe-tp (TWO_CARD: --cards 2) is Q3MoETP with
             Q3MPrefillTP, offered beside the one-card default

Any other architecture takes the Coder's path, as before the registry: FlashNextTP refuses a GGUF it cannot load.

An entry (Arch) answers: kind ('coder' | 'one-card' | 'tp'), cards, T (its tokens per pass when served), drafter (the drafter
it loads: 'dflash', the catalog role of its file, or None), alias (the architecture's default model id), engine (its engine
class); build(opt) -> (engine, prompt path, DFlash engine or None, drafter or None) in the serving order; auto_max_ctx(opt)
(--max-ctx auto); physics() (the stats hook's card model; None: the Coder's two-card default); set_ids(alias, tok) (the
template's model ids); rope_refusal(path) (why a file's rope.scaling.* keys refuse it: no entry applies them yet);
rope_facts (the model card's native context and long-context recipe, data only). Module functions: of(path) (the entry for a file), default_alias(path) (the id a server answers to:
the catalog's for a catalog file, else the architecture's), usable_ctx(max_ctx, T), meta(path) (the header facts, read once
per file).

Imports nothing heavy at module import: caminus.models and caminus.cmd.manage read ARCHS without torch. Before 11-07 this was
spread over caminus.serve.cli (11 architecture checks, a second table of model ids: AREX-2 answered qwen3.5 under a bare
caminus serve and arex-2 under caminus start), caminus.cmd.manage (the Coder's drafter by model name) and the test rigs, and
a start read the GGUF's header up to 4 times.
"""
import os
import struct
from dataclasses import dataclass

from . import config as _config

AUTO_STEP = 4096             # max_ctx auto: a multiple of this (or the model's context_length itself)
MIN_CTX = 32768              # the smallest context a model is placed on a card for (D-292 G4: below it the setup is refused)
AUTO_SLACK_MIB = 384.0       # max_ctx auto: kept on top (the projections' own error, allocator rounding: AREX-2 measured
                             # ~270 MiB over its projection with Prefill35's buffers counted exactly, 11-07)
PREFILL_MODULES_MIB = 64.0   # Prefill35's code objects, streams and its padded alpha / beta copies (a few MiB each)
CARD_BW = {'XTX': 881e9, 'XT': 774e9}           # measured streaming bandwidth (stats.PhysicsConfig's defaults, per card name)
CODER_CTX = 262144           # the Coder's max_ctx auto: its context_length, the measured fit with DFlash
LONG_NTK = 4.0               # the Coder's long-context mode (Coder.long_context): NTK RoPE scaling factor
LONG_CACHE_CAP_GB = 64.0     # its disk tier's default cap: a full 1M park is about 16 GB with its checkpoints

# the RoPE scaling keys a GGUF can carry, under the architecture's prefix (gguf-py's Keys.Rope.SCALING_*), and
# rope.scale_linear, llama.cpp's older name for a linear factor, which it still reads when rope.scaling.factor is absent.
# No engine applies them yet: Arch.rope_refusal refuses a file that sets them rather than serve it unscaled
ROPE_KEYS = ('rope.scaling.type', 'rope.scaling.factor', 'rope.scaling.alpha', 'rope.scaling.attn_factor',
             'rope.scaling.original_context_length', 'rope.scaling.finetuned', 'rope.scaling.yarn_truncate',
             'rope.scaling.yarn_log_multiplier', 'rope.scaling.yarn_ext_factor', 'rope.scaling.yarn_attn_factor',
             'rope.scaling.yarn_beta_fast', 'rope.scaling.yarn_beta_slow', 'rope.scale_linear')
# the header keys meta() keeps, under the architecture's prefix: what the projections below read, and the RoPE scaling keys
HP_KEYS = ('context_length', 'block_count', 'full_attention_interval', 'attention.key_length', 'ssm.time_step_rank',
           'ssm.state_size', 'rope.freq_base') + ROPE_KEYS
_META = {}
_TYPES = {}


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
        from . import ggufio
        try:                       # (Caminus's own reader; gguf-py only for a file it cannot read: a tensor type it lacks)
            F = ggufio.GGUFFile(path)
        except ggufio.GGUFError:
            F = None
        if F is not None:
            try:
                a = F.meta['general.architecture']
                hp = {h: _plain(F.meta[f'{a}.{h}']) for h in HP_KEYS if f'{a}.{h}' in F.meta}
                w = sum(int(t.n_bytes) for t in F.tensors if t.name != 'token_embd.weight')
            finally:
                F.close()
        else:
            _config.ensure_gguf()
            from gguf import GGUFReader
            rd = GGUFReader(path, 'r'); f = {x.name: x for x in rd.fields.values()}
            a = f['general.architecture'].contents()
            hp = {h: f[f'{a}.{h}'].contents() for h in HP_KEYS if f'{a}.{h}' in f}
            w = sum(int(t.n_bytes) for t in rd.tensors if t.name != 'token_embd.weight')
            del rd, f
        _META[k] = dict(arch=a, hp=hp, weights=w)
    return _META[k]


def tensor_types(path):
    """the weight formats of a file's tensors but token_embd (which the engines keep on the host), read once per file by
    Caminus's own reader; None when that reader refuses the file (the engines read it with the same reader, so they refuse
    it then too)"""
    k = _file_key(path)
    if k not in _TYPES:
        from . import ggufio
        try:
            F = ggufio.GGUFFile(path)
        except ggufio.GGUFError:
            _TYPES[k] = None
        else:
            try: _TYPES[k] = frozenset(t.type for t in F.tensors if t.name != 'token_embd.weight')
            finally: F.close()
    return _TYPES[k]


def rope_scaling(hp):
    """the RoPE scaling a file's header asks for: {key: value} of the ROPE_KEYS present, or None when it asks for none.
    None: no scaling key; rope.scaling.type 'none' (llama.cpp never scales then); or an effective factor of 1 (the
    factor, else rope.scale_linear; llama.cpp reads 0 as 1) with no attention factor other than 1 and no alpha other
    than 1. The other yarn_* / original_context_length / finetuned keys do nothing at factor 1, so alone they ask for
    nothing. Any other combination counts, typed or not: llama.cpp's default type is linear"""
    got = {k: hp[k] for k in ROPE_KEYS if k in hp}
    if not got or got.get('rope.scaling.type') == 'none': return None
    def one(k):
        v = got.get(k)
        try: return v is None or float(v) in (0.0, 1.0)
        except (TypeError, ValueError): return False
    f = 'rope.scaling.factor' if 'rope.scaling.factor' in got else 'rope.scale_linear'
    if one(f) and one('rope.scaling.attn_factor') and one('rope.scaling.yarn_attn_factor') and one('rope.scaling.alpha'):
        return None
    return got


@dataclass(frozen=True)
class RopeFacts:
    """a model line's context facts from its model card: the native context, the GGUF's, the card's recipe, and the scaling
    Caminus ships past gguf_ctx (shipped: None refuses a longer max_ctx; 'yarn' is Qwen3-30B-A3B's opt-in long-context mode,
    caminus serve --long-context, at shipped_factor from native_ctx; 'ntk' is the Coder's LONG_NTK mode)"""
    native_ctx: int                # the context the model was trained for (the card's original_max_position_embeddings)
    gguf_ctx: int                  # the GGUF's context_length: how far Caminus serves it unscaled today
    card_kind: str                 # the card's long-context recipe ('yarn')
    card_factors: tuple            # the factors the card names: x2 for twice the native context, x4 the most it recommends
    shipped: 'str | None' = None   # the scaling Caminus applies past gguf_ctx today: None (a longer max_ctx is refused)
    shipped_factor: 'float | None' = None
    source: str = ''               # the model card the recipe is from


# Qwen3-30B-A3B: "32,768 natively and 131,072 tokens with YaRN" (factor 4, original_max_position_embeddings 32768; the card
# advises factor 2 for about 65,536 and no YaRN while contexts stay under 32,768). Its GGUF's context_length, 40,960, is
# 32,768 for the output plus 8,192 for the prompt: served unscaled up to there, as the card allows (the default). The
# long-context mode (--long-context, or an explicit --max-ctx past 40,960) serves YaRN x4 up to 131,072 at every position:
# the card's note that static YaRN can cost short texts some quality is the mode's trade-off, so it is never the default
QWEN3MOE_ROPE = RopeFacts(32768, 40960, 'yarn', (2.0, 4.0), shipped='yarn', shipped_factor=4.0,
                          source='https://huggingface.co/Qwen/Qwen3-30B-A3B')
# the dense Qwen3.8-27B line: 262,144 native, YaRN factor 4 from original 262,144 ("up to 1,000,000"; factor 2 for about
# 524,288). Swift 1.5 and ThinkingCap keep the base's RoPE; their own cards make no RoPE-scaling statement, so the recipe
# is the base card's, not one their authors claim
QWEN35_ROPE = RopeFacts(262144, 262144, 'yarn', (2.0, 4.0), source='https://huggingface.co/Qwen/Qwen3.8-27B')
# the Coder: its base card (Qwen3.8-Flash-Next) recommends YaRN factor 4 from 262,144; Caminus ships NTK base scaling by
# LONG_NTK instead (Coder.long_context, an explicit max_ctx past 262,144)
CODER_ROPE = RopeFacts(262144, 262144, 'yarn', (4.0,), shipped='ntk', shipped_factor=LONG_NTK,
                       source='https://huggingface.co/Qwen/Qwen3.8-Flash-Next')


def _plain(v):
    """a header value as gguf-py's contents() gives it (a numeric array as a list)"""
    return v.tolist() if hasattr(v, 'tolist') else v


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
    rope_facts = None        # RopeFacts: the model card's context facts (data only), None where the registry has none
    kind = None
    cards = None
    T = 1
    drafter = None
    engine = None            # 'module:Class' under caminus
    tokenizer = 'hf'         # caminus.serve.tokenizer.QwenTokenizer's BPE engine: 'hf' (the tokenizers library) or 'own'

    # the KV cache formats opt['kv'] (caminus serve --kv) selects for this entry (kv_check); only the qwen3moe entries
    # (Q3MoE, Q3MoETP) build 'q8'. 'f16' is the switch's default and selects nothing on an entry without the choice: the
    # dense own-code line keeps its F16 cache, and the GGUF engines (the Coder, the paused one-card / Qwen3-Coder-Next
    # entries) keep their own 8-bit q8_0 cache (kv_fixed), which /props reports from the built engine
    kv_formats = ('f16',)
    kv_fixed = None          # the cache an entry without the choice always builds, when it is not F16 (the GGUF engines: 'q8_0')

    def __init__(self, name, alias):
        self.name, self.alias = name, alias

    def kv_check(self, opt, who=None):
        """opt['kv'] for this entry: True for 'q8', False for 'f16' (the default). A format the entry has no build for is
        refused with a ValueError before anything is read or allocated, so a Q8 request never runs an F16 engine. who: the
        model's name for the message (caminus serve: the catalog's id), default the entry's alias"""
        q8 = kv_q8(opt)
        if q8 and 'q8' not in self.kv_formats:
            raise ValueError(f"kv 'q8': {who or self.alias} keeps its own KV cache ("
                             + ("its 8-bit q8_0 rows" if self.kv_fixed else "F16") + f"; no --kv choice in this release): "
                             f"--kv q8 serves {kv_q8_models()} only")
        return q8

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

    def context_fit(self, opt, cards=None):
        """One allocation recipe and room calculation for placement, auto and picker reports.

        Explicit host cards use metadata/sysfs only, before HIP starts. Runtime
        callers omit cards and read capacity's actual guard caps and usage.
        """
        from .device import capacity
        from .device.context_fit import ContextFit
        n_ctx, devs, slot, need = self.auto_need(opt, int(opt.get('auto_cache_slots', 4)), cards=cards)
        extra = self.extras_mib(opt, devs, cards=cards)
        if cards is None:
            caps, reserve = capacity.caps_mib(devs), capacity.cache_reserve_mib(devs)
        else:
            # caps_mib already subtracts the allocator margin from the guard
            # line. Placement must do the same before its cache/display reserve.
            caps = []
            for c in cards:
                used = _read_int(os.path.join(c['dev'], 'mem_info_vram_used'))
                if used is None or used < 0:
                    from .device.host import PlacementError
                    raise PlacementError(f'cannot read valid current VRAM usage for {c.get("pci") or c["dev"]}')
                caps.append((c['line_mib'] - capacity.MARGIN_MIB, used / 2 ** 20))
            reserve = [capacity.DISPLAY_RESERVE_MIB if c['display'] else capacity.MARGIN_MIB for c in cards]
        if not (len(devs) == len(caps) == len(reserve) == len(extra)):
            raise ValueError('context fit: card, projection and reserve counts differ')
        slack = self.slack_mib(opt, cards)
        room = [cap - used - res - capacity.CACHE_WORK_MIB[0] - slack - ex
                for (cap, used), res, ex in zip(caps, reserve, extra)]
        return ContextFit(n_ctx, devs, slot, room, need, AUTO_STEP)

    def auto_max_ctx(self, opt, log=print):
        """--max-ctx auto (the default): the largest context the cards hold now, under the guard's line, together with a
        working prompt cache: its reserve (capacity.cache_reserve_mib + CACHE_WORK_MIB), its side slot's rows, its io / side
        checkpoint slots and --auto-cache-slots checkpoint slots (default 4: more slots, less context), estimated from the
        engine's own host projection (the one its build checks) before anything is read. A multiple of AUTO_STEP, at most
        the model's context_length; extras_mib (a drafter's VRAM) comes off the room first"""
        from .device import capacity
        slots = int(opt.get('auto_cache_slots', 4))
        plan = self.context_fit(opt)
        c, room = plan.maximum(), plan.room_mib
        if c is not None:
            n = plan.need_mib(c)
            log(f'caminus serve: max_ctx auto -> {c} (the model\'s {plan.native_ctx}; room {" / ".join(f"{r:.0f}" for r in room)} MiB, '
                f'need {" / ".join(f"{v:.0f}" for v in n)} MiB with {slots} cache slots of ~{plan.slot_mib:.0f} MiB)', flush=True)
            return c
        lowest = min(AUTO_STEP, plan.native_ctx)
        raise capacity.CapacityError(f'max_ctx auto: not even {lowest} positions fit (room {" / ".join(f"{r:.0f}" for r in room)} '
                                     f'MiB, need {" / ".join(f"{v:.0f}" for v in plan.need_mib(lowest))} MiB)')

    def fits(self, opt, cards):
        """whether the model holds MIN_CTX positions (or its context_length, if smaller) with a working prompt cache on these
        cards (caminus.device.host dicts, engine order) as they are now, from sysfs alone (no GPU runtime: caminus serve picks
        its cards with this before HIP starts) and the same projection max_ctx auto reads. True when the architecture has no
        host projection here (the Coder: its engine's own check decides)"""
        from .device import capacity
        try:
            plan = self.context_fit(opt, cards)
        except NotImplementedError:
            return True
        try:
            want = int(opt.get('max_ctx'))              # an explicit --max-ctx: that context; auto: the architecture's floor
        except (TypeError, ValueError):
            want = self.min_ctx(opt)
        return plan.fits(min(want, plan.native_ctx))

    def draft_refusal(self, draft, path=None):
        """why this entry must not serve --draft `draft` for the GGUF at path (a sentence for the error), or None. Without a
        path (the model picker: no file is read) only what the entry knows without a file"""
        return None

    def placement_extra(self, opt, cards):
        """what the entry decides once on the placed cards (caminus.device.host dicts, engine order) and reads back from
        opt['placement'] at build: nothing unless an entry needs it (Dense35TP: the drafter's card)"""
        return {}

    def default_draft(self, opt):
        """the drafter kind a serve runs without --draft: the architecture's own when it needs no extra file choice (the
        Coder's DFlash), else off"""
        return 'dflash' if self.drafter == 'dflash' else 'off'

    def min_ctx(self, opt):
        """the smallest context this serve is placed on a card for (max_ctx auto): MIN_CTX unless the architecture trades
        context for a drafter"""
        return MIN_CTX

    def long_mode(self, opt):
        """the long-context mode, resolved before placement (caminus serve's recipe, caminus fit, caminus start): sets what the
        projections, placement and build read into opt and returns a short description, or None when the serve is not in
        the mode. --long-context (opt['long_context']) on an entry without a mode is refused here, before anything is read
        on a card; an explicit max_ctx past the context_length stays the engine's refusal (as before). The Coder and the
        qwen3moe entries override it"""
        if opt.get('long_context'):
            raise ValueError(f'--long-context: {self.alias} ({self.name}) has no long-context mode (Qwen3-30B-A3B has YaRN x4, '
                             f'the Coder its NTK mode); serve it without the flag, at most at its own context_length')
        return None

    def rope_mode(self, opt):
        """the engine's rope argument for this serve: None (the trained RoPE) unless a long_mode set opt['rope']"""
        return None

    def long_context(self, opt):
        """the long-context mode's settings into opt for an explicit max_ctx past the model's context_length (a note for the
        log), or None: the Coder (its KV in host memory, NTK) and the qwen3moe entries (YaRN, resolved by long_mode before
        placement: this only words the note) have one; every other model refuses such a max_ctx in its engine"""
        return None

    def rope_refusal(self, path):
        """why this entry must not serve this file's RoPE scaling, or None (the file asks for none: rope_scaling). No engine
        applies a file's rope.scaling.* keys yet, so every entry refuses such a file rather than serve it unscaled (its
        attention would not be the model's). A header neither reader can read is left to the engine's own reader, which
        refuses the file. The Coder's NTK long mode is a serve option (an explicit max_ctx past 262,144), not a file key"""
        try:
            md = meta(path)
        except (OSError, ValueError, KeyError, IndexError, struct.error):
            return None
        rs = rope_scaling(md['hp'])
        if rs is None: return None
        keys = ', '.join(f'{md["arch"]}.{k} = {v}' for k, v in rs.items())
        return (f'{os.path.basename(path)} asks for RoPE scaling ({keys}): {self.alias} ({self.name}) does not apply a '
                f'file\'s RoPE scaling yet, and serving it unscaled would not be the model\'s attention; use a file without '
                f'these keys (or rope.scaling.type none)')

    def physics(self, d=0):
        """the stats hook's card model (caminus.serve.stats.PhysicsConfig), or None: EngineStats' default, the Coder's cards"""
        return None

    def variant(self, cards):
        """the entry that serves this model on `cards` cards (None: its own count). An architecture with a two-card form names
        it in TWO_CARD (the dense own line: Dense35 -> Dense35TP; Qwen3-30B-A3B: Q3MoE -> Q3MoETP); any other count than its
        own is refused (ValueError)"""
        if cards is None or int(cards) == self.cards: return self
        alt = TWO_CARD.get(self.name) if int(cards) == 2 else None
        if alt is None:
            raise ValueError(f'{self.alias}: runs on {self.cards} card{"s" if self.cards > 1 else ""} only (not {cards})')
        return ARCHS[alt]

    def set_ids(self, alias, tok):
        """the model ids the server lists (caminus.serve.template): alias (thinking on, unless the model's chat template never
        opens a think block: Qwen3-Coder-Next has no thinking mode) and alias-nothink, plus alias-low / alias-xhigh when the
        template also pins a reasoning_effort line (the dense Qwen3.8-27B line does)"""
        from .serve.template import set_model_alias
        tpl = tok.vocab.chat_template or ''
        set_model_alias(alias, thinking='<think>' in tpl, effort='reasoning_effort' in tpl)


class Coder(Arch):
    kind, cards, T, drafter, engine = 'coder', 2, 4, 'dflash', 'engine.flashnext_tp:FlashNextTP'
    rope_facts = CODER_ROPE
    kv_fixed = 'q8_0'        # (FlashNextTP's own cache: q8=True, its default; build passes none)

    def long_context(self, opt):
        """an explicit max_ctx past the Coder's context_length (262,144): the KV cache in pinned host memory (kv_host: 13 GB at
        1M for both cards, checked against MemAvailable and the GTT limit), ctx_extend, NTK RoPE scaling by LONG_NTK (measured at
        1,040,384: 3/3 needles to 936K back, 2/3 without; its cost at short context: KLD 0.0077 vs 0.0057 unscaled, llama.cpp's
        floor 0.0076). The disk tier stays on (0.2.2; it was off in 0.2.1): a park writes the conversation's KV rows, about 12.4 KB
        per token (5 GB at 400K, a full 1M one about 16 GB with its checkpoints), so the cap is LONG_CACHE_CAP_GB unless set: a
        conversation another one pushed out (a harness's background job too big for the side slot, a second session) comes
        back from disk in seconds instead of being read again (a 494,882-token conversation took 587 s, 11-09). The daily
        mode (<= 262,144) is unchanged"""
        n_ctx = int(coder_facts(opt['model'])['geom']['ctx_train'])
        if int(opt['max_ctx']) <= n_ctx: return None
        opt['kv_host'] = True; opt['long_ntk'] = LONG_NTK
        disk = not opt.get('no_disk_cache')
        if disk and not opt.get('cache_cap_gb'): opt['cache_cap_gb'] = LONG_CACHE_CAP_GB
        return (f'long-context mode: max_ctx {int(opt["max_ctx"])} past the model\'s {n_ctx}: KV in host memory, RoPE base x '
                f'{LONG_NTK:g}^(n_rot / (n_rot - 2)), '
                f'{"the disk tier capped at " + format(float(opt["cache_cap_gb"]), "g") + " GB (a park writes about 12.4 KB per token)" if disk else "no disk tier"}'
                f'{", plain decoding at T = 1" if opt.get("no_drafter") else ""}')

    def long_mode(self, opt):
        """the Coder's mode is chosen by an explicit --max-ctx past its context_length (long_context below, unchanged);
        --long-context with it is accepted, without it refused: the mode's context lives in host memory and has no auto size"""
        if not opt.get('long_context'): return None
        n = _explicit_ctx(opt)
        if n is None or n <= CODER_CTX:
            raise ValueError(f'--long-context: the Coder\'s long-context mode takes an explicit --max-ctx past {CODER_CTX:,} '
                             f'(its KV cache in host memory, NTK RoPE x {LONG_NTK:g}; 1,048,576 at most measured)')
        return 'NTK long-context mode'

    def _engine_kw(self, opt):
        """FlashNextTP's placement / long-context arguments from opt (kv_host, the long mode's ctx_extend + rope_ntk)"""
        kw = dict(kv_host=True) if opt.get('kv_host') else {}
        if opt.get('long_ntk'): kw.update(ctx_extend=True, rope_ntk=float(opt['long_ntk']))
        return kw

    def build(self, opt, log=print, T=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        T = self.T if T is None else int(T)
        from .engine.flashnext_tp import FlashNextTP
        from .prefill import coder as P
        chunk, max_ctx = int(opt['chunk']), int(opt['max_ctx'])
        sp = dict((opt.get('placement') or {}).get('split') or {})     # (caminus.device.host.split of the placed pair; else the default)
        sp.update(self._engine_kw(opt))
        if opt['no_drafter']:
            if opt.get('long_ntk'): T = 1   # (the long mode without a drafter: plain T = 1 passes, ~29.7 ms/token at 1M)
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
        from . import models                    # (the one 24 GB rule: models.CLASS_MIB; models imports this module)
        if not ok and len(cards) >= 2 and models.card_class(cards[0]['total_mib']) != 'XTX':
            from .device import host
            raise host.PlacementError(f"the Coder needs a 24 GB RX 7900 XTX as card 0 (it holds the bigger share of the split): "
                                      f"{' + '.join(host.describe(c) for c in cards)} is too small")
        return ok

    def auto_need(self, opt, slots, cards=None):
        """the Coder's projection per card at its pair's split (D-292): FlashNextTP's own check (capacity.fixed_mib with the
        split's moved tensors, the context-sized buffers, the QSA tapes, the batched prefill's plan) + the prompt cache's
        slots + 2 checkpoint slots (its side rows are host memory); the DFlash drafter in extras_mib"""
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .device import capacity as C
        from .prefill.plan import PrefillPlan
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
        # engine's own check, the tier taking what is left (the side slot's rows are host memory since 0.2.2, not VRAM); any other split adds the checkpoint slots and
        # slack_mib (unmeasured: conservative, so a pair is refused rather than guessed)
        measured = sp == C.MEASURED_SPLIT
        def need(c):
            ctx = C.tp_ctx_bytes(g, c, True)
            if c > C.QSA_EXACT: ctx.update(C.tp_qsa_bytes(g, c, T))
            pf = PrefillPlan.of(g, chunk, gdn_keys=sp[0], moe_blocks=sp[1], T_pass=T, n_feat=nf, feature_card=0,
                                qsa_ctx=c if c > C.QSA_EXACT else 0).need_mib()
            return [fx[k] + sum(ctx.values()) / C.MIB + pf[k] + (0 if measured else (slots + 2) * slot[k]) for k in (0, 1)]
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


def kv_q8(opt):
    """the KV cache format an entry builds from opt['kv'] ('f16', the default, or 'q8': the Q8_0 rows of the entries that
    have them; caminus serve --kv / caminus start --kv set it): True for q8; anything else is refused"""
    v = opt.get('kv') or 'f16'
    if v not in ('f16', 'q8'): raise ValueError(f"kv {v!r}: 'f16' or 'q8'")
    return v == 'q8'


# The own Q8_0 KV rows' format ids: an engine built with q8 names its rows by one (kv_format_id; the prompt cache keys
# parks by it). A built engine holds a Q8 cache exactly when its id is one of these, so an F16 engine that names its own
# format never reads as Q8 (tests/test_arch.py checks every kv_format_id an engine assigns is listed here).
KV_Q8_FORMATS = ('q8_0-v1:qk-hadamard128:f16-rounded-v1',   # Qwen3-30B-A3B, one card and two (engine.q3moe_one)
                 'q8_0-v1:qk-hadamard256:v-hadamard64-v1')  # the dense own-code line's staged Q8 (engine.q35_dense)


def kv_q8_built(m):
    """True when a built engine holds the own Q8_0 rows: its kv_format_id is one of KV_Q8_FORMATS"""
    return getattr(m, 'kv_format_id', None) in KV_Q8_FORMATS


def kv_q8_models():
    """the models --kv q8 serves, for the refusals: each alias whose entries list 'q8' in kv_formats, with its card counts
    ('qwen3-30b-a3b (one card or two)')"""
    by = {}
    for a in ARCHS.values():
        if 'q8' in a.kv_formats: by.setdefault(a.alias, set()).add(a.cards)
    words = {(1,): 'one card', (2,): 'two cards', (1, 2): 'one card or two'}
    return ', '.join(f'{al} ({words.get(tuple(sorted(cs)), "cards " + "/".join(map(str, sorted(cs))))})'
                     for al, cs in sorted(by.items())) or 'no model'


def kv_cache_of(A, m):
    """the KV cache a built engine holds, for /props and /caminus/status: {'type': 'f16' | 'q8', 'format': the engine's
    format id (kv_format_id, one of KV_Q8_FORMATS: the own Q8_0 rows), 'q8_0' (the GGUF engines' own 8-bit cache,
    A.kv_fixed) or 'f16' (an F16 engine's own kv_format_id when it names one)}"""
    fid = getattr(m, 'kv_format_id', None)
    if kv_q8_built(m): return {'type': 'q8', 'format': str(fid)}
    if A is not None and A.kv_fixed: return {'type': 'q8', 'format': str(A.kv_fixed)}
    return {'type': 'f16', 'format': str(fid) if fid else 'f16'}   # (an F16 engine may name its own format)


def _explicit_ctx(opt):
    """opt['max_ctx'] as an int when it is an explicit number, else None ('auto', unset)"""
    v = opt.get('max_ctx')
    if v is None or isinstance(v, bool) or str(v).strip().lower() == 'auto': return None
    try: return int(v)
    except (TypeError, ValueError): return None


class _Q3MLong:
    """the qwen3moe entries' long-context mode (Q3MoE one card, Q3MoETP two): YaRN at the model card's factor (x4 from
    32,768: RopeFacts.shipped) on the YaRN builds of the decode modules and the prefill's qk_rope (caminus.rope).

    In the mode: max_ctx auto and explicit up to the mode's 131,072 positions (the projection counts them; native_ctx of
    the context fit is the mode's cap), the engine key carries the RoPE (a YaRN park never restores into a native server:
    the module names differ too). The KV format: what opt['kv'] says when set; otherwise F16 where the asked context (an
    explicit max_ctx, or the cap for auto) fits in F16, else Q8_0 rows where they give more positions (one XTX: about 80K
    instead of about 45K), and the note says so. Native (no mode) stays the default: the card advises no YaRN under 32,768
    and static YaRN at every position can cost short texts some quality"""

    # the file the mode's recipe is for: the model card's YaRN numbers (x4 from 32,768) hold for Qwen3-30B-A3B's own weights,
    # whose GGUF says context_length 40,960 and rope.freq_base 1e6. Another qwen3moe file (the 2507 Instruct / Thinking
    # releases, Qwen3-Coder-30B-A3B: 262,144 positions natively, another base) reaches this entry by its header too; the
    # mode would give it a scaling it was not trained with, so the mode refuses it (its native serve is unchanged)
    LONG_FREQ_BASE = 1e6
    # the weight formats the mode's builds run (the lean decode modules, q3m_prefill): the codebook (k3.OWN_IQ_FMT) and
    # row-scale (own_layout.RS_CONT) formats run on their own modules, which have no YaRN build; refused here, before a card
    # is placed, rather than at engine build (tests/engine/test_q3moe_yarn_cpu.py checks these names against those tables)
    LONG_OFF_FORMATS = frozenset(('IQ3_XXS', 'IQ4_XS', 'IQ3_S', 'IQ2_XXS', 'Q4_0', 'Q2_0'))

    def long_refusal(self, path):
        """why the long-context mode refuses this file (None: it is the model the recipe is for, in formats the YaRN builds run)"""
        hp, rf = meta(path)['hp'], self.rope_facts
        ctx, base = hp.get('context_length'), hp.get('rope.freq_base')
        if ctx is None or int(ctx) != rf.gguf_ctx or base is None or float(base) != self.LONG_FREQ_BASE:
            return (f'the long-context mode (YaRN x{rf.shipped_factor:g} from {rf.native_ctx:,}) is Qwen3-30B-A3B\'s recipe '
                    f'(context_length {rf.gguf_ctx:,}, rope.freq_base {self.LONG_FREQ_BASE:g}); this file says context_length '
                    f'{ctx if ctx is None else f"{int(ctx):,}"}, rope.freq_base {base if base is None else f"{float(base):g}"}: '
                    f'another model, whose RoPE the recipe does not fit (serve it natively, up to its own context_length)')
        off = sorted((tensor_types(path) or frozenset()) & self.LONG_OFF_FORMATS)
        if off:
            return (f'the long-context mode runs on the default formats\' modules only; this file has {", ".join(off)} tensors '
                    f'(codebook or row-scale formats: their modules have no YaRN build)')
        return None

    def long_mode(self, opt):
        n, gguf_ctx = _explicit_ctx(opt), int(meta(opt['model'])['hp'].get('context_length') or self.rope_facts.gguf_ctx)
        if not (opt.get('long_context') or (n is not None and n > gguf_ctx)):
            opt.pop('rope', None)
            return None
        why = self.long_refusal(opt['model'])
        if why: raise ValueError(why)
        mode = self._yarn()
        if n is not None and n > mode.max_ctx:
            raise ValueError(f'--max-ctx {n:,}: Qwen3-30B-A3B serves at most {mode.max_ctx:,} positions (its long-context mode, '
                             f'YaRN x{mode.factor:g} from {mode.n_ctx_orig:,}; the model card names no longer context)')
        opt['long_context'], opt['rope'] = True, 'yarn'
        return f'YaRN x{mode.factor:g}'

    def _yarn(self):
        from .rope import YarnMode
        rf = self.rope_facts
        return YarnMode(rf.shipped_factor, rf.native_ctx)

    def rope_mode(self, opt):
        return self._yarn() if opt.get('rope') == 'yarn' else None

    def _ctx_cap(self, opt):
        """the context fit's native_ctx: the file's context_length, or the long-context mode's positions"""
        if opt.get('rope') == 'yarn': return self._yarn().max_ctx
        return int(meta(opt['model'])['hp'].get('context_length') or self.rope_facts.gguf_ctx)

    def context_fit(self, opt, cards=None):
        """Arch.context_fit, and in the long-context mode with no opt['kv'] the format the mode would choose (F16 where the
        asked context fits, else Q8 where it holds more): the plan carries it as .kv"""
        if opt.get('rope') != 'yarn' or opt.get('kv') is not None:
            plan = super().context_fit(opt, cards); plan.kv = 'q8' if kv_q8(opt) else 'f16'
            return plan
        f16 = super().context_fit(dict(opt, kv='f16'), cards); f16.kv = 'f16'
        n = _explicit_ctx(opt)
        want = min(n if n is not None else f16.native_ctx, f16.native_ctx)
        if f16.fits(want): return f16
        q8 = super().context_fit(dict(opt, kv='q8'), cards); q8.kv = 'q8'
        if (q8.fits(want) if n is not None else (q8.maximum() or 0) > (f16.maximum() or 0)):
            q8.kv_why = (f'F16 rows hold {f16.maximum() or 0:,} positions here, Q8_0 rows ' +
                         (f'hold the asked {want:,}' if n is not None else f'{q8.maximum() or 0:,}'))
            return q8
        return f16

    def placement_extra(self, opt, cards):
        """the long-context mode's KV format, decided once on the placed cards (context_fit) and written to opt['kv'] (auto,
        build and the engine key read it); recorded in the placement"""
        if opt.get('rope') != 'yarn' or opt.get('kv') is not None: return {}
        plan = self.context_fit(opt, cards)
        opt['kv'] = plan.kv
        if plan.kv == 'q8': opt['kv_auto_why'] = getattr(plan, 'kv_why', '')
        return {'kv': plan.kv}

    def long_context(self, opt):
        if opt.get('rope') != 'yarn': return None
        mode = self._yarn()
        kv = 'q8' if kv_q8(opt) else 'f16'
        why = opt.get('kv_auto_why')
        return (f'long-context mode: YaRN x{mode.factor:g} from {mode.n_ctx_orig:,} (up to {mode.max_ctx:,} positions), max_ctx '
                f'{int(opt["max_ctx"]):,}, KV {"Q8_0 rows" if kv == "q8" else "F16"}'
                f'{" (chosen by the mode: " + why + ")" if why else ""}; YaRN applies at every position, so short texts can '
                f'read slightly differently than in the default mode (the model card advises it only past 32,768)')


class OneCard(Arch):
    kind, cards, T, engine = 'one-card', 1, 1, 'engine.qwen35_one:Qwen35One'
    kv_fixed = 'q8_0'        # (Qwen35One's q8_0 cache: q8=True, its default; the own-code subclasses below set None)

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
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        sh = self.shape(opt['model']); p = sh.projection_parts(); chunk = int(opt['chunk'])
        slot = (p['gdn'] + p['kv_pos']) / 2 ** 20 * 1.05 + 1.0          # a checkpoint: the recurrent state + T = 1 tail rows
        need = lambda c: [sh.projection_mib(c) + one_card_extra_mib(sh, chunk, c) + (slots + 2) * slot]   # (the side rows: host memory)
        return sh.n_ctx_train or 262144, [0], slot, need

    def build(self, opt, log=print, T=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
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


class Q3MoE(_Q3MLong, OneCard):
    """qwen3moe (Qwen3-30B-A3B): Q3MoEOne on one card, T = 1, Caminus's own code from the GGUF to the logits; no drafter yet;
    the prompt through its own batched prefill (caminus.prefill.q3m.Q3MPrefill, chunks of PREFILL_CHUNK). Full depth
    (context_length 40,960) fits the 24 GB card only: weights 17.3 GiB + f16 KV 96 KiB per position (3.75 GiB at 40,960).
    The long-context mode (_Q3MLong: YaRN x4) on one XTX: about 80K positions with Q8_0 rows (the mode picks them)."""
    engine = 'engine.q3moe_one:Q3MoEOne'
    rope_facts = QWEN3MOE_ROPE
    tokenizer = 'own'          # (caminus.bpe: the qwen2 pre-tokenizer; the GGUF read by caminus.ggufio: no gguf-py, no tokenizers)
    kv_formats = ('f16', 'q8')   # (caminus serve --kv q8: Q3MoEOne(q8=True), docs/dev/q8-kv.md; F16 the default)
    kv_fixed = None
    PREFILL_CHUNK = 2048       # (XTX, from empty: 512 / 4K / 32K at 1024: 5,183 / 5,787 / 2,609 tok/s; 2048: 5,196 / 6,436 / 2,771; 281 MiB)

    def auto_need(self, opt, slots, cards=None):
        from .engine.q3moe_one import Q3MoEOne
        from .prefill import q3m
        path = opt['model']
        n_ctx = self._ctx_cap(opt)                          # (the file's 40,960, or the long-context mode's 131,072)
        q8 = self.kv_check(opt)                             # (opt['kv']: caminus serve --kv; 'f16' by default)
        slot = Q3MoEOne.kv_pos_mib(path, q8) * 1.05 + 1.0          # a checkpoint: one tail row of every layer (T = 1)
        pf = q3m.buffer_need_mib(q3m.shapes_of(path), self.PREFILL_CHUNK)
        need = lambda c: [Q3MoEOne.projection_mib(path, c, q8=q8) + pf + (slots + 2) * slot]   # (the side rows: host memory)
        return n_ctx, [0], slot, need

    def build(self, opt, log=print, T=None):
        from .engine.q3moe_one import Q3MoEOne
        from .prefill.q3m import Q3MPrefill
        m = Q3MoEOne(opt['model'], device=0, max_ctx=int(opt['max_ctx']), q8=self.kv_check(opt), rope=self.rope_mode(opt))
        return m, Q3MPrefill(m, chunk=self.PREFILL_CHUNK), None, None


class Dense35(OneCard):
    """the dense qwen35 line on Caminus's own code (Swift 1.5 Q4_K_M, ThinkingCap RCO 4.2 bpw: Dense35Own picks its decode
    module from the file's formats, the prefill its object): Dense35Own on one card, T = 1, the own reader,
    kernels and tokenizer; the prompt through its own batched prefill (caminus.prefill.q35own.Q35OwnPrefill). Selected
    through its catalog model (of(): the file's catalog model names this entry), never by the header: AREX-2 is a qwen35
    file too and keeps OneCard's engine. Drafter: the model's own MTP block (--draft mtp, one card; Swift 1.5 first:
    the drafter module runs the block's Q4_0 and the file's Q6_K head)"""
    engine = 'engine.q35_dense:Dense35Own'
    rope_facts = QWEN35_ROPE
    kv_fixed = None            # (F16: its Q8 objects are not served, kv_check refuses 'q8')
    drafter = 'mtp'
    drafters = ('mtp', 'dflash')   # (--draft dflash: the base model's DFlash 2 drafter, decode/dflash35.py; one card, Swift 1.5 first)
    tokenizer = 'own'          # (caminus.bpe: the qwen35 pre-tokenizer)
    PREFILL_CHUNK = 2048       # (XT, from empty, 2,048 positions: chunk 2048 771 tok/s, 1024 760; 538 MiB at 2048)
    MTP_K = 3                  # drafts per round of the served MTP drafter
    MTP_T = 4                  # the verify width an MTP session's passes run at (the engine's spec_T)
    # --draft dflash: the drafter's 2 GB on the one card leaves Swift 1.5 ~12-16K of context on a 24 GB XTX with 4 cache
    # slots (32K needs ~23.2 GB of ~22.3 GB room, projected), so a drafted serve is placed from 8K up and max_ctx auto
    # takes the largest that fits; --draft mtp / off hold the 32K floor and more
    DFLASH_MIN_CTX = 8192

    def min_ctx(self, opt):
        # (a default drafter kept by serve.cli.keep_default_draft carries its floor: placement keeps it too)
        if (opt.get('draft') or 'off') == 'dflash': return opt.get('draft_floor') or self.DFLASH_MIN_CTX
        return MIN_CTX

    # the drafter a serve runs when --draft is not given, per (decode module, cards); a module not listed serves plain.
    # One card: the DFlash2 drafter for all three files. Served A/B (docs/dev/benchmarks.md), greedy, every reply the
    # plain reply token for token: short prompts Swift ~2.2x (XTX), ThinkingCap 1.5-1.9x (XT), the plain 27B 1.17-1.37x
    # (XTX); public 10K / 34K prompts ThinkingCap 1.18x / 1.23x (XTX), the plain 27B 1.11x / 1.17x (XT), Swift 1.34x at
    # 10K (XTX). The drafter's 2 GB comes out of the context (min_ctx: 8K floor). Without the drafter file, plain.
    # Two cards (XTX + XT, served A/B with every drafted reply token for token the plain one): DFlash2 for Swift 1.5 and
    # ThinkingCap, 1.13-1.44x on public 10K / 34K prompts and 1.20-1.45x on 11K / 29K agent prompts; the drafter's VRAM
    # comes out of the two-card context. The plain 27B has no two-card form (no k3_q35_gsq entry: plain).
    DRAFT_DEFAULT = {('k3_q35_swift', 1): 'dflash', ('k3_q35_tcap', 1): 'dflash', ('k3_q35_gsq', 1): 'dflash',
                     ('k3_q35_swift', 2): 'dflash', ('k3_q35_tcap', 2): 'dflash'}

    def default_draft(self, opt):
        from . import ggufio
        from .engine.q35_dense import module_of
        try:
            F = ggufio.GGUFFile(opt['model'])
        except Exception:                       # noqa: BLE001 - a file this reader refuses: the server's own checks say why
            return 'off'
        try:
            d = self.DRAFT_DEFAULT.get((module_of(F), self.cards), 'off')
            if d == 'mtp':
                from .engine.q35_mtp import block_of
                if block_of(F) is None: d = 'off'   # (a file without the multi-token-prediction block: plain)
        except Exception:                       # noqa: BLE001 - formats module_of cannot place: plain
            d = 'off'
        finally:
            F.close()
        return d

    def _mtp(self, opt):
        """whether this serve loads the model's MTP drafter (--draft mtp), and its block index"""
        if (opt.get('draft') or 'off') != 'mtp': return None
        from . import ggufio
        from .engine.q35_mtp import block_of
        F = ggufio.GGUFFile(opt['model'])
        try: return block_of(F)
        finally: F.close()

    def _dflash(self, opt):
        """the DFlash 2 drafter's (GGUF path, target layers) when this serve loads it (--draft dflash), else None"""
        if (opt.get('draft') or 'off') != 'dflash': return None
        from . import ggufio
        F = ggufio.GGUFFile(opt['drafter'])
        try: return opt['drafter'], [int(v) for v in F.meta['dflash.target_layers']]
        finally: F.close()

    def _dflash_mib(self, opt, c):
        """MiB the DFlash 2 drafter adds at context c: the drafter, the verify tape with the feature export"""
        from .engine.q35_dense_verify import Dense35Verify
        from .engine.q35_dflash import Dflash35
        path, layers = self._dflash(opt)
        from . import ggufio
        F = ggufio.GGUFFile(opt['model'])
        try: vocab = int(F.T['output.weight'].shape[1])
        finally: F.close()
        return Dflash35.need_mib(path, vocab, self.PREFILL_CHUNK) + Dense35Verify.need_mib(self.MTP_T, max_ctx=c, nfeat=len(layers))

    def auto_need(self, opt, slots, cards=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .engine.q35_dense import Dense35Own
        from .engine.q35_dense_verify import Dense35Verify
        from .engine.q35_mtp import Mtp35
        from .decode.mtp35 import Mtp35Drafter
        from .prefill import q35own
        path = opt['model']
        n_ctx = int(meta(path)['hp'].get('context_length') or 262144)
        il = self._mtp(opt)
        dfl = self._dflash(opt)
        if dfl is not None:
            spec_T, sh = self.MTP_T, q35own.shapes_of(path)
            slot = (Dense35Own.recurrent_mib(path, spec_T) + Dense35Own.kv_pos_mib(path)) * 1.05 + 1.0
            # (+ the checkpoint's drafter window: W + T ring rows of every drafter layer)
            slot += self._dfl_win_mib(dfl[0])
            pf = q35own.buffer_need_mib(sh, self.PREFILL_CHUNK, nfeat=len(dfl[1])) + PREFILL_MODULES_MIB
            need = lambda c: [Dense35Own.projection_mib(path, c, spec_T) + pf + self._dflash_mib(opt, c) + (slots + 2) * slot]
            return n_ctx, [0], slot, need
        spec_T = self.MTP_T if il is not None else 1
        sh = q35own.shapes_of(path)
        slot = (Dense35Own.recurrent_mib(path, spec_T) + Dense35Own.kv_pos_mib(path)) * 1.05 + 1.0   # (a checkpoint: the states + tail rows)
        pf = q35own.buffer_need_mib(sh, self.PREFILL_CHUNK, export_h=il is not None) + PREFILL_MODULES_MIB
        def extra(c):
            if il is None: return 0.0
            return (Dense35Verify.need_mib(self.MTP_T, max_ctx=c) + Mtp35.projection_mib(path, c, k=self.MTP_K)   # (the verify tape + the drafter at this context)
                    + Mtp35Drafter.ring_mib(sh.D))                                # (the serve adapter's h ring + gather workspace)
        need = lambda c: [Dense35Own.projection_mib(path, c, spec_T) + pf + extra(c) + (slots + 2) * slot]   # (the side rows: host memory)
        return n_ctx, [0], slot, need

    @staticmethod
    def _dfl_win_mib(path):
        from . import ggufio
        F = ggufio.GGUFFile(path)
        try:
            hp = lambda k: int(F.meta[f'dflash.{k}'])
            return hp('block_count') * (hp('attention.sliding_window') + 4) * 2 * hp('attention.head_count_kv') * hp('attention.key_length') * 2 / 2 ** 20
        finally: F.close()

    def build(self, opt, log=print, T=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .engine.q35_dense import Dense35Own
        from .engine.q35_dense_verify import Dense35Verify
        from .engine.q35_mtp import Mtp35
        from .decode.mtp35 import Mtp35Drafter
        from .prefill import q35own
        path = opt['model']
        dfl = self._dflash(opt)
        if dfl is not None:
            from .engine.q35_dflash import Dflash35
            from .decode.dflash35 import Dflash35Drafter
            dpath, layers = dfl
            sh, mc = q35own.shapes_of(path), int(opt['max_ctx'])
            extra = q35own.buffer_need_mib(sh, self.PREFILL_CHUNK, nfeat=len(layers)) + PREFILL_MODULES_MIB + self._dflash_mib(opt, mc)
            m = Dense35Own(path, device=0, max_ctx=mc, vram_extra_mib=extra, spec_T=self.MTP_T, feature_layers=layers)
            m.ver = Dense35Verify(m, self.MTP_T, feature_layers=layers)
            d = Dflash35(dpath, m, chunk=self.PREFILL_CHUNK)   # (+ the batched prefill inject's buffers for its chunk)
            pf = q35own.Q35OwnPrefill(m, chunk=self.PREFILL_CHUNK, feature_layers=layers)
            dr = Dflash35Drafter(d)
            log(f'caminus serve: DFlash 2 drafter {os.path.basename(dpath)} (target layers {layers}, verify T {self.MTP_T}): '
                f'+{self._dflash_mib(opt, mc):.0f} MiB' + (f'; context {mc} (the drafter\'s VRAM: --draft mtp or off serve more '
                f'context on this card)' if mc < MIN_CTX else ''), flush=True)
            return m, pf, d, dr
        il = self._mtp(opt)
        spec_T = self.MTP_T if il is not None else 1
        sh = q35own.shapes_of(path)
        extra = q35own.buffer_need_mib(sh, self.PREFILL_CHUNK, export_h=il is not None) + PREFILL_MODULES_MIB
        if il is not None:
            extra += (Dense35Verify.need_mib(self.MTP_T, max_ctx=int(opt['max_ctx'])) + Mtp35.projection_mib(path, int(opt['max_ctx']), k=self.MTP_K)
                      + Mtp35Drafter.ring_mib(sh.D))
        m = Dense35Own(path, device=0, max_ctx=int(opt['max_ctx']), vram_extra_mib=extra, spec_T=spec_T)
        if il is not None:
            m.ver = Dense35Verify(m, self.MTP_T)          # (checked before anything reads it: the session's draft passes)
            m.mtp35 = Mtp35(m, k=self.MTP_K)
        pf = q35own.Q35OwnPrefill(m, chunk=self.PREFILL_CHUNK)
        dr = Mtp35Drafter(m.mtp35) if il is not None else None
        if dr is not None:
            log(f'caminus serve: MTP drafter blk.{il} (k {self.MTP_K}, verify T {self.MTP_T}, spec_T {spec_T}): '
                f'+{Dense35Verify.need_mib(self.MTP_T, max_ctx=int(opt["max_ctx"])) + Mtp35.projection_mib(path, int(opt["max_ctx"]), k=self.MTP_K) + Mtp35Drafter.ring_mib(sh.D):.0f} MiB '
                f'(incl. the adapter\'s h ring {Mtp35Drafter.ring_mib(sh.D):.2f} MiB, checked at its allocation)', flush=True)
        return m, pf, m.mtp35 if dr is not None else None, dr


class Dense35TP(Arch):
    """the dense qwen35 line on Caminus's own code on TWO cards (caminus serve --cards 2 for swift-1.5 / thinkingcap): Dense35TP
    (engine/q35_dense_tp.py: tensor parallelism, the two-card build of the file's module), T = 1, its own two-card prefill
    (caminus.prefill.q35own_tp.Q35OwnPrefillTP). Never picked from a header or the catalog: Dense35.variant(2) names it. The
    feed-forward rows split by the cards' streaming bandwidth (ffn_share: card 0's part; the attention and DeltaNet heads
    split evenly).

    Drafting (--draft mtp / dflash) runs the one-card drafters beside the trunk: the two-card verify tape
    (engine/q35_dense_verify_tp.py) on both cards, the drafter and a complete output head on one card
    (engine/q35_draft_tp.py). The default is DFlash2 for both models (Dense35.DRAFT_DEFAULT, keyed by (module, 2)). A file
    whose decode module has no two-card verify object (k3.Q35_TPV_OF; the plain 27B's k3_q35_gsq has no two-card form at
    all) is refused two-card drafting (draft_refusal); two_card_models names the catalog models that have one, for the
    model picker (it reads no file)."""
    kind, cards, T, engine = 'tp', 2, 1, 'engine.q35_dense_tp:Dense35TP'
    rope_facts = QWEN35_ROPE
    tokenizer = 'own'
    drafter = 'mtp'
    drafters = ('mtp', 'dflash')
    PREFILL_CHUNK = 2048
    MTP_K, MTP_T = Dense35.MTP_K, Dense35.MTP_T
    _mtp, _dflash = Dense35._mtp, Dense35._dflash

    # the catalog models (caminus.models) whose files' decode modules have a two-card build and its verify object
    # (k3.Q35_TP_OF, k3.Q35_TPV_OF: Swift's and ThinkingCap's). A fact about the catalog's files: the picker reads no file,
    # so nothing derives it; a new two-card module adds its model here
    two_card_models = ('swift-1.5', 'thinkingcap')

    def draft_refusal(self, draft, path=None):
        if (draft or 'off') == 'off' or path is None: return None
        from . import ggufio
        from .engine import k3
        from .engine.q35_dense import module_of
        try:
            F = ggufio.GGUFFile(path)
        except Exception:                       # noqa: BLE001 - a file this reader refuses: the engine's own checks say why
            return None
        try:
            one = module_of(F)
        except Exception:                       # noqa: BLE001 - formats module_of cannot place: the engine refuses them
            return None
        finally:
            F.close()
        if k3.Q35_TPV_OF.get(k3.Q35_TP_OF.get(one)) is None:
            return (f'--draft {draft} on two cards: the decode module {one} has no two-card verify object (two-card drafting '
                    f'serves Swift 1.5 and ThinkingCap): --cards 1, or --draft off')
        return None

    @property
    def DRAFT_DEFAULT(self):
        return Dense35.DRAFT_DEFAULT          # (one table, keyed by (module, cards))

    def default_draft(self, opt):
        return Dense35.default_draft(self, opt)

    def split(self, cards):
        """{'ffn_share': card 0's part of the feed-forward rows}: the cards' measured streaming bandwidth (CARD_BW) by label"""
        if not cards: return {'ffn_share': 0.5}
        bw = [CARD_BW.get(c.get('label'), CARD_BW['XT']) for c in cards]
        return {'ffn_share': round(bw[0] / (bw[0] + bw[1]), 3)}

    def _share(self, opt, cards=None):
        sp = self.split(cards) if cards is not None else ((opt.get('placement') or {}).get('split') or {})
        return float(sp.get('ffn_share', 0.5))

    def draft_card(self, opt, cards):
        """the card the drafter runs on, from the placed cards (sysfs only, before HIP starts), or None without a drafter.
        MTP: card 0 (its input is the trunk's final h, replicated; card 0's is read). DFlash: the card with more room left
        after the plain two-card projection at this serve's floor context, from the same context_fit as placement and max_ctx
        auto (draft='off', so no drafter term enters). The KV and states split evenly, so the difference between the cards'
        rooms is the same at every context: the choice does not depend on the context picked later"""
        d = opt.get('draft') or 'off'
        if d == 'off': return None
        if d == 'mtp': return 0
        plan = self.context_fit(dict(opt, draft='off'), cards)
        need = plan.need_mib(min(self.min_ctx(opt), plan.native_ctx))
        left = [r - n for r, n in zip(plan.room_mib, need)]
        return 0 if left[0] >= left[1] else 1

    def placement_extra(self, opt, cards):
        dc = self.draft_card(opt, cards)
        return {} if dc is None else {'draft_card': dc}

    def _resources(self, opt, cards=None):
        """the drafter's part of the projection, one recipe for placement, max_ctx auto and build: (spec_T, the drafter's
        card or None, the DFlash target layers, the DFlash (path, layers) or None, extra(c) -> per card MiB beside the
        engine at context c). extra: the prefill's buffers; with a drafter also the verify tape on both cards and, on the
        drafter's card, the full head copy, the verify pass's exported residuals and the drafter (MTP: its block, KV and
        tapes, the adapter's ring; DFlash: Dflash35.need_mib). cards given (placement, caminus fit): the drafter's card from
        them; else the one the placement stored (opt['placement']['draft_card'], computed once by placement_extra)"""
        from .engine.q35_dense_tp import Dense35TP as E
        from .engine.q35_dense_verify_tp import Dense35VerifyTP
        from .prefill.q35own_tp import buffer_need_tp_mib, tp_shapes
        path, share = opt['model'], self._share(opt, cards)
        mtp, dfl = self._mtp(opt), self._dflash(opt)
        if (opt.get('draft') or 'off') == 'mtp' and mtp is None: raise ValueError('Dense35TP: --draft mtp, but the file has no MTP block')
        drafting = mtp is not None or dfl is not None
        spec_T = self.MTP_T if drafting else 1
        layers = [] if dfl is None else list(dfl[1])
        dc = None
        if drafting:
            dc = self.draft_card(opt, cards) if cards is not None else (opt.get('placement') or {}).get('draft_card')
            if dc not in (0, 1):
                raise ValueError('Dense35TP: no drafter card in the placement (caminus serve stores it: Arch.placement_extra)')
        sh = tp_shapes(path, share)
        pf = [v + PREFILL_MODULES_MIB for v in buffer_need_tp_mib(sh, self.PREFILL_CHUNK, mtp is not None, len(layers), dc or 0)]
        if not drafting:
            return spec_T, dc, layers, dfl, lambda c: pf[:]
        from .engine.q35_draft_tp import head_mib
        head, D = head_mib(path), sh[0].D
        if dfl is not None:
            from . import ggufio
            from .engine.q35_dflash import Dflash35
            F = ggufio.GGUFFile(path)
            try: vocab = int(F.T['output.weight'].shape[1])
            finally: F.close()
            drafter_mib = lambda c: Dflash35.need_mib(dfl[0], vocab, self.PREFILL_CHUNK)
        else:
            from .engine.q35_mtp import Mtp35
            from .decode.mtp35 import Mtp35Drafter
            drafter_mib = lambda c: Mtp35.projection_mib(path, c, k=self.MTP_K) + Mtp35Drafter.ring_mib(D)
        def extra(c):
            out = [v + Dense35VerifyTP.need_mib(spec_T, max_ctx=c) for v in pf]
            out[dc] += head + E.feature_mib(D, spec_T, len(layers)) + drafter_mib(c)
            return out
        return spec_T, dc, layers, dfl, extra

    def ckpt_mib(self, opt, spec_T, dc, dfl):
        """per card, MiB of one prompt-cache checkpoint slot before its margin, as cache/prompt.py Layout lays it out: the
        recurrent states (spec_T tokens' saved inputs), T = spec_T KV tail rows of every attention layer (a drafted
        session's passes are T wide: PromptCache passes T = spec_T to Layout), and on the drafter's card its window
        (W + T rows of each drafter layer: Dflash's sliding window, Mtp35.WINDOW for MTP)"""
        from .engine.q35_dense_tp import Dense35TP as E
        path = opt['model']
        base = E.recurrent_mib(path, spec_T) + spec_T * E.kv_pos_mib(path)
        out = [base, base]
        if dc is not None:
            if dfl is not None: out[dc] += Dense35._dfl_win_mib(dfl[0])
            else:
                from .engine.q35_mtp import Mtp35
                from .prefill.q35own import shapes_of
                g = shapes_of(path)
                out[dc] += Mtp35.window_mib(g.nkv, g.hd)
        return out

    def auto_need(self, opt, slots, cards=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .engine.q35_dense_tp import Dense35TP as E
        path, share = opt['model'], self._share(opt, cards)
        spec_T, dc, layers, dfl, extra = self._resources(opt, cards)
        n_ctx = int(meta(path)['hp'].get('context_length') or 262144)
        per_slot = [v * 1.05 + 1.0 for v in self.ckpt_mib(opt, spec_T, dc, dfl)]   # (a checkpoint per card, with its margin)
        need = lambda c: [v + p + (slots + 2) * s for v, p, s in
                          zip(E.projection_mib(path, c, share, spec_T=spec_T, nfeat=len(layers), feature_card=dc or 0),
                              extra(c), per_slot)]
        return n_ctx, [0, 1], max(per_slot), need

    def build(self, opt, log=print, T=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .engine.q35_dense_tp import Dense35TP as E
        from .prefill.q35own_tp import Q35OwnPrefillTP
        why = self.draft_refusal(opt.get('draft'), opt['model'])
        if why: raise ValueError(why)
        path, share = opt['model'], self._share(opt)
        mc = int(opt['max_ctx'])
        spec_T, dc, layers, dfl, extra = self._resources(opt)
        m = E(path, devices=(0, 1), max_ctx=mc, ffn_share=share, vram_extra_mib=extra(mc), spec_T=spec_T,
              feature_layers=layers, feature_card=dc or 0)
        log(f'caminus serve: two cards: feed-forward rows {m.f0} / {m.ffn - m.f0} (ffn_share {share})', flush=True)
        d = dr = None
        if spec_T > 1:
            from .engine.q35_dense_verify_tp import Dense35VerifyTP
            from .engine.q35_draft_tp import DraftTarget35TP
            m.ver = Dense35VerifyTP(m, spec_T, feature_layers=layers)
            target = DraftTarget35TP(m, dc)
            if dfl is not None:
                from .engine.q35_dflash import Dflash35
                from .decode.dflash35 import Dflash35Drafter
                d = Dflash35(dfl[0], target, chunk=self.PREFILL_CHUNK)
                dr = Dflash35Drafter(d)
            else:
                from .engine.q35_mtp import Mtp35
                from .decode.mtp35 import Mtp35Drafter
                m.mtp35 = d = Mtp35(target, k=self.MTP_K)
                dr = Mtp35Drafter(d)
            log(f'caminus serve: two-card --draft {opt["draft"]} on card {dc}: full head '
                f'{target.mib:.0f} MiB, verify T {spec_T}', flush=True)
        pf = Q35OwnPrefillTP(m, chunk=self.PREFILL_CHUNK, feature_layers=layers)
        return m, pf, d, dr


class Q3MoETP(_Q3MLong, Arch):
    """qwen3moe (Qwen3-30B-A3B) on TWO cards (caminus serve --cards 2 / caminus start qwen3-30b-a3b --cards 2; one card stays
    the model's default): engine/q3moe_tp.py's Q3MoETP (tensor parallel on the two-card build of the lean decode module), T = 1,
    no drafter, its own two-card batched prefill (caminus.prefill.q3m_tp.Q3MPrefillTP). Never picked from a header or the
    catalog: Q3MoE.variant(2) names it. The split: card 0 takes 2 of the 4 kv heads and rows 0 .. 511 of every expert's 768
    (the down projection's K cut must fall on a whole 256-value block, so the only cuts are 512 / 256 and 256 / 512), the head
    rows by the cards' streaming bandwidth (CARD_BW); the router, norms and embedding are on both cards.

    Why 512 on card 0: placement (caminus.device.host.pair) puts the bigger card first, and the bigger card takes the bigger
    expert share, as the Coder's split does; on an XTX + XT that is also the faster card (its 512 rows balance the XT's higher
    fixed cost per segment). On two equal cards (XT + XT, XTX + XTX) the cut is still 512 / 256: no 50 / 50 cut exists on
    256-value blocks, and the memory projection decides whether card 0 holds it. A pair whose card 0 is the smaller card (only
    reachable by bypassing the placement's order) is refused (pair_refusal), not given the 256 / 512 cut: that cut is not
    gated or measured.

    max_ctx auto is capped at the file's context_length (40,960) as on one card, unless the serve is in the long-context
    mode (_Q3MLong: YaRN x4, up to 131,072 positions; F16 holds them on two cards by the projection: about 213K would fit at
    F16 on an XTX + XT). Two cards are slower than one at short context and faster past about 16K positions (engine level,
    docs/dev/benchmarks.md), which is why one card stays the default."""
    kind, cards, T, engine = 'tp', 2, 1, 'engine.q3moe_tp:Q3MoETP'
    rope_facts = QWEN3MOE_ROPE
    tokenizer = 'own'          # (caminus.bpe, as the one-card entry: no gguf-py, no tokenizers on the two-card path either)
    PREFILL_CHUNK = Q3MoE.PREFILL_CHUNK
    EXP_ROWS = 512             # card 0's rows of every expert (= caminus.engine.q3moe_tp.EXP_ROWS; tests/test_arch.py checks it)
    # the catalog models this entry serves on two cards (the model picker reads no file: a fact about the catalog)
    two_card_models = ('qwen3-30b-a3b',)
    kv_formats = ('f16', 'q8')   # (caminus serve --cards 2 --kv q8: Q3MoETP(q8=True) on k3_q3m_lean_tp_kvq8 + Q3MPrefillTP's Q8 path)

    def split(self, cards):
        """{'exp_rows': card 0's rows of every expert, 'vocab_share': card 0's part of the head rows} for the pair in engine
        order (the bigger card first): EXP_ROWS, and the cards' measured streaming bandwidth (CARD_BW) by label"""
        if not cards: return {'exp_rows': self.EXP_ROWS, 'vocab_share': 0.5}
        bw = [CARD_BW.get(c.get('label'), CARD_BW['XT']) for c in cards]
        return {'exp_rows': self.EXP_ROWS, 'vocab_share': round(bw[0] / (bw[0] + bw[1]), 3)}

    def _split(self, opt, cards=None):
        """(exp_rows, vocab_share): from these cards (placement, caminus fit), else the placement's stored split (build, max_ctx
        auto), else the default"""
        sp = self.split(cards) if cards is not None else ((opt.get('placement') or {}).get('split') or self.split(None))
        return int(sp.get('exp_rows', self.EXP_ROWS)), float(sp.get('vocab_share', 0.5))

    def pair_refusal(self, cards):
        if cards and len(cards) >= 2 and int(cards[0].get('total_mib') or 0) < int(cards[1].get('total_mib') or 0):
            from .device import host
            return (f'Qwen3-30B-A3B on two cards puts {self.EXP_ROWS} of each expert\'s 768 rows on card 0, so card 0 must be the '
                    f'bigger card: {" + ".join(host.describe(c) for c in cards)} is in the other order (let caminus pick the '
                    f'cards, or list the bigger card first in HIP_VISIBLE_DEVICES)')
        return None

    def fits(self, opt, cards):
        why = self.pair_refusal(cards)
        if why:
            from .device import host
            raise host.PlacementError(why)
        return super().fits(opt, cards)

    def _prefill_mib(self, path, exp_rows):
        """per card, everything Q3MPrefillTP checks at construction (its buffers, receive rows, code objects and streams)"""
        from .prefill.q3m_tp import need_tp_mib, tp_shapes
        return need_tp_mib(tp_shapes(path, exp_rows), self.PREFILL_CHUNK)

    def auto_need(self, opt, slots, cards=None):
        """per card: the engine's own projection (Q3MoETP.projection_mib, the terms its check counts) + the two-card prefill's
        construction check + the prompt cache's slots + 2 checkpoint slots, each one T = 1 tail row of that card's kv heads in
        every layer (cache/prompt.py Layout: no recurrent state); the side rows are host memory"""
        q8 = self.kv_check(opt)                             # (opt['kv']: caminus serve --kv; 'f16' by default)
        from .engine.q3moe_tp import Q3MoETP as E
        path = opt['model']
        er, vs = self._split(opt, cards)
        n_ctx = self._ctx_cap(opt)                          # (the file's 40,960, or the long-context mode's 131,072)
        yarn = opt.get('rope') == 'yarn'
        pf = self._prefill_mib(path, er)
        slot = [v * 1.05 + 1.0 for v in E.kv_pos_mib(path, q8=q8)]
        need = lambda c: [p + f + (slots + 2) * s for p, f, s in zip(E.projection_mib(path, c, exp_rows=er, vocab_share=vs, q8=q8, yarn=yarn), pf, slot)]
        return n_ctx, [0, 1], max(slot), need

    def build(self, opt, log=print, T=None):
        q8 = self.kv_check(opt)
        from .engine.q3moe_tp import Q3MoETP as E
        from .prefill.q3m_tp import Q3MPrefillTP
        path = opt['model']
        er, vs = self._split(opt)
        # (the prefill's construction check counted in the engine's own, before anything is read)
        m = E(path, devices=(0, 1), max_ctx=int(opt['max_ctx']), exp_rows=er, vocab_share=vs,
              vram_extra_mib=self._prefill_mib(path, er), q8=q8, rope=self.rope_mode(opt))
        log(f'caminus serve: two cards: expert rows {m.exp_rows[0]} / {m.exp_rows[1]}, kv heads {m.kvc[0]} / {m.kvc[1]}, '
            f'head rows {m.vh} / {m.vocab - m.vh} (vocab_share {vs})', flush=True)
        return m, Q3MPrefillTP(m, chunk=self.PREFILL_CHUNK), None, None


class TwoCardTP(Arch):
    kind, cards, T, engine = 'tp', 2, 1, 'engine.qwen3next_tp:Qwen3NextTP'
    kv_fixed = 'q8_0'          # (Qwen3NextTP's q8_0 cache: q8=True, its default)

    def auto_need(self, opt, slots, cards=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        sh = tp_shape(opt['model'])
        slot = (sh['state'] + sh['kv_pos']) / 2 ** 20 * 1.05 + 1.0
        need = lambda c: [sh['need_mib'](c) + (slots + 2) * slot] * 2   # (the side rows: host memory)
        return sh['n_ctx'], [0, 1], slot, need

    def build(self, opt, log=print, T=None):
        self.kv_check(opt)                                  # (opt['kv'] = 'q8': refused, never built as F16)
        from .engine.qwen3next_tp import Qwen3NextTP
        from .prefill.kpass import KPassOnly
        # (its own check, before anything is read: qwen3next_tp.projection, the one auto_need reads)
        m = Qwen3NextTP(opt['model'], max_ctx=int(opt['max_ctx']), ntok=1, vram_check=True)
        return m, KPassOnly(m), None, None


ARCHS = {a.name: a for a in (Coder('qwen4exp', 'k3-coder'), OneCard('qwen35moe', 'qwen3.6'), OneCard('qwen35', 'qwen3.5'),
                             Q3MoE('qwen3moe', 'qwen3-30b-a3b'), Dense35('qwen35-own', 'swift-1.5'),
                             TwoCardTP('qwen3next', 'qwen3-coder-next'), Dense35TP('qwen35-own-tp', 'swift-1.5'),
                             Q3MoETP('qwen3moe-tp', 'qwen3-30b-a3b'))}
# a one-card architecture's two-card form: Arch.variant(2), caminus serve --cards 2. Offered, never the default: a model's own
# card count (its catalog entry, Arch.cards) is what a serve without --cards runs
TWO_CARD = {'qwen35-own': 'qwen35-own-tp', 'qwen3moe': 'qwen3moe-tp'}
CODER = ARCHS['qwen4exp']


# architectures caminus serve refuses while their models are paused (caminus.models.PAUSED; CAMINUS_ALLOW_PAUSED=1 serves them)
PAUSED = {'qwen35moe': 'Qwen3.6-35B-A3B', 'qwen3next': 'Qwen3-Coder-Next', 'qwen35': 'AREX-2'}


def paused(A):
    """the refusal when A is a paused architecture's entry, else None"""
    from . import models
    if A.name not in PAUSED or models.allow_paused(): return None
    return (f'{PAUSED[A.name]} ({A.name}) is paused: not in this release (it is not measured on the current code); '
            'CAMINUS_ALLOW_PAUSED=1 serves it anyway')


def lookup(name):
    """the entry for an architecture name (any other: the Coder's, whose engine refuses what it cannot load)"""
    return ARCHS.get(name, CODER)


def of(path):
    """the entry for a GGUF: its catalog model's (caminus.models.model_of: the same file name and size), else its header's
    general.architecture (meta: read once per file). The catalog decides where two models share a header architecture
    and run on different engines: Swift 1.5 (Dense35, own code) and AREX-2 (OneCard) are both qwen35 files"""
    from . import models
    m = models.model_of(path)
    if m is not None and m.arch in ARCHS: return ARCHS[m.arch]
    return lookup(meta(path)['arch'])


def default_alias(path):
    """the model id a server answers to for this GGUF when --model-alias is not given: the catalog's alias when the file is a
    catalog model's (caminus.models.model_of: its first file, same name and size; so a bare caminus serve and caminus start
    give AREX-2 the same id), else its architecture's"""
    from . import models
    m = models.model_of(path)
    return m.alias if m is not None else of(path).alias
