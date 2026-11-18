"""The model files Caminus serves: where each comes from, its size and sha256 (docs/dev/packaging.md, `caminus download`).

Nothing here is redistributed: the files are downloaded from their public Hugging Face repositories at the pinned revisions.
The local layout under the model directory (caminus.config.model_dir) is `<repository name>/<path in the repository>`, so the
Coder's two shards share a directory (the loader finds shard 2 from shard 1's name).

Sources of the numbers (2026-11-06):
  * the Coder's shards: the revision and the LFS sha256 Hugging Face served when this machine downloaded them (the
    `hf download --local-dir` metadata beside the files); sizes from the files. Hashing the local files (caminus doctor
    --verify, 11-06) gave the same sha256 for both shards.
  * the DFlash2 Q8_0 drafter: this machine's file (sha256 and size computed from it).
  * all three checked against Hugging Face's API on 2026-11-06: each repository serves the file at its pinned revision, under
    this path, with this size and this LFS sha256; both repositories are public and not gated (the drafter: Apache-2.0).
`caminus download` still refuses any mismatch (size or sha256) and keeps nothing it could not verify.
"""
import os
from collections import namedtuple

from . import arch as _arch

Item = namedtuple('Item', 'key role repo revision path size sha256 source')

ITEMS = (
    Item('coder-1', 'coder', 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF', 'bba88856f8c3c0c9e40e45f82d05e228b814ac43',
         'IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf', 29608446496,
         'bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211', 'hf-lfs'),
    Item('coder-2', 'coder-ple', 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF', 'bba88856f8c3c0c9e40e45f82d05e228b814ac43',
         'IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf', 28800138432,
         '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113', 'hf-lfs'),
    Item('dflash', 'dflash', 'incoai/Qwen3.8-Flash-Next-DFlash2-GGUF', '4f5a39be8f0a9e9a573ac9824251830a8f4096d9',
         'Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf', 652101024,
         '541f4d6ee230747a34d4a2c6890e6e08cc8b19c461226dc1829701a544e46e8a', 'hf-lfs'),
    # the one-card and Qwen3-Coder-Next files (11-07): revision and LFS sha256 from this machine's `hf download` metadata,
    # each checked against Hugging Face's API (paths-info at the pinned revision: path, size, LFS sha256) on 2026-11-07
    Item('qwen3.6', 'qwen3.6', 'chtisgit/Qwen3.6-35B-A3B-2Bit-GSQ-hybrid-GGUF', '34a912bde355affe80eb1af654475a6fe9feff5f',
         'Qwen3.6-35B-A3B-GSQ-hybrid.gguf', 12217776832,
         '8e50912f5d0703401ca21b09deb2ee8a84d03536ade6fb44815fcb40a0b03596', 'hf-lfs'),
    Item('qwen3.6-hauhau', 'qwen3.6-hauhau', 'HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive',
         'f12a584fecbeb5f20001130d8ecd66c9327ae685', 'Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q3_K_P.gguf', 19023337600,
         '18508f37fc7787f260df7ac1a3934b15dd9496d050b2d4b864241b493588fe95', 'hf-lfs'),
    Item('arex-2', 'arex-2', 'bartowski/BAAI_AREX-2-GGUF', 'b5817eb07865e9619f1516140472f61d66268922',
         'BAAI_AREX-2-Q4_K_M.gguf', 17203416160,
         '45453eed75a1c32b0ee203c1864ed388b46868305fb523d88f4484faa0837b4a', 'hf-lfs'),
    Item('qwen3-coder-next', 'qwen3-coder-next', 'unsloth/Qwen3-Coder-Next-GGUF', 'ce09c67b53bc8739eef83fe67b2f5d293c270632',
         'Qwen3-Coder-Next-UD-IQ3_S.gguf', 29690687488,
         'fddff90a6e3b9f0aee320a2710024c458cd5f3b07f0b7bdfb33fcf23c08e5389', 'hf-lfs'),
)
# Qwen3-30B-A3B (11-10): the official Q4_K_M; revision, size and LFS sha256 from Hugging Face's API on 2026-11-10 (the
# local file's sha256 computed and equal)
ITEMS = ITEMS + (
    Item('qwen3-30b-a3b', 'qwen3-30b-a3b', 'Qwen/Qwen3-30B-A3B-GGUF', 'e4d4bafdfb96a411a163846265362aceb0b9c63a',
         'Qwen3-30B-A3B-Q4_K_M.gguf', 18556685824,
         '0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5', 'hf-lfs'),
)
# Two dense Qwen3.8-27B finetunes and the DFlash2 drafter of their base (11-11): revision, size and LFS sha256 from Hugging
# Face's API on 2026-11-11 (each local file's sha256 computed and equal). CATALOG names Swift 1.5 and ThinkingCap (their
# own-code engines serve: arch.Dense35, the module from the file's formats); the drafter is part of both models' files
# (`caminus pull` fetches it once for both; `caminus start` passes it; `--draft dflash` uses it, the default where the file is present and the context fits).
# Licences: Swift 1.5 the Swift Open License 1.0, ThinkingCap PolyForm Small Business 1.0.0, the drafter Apache-2.0.
ITEMS = ITEMS + (
    Item('swift-1.5', 'swift-1.5', 'ukisai/Swift-1.5-Qwen3.8-27B-GGUF', '14bfe4b42be4a925d98816db830155f476c605e7',
         'Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf', 17442399936,
         '2ebba0ff1e63c1ac3fadd4e83efcea189f47f33ec72c91877af94de6ebe30590', 'hf-lfs'),
    Item('thinkingcap', 'thinkingcap', 'bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw',
         '67c3324641f39f169e1cbc8280f9748e890f3046', 'ThinkingCap-Qwen3.8-27B-RCO-4.2bpw.gguf', 14309041504,
         'bfb7712f21ad131d64321c4f9ce42417be0fe7a52bd2ff69d263012ddf631654', 'hf-lfs'),
    Item('qwen3.8-27b', 'qwen3.8-27b', 'ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF', 'd562806dbafae37109975e970aae91b43e73b440',
         'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf', 12120016960,
         '58fd826723939933dc86f45b7fe04545cbc2de1c70f6fe2cdd3858c87a98c12f', 'hf-lfs'),
    Item('dflash2-27b', 'dflash', 'incoai/Qwen3.8-27B-DFlash2-GGUF', '144eb06583b4fb0b6e54bc814b16a206a240f4d2',
         'Qwen3.8-27B-DFlash2-Q8_0.gguf', 2056414816,
         '651a1b2930c9825080480f79cf38057e681c9b175d580a357c14f67cf77ccf49', 'hf-lfs'),
)
BY_KEY = {it.key: it for it in ITEMS}

# The named models `caminus models` / `caminus pull` / `caminus start` know: the files (ITEMS keys; the first is what caminus
# serve --model takes), the GGUF architecture, the cards it needs (from the architecture registry, caminus.arch: 1 = any one
# card it fits, by caminus serve's max_ctx auto; 2 = both), the model id the server answers to (a server finds it here by its
# file: arch.default_alias), and the measured speeds (docs: decode and prefill tok/s, the cards alone).
Model = namedtuple('Model', 'name title arch cards items alias speed notes')


def _model(name, title, arch, items, alias, speed, notes):
    return Model(name, title, arch, _arch.lookup(arch).cards, items, alias, speed, notes)


ALL = (
    _model('coder', 'Qwen3.8-Flash-Next Coder IQ1_M + DFlash2', 'qwen4exp', ('coder-1', 'coder-2', 'dflash'), 'k3-coder',
           '~104 tok/s decode, 123-203 with DFlash; prefill ~2,800 tok/s (both cards)', '262K context'),
    _model('qwen3.6', 'Qwen3.6-35B-A3B GSQ-hybrid (2-bit)', 'qwen35moe', ('qwen3.6',), 'qwen3.6',
           '175 tok/s decode, 5,227 tok/s prefill 2K (one XTX)', 'fits either card at 262K'),
    _model('qwen3.6-hauhau', 'Qwen3.6-35B-A3B HauhauCS Q3_K_P', 'qwen35moe', ('qwen3.6-hauhau',), 'qwen3.6',
           '192 tok/s decode, 4,136 tok/s prefill 2K (one XTX)', 'the 24 GB card only'),
    _model('arex-2', 'AREX-2 27B dense Q4_K_M', 'qwen35', ('arex-2',), 'arex-2',
           '41 tok/s decode, 928 tok/s prefill 2K (one XTX)', 'the 24 GB card only; ~119K context'),
    _model('qwen3-coder-next', 'Qwen3-Coder-Next 80B-A3B IQ3_S', 'qwen3next', ('qwen3-coder-next',), 'qwen3-coder-next',
           '136 tok/s decode; prompts read at ~135 tok/s (no batched prefill yet)', '262K context, both cards'),
    _model('swift-1.5', 'Swift 1.5 (Qwen3.8 27B dense) Q4_K_M (own code end to end)', 'qwen35-own', ('swift-1.5', 'dflash2-27b'), 'swift-1.5',
           '42 tok/s decode on an XTX (short context); prefill 771 tok/s at 2K (XT)', 'the 24 GB card only, or two cards; --draft mtp (its own MTP block)'),
    _model('thinkingcap', 'ThinkingCap (Qwen3.8 27B dense) RCO 4.2 bpw (own code end to end)', 'qwen35-own', ('thinkingcap', 'dflash2-27b'),
           'thinkingcap', '47 tok/s decode on an XTX, 39 on an XT (short context); prefill 942 tok/s at 2K (XTX)',
           'fits either card (an XT: ~53K context), or two; --draft dflash: its base model\'s DFlash2 drafter'),
    _model('qwen3.8-27b', 'Qwen3.8 27B dense GSQ-RCO IQ3_S (own code end to end)', 'qwen35-own', ('qwen3.8-27b', 'dflash2-27b'),
           'qwen3.8-27b', '55 tok/s decode on an XTX (short context); --draft dflash 1.2-1.4x, --draft mtp 1.1-1.3x',
           'the plain Qwen3.8-27B (ISTA-DASLab GSQ-RCO, Apache-2.0); fits either card'),
    _model('qwen3-30b-a3b', 'Qwen3-30B-A3B Q4_K_M (own code end to end)', 'qwen3moe', ('qwen3-30b-a3b',), 'qwen3-30b-a3b',
           '248 tok/s decode at short context, 213 at 4K, 111 at 40.9K; prefill 6,432 tok/s at 4K (one XTX, the published row)', '40,960 context: the 24 GB card only, or two cards (--cards 2; one card stays the default)'),
)
# Each served model's recommended sampling for agentic work (thinking on), from its model card: what caminus serve uses as a
# request's defaults (caminus.serve.sampler.model_defaults: these first, then the GGUF's general.sampling.* where a field is
# None here, then SamplingParams' own defaults; a request's own fields still win). None = the card does not state it.
# max_output: the card's recommended output length (reasoning included), a hint /props reports; the server never cuts a reply
# there by itself (max_tokens stays the client's, clamped to the context's room). nothink: the card's separate set for answers
# without thinking (a Sampling, or None: the same set either way), used where the request's profile turns thinking off (the
# -nothink id, enable_thinking false). A greedy request (temperature 0) takes none of these penalties unless it sends its own
# (caminus.serve.sampler.greedy_defaults). Read on 2026-11-16 at these revisions.
Sampling = namedtuple('Sampling', 'temperature top_p top_k min_p presence_penalty repeat_penalty max_output nothink',
                      defaults=(None,))
_QWEN38 = Sampling(1.0, 0.95, 20, 0.0, 0.0, 1.0, 131072 + 262144)   # Thinking Mode; 131,072 answer + 262,144 reasoning
SAMPLING = {
    # https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md
    # ("Thinking Mode: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0" and
    # its output length section); the GSQ-RCO GGUF card states none, its header carries 1.0 / 20 / 0.95
    'coder': _QWEN38,
    # https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/README.md (the same Thinking Mode
    # line and output length); the GSQ-RCO GGUF card states none, its header carries 1.0 / 20 / 0.95 / min_p 0
    'qwen3.8-27b': _QWEN38,
    # https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF/blob/14bfe4b42be4a925d98816db830155f476c605e7/README.md (its
    # evaluation sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1, the base
    # Qwen3.8-27B card's Thinking Mode set; the output length from that base card)
    'swift-1.5': _QWEN38,
    # https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw/blob/67c3324641f39f169e1cbc8280f9748e890f3046/README.md
    # ("Use the sampling settings from the main model card (temperature 1.0, top_p 0.95, top_k 20, min_p 0.0). Greedy
    # decoding can loop."); the penalties and the output length from the base Qwen3.8-27B card it defers to
    'thinkingcap': _QWEN38,
    # https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/blob/e4d4bafdfb96a411a163846265362aceb0b9c63a/README.md (thinking mode:
    # temperature 0.6, top_p 0.95, top_k 20, min_p 0; "We recommend setting presence_penalty to 1.5 for quantized models to
    # suppress repetitive outputs"; "an output length of 32,768 tokens for most queries"); also the base card
    # https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md. The GGUF carries no
    # general.sampling.* keys, so before this table it was served at temperature 1.0 and presence 0. No repetition penalty
    # is stated (None: the GGUF's, else off). nothink: the same section's non-thinking set ("For non-thinking mode
    # (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, MinP=0, and PresencePenalty=1.5").
    'qwen3-30b-a3b': Sampling(0.6, 0.95, 20, 0.0, 1.5, None, 32768,
                              nothink=Sampling(0.7, 0.8, 20, 0.0, 1.5, None, 32768)),
}


def sampling_of(path):
    """the recommended sampling (Sampling) of the catalog model whose first file this is (model_of: same name and size), or
    None: a file outside the catalog, or a model with no entry (the paused ones)"""
    m = model_of(path)
    return SAMPLING.get(m.name) if m is not None else None


# Paused models: in the tree (their engines, kernels and files stay) but not offered: `caminus models`, `caminus pull`,
# `caminus start` and the panel leave them out, and caminus serve refuses their architectures (caminus.arch.PAUSED), until
# they are measured again on the current code. CAMINUS_ALLOW_PAUSED=1 offers them anyway (development and gates).
PAUSED = {
    'qwen3.6': 'Qwen3.6 is paused: not in this release (it is not measured on the current code)',
    'qwen3.6-hauhau': 'Qwen3.6 is paused: not in this release (it is not measured on the current code)',
    'qwen3-coder-next': 'Qwen3-Coder-Next is paused: not in this release (it is not measured on the current code)',
    'arex-2': 'AREX-2 is paused: not in this release (the 0.3.0 set is the Coder, Qwen3-30B-A3B and the three Qwen3.8 27B models)',
}


# The model to start with on each setup: the README's "Get started" table (tests/docs/test_docs.py compares it with this row
# by row) and what `caminus wizard` preselects. A plain literal, so the docs check reads it without importing the package.
# cards: the setup's card classes, the bigger first ('XTX': a 24 GB card, 'XT': a 20 GB card; setup_of); runs: the catalog
# models that run there, a two-card form as 'name:2' (the wizard's list before the files are on disk: the real projection
# runs at caminus start); row / what / start_text: the README cells.
SETUPS = (
    {'cards': ('XT',), 'runs': ('thinkingcap', 'qwen3.8-27b'), 'start': 'thinkingcap',
     'row': 'One XT', 'what': 'ThinkingCap, the plain Qwen3.8 27B', 'start_text': 'ThinkingCap'},
    {'cards': ('XTX',), 'runs': ('swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b'), 'start': 'swift-1.5',
     'row': 'One XTX', 'what': 'Those two, plus Qwen3-30B-A3B and Swift 1.5', 'start_text': 'Swift 1.5'},
    {'cards': ('XTX', 'XT'), 'runs': ('coder', 'swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b', 'swift-1.5:2',
                                     'thinkingcap:2', 'qwen3-30b-a3b:2'), 'start': 'coder',
     'row': 'XTX + XT', 'what': 'Everything, including the Coder and the two-card forms', 'start_text': 'the Coder (both cards)'},
    {'cards': ('XTX', 'XTX'), 'runs': ('coder', 'swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b', 'swift-1.5:2',
                                      'thinkingcap:2', 'qwen3-30b-a3b:2'), 'start': 'coder',
     'row': 'Two XTX', 'what': 'The same as XTX + XT. Untested: reports are welcome.', 'start_text': 'the Coder (both cards)'},
    {'cards': ('XT', 'XT'), 'runs': ('thinkingcap', 'qwen3.8-27b', 'swift-1.5:2', 'thinkingcap:2', 'qwen3-30b-a3b:2'),
     'start': 'thinkingcap', 'row': 'Two XT', 'what': 'Two-card Swift 1.5, ThinkingCap or Qwen3-30B-A3B, or two one-card models '
     'side by side; no Coder. Untested: reports are welcome.', 'start_text': 'ThinkingCap'},
)
CLASS_MIB = (('XTX', 22 * 1024), ('XT', 19 * 1024))   # a card's class by its VRAM: 24,560 MiB the XTX, 20,464 the XT


def card_class(total_mib):
    """'XTX' (a 24 GB card or bigger), 'XT' (a 20 GB card) or None (smaller: no setup row was measured for it)"""
    return next((c for c, lo in CLASS_MIB if (total_mib or 0) >= lo), None)


def setup_of(cards):
    """the SETUPS row of these usable cards (caminus.device.host dicts in placement order, the biggest first: host.ranked),
    by their classes: the first two with a class, else the first one; None when no row matches (no card of 20 GB or more).
    An untested gfx1100 card counts by its VRAM (a 48 GB W7900 as a 24 GB card)"""
    cls = [c for c in (card_class(g['total_mib']) for g in cards) if c]
    for key in (tuple(cls[:2]), tuple(cls[:1])):
        row = next((s for s in SETUPS if s['cards'] == key), None)
        if row is not None:
            return row
    return None


def setup_here(gs):
    """(row, why) for a host's GPUs (caminus.device.host.gpus): setup_of the pair host.pair allows, else of the first
    usable card alone (a second card on too slow a link, or one card); why: the pair's refusal when two or more usable
    cards make no pair, else None. caminus wizard and Ember's default brain both read it"""
    from .device import host
    cards = host.ranked(gs)
    try:
        return setup_of(host.pair(gs)), None
    except host.PlacementError as e:
        return setup_of(cards[:1]), (str(e) if len(cards) >= 2 else None)


def allow_paused():
    """CAMINUS_ALLOW_PAUSED=1: paused models and architectures are offered and served anyway"""
    return os.environ.get('CAMINUS_ALLOW_PAUSED') == '1'


CATALOG = tuple(m for m in ALL if allow_paused() or m.name not in PAUSED)
BY_NAME = {m.name: m for m in CATALOG}


def paused(name):
    """the refusal for a paused model's name (None: not paused, or CAMINUS_ALLOW_PAUSED=1)"""
    return None if allow_paused() else PAUSED.get(name)


def model_of(path):
    """the catalog model whose first file this is (the same file name and size: what caminus start finds and passes as
    --model), or None"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    base = os.path.basename(path)
    for m in ALL:
        it = BY_KEY[m.items[0]]
        if os.path.basename(it.path) == base and it.size == size:
            return m
    return None


def drafter_item(m):
    """the ITEMS key of the drafter file m's architecture loads (its role is a drafter's name: 'dflash'; the architecture's
    own drafter or one of its optional ones, Arch.drafters: the dense 27B line's --draft dflash), or None"""
    A = _arch.lookup(m.arch)
    ds = {d for d in (A.drafter, *getattr(A, 'drafters', ())) if d}
    return next((k for k in m.items if BY_KEY[k].role in ds), None)


def local_rel(it):
    """an item's path under the model directory"""
    return it.repo.split('/', 1)[1] + '/' + it.path
