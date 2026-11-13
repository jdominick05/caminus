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
# (`caminus pull` fetches it once for both; `caminus start` passes it, and `--draft dflash` uses it: the default stays off).
# Licences: Swift 1.5 the Swift Open License 1.0, ThinkingCap PolyForm Small Business 1.0.0, the drafter Apache-2.0.
ITEMS = ITEMS + (
    Item('swift-1.5', 'swift-1.5', 'ukisai/Swift-1.5-Qwen3.8-27B-GGUF', '14bfe4b42be4a925d98816db830155f476c605e7',
         'Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf', 17442399936,
         '2ebba0ff1e63c1ac3fadd4e83efcea189f47f33ec72c91877af94de6ebe30590', 'hf-lfs'),
    Item('thinkingcap', 'thinkingcap', 'bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw',
         '67c3324641f39f169e1cbc8280f9748e890f3046', 'ThinkingCap-Qwen3.8-27B-RCO-4.2bpw.gguf', 14309041504,
         'bfb7712f21ad131d64321c4f9ce42417be0fe7a52bd2ff69d263012ddf631654', 'hf-lfs'),
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
           '42 tok/s decode on an XTX (short context); prefill 771 tok/s at 2K (XT)', 'the 24 GB card only; --draft mtp (its own MTP block, one card)'),
    _model('thinkingcap', 'ThinkingCap (Qwen3.8 27B dense) RCO 4.2 bpw (own code end to end)', 'qwen35-own', ('thinkingcap', 'dflash2-27b'),
           'thinkingcap', '47 tok/s decode on an XTX, 39 on an XT (short context); prefill 942 tok/s at 2K (XTX)',
           'fits either card (an XT: ~53K context); no drafter yet (its MTP block is Q6_K: the drafter module runs Q4_0)'),
    _model('qwen3-30b-a3b', 'Qwen3-30B-A3B Q4_K_M (own code end to end)', 'qwen3moe', ('qwen3-30b-a3b',), 'qwen3-30b-a3b',
           '248 tok/s decode at short context, 213 at 4K, 111 at 40.9K; prefill 6,493 tok/s at 4K (one XTX)', '40,960 context: the 24 GB card only'),
)
# Paused models: in the tree (their engines, kernels and files stay) but not offered: `caminus models`, `caminus pull`,
# `caminus start` and the panel leave them out, and caminus serve refuses their architectures (caminus.arch.PAUSED), until
# they are measured again on the current code. CAMINUS_ALLOW_PAUSED=1 offers them anyway (development and gates).
PAUSED = {
    'qwen3.6': 'Qwen3.6 is paused: not in this release (it is not measured on the current code)',
    'qwen3.6-hauhau': 'Qwen3.6 is paused: not in this release (it is not measured on the current code)',
    'qwen3-coder-next': 'Qwen3-Coder-Next is paused: not in this release (it is not measured on the current code)',
}


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
