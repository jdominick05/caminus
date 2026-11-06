"""The model files Caminus serves: where each comes from, its size and sha256 (docs/RELEASE-PLAN.md, `caminus download`).

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
from collections import namedtuple

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
)
BY_KEY = {it.key: it for it in ITEMS}


def local_rel(it):
    """an item's path under the model directory"""
    return it.repo.split('/', 1)[1] + '/' + it.path
