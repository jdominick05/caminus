"""Tokenizer metadata from a GGUF file: tokens, token types, merges, chat template and the special ids.

Only the key/value header is parsed (gguf-py's GGUFReader memory-maps the file; no tensor data is touched).
gguf-py is the package's vendored reader (caminus.config.gguf_py: CAMINUS_GGUF_PY or K3_GGUF_PY name another tree)."""
import os
import sys
from dataclasses import dataclass, field

from .. import config as _config

GGUF_PY = _config.gguf_py()

# llama.cpp token types (gguf TokenType)
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


@dataclass
class Vocab:
    tokens: list            # id -> token text (byte-level unicode for NORMAL tokens, raw text for the rest)
    types: list             # id -> llama.cpp token type
    merges: list            # "a b" strings, rank order
    pre: str                # tokenizer.ggml.pre (e.g. 'qwen35')
    model: str              # tokenizer.ggml.model ('gpt2')
    chat_template: str = ''
    eos_id: int = -1
    bos_id: int = -1
    pad_id: int = -1
    add_bos: bool = False
    meta: dict = field(default_factory=dict)


def _field_str(f, i):
    return bytes(f.parts[f.data[i]]).decode('utf-8')


def load_gguf_vocab(path):
    if GGUF_PY not in sys.path:
        sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader                      # noqa: E402  (gguf-py, see GGUF_PY)
    r = GGUFReader(path, 'r')
    fl = r.fields
    tf, ty, mf = fl['tokenizer.ggml.tokens'], fl['tokenizer.ggml.token_type'], fl['tokenizer.ggml.merges']
    tokens = [_field_str(tf, i) for i in range(len(tf.data))]
    types = [int(ty.parts[ty.data[i]][0]) for i in range(len(ty.data))]
    merges = [_field_str(mf, i) for i in range(len(mf.data))]

    def get(k, d=None):
        return fl[k].contents() if k in fl else d

    v = Vocab(tokens=tokens, types=types, merges=merges, pre=get('tokenizer.ggml.pre', ''),
              model=get('tokenizer.ggml.model', ''), chat_template=get('tokenizer.chat_template', '') or '',
              eos_id=int(get('tokenizer.ggml.eos_token_id', -1)), bos_id=int(get('tokenizer.ggml.bos_token_id', -1)),
              pad_id=int(get('tokenizer.ggml.padding_token_id', -1)),
              add_bos=bool(get('tokenizer.ggml.add_bos_token', False)))
    for k in fl:
        if k in ('general.name', 'general.architecture') or k.startswith('general.sampling.'):
            v.meta[k] = fl[k].contents()          # general.sampling.* = the model's default sampler settings
    return v
