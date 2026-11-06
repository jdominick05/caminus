"""Tokenizer and streaming detokenizer for the K3 server, built from the GGUF's own vocab and merges.

Encoding follows llama.cpp (llama-vocab.cpp) so that ids match `llama-tokenize` and llama-server:
  * special-token partition first: USER_DEFINED tokens (<think>, <tool_call>, ...) are always split out, CONTROL
    tokens (<|im_start|>, <|im_end|>, ...) only when parse_special is true (tokenizer_st_partition); UNUSED tokens
    ([PAD248077] ...) never are;
  * each text fragment is BPE-encoded with the qwen35 pre-tokenizer regex and GPT-2 byte-level mapping;
  * no Unicode normalization (llama.cpp's BPE path has none; HF's tokenizer.json applies NFC: the two differ only
    on non-NFC input, see `nfc`).
The BPE runs in HF `tokenizers` (Rust, releases the GIL); the special-token split is a Python regex.

Literal spans: the chat template output is passed with literal-tag marks (template.py: MARK/ESC). encode_prompt()
splits only at real special-token text, then un-marks each fragment and encodes it as plain text, so a literal
"<tool_call>" or "<|im_end|>" inside a tool result or user text becomes ordinary BPE tokens, never a special id."""
import codecs
import re

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from .vocab import CONTROL, NORMAL, UNUSED, USER_DEFINED, load_gguf_vocab

# llama.cpp's qwen35 pre-tokenizer (llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35); the tokenizer.json original has
# (?i:'s|'t|'re|'ve|'m|'ll|'d) for the first group, which is the same set.
QWEN35_PRE = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
              r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PRE_REGEX = {'qwen35': QWEN35_PRE}

# literal-tag marks (template.py writes them, encode_prompt reads them): MARK replaces the '<' of a literal special
# token text, ESC escapes MARK/ESC characters that were already in the input. Both are Unicode private-use.
MARK = ''
ESC = ''
_UNMARK_RE = re.compile('([])|')


def unmark(s):
    """inverse of template.mark_text: MARK -> '<', ESC x -> x"""
    if MARK not in s and ESC not in s:
        return s
    return _UNMARK_RE.sub(lambda m: m.group(1) or '<', s)


def bytes_to_unicode():
    """GPT-2 byte-level map: byte -> printable unicode char"""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class QwenTokenizer:
    def __init__(self, vocab, nfc=False):
        if vocab.model != 'gpt2' or vocab.pre not in PRE_REGEX:
            raise ValueError(f'unsupported tokenizer {vocab.model}/{vocab.pre}')
        self.vocab = vocab
        self.n_vocab = len(vocab.tokens)
        self.eos_id, self.bos_id = vocab.eos_id, vocab.bos_id
        bpe_vocab = {t: i for i, (t, ty) in enumerate(zip(vocab.tokens, vocab.types)) if ty == NORMAL}
        merges = [tuple(m.split(' ')) for m in vocab.merges]
        tk = Tokenizer(models.BPE(vocab=bpe_vocab, merges=merges, ignore_merges=False, fuse_unk=False,
                                  byte_fallback=False))
        if nfc:
            tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(PRE_REGEX[vocab.pre]), behavior='isolated', invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
        tk.decoder = decoders.ByteLevel()
        self._bpe = tk
        # special tokens: llama.cpp's cache_special_tokens = CONTROL | USER_DEFINED | UNKNOWN (minus "unused" control)
        self.control = {t: i for i, (t, ty) in enumerate(zip(vocab.tokens, vocab.types))
                        if ty == CONTROL and 'unused' not in t}
        self.user_defined = {t: i for i, (t, ty) in enumerate(zip(vocab.tokens, vocab.types)) if ty == USER_DEFINED}
        self.special = {**self.control, **self.user_defined}         # text -> id
        self.special_ids = set(self.special.values())
        self.unused_ids = {i for i, ty in enumerate(vocab.types) if ty == UNUSED}

        def alt(texts):
            return re.compile('|'.join(re.escape(t) for t in sorted(texts, key=len, reverse=True)))
        self._re_all = alt(self.special)
        self._re_user = alt(self.user_defined) if self.user_defined else None
        # id -> bytes for detokenization
        b2u = bytes_to_unicode()
        u2b = {u: b for b, u in b2u.items()}
        self._bytes = []
        for t, ty in zip(vocab.tokens, vocab.types):
            if ty == NORMAL:
                self._bytes.append(bytes(u2b[c] for c in t))
            elif ty == UNUSED:
                self._bytes.append(b'')                  # llama.cpp token_to_piece prints nothing for UNUSED
            else:
                self._bytes.append(t.encode('utf-8'))

    @classmethod
    def from_gguf(cls, path, **kw):
        return cls(load_gguf_vocab(path), **kw)

    # ------------------------------------------------------------------------------------------------ encode
    def encode_plain(self, text):
        """BPE only: no special-token matching at all (the text is ordinary text)"""
        if not text:
            return []
        return self._bpe.encode(text, add_special_tokens=False).ids

    def _split(self, text, parse_special):
        """[(is_special, str_or_id)] in order, llama.cpp tokenizer_st_partition semantics"""
        rx = self._re_all if parse_special else self._re_user
        out, pos = [], 0
        if rx is not None:
            for m in rx.finditer(text):
                if m.start() > pos:
                    out.append((False, text[pos:m.start()]))
                out.append((True, self.special[m.group(0)]))
                pos = m.end()
        if pos < len(text):
            out.append((False, text[pos:]))
        return out

    def _encode_parts(self, parts, fragment_fn=None):
        frags = [p for sp, p in parts if not sp]
        if fragment_fn is not None:
            frags = [fragment_fn(f) for f in frags]
        enc = self._bpe.encode_batch(frags, add_special_tokens=False) if frags else []
        ids, k = [], 0
        for sp, p in parts:
            if sp:
                ids.append(p)
            else:
                ids.extend(enc[k].ids)
                k += 1
        return ids

    def encode(self, text, parse_special=True):
        """llama_tokenize(text, add_special=False, parse_special) equivalent"""
        return self._encode_parts(self._split(text, parse_special))

    def encode_prompt(self, marked_text):
        """a rendered prompt that may carry literal-tag marks: special tokens only where the template wrote them,
        literal tags (marked) encoded as plain text in the context of their fragment"""
        return self._encode_parts(self._split(marked_text, True), unmark)

    # ------------------------------------------------------------------------------------------------ decode
    def token_bytes(self, i):
        return self._bytes[i]

    def decode_bytes(self, ids, skip_special=False):
        if skip_special:
            return b''.join(self._bytes[i] for i in ids if i not in self.special_ids)
        return b''.join(self._bytes[i] for i in ids)

    def decode(self, ids, skip_special=False):
        return self.decode_bytes(ids, skip_special).decode('utf-8', errors='replace')

    def stream_decoder(self, skip_special=False):
        return StreamDecoder(self, skip_special)


class StreamDecoder:
    """Incremental detokenizer: feed ids, get text. Bytes of an incomplete UTF-8 sequence are held until the
    sequence completes (a token can end inside a multi-byte character); invalid bytes become U+FFFD exactly as a
    one-shot decode with errors='replace' would. flush() returns what is left at the end of the stream."""

    def __init__(self, tok, skip_special=False):
        self.tok = tok
        self.skip_special = skip_special
        self._dec = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def feed(self, ids):
        if isinstance(ids, int):
            ids = (ids,)
        return self._dec.decode(self.tok.decode_bytes(ids, self.skip_special), final=False)

    def flush(self):
        return self._dec.decode(b'', final=True)

    @property
    def pending(self):
        """number of held bytes (an incomplete UTF-8 sequence)"""
        return len(self._dec.getstate()[0])
