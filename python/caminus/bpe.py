"""Caminus's own byte-level BPE tokenizer, built from a GGUF's tokenizer metadata (tokenizer.ggml.model 'gpt2'; the
pre-tokenizer 'qwen2'). Written from the format's definition (docs/specs/spec-own-quant.md: every step from the GGUF file
to the output token on Caminus's own code); the Hugging Face tokenizers library and llama.cpp are test oracles only.

  Encoding: (1) special tokens (token types 3 control and 4 user-defined) are split out first, longest match first;
  (1b) the text between them is normalized to Unicode NFC (Qwen2's tokenizer declares an NFC normalizer; the GGUF does
  not record it, so it goes with the 'qwen2' pre-tokenizer here); (2) that text is cut into pieces by the qwen2
  pre-tokenizer: the pattern
        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+ | \\p{N} | ' '?[^\\s\\p{L}\\p{N}]+[\\r\\n]* | \\s*[\\r\\n]+
        | \\s+(?!\\S) | \\s+
  scanned by hand here (the first alternative that matches at a position wins, as in the regex); (3) each piece's UTF-8
  bytes become the GPT-2 byte alphabet's characters; (4) merges apply by rank (the lowest-ranked adjacent pair first,
  left to right), and the resulting symbols are the token ids.
  Decoding reverses (3) (special tokens decode to their text); decode(encode(t)) is NFC(t)."""
import unicodedata

# Unicode White_Space (what the regex engines the reference tokenizers use mean by \s)
_WS = frozenset('\t\n\v\f\r \x85\xa0     　' + ''.join(chr(c) for c in range(0x2000, 0x200b)))


def _is_ws(c):
    return c in _WS


def _is_l(c):
    return unicodedata.category(c)[0] == 'L'


def _is_n(c):
    return unicodedata.category(c)[0] == 'N'


def byte_alphabet():
    """the GPT-2 byte -> character map: printable Latin-1 bytes stand for themselves, the other 68 bytes take the
    characters from U+0100 up, in byte order"""
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(0xa1, 0xac + 1)) + list(range(0xae, 0xff + 1))
    m, n = {}, 0
    for b in range(256):
        if b in keep: m[b] = chr(b)
        else: m[b] = chr(256 + n); n += 1
    return m


_CONTRACTIONS = ('s', 't', 're', 've', 'm', 'll', 'd')   # after an ASCII apostrophe, any case


def _fold(c):
    # the case-insensitive contraction match folds case as the reference regex engines do (U+017F long s folds to s)
    return 's' if c == 'ſ' else c.lower()


def pretokenize(text):
    """the qwen2 pre-tokenizer's pieces of text (no special tokens in it)"""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        # (a) contractions
        if c == "'":
            hit = 0
            for w in _CONTRACTIONS:
                if i + 1 + len(w) <= n and all(_fold(text[i + 1 + k]) == w[k] for k in range(len(w))):
                    hit = len(w); break
            if hit:
                out.append(text[i:i + 1 + hit]); i += 1 + hit; continue
        # (b) one optional non-letter / non-digit / non-CR-LF character, then letters
        j = i
        if not (c in '\r\n' or _is_l(c) or _is_n(c)) and i + 1 < n and _is_l(text[i + 1]): j = i + 1
        if _is_l(text[j]):
            k = j
            while k < n and _is_l(text[k]): k += 1
            out.append(text[i:k]); i = k; continue
        # (c) one digit
        if _is_n(c):
            out.append(c); i += 1; continue
        # (d) an optional space, symbols (not whitespace, letters or digits), then any CR / LF
        j = i + 1 if c == ' ' else i
        if j < n and not (_is_ws(text[j]) or _is_l(text[j]) or _is_n(text[j])):
            k = j
            while k < n and not (_is_ws(text[k]) or _is_l(text[k]) or _is_n(text[k])): k += 1
            while k < n and text[k] in '\r\n': k += 1
            out.append(text[i:k]); i = k; continue
        if _is_ws(c):
            k = i
            while k < n and _is_ws(text[k]): k += 1          # the whitespace run [i, k)
            # (e) whitespace up to and including the run's last CR / LF
            last = max((p for p in range(i, k) if text[p] in '\r\n'), default=-1)
            if last >= 0:
                out.append(text[i:last + 1]); i = last + 1; continue
            # (f) whitespace not followed by a non-space: the run, less its last character when more text follows
            if k == n:
                out.append(text[i:k]); i = k; continue
            if k - i >= 2:
                out.append(text[i:k - 1]); i = k - 1; continue
            # (g) the single whitespace character
            out.append(text[i:k]); i = k; continue
        # (nothing else can reach here: a non-space character matched (b), (c) or (d))
        out.append(c); i += 1
    return out


class BPE:
    """BPE(meta): meta = a GGUF's metadata (caminus.ggufio.GGUFFile.meta). encode(text, special=True) -> [ids];
    decode(ids) -> str; .eos, .bos, .vocab_size"""

    def __init__(self, meta):
        if meta.get('tokenizer.ggml.model') != 'gpt2':
            raise ValueError(f"bpe: tokenizer.ggml.model {meta.get('tokenizer.ggml.model')!r} (byte-level BPE 'gpt2' only)")
        pre = meta.get('tokenizer.ggml.pre')
        if pre not in ('qwen2',):
            raise ValueError(f"bpe: pre-tokenizer {pre!r} (qwen2 only so far)")
        self.tokens = list(meta['tokenizer.ggml.tokens'])
        types = list(meta.get('tokenizer.ggml.token_type', [1] * len(self.tokens)))
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {}
        for r, m in enumerate(meta['tokenizer.ggml.merges']):
            a, b = m.split(' ', 1)
            self.rank[(a, b)] = r
        self.special = sorted((t for t, ty in zip(self.tokens, types) if ty in (3, 4)), key=len, reverse=True)
        self.special_ids = {self.id_of[t] for t in self.special}
        self._first = {}
        for t in self.special: self._first.setdefault(t[0], []).append(t)
        self.b2c = byte_alphabet(); self.c2b = {c: b for b, c in self.b2c.items()}
        self.eos = meta.get('tokenizer.ggml.eos_token_id'); self.bos = meta.get('tokenizer.ggml.bos_token_id')
        self.vocab_size = len(self.tokens)
        self._cache = {}

    def _bpe(self, piece):
        word = ''.join(self.b2c[b] for b in piece.encode('utf-8'))
        hit = self._cache.get(word)
        if hit is not None: return hit
        sym = list(word)
        while len(sym) > 1:
            best, bi = None, -1
            for k in range(len(sym) - 1):
                r = self.rank.get((sym[k], sym[k + 1]))
                if r is not None and (best is None or r < best): best, bi = r, k
            if best is None: break
            a, b = sym[bi], sym[bi + 1]
            out, k = [], 0
            while k < len(sym):                      # every occurrence of the pair, left to right
                if k < len(sym) - 1 and sym[k] == a and sym[k + 1] == b: out.append(a + b); k += 2
                else: out.append(sym[k]); k += 1
            sym = out
        ids = []
        for s in sym:
            t = self.id_of.get(s)
            if t is None:                            # (a symbol outside the vocabulary: its bytes one by one)
                ids += [self.id_of[ch] for ch in s]
            else: ids.append(t)
        if len(self._cache) < 1 << 16: self._cache[word] = ids
        return ids

    def encode(self, text, special=True):
        ids, i, start, n = [], 0, 0, len(text)
        while special and i < n:
            m = next((t for t in self._first.get(text[i], ()) if text.startswith(t, i)), None)
            if m is None: i += 1; continue
            for p in pretokenize(unicodedata.normalize('NFC', text[start:i])): ids += self._bpe(p)
            ids.append(self.id_of[m]); i += len(m); start = i
        for p in pretokenize(unicodedata.normalize('NFC', text[start:])): ids += self._bpe(p)
        return ids

    def decode_bytes(self, ids):
        out = bytearray()
        for t in ids:
            s = self.tokens[t]
            if t in self.special_ids: out += s.encode('utf-8')
            else: out += bytes(self.c2b[c] for c in s)
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode('utf-8', errors='replace')
