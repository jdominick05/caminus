"""caminus.bpe (our byte-level BPE, built from the GGUF's metadata) against the Hugging Face tokenizers library (the
oracle, from the model's tokenizer.json): ids equal on every text, and decode(encode(text)) == NFC(text). Texts: every
.py / .md / .hip / .h / .sh / .go file of this repository, split into 4 KB chunks, plus a Unicode stress set (scripts,
digits, contractions, mixed whitespace, special tokens). Prints PASS / FAIL and RESULT PASS | FAIL.
  python -I tests/io/test_bpe.py GGUF TOKENIZER_JSON"""
import os, sys, unicodedata
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus import ggufio
from caminus.bpe import BPE, pretokenize
from tokenizers import Tokenizer   # (the oracle only)

F = ggufio.GGUFFile(sys.argv[1]); ours = BPE(F.meta); ref = Tokenizer.from_file(sys.argv[2])
texts = []
for d, _, fs in os.walk(ROOT):
    if '/.git' in d or '/build' in d or 'node_modules' in d: continue
    for f in sorted(fs):
        if f.rsplit('.', 1)[-1] in ('py', 'md', 'hip', 'h', 'sh', 'go'):
            try: s = open(os.path.join(d, f), encoding='utf-8').read()
            except (UnicodeDecodeError, OSError): continue
            texts += [s[k:k + 4096] for k in range(0, len(s), 4096)]
stress = [
    "Hello world! It's, you're, we've, I'm, they'll, he'd. IT'S YOU'RE WE'VE I'M THEY'LL HE'D 'S 'ſ",
    "  leading spaces and trailing   ", "tabs\tand\t\tmore\t\t\n\n\nnewlines \r\n windows\r\n\r\n x",
    "1234567890 3.14159 1e-6 0x1F 100,000 ١٢٣ ⅷ ²³ 五六", "日本語のテキスト、中文文本，한국어 텍스트",
    "Ünïcödé çømbining: e\u0301 a\u0308 n\u0303 — emoji: 😀👍🏽👨‍👩‍👧 🇺🇸", "!!!???... ;;; ::: --> <== {}[]() @#$%^&*",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><|endoftext|>", "\u00a0nbsp\u2003em\u3000ideo\u2028ls\u2029ps\x85nel",
    "a\n \nb \n  \n c", "   \n", " ", "\n", "", "x" * 300, "ab" * 200 + "!", "'''", "''s", "don't'S", "x'll'Ll'LL",
    "def f(x):\n    return x  # comment\n\n\n\tclass A:\n\t\tpass\n", "Ελληνικά Русский עברית العربية हिन्दी ไทย",
]
texts += stress
bad = 0
for i, t in enumerate(texts):
    a, b = ours.encode(t), ref.encode(t, add_special_tokens=False).ids
    if a != b:
        bad += 1
        if bad <= 5:
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            print(f'  mismatch text {i}: at token {k}: ours {a[max(0,k-2):k+3]} ref {b[max(0,k-2):k+3]} '
                  f'ours {[ours.tokens[x] for x in a[max(0,k-2):k+3]]} ref {[ours.tokens[x] for x in b[max(0,k-2):k+3]]}')
    elif ours.decode(a) != unicodedata.normalize('NFC', t):
        bad += 1; print(f'  round trip differs on text {i}')
ntok = sum(len(ours.encode(t)) for t in stress)
print(('PASS' if bad == 0 else f'FAIL {bad}') + f' of {len(texts)} texts: ids equal the oracle and decode round-trips ({ntok} stress tokens)')
print('RESULT', 'PASS' if bad == 0 else 'FAIL')
