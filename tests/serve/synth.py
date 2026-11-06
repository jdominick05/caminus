"""Synthetic test data for the server tests: tool definitions, agent conversations and a tokenizer corpus.
Everything here is made up for the tests (no session transcripts, no harness data)."""
import random

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'read_file', 'description': 'Read a file from the workspace and return its text.',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': 'Path relative to the repo root'},
            'offset': {'type': 'integer', 'description': 'First line (1-based)'},
            'limit': {'type': 'integer'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {
        'name': 'run_shell', 'description': 'Run a shell command; returns stdout, stderr and the exit code.',
        'parameters': {'type': 'object', 'properties': {
            'command': {'type': 'string'}, 'timeout_s': {'type': 'number'},
            'env': {'type': 'object', 'additionalProperties': {'type': 'string'}}}, 'required': ['command']}}},
    {'type': 'function', 'function': {
        'name': 'edit_file', 'description': 'Replace old_str with new_str in a file (must match exactly once).',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string'}, 'old_str': {'type': 'string'}, 'new_str': {'type': 'string'},
            'replace_all': {'type': 'boolean'}}, 'required': ['path', 'old_str', 'new_str']}}},
    {'type': 'function', 'function': {
        'name': 'search', 'description': 'Search files with a regex. Données & 検索 → résultats.',
        'parameters': {'type': 'object', 'properties': {
            'pattern': {'type': 'string'}, 'globs': {'type': 'array', 'items': {'type': 'string'}},
            'max_results': {'type': 'integer', 'default': 50}}, 'required': ['pattern']}}},
]

CODE = [
    'def fib(n):\n    """Return the n-th Fibonacci number."""\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n',
    '#include <stdio.h>\nint main(void) {\n\tprintf("%d\\n", 42);\n\treturn 0;\n}\n',
    'fn main() {\n    let v: Vec<u32> = (1..=10).map(|x| x * x).collect();\n    println!("{:?}", v);\n}\n',
    'const re = /^(?:\\d{3})-\\d{4}$/u;\nexport default async function f(x) { return await fetch(`/api/${x}`); }\n',
    'SELECT u.id, COUNT(*) AS n FROM users u JOIN orders o ON o.user_id = u.id WHERE o.ts >= \'2026-01-01\' GROUP BY 1;',
    '{"name": "k3", "version": "0.1.0", "deps": {"numpy": ">=2.0"}, "flags": [true, false, null], "pi": 3.14159}',
    'class Foo(Base):\n    def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs)\n        self._cache: dict[str, int] = {}\n',
    'diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,4 @@\n import os\n+import sys\n def f():\n-    pass\n+    return 1\n',
    'template <typename T>\nconstexpr T clamp(T v, T lo, T hi) noexcept { return v < lo ? lo : (hi < v ? hi : v); }\n',
    '    if (x == NULL) {\r\n        return -EINVAL;\r\n    }\r\n',
    'print("tabs\\tand\\\\backslashes\\n")  # \\x00 \\u00e9 \\N{BULLET}',
    'matrix = [[1, 2, 3],\n          [4, 5, 6]]\nassert sum(map(sum, matrix)) == 21\n',
]

SHELL = [
    '$ ls -la /tmp\ntotal 12\ndrwxrwxrwt  3 root root 4096 Nov  1 10:00 .\n-rw-r--r--  1 user user  220 Nov  1 09:59 notes.txt\n',
    'git log --oneline -3 && git status --short | grep -v "^??" || echo "clean"',
    'find . -name "*.py" -print0 | xargs -0 grep -nE "TODO|FIXME" | head -n 20',
    'for f in *.log; do gzip -9 "$f" && mv "$f.gz" archive/; done 2>/dev/null',
    'Traceback (most recent call last):\n  File "/app/main.py", line 12, in <module>\n    main()\nZeroDivisionError: division by zero\n',
    'curl -sS -H "Authorization: Bearer $TOKEN" https://example.invalid/api/v1/items?limit=10&page=2 | jq \'.items[] | .id\'',
    '===== 3 passed, 1 failed, 2 warnings in 0.42s =====\nFAILED tests/test_x.py::test_edge - AssertionError: assert 1 == 2',
    'export PATH="$HOME/.local/bin:$PATH"; echo ${#PATH}; printf \'%s\\n\' "${arr[@]}"',
]

PROSE = [
    "The quick brown fox jumps over the lazy dog. It's a pangram; they're common, aren't they?",
    "I'LL SAY IT AGAIN: THE CACHE ISN'T WARM. We'd better check what's going on, shouldn't we?",
    'Numbers: 1234567890, 3.14159, -42, 1e-9, 0xFF, 1,000,000 and 12:30:45.',
    'Mixed   spaces\tand tabs\t\tand\n\n\nnewlines   \n  trailing   ',
    'Ellipsis… em—dash – en-dash “quotes” ‘single’ «guillemets» ¿qué? ¡sí!',
    'Résumé naïve café façade coöperate über straße Ærøskøbing.',
    'Привет, мир! Это тест токенизатора. Ελληνικά: γειά σου κόσμε.',
    'مرحبا بالعالم ٠١٢٣٤٥٦٧٨٩ — עברית: שלום עולם',
    'हिन्दी में परीक्षण। தமிழ் சோதனை. ภาษาไทยทดสอบ',
    'Superscripts x² + y³ = z⁴, fractions ½ ¼ ¾, roman Ⅻ ⅷ, circled ① ②.',
    'Combining: é à ñ ö (NFD) vs é à ñ ö (NFC); Hangul jamo 각 vs 각.',
    'NBSP here, thin space, zero​width, line separator, para sep, BOM﻿inside.',
    "Contractions: don't DON'T we've WE'VE she'd SHE'D they'll THEY'LL I'm I'M it's IT's 's 'S",
    'URLs https://example.invalid/a/b?c=d&e=f#g and emails someone@example.invalid.',
]

CJK = [
    '你好，世界！这是一个分词器测试。我们需要确保中文文本被正确处理。',
    '日本語のテキスト：こんにちは世界。カタカナとひらがなと漢字が混在しています。',
    '한국어 텍스트: 안녕하세요 세계. 토크나이저 테스트입니다.',
    '混合 mixed 文本 text 123 数字 ４５６ 全角ＡＢＣ',
    '「引用」『二重引用』【括弧】（丸括弧）〔亀甲〕《書名》',
    'def 函数(参数):\n    return 参数 * 2  # 注释',
]

EMOJI = [
    'Emoji: 😀 😃 😄 🙂 🤖 🚀 ✨ 🔥 ✅ ❌',
    'ZWJ family 👨‍👩‍👧‍👦 and 🧑‍💻 and 🏳️‍🌈 rainbow flag',
    'Skin tones 👍🏻 👍🏼 👍🏽 👍🏾 👍🏿 and flags 🇯🇵 🇺🇸 🇩🇪',
    'Keycaps 1️⃣ 2️⃣ #️⃣ and variation ☺︎ ☺️ and math 𝔸𝕓𝕔 𝟙𝟚𝟛',
    'Old italic 𐌀𐌁 and cuneiform 𒀀 and musical 𝄞 𝄢',
]

SPECIALS = [
    '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\n\nhi<|im_end|>',
    '<|endoftext|><|im_start|>system\nYou are helpful.<|im_end|>',
    '<tool_call>\n<function=run_shell>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>',
    '<tool_response>\nok\n</tool_response><|im_end|>\n',
    'text<think>inline</think>more <|vision_start|><|image_pad|><|vision_end|> x',
    '<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    pass<|fim_pad|><|repo_name|>r<|file_sep|>',
    '<|audio_start|><|audio_pad|><|audio_end|><tts_pad><tts_text_bos><tts_text_eod><tts_text_bos_single>',
    '[PAD248077] [PAD248319] are unused tokens and stay plain text',
    '<|object_ref_start|>a<|object_ref_end|><|box_start|>(1,2)<|box_end|><|quad_start|>q<|quad_end|><|vision_pad|><|video_pad|>',
    '<<|im_end|>> <tool_call<tool_call>> </think</think>> <|im_start|<|im_start|>',
    ' <think> spaced  </think>  \n<tool_call>\n',
]

WHITESPACE = [
    'next\u0085line, vt\x0bff\x0c and a lone\rcarriage return, then \r\n crlf',
    'ideographic\u3000space, ogham\u1680mark, en\u2000em\u2001thin\u2009hair\u200a narrow\u202fmath\u205fend',
    ''.join(chr(c) for c in range(0, 32)) + ' ctl \x7f del \x00nul',
    'tabs\t\t\tthen spaces    \n\t \n \t\r\n\r\n\n\x0b\x0c\u0085 mixed runs',
    '\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a word \u3000\u3000',
]

LONG = [
    ''.join('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'[(i * 7919 + 13) % 64]
            for i in range(10000)) + '==',
    'var a=function(b){return b.map(function(c){return c*2}).filter(Boolean)};' * 150,
    ''.join(str((i * 31337) % 10) for i in range(2000)),
    ' ' * 240 + 'word',
    '\n' * 300 + 'x' + '\t' * 100,
    'é' * 3000,
]

REASONING_STARTS = [
    'The user wants me to look at the file first.',
    '1. Read the config.\n2. Run the tests.',
    '(First) check the failing test, then fix it.',
    '`pytest -q` failed; I need the traceback.',
    '用户希望我先读取文件。',
    '🤔 Let me think about the edge cases.',
    '#include guard is missing; add one.',
    '- bullet one\n- bullet two',
    "'quoted' start and a contraction it's fine",
    '   leading spaces are trimmed by the template',
]


def text_corpus():
    return CODE + SHELL + PROSE + CJK + EMOJI + SPECIALS + WHITESPACE + LONG


def fuzz_corpus(n, seed=0):
    """random strings from a pool that hits the pre-tokenizer's edge cases"""
    rng = random.Random(seed)
    pool = list('abcXYZ019 \t\n\r.,;:!?\'"`()[]{}<>/\\|_-+=*&^%$#@~') + [
        'é', 'é', 'ß', 'Ж', 'λ', '中', '文', 'の', '한', '٣', '²', '½', 'Ⅻ', ' ', ' ', '​',
        '😀', '👨‍👩‍👧', '🇯🇵', '̀', '॑', 'ก', 'ั', "'s", "'LL", "'ve", '\r\n', '  ', '\n\n',
        '<think>', '</tool_call>', '<|im_end|>', '<|endoftext|>', '[PAD248100]', '﻿', '𝟙', '\U0001F3FB']
    out = []
    for _ in range(n):
        k = rng.randint(1, 120)
        out.append(''.join(rng.choice(pool) for _ in range(k)))
    return out


def _args_for(rng, name):
    if name == 'read_file':
        a = {'path': rng.choice(['src/main.py', 'README.md', 'tests/test_x.py', 'docs/设计.md'])}
        if rng.random() < 0.5:
            a['offset'] = rng.randint(1, 400)
            a['limit'] = rng.choice([50, 200])
        return a
    if name == 'run_shell':
        a = {'command': rng.choice(SHELL[1:4] + ['pytest -q tests/', 'python -c "print(1/0)"', 'make -j8'])}
        if rng.random() < 0.3:
            a['timeout_s'] = rng.choice([30, 120.5])
        if rng.random() < 0.2:
            a['env'] = {'LANG': 'C.UTF-8', 'X': 'a "quoted" value'}
        return a
    if name == 'edit_file':
        return {'path': 'src/main.py', 'old_str': rng.choice(CODE[:3]), 'new_str': rng.choice(CODE[3:6]),
                'replace_all': rng.random() < 0.3}
    return {'pattern': rng.choice([r'def \w+\(', 'TODO|FIXME', '<tool_call>', r'\bimport\s+os\b']),
            'globs': ['*.py', '*.md'], 'max_results': rng.randint(1, 100)}


def _tool_output(rng):
    out = rng.choice(SHELL + CODE + CJK + EMOJI)
    r = rng.random()
    if r < 0.15:          # tool output that prints special-token text (must stay plain text)
        out += '\nlog: <tool_call> seen, then <|im_end|> and </think> in the output\n'
    elif r < 0.3:
        out = '\n' + out + '\n\n'
    return out


def agent_conversation(seed, rounds=4, with_system=True, with_tools=True, user_turns=2, reasoning_starts=None):
    """A synthetic agent session as a list of 'requests': each request is (messages, reply) where messages is what
    the client sends and reply is the assistant message the model produces (canonical form). Assistant replies
    alternate between tool calls (1-3, parallel) and a final answer; each user turn runs `rounds` tool rounds."""
    rng = random.Random(seed)
    starts = reasoning_starts or REASONING_STARTS
    msgs = []
    if with_system:
        msgs.append({'role': 'system', 'content': rng.choice([
            'You are a coding agent. Work in the repo; use the tools.',
            'You are K3, a careful assistant.\nAlways run the tests before finishing.',
            'Système: réponds en français si on te parle français. 中文也可以。'])})
    tools = TOOLS if with_tools else None
    steps = []
    for u in range(user_turns):
        msgs.append({'role': 'user', 'content': rng.choice(PROSE[:5] + CJK[:2] + [
            'Fix the failing test in tests/test_x.py.', 'Add a --verbose flag to the CLI and document it.',
            'Why does `make` fail? Paste: <|im_end|> appears in the log.'])})
        for r in range(rounds if with_tools else 0):
            n = rng.choice([1, 1, 2, 3])
            calls = []
            for k in range(n):
                name = rng.choice([t['function']['name'] for t in TOOLS])
                calls.append({'id': f'call_{u}_{r}_{k}', 'type': 'function',
                              'function': {'name': name, 'arguments': _args_for(rng, name)}})
            reply = {'role': 'assistant', 'reasoning_content': rng.choice(starts) + ' ' + rng.choice(PROSE[:3]),
                     'content': rng.choice(['', '', 'Let me check.', 'Running the tests now.']), 'tool_calls': calls}
            steps.append((list(msgs), reply))
            msgs.append(reply)
            for k in range(n):
                msgs.append({'role': 'tool', 'content': _tool_output(rng)})
        reply = {'role': 'assistant', 'reasoning_content': rng.choice(starts),
                 'content': rng.choice(['Done. The test passes now.', CODE[0], 'All set ✅ — 完成。'])}
        steps.append((list(msgs), reply))
        msgs.append(reply)
    return tools, steps
