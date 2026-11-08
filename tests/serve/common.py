"""Shared fixtures for the server tests (CPU only). Run with the drafter env:  python -I tests/serve/run_all.py"""
import functools
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python'))
sys.path.insert(0, HERE)

from caminus import devpaths  # noqa: E402

MODEL = devpaths.coder()      # CODER, else caminus.config's coder (CAMINUS_CODER / K3_GGUF / config.toml / the download location)
LLAMA_TOKENIZE = os.environ.get('K3_LLAMA_TOKENIZE',
                                os.path.expanduser('~/src/ignite-llama/build-cpu/bin/llama-tokenize'))
HF_TOKENIZER_JSON = os.path.expanduser(os.environ.get('HF_TOKENIZER_JSON', ''))   # the base model's tokenizer.json (optional)


def need_llama_tokenize(test):
    """skip `test` (a unittest.TestCase) when llama.cpp's llama-tokenize is not there (a machine without the fork's CPU build,
    e.g. CI): the parity tests compare against it"""
    if not os.access(LLAMA_TOKENIZE, os.X_OK):
        test.skipTest(f'no llama-tokenize at {LLAMA_TOKENIZE} (K3_LLAMA_TOKENIZE; the ignite-llama fork\'s CPU build)')


@functools.lru_cache(None)
def tokenizer():
    from caminus.serve.tokenizer import QwenTokenizer
    return QwenTokenizer.from_gguf(MODEL)


@functools.lru_cache(None)
def builder():
    from caminus.serve.template import PromptBuilder
    return PromptBuilder.from_tokenizer(tokenizer())


@functools.lru_cache(None)
def hf():
    """transformers' apply_chat_template with the GGUF's chat template (the T0 reference)"""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from transformers import PreTrainedTokenizerFast
    tk = tokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=tk._bpe, chat_template=tk.vocab.chat_template)


def hf_render(messages, tools=None, add_generation_prompt=True, **kw):
    import copy
    return hf().apply_chat_template(copy.deepcopy(messages), tools=copy.deepcopy(tools), tokenize=False,
                                    add_generation_prompt=add_generation_prompt, **kw)


def llama_tokenize(text, parse_special=True):
    """ids from llama.cpp's llama-tokenize (CPU build, vocab only), raw file bytes, no escape processing"""
    with tempfile.NamedTemporaryFile('wb', suffix='.txt', delete=False) as f:
        f.write(text.encode('utf-8'))
        path = f.name
    try:
        cmd = [LLAMA_TOKENIZE, '-m', MODEL, '-f', path, '--ids', '--no-escape', '--log-disable']
        if not parse_special:
            cmd.append('--no-parse-special')
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f'llama-tokenize failed: {r.stderr[-500:]}')
        out = r.stdout.strip()
        i, j = out.rfind('['), out.rfind(']')
        body = out[i + 1:j].strip()
        return [int(x) for x in body.split(',')] if body else []
    finally:
        os.unlink(path)


QWEN_PARSE_REF = os.environ.get('K3_QWEN_PARSE_REF', os.path.join(HERE, '..', '..', 'build', 'qwen_parse_ref'))


@functools.lru_cache(None)
def qwen_parse_ref_binary():
    """llama.cpp's chat-output parser behind a JSONL driver (tools/reference/qwen_parse_ref.cpp), built on first use with
    tools/reference/build_qwen_parse_ref.sh against ignite-llama's CPU build; None when it cannot be built"""
    path = os.path.abspath(QWEN_PARSE_REF)
    if not os.path.exists(path):
        script = os.path.join(HERE, '..', '..', 'tools', 'reference', 'build_qwen_parse_ref.sh')
        r = subprocess.run([script], env=dict(os.environ, OUT=path), capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(path):
            return None
    return path


@functools.lru_cache(None)
def chat_template_file():
    path = os.path.join(tempfile.gettempdir(), f'k3-serve-tmpl-{os.getpid()}.jinja')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tokenizer().vocab.chat_template)
    return path


def qwen_parse_ref(cases):
    """run llama.cpp's parser on cases ({'tools', 'enable_thinking', 'tool_choice', 'parallel', 'text',
    'partial_at'}); returns one result dict per case (see tools/reference/qwen_parse_ref.cpp)"""
    import json
    binary = qwen_parse_ref_binary()
    if binary is None:
        raise RuntimeError('qwen_parse_ref is not available (tools/reference/build_qwen_parse_ref.sh failed)')
    inp = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases)
    r = subprocess.run([binary, chat_template_file()], input=inp.encode('utf-8'), capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f'qwen_parse_ref failed: {r.stderr[-1000:]!r}')
    out = [json.loads(line) for line in r.stdout.decode('utf-8').split('\n') if line]
    assert len(out) == len(cases), (len(out), len(cases))
    return out
