"""The own-code gate of the qwen3moe line (Qwen3-30B-A3B), CPU only:
  - the decode module (kernels/k3_q3m_lean.hip, + _trace) and the prefill object (kernels/q3m_prefill.hip): no ggml /
    llama.cpp header in their include closure (every #include "..." followed inside kernels/), and build.sh gives them no
    ggml include path (their compile lines are the generic loop's or the k3_q3m case, not the k3_gguf one)
  - the serve path for the architecture imports neither gguf (gguf-py) nor tokenizers: the modules caminus serve loads
    for it, imported under a hook that refuses both, and arch.Q3MoE selects the own BPE ('own')
  - with a qwen3moe GGUF given (--model; skipped otherwise): its header facts (arch.meta) and vocabulary through
    caminus.ggufio equal gguf-py's, and QwenTokenizer(engine='own') gives the 'hf' engine's ids on every source file of
    this tree (4 KB chunks) and a Unicode stress set, special tokens parsed and not
The end-to-end form: tools/serve_smoke1.py --block-imports gguf,tokenizers (a GPU run: the server under an import hook, and
the include closure of every code object it reports having loaded).
  python -I tests/test_own_code.py [--model GGUF]"""
import argparse, os, re, subprocess, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tools'))
from own_closure import audit   # noqa: E402
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


for top in ('k3_q3m_lean.hip', 'k3_q3m_lean_trace.hip', 'q3m_prefill.hip'):
    cl, missing, bad = audit(top)
    check(not bad and not missing, f'{top}: includes {sorted(cl - {top})} (no ggml header; every one in kernels/)')
ref = audit('k3_gguf_q3m_own.hip')[2]
check(bool(ref), f'the detector itself: the full GGUF module (k3_gguf_q3m_own.hip) is flagged ({ref})')
bs = open(os.path.join(ROOT, 'build.sh')).read()
check(re.search(r'kernels/k3_q3m\*\.hip\)\s*extra=\(-mllvm -amdgpu-sched-strategy=max-ilp\)', bs) is not None and
      'k3_q3m' not in ''.join(re.findall(r'\*k3_gguf\*\.hip\).*', bs)),
      'build.sh: k3_q3m*.hip get max-ilp and no ggml include path; q3m_prefill.hip the generic loop')

# the serve path under an import hook refusing gguf and tokenizers
MODS = ('caminus.serve.cli', 'caminus.serve.server', 'caminus.serve.k3_session', 'caminus.serve.worker', 'caminus.serve.generate',
        'caminus.cache.prompt', 'caminus.decode.longctx', 'caminus.engine.q3moe_one', 'caminus.prefill.q3m', 'caminus.serve.tokenizer')
code = f'''import sys, importlib.abc
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('gguf', 'tokenizers'): raise ImportError('refused: ' + name)
sys.meta_path.insert(0, B()); sys.path.insert(0, {os.path.join(ROOT, 'python')!r})
for m in {MODS!r}: __import__(m)
from caminus import arch
assert arch.ARCHS['qwen3moe'].tokenizer == 'own'
print('imported', len({MODS!r}))
'''
r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
check(r.returncode == 0, f'serve modules for qwen3moe import with gguf and tokenizers refused: {(r.stdout + r.stderr).strip().splitlines()[-1][:160]}')

ap = argparse.ArgumentParser(); ap.add_argument('--model', default=None)
A = ap.parse_args()
if not A.model or not os.path.isfile(A.model):
    print('SKIP the model checks: no --model qwen3moe GGUF')
else:
    from caminus import arch, config
    from caminus.serve import vocab
    from caminus.serve.tokenizer import QwenTokenizer
    config.ensure_gguf()
    from gguf import GGUFReader   # (the comparison only)
    arch._META.clear(); m1 = arch.meta(A.model)
    rd = GGUFReader(A.model, 'r'); f = {x.name: x for x in rd.fields.values()}; a = f['general.architecture'].contents()
    m2 = dict(arch=a, hp={h: f[f'{a}.{h}'].contents() for h in arch.HP_KEYS if f'{a}.{h}' in f},
              weights=sum(int(t.n_bytes) for t in rd.tensors if t.name != 'token_embd.weight'))
    check(m1 == m2, f'arch.meta through caminus.ggufio == gguf-py\'s ({m1["arch"]}, {len(m1["hp"])} keys, {m1["weights"]} B)')
    check(vocab.load_gguf_vocab(A.model) == vocab._load_gguf_vocab_ggufpy(A.model), 'the vocabulary through caminus.ggufio == gguf-py\'s')
    own, hf = QwenTokenizer.from_gguf(A.model, engine='own'), QwenTokenizer.from_gguf(A.model)
    texts = []
    for d, _, fs in os.walk(ROOT):
        if '/.git' in d or '/build' in d: continue
        for fn in sorted(fs):
            if fn.rsplit('.', 1)[-1] in ('py', 'md', 'hip', 'h', 'sh', 'go', 'toml', 'json'):
                try: s = open(os.path.join(d, fn), encoding='utf-8').read()
                except (UnicodeDecodeError, OSError): continue
                texts += [s[k:k + 4096] for k in range(0, len(s), 4096)]
    texts += ["é ä Ünïcödé — 😀👍🏽👨‍👩‍👧 🇺🇸 日本語 한국어 ١٢٣   　 \x85", "<|im_start|>user\nhi<|im_end|>\n<think>\n</think>",
              "  \n\n\t x'll'S 1234567 3.14", "", " ", "\n\n\n", "x" * 300]
    bad = n = 0
    for t in texts:
        for ps in (False, True):
            b = hf.encode(t, parse_special=ps); n += len(b); bad += own.encode(t, parse_special=ps) != b
    check(bad == 0, f'QwenTokenizer own == hf: {len(texts)} texts x 2 (special parsed / not), {n} tokens, {bad} differ')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
