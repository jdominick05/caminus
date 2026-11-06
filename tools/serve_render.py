"""Render a chat request the way the K3 server will: prompt text and token ids (CPU only).

  python -I tools/serve_render.py request.json [--api openai|anthropic] [--gguf PATH] [--ids] [--no-bos]

request.json is an OpenAI /v1/chat/completions or Anthropic /v1/messages body."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from caminus.serve.normalize import anthropic_to_template, openai_chat_to_template  # noqa: E402
from caminus.serve.template import PromptBuilder  # noqa: E402
from caminus.serve.tokenizer import QwenTokenizer  # noqa: E402

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
DEFAULT_GGUF = devpaths.coder()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('request')
    ap.add_argument('--api', choices=('openai', 'anthropic'), default=None)
    ap.add_argument('--gguf', default=os.environ.get('K3_GGUF', DEFAULT_GGUF))
    ap.add_argument('--ids', action='store_true', help='print the token ids too')
    ap.add_argument('--no-bos', action='store_true')
    a = ap.parse_args()
    with open(a.request) as f:
        req = json.load(f)
    blocks = {b.get('type') for m in req.get('messages', []) if isinstance(m.get('content'), list)
              for b in m['content'] if isinstance(b, dict)}
    api = a.api or ('anthropic' if 'system' in req or blocks & {'tool_use', 'tool_result', 'thinking'} else 'openai')
    n = (anthropic_to_template if api == 'anthropic' else openai_chat_to_template)(req)
    b = PromptBuilder.from_tokenizer(QwenTokenizer.from_gguf(a.gguf), prepend_bos=not a.no_bos)
    p = b.build(n['messages'], n['tools'], model=n['model'], client_kwargs=n['client_kwargs'])
    sys.stdout.write(p.text)
    sys.stderr.write(f'\n[{api}] profile={p.profile} kwargs={p.kwargs} tokens={len(p.ids)}\n')
    if a.ids:
        print(json.dumps(p.ids))


if __name__ == '__main__':
    main()
