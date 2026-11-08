# The Jinja environment setup and _tojson (ChatTemplate.__init__'s sandboxed environment with trim_blocks / lstrip_blocks /
# loopcontrols, the tojson override, raise_exception, strftime_now) follow Hugging Face transformers'
# src/transformers/utils/chat_template_utils.py (https://github.com/huggingface/transformers), modified:
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# (a copy of the License: LICENSES/Apache-2.0.txt; THIRD_PARTY.md)
#
# Parts of this file (the Jinja environment setup; LiteralMarker's mark/unmark scheme and its kept <tool_response> wrapper) are rewritten, line-reviewed, from Strata (serve/frontend.py),
# which is distributed under the following license:
#
# MIT License
#
# Copyright (c) 2026 Niko1221 and the Strata contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Chat template rendering for the K3 server (spec-server.md section 3, step S1).

The template is the GGUF's tokenizer.chat_template, rendered with the same Jinja environment as transformers'
apply_chat_template (transformers/utils/chat_template_utils.py: ImmutableSandboxedEnvironment, trim_blocks,
lstrip_blocks, loopcontrols, tojson = json.dumps with ensure_ascii=False, raise_exception, strftime_now), so the
output is byte-equal to transformers' (gate T0, tests/serve/test_template.py).

On top of the plain render, three fixes:
  1. tool-call arguments that arrive as JSON strings are parsed into objects (the template raises on a non-empty
     string): normalize.parse_tool_arguments, applied in normalize.py;
  2. reasoning effort is pinned per model id (MODEL_PROFILES): the effort line is the first thing in the system
     message, so a per-request effort would invalidate the whole prompt cache;
  3. literal-tag marking: every special-token text that appears inside message content, reasoning, tool
     arguments or tool definitions is marked before rendering (mark_messages) and encoded as plain text after
     (tokenizer.encode_prompt), so a tool result that prints "<|im_end|>" or "<tool_call>" cannot inject control
     tokens. unmark(render(mark(m))) == render(m) byte for byte (tested).

APPEND-ONLY (what makes the prompt cache hit; tests/serve/test_append_only.py). With preserve_thinking true (the
template's default) the prompt of turn n, generation prompt included, is a prefix of the prompt of turn n+1, and the
tokens of prompt n followed by the canonical tokens of the model's reply are a prefix of the tokens of prompt n+1.
Cases where the template forces a rewrite (all tested):
  * preserve_thinking=false: when a new user query arrives, earlier assistant turns lose their <think> block
    (template lines 119-123), so the prompt changes from the first assistant turn on;
  * the client drops reasoning_content (OpenAI clients often do): the turn renders an empty think block and the
    tokens diverge at the last token of the previous prompt ('<think>\n' -> '<think>\n\n</think>'), so the whole
    previous reply is processed again (needs the S5 reasoning stash);
  * a change of reasoning effort (k3-coder-xhigh <-> others), of the tool list, or of the system text: the
    system message is rewritten, the whole cache goes. With the medium effort (k3-coder) the system message is
    the same with thinking on or off, so a per-request enable_thinking does not invalidate it;
  * the model's reply is not in canonical form: the template trims reasoning and content, puts exactly '\\n\\n'
    between content and the first <tool_call>, and re-serializes non-string tool arguments with tojson
    ({"a":1} -> {"a": 1}); a reply that differs is rewritten from that point;
  * empty reasoning with thinking on: the prompt ends '<think>\\n' (token 198) and the model writes '\\n</think>',
    but the re-render has '<think>\\n\\n</think>' whose '\\n\\n' is one token (271): the token sequence diverges right
    after <think> although the text is append-only (one token of the prompt plus the reply are processed again;
    a fix belongs to S5: reuse the generated ids when the re-rendered text equals their decode);
  * parallel tool results must arrive together: a tool message that is last gets '<|im_end|>\\n' after it;
  * (from the marking choice, not the template) the model's own content, reasoning and tool arguments are marked
    too: if the model emitted a special id inside its output (e.g. <tool_call> inside a file body passed to
    edit_file), the re-render encodes it as plain text and the tokens diverge there.
"""
import json
from datetime import datetime

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .tokenizer import ESC, MARK, unmark


class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """the template refused the messages (raise_exception): the server answers 400 with the message"""


def _raise_exception(message):
    raise TemplateRequestError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # transformers' override: no HTML escaping, ensure_ascii False by default
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _strftime_now(fmt):
    return datetime.now().strftime(fmt)


# The sandbox below renders the chat template stored in the model file: input from whoever made the GGUF. Jinja2 before 3.1.6
# has sandbox escapes (CVE-2024-56326, an indirect reference to str.format, fixed in 3.1.5; CVE-2025-27516, the attr filter
# selecting str.format, fixed in 3.1.6), so an older Jinja2 is refused here, whatever installed it (pyproject declares 3.1.6)
JINJA2_MIN = (3, 1, 6)


def jinja2_version_ok(v=None):
    """True when Jinja2 version string v (default: the installed one) is at least JINJA2_MIN"""
    import re
    m = re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', jinja2.__version__ if v is None else v)
    return bool(m) and tuple(int(x or 0) for x in m.groups()) >= JINJA2_MIN


class ChatTemplate:
    def __init__(self, source):
        if not jinja2_version_ok():
            raise RuntimeError(f'Jinja2 {jinja2.__version__} has known sandbox escapes and the chat template comes from the model '
                               f'file: pip install "jinja2>={".".join(map(str, JINJA2_MIN))}"')
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters['tojson'] = _tojson
        env.globals['raise_exception'] = _raise_exception
        env.globals['strftime_now'] = _strftime_now
        self.source = source
        self._t = env.from_string(source)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        """the same call transformers makes: render(messages=, tools=, documents=None, add_generation_prompt=, **kw)"""
        return self._t.render(messages=messages, tools=tools, documents=None,
                              add_generation_prompt=add_generation_prompt, **kwargs)


# ---------------------------------------------------------------------------------------------- effort per model id
# k3-coder runs the medium effort: the template writes no effort line for it, so its system message is the same
# with thinking on or off (cache-friendly). The template's own default (reasoning_effort undefined) is xhigh.
MODEL_PROFILES = {
    'k3-coder': {'enable_thinking': True, 'reasoning_effort': 'medium'},
    'k3-coder-xhigh': {'enable_thinking': True, 'reasoning_effort': 'xhigh'},
    'k3-coder-low': {'enable_thinking': True, 'reasoning_effort': 'low'},
    'k3-coder-nothink': {'enable_thinking': False, 'reasoning_effort': 'medium'},
}
DEFAULT_MODEL = 'k3-coder'


def set_model_alias(alias, thinking=True):
    """the model ids of a server whose model is not the Coder (caminus serve --model-alias, the one-card engines): alias
    (thinking on) and alias-nothink (off) replace the k3-coder profiles in place (the server lists MODEL_PROFILES, imported
    by name), and alias becomes the default for unknown ids. The k3-coder ids (and their reasoning_effort profiles: the
    Coder's template has the effort line, Qwen3.6's has none) stay only for the Coder. thinking=False: a model whose
    template has no think block (Qwen3-Coder-Next): both ids render and parse with thinking off (the reply is content; a
    thinking parse would take all of it as reasoning, since no '</think>' ever comes)"""
    global DEFAULT_MODEL
    if alias == 'k3-coder': return
    MODEL_PROFILES.clear()
    MODEL_PROFILES.update({alias: {'enable_thinking': bool(thinking), 'reasoning_effort': 'medium'},
                           f'{alias}-nothink': {'enable_thinking': False, 'reasoning_effort': 'medium'}})
    DEFAULT_MODEL = alias
CLIENT_KWARGS = ('enable_thinking', 'reasoning_effort', 'preserve_thinking')


def resolve_template_kwargs(model, client=None, allow_client_effort=False):
    """Template kwargs for a request: (profile name, kwargs).

    Precedence: the model id pins reasoning_effort (a client's reasoning_effort is ignored unless
    allow_client_effort), and a -nothink model never thinks. A client may turn thinking OFF for one request
    (enable_thinking false: Claude Code side calls, Anthropic requests without "thinking"); it cannot turn it on for
    a -nothink model. preserve_thinking is the client's (default true; false breaks append-only, see above).
    Unknown model ids get DEFAULT_MODEL's profile."""
    name = model if model in MODEL_PROFILES else DEFAULT_MODEL
    kw = dict(MODEL_PROFILES[name])
    client = client or {}
    if client.get('enable_thinking') is False:
        kw['enable_thinking'] = False
    if allow_client_effort and kw['enable_thinking'] and client.get('reasoning_effort') is not None:
        kw['reasoning_effort'] = client['reasoning_effort']
    if client.get('preserve_thinking') is not None:
        kw['preserve_thinking'] = bool(client['preserve_thinking'])
    return name, kw


# ---------------------------------------------------------------------------------------------- literal-tag marking
def literal_tags(tokenizer):
    """every special-token text llama.cpp would split out (CONTROL and USER_DEFINED)"""
    return sorted(tokenizer.special, key=len, reverse=True)


class LiteralMarker:
    """Marks literal special-token text in request content so it is encoded as plain text.

    mark_text(s): ESC/MARK chars already in s are escaped (ESC x), then each literal tag's leading '<' becomes MARK.
    The template only trims, concatenates and tojson's these strings (tojson keeps private-use chars as-is), and
    its few content tests (startswith('<tool_response>'), endswith) give the same answer on marked text because
    the one place where those tags are structural (a user message that is a whole <tool_response> wrapper, the
    Qwen Code convention) keeps its outer pair unmarked."""

    WRAP_OPEN, WRAP_CLOSE = '<tool_response>', '</tool_response>'

    def __init__(self, tags):
        import re
        self.tags = list(tags)
        self._re = re.compile('|'.join(re.escape(t) for t in self.tags))

    def has_tag(self, s):
        return isinstance(s, str) and (self._re.search(s) is not None or MARK in s or ESC in s)

    def mark_text(self, s):
        if not isinstance(s, str) or not self.has_tag(s):
            return s
        s = s.replace(ESC, ESC + ESC).replace(MARK, ESC + MARK)
        return self._re.sub(lambda m: MARK + m.group(0)[1:], s)

    def mark_deep(self, x):
        if isinstance(x, str):
            return self.mark_text(x)
        if isinstance(x, list):
            return [self.mark_deep(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self.mark_deep(v) for v in x)
        if isinstance(x, dict):
            return {self.mark_deep(k): self.mark_deep(v) for k, v in x.items()}
        return x

    def _mark_user_content(self, c):
        if not isinstance(c, str):
            return self.mark_deep(c)
        t = c.strip()
        if t.startswith(self.WRAP_OPEN) and t.endswith(self.WRAP_CLOSE) and len(t) >= len(self.WRAP_OPEN) + len(
                self.WRAP_CLOSE):
            i = c.index(self.WRAP_OPEN) + len(self.WRAP_OPEN)
            j = c.rindex(self.WRAP_CLOSE)
            return c[:i] + self.mark_text(c[i:j]) + c[j:]
        return self.mark_text(c)

    def mark_messages(self, messages, tools=None):
        """(messages, tools) with literal tags marked; the input is not modified"""
        out = []
        for m in messages:
            m2 = {}
            for k, v in m.items():
                if k == 'role':
                    m2[k] = v
                elif k == 'content' and m.get('role') == 'user':
                    m2[k] = self._mark_user_content(v)
                else:
                    m2[k] = self.mark_deep(v)
            out.append(m2)
        return out, (self.mark_deep(tools) if tools is not None else None)


# ---------------------------------------------------------------------------------------------- prompt builder
class Prompt:
    __slots__ = ('text', 'ids', 'profile', 'kwargs', 'marks')

    def __init__(self, text, ids, profile, kwargs, marks=None):
        self.text, self.ids, self.profile, self.kwargs, self.marks = text, ids, profile, kwargs, marks


class PromptBuilder:
    """messages -> (prompt text, token ids). ids start with the BOS id 248044 (<|endoftext|>) when prepend_bos:
    every K3 chain starts from zeroed state with 248044 at position 0 (D-277); the GGUF itself has add_bos=False,
    so the text does not carry it."""

    def __init__(self, template, tokenizer, prepend_bos=True, mark_literals=True, allow_client_effort=False):
        self.template = template if isinstance(template, ChatTemplate) else ChatTemplate(template)
        self.tok = tokenizer
        self.prepend_bos = prepend_bos
        self.mark_literals = mark_literals
        self.allow_client_effort = allow_client_effort
        self.marker = LiteralMarker(literal_tags(tokenizer))

    @classmethod
    def from_tokenizer(cls, tokenizer, **kw):
        return cls(ChatTemplate(tokenizer.vocab.chat_template), tokenizer, **kw)

    def render_marked(self, messages, tools=None, kwargs=None, add_generation_prompt=True):
        if self.mark_literals:
            messages, tools = self.marker.mark_messages(messages, tools)
        return self.template.render(messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **(kwargs or {}))

    def build(self, messages, tools=None, model=DEFAULT_MODEL, client_kwargs=None, add_generation_prompt=True, marks=False):
        """marks: also give prompt.marks, the prompt cache's message boundaries (caminus.cache.prompt, docs/specs/
        spec-cache-api.md) as token positions: 'root' = the end of the system block (system text + tools: what a sub-agent or a
        new conversation with the same setup shares), 'user' = the start of the last user turn (a user message, or the user
        turn that carries the trailing tool results). Found in the marked render (literal special-token text inside content
        is marked, so every '<|im_start|>' / '<|im_end|>' left is structural) and tokenized up to there: the text before a
        special token tokenizes on its own, so each mark's tokens are a prefix of the prompt's (checked; a mark that is not
        is None)"""
        profile, kw = resolve_template_kwargs(model, client_kwargs, self.allow_client_effort)
        marked = self.render_marked(messages, tools, kw, add_generation_prompt)
        ids = self.tok.encode_prompt(marked)
        if self.prepend_bos and self.tok.bos_id is not None:     # (no BOS in the vocabulary: none, as llama.cpp tokenizes it)
            ids = [self.tok.bos_id] + ids
        mk = self.marks_of(marked, ids) if marks else None
        return Prompt(unmark(marked), ids, profile, kw, mk)

    def marks_of(self, marked, ids):
        """the prompt cache's marks of a marked render whose tokens are ids (see build)"""
        out = dict(root=None, user=None)
        b = 1 if self.prepend_bos else 0
        def pos(ch):
            t = self.tok.encode_prompt(marked[:ch]) if ch else []
            return len(t) + b if list(ids[b:b + len(t)]) == list(t) else None
        if marked.startswith('<|im_start|>system'):
            e = marked.find('<|im_end|>\n')
            if e >= 0: out['root'] = pos(e + len('<|im_end|>\n'))
        u = marked.rfind('<|im_start|>user')
        if u >= 0: out['user'] = pos(u)
        return out
