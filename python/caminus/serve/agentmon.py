"""Agent-behaviour signals for the metrics (docs/specs/spec-metrics.md, 'agent behaviour' and 'DFlash'): the content
class of the text being generated, the harness behind a User-Agent, the tool-call outcome of a parsed reply, and the
loop detector over the reasoning. None of them keeps any text: classes, counts and line hashes only.

ContentClass (engine thread, fed the released text of each pass): 'tool' inside <tool_call> ... </tool_call>, 'shell'
inside a call of a shell tool (run_shell_command, bash, ...) or a fenced block tagged bash / sh / console / ...,
'code' inside any other fenced block, else 'prose' (reasoning and answer text alike). A few string finds per pass.

LoopDetector (HTTP thread, fed the raw text as it arrives): counts each complete reasoning line (stripped, at least
MIN_LINE characters) by its hash; a line seen REPEAT times or more flags the request as a loop (UkisAI's
swift-overthinking report and our 10-30 CTF runs: runaway turns repeat whole reasoning lines). Reasoning is the text
before '</think>' when the request thinks; a request without thinking has none. At most MAX_LINES distinct lines are
counted per request."""
import re

SHELL_TOOLS = frozenset(('run_shell_command', 'bash', 'shell', 'execute_command', 'run_command', 'terminal', 'exec',
                         'execute_bash', 'run_terminal_cmd', 'shell_command', 'cmd', 'powershell'))
SHELL_LANGS = frozenset(('bash', 'sh', 'shell', 'console', 'zsh', 'fish', 'powershell', 'ps1', 'pwsh', 'cmd', 'bat',
                         'shell-session', 'terminal'))
FENCE = '```'
TOOL_OPEN, TOOL_CLOSE = '<tool_call>', '</tool_call>'
FN_OPEN = '<function='
THINK_END = '</think>'


class ContentClass:
    """the class of the text being generated (see the module docstring); feed(text) after each pass"""
    __slots__ = ('in_call', 'shell_call', 'fn_pending', 'in_fence', 'fence_lang', 'lang_pending', 'tail')

    def __init__(self):
        self.in_call = False
        self.shell_call = False
        self.fn_pending = None      # the text after '<function=' while its '>' has not arrived
        self.in_fence = False
        self.fence_lang = ''
        self.lang_pending = None    # the text after an opening fence while its newline has not arrived
        self.tail = ''              # the last characters of the previous piece (a fence split across pieces)

    @property
    def cls(self):
        if self.in_call:
            return 'shell' if self.shell_call else 'tool'
        if self.in_fence:
            return 'shell' if self.fence_lang in SHELL_LANGS else 'code'
        return 'prose'

    def feed(self, text):
        if not text:
            return self.cls
        s = self.tail + text
        start = len(self.tail)
        if self.fn_pending is not None:
            buf = self.fn_pending + text
            j = buf.find('>')
            if j >= 0:
                self.shell_call = buf[:j].strip().lower() in SHELL_TOOLS
                self.fn_pending = None
            elif len(buf) > 128:
                self.fn_pending = None
            else:
                self.fn_pending = buf
        if self.lang_pending is not None:
            buf = self.lang_pending + text
            j = buf.find('\n')
            if j >= 0:
                self.fence_lang = buf[:j].strip().lower()
                self.lang_pending = None
            elif len(buf) > 64:
                self.fence_lang, self.lang_pending = buf.strip().lower()[:32], None
            else:
                self.lang_pending = buf
        if '<' in text or '`' in s:
            self._scan(s, start)
        self.tail = s[-2:]
        return self.cls

    def _scan(self, s, start):
        """markers whose last character lies in the new text (s[start:])"""
        i = 0
        while True:
            hits = []
            for m in (TOOL_OPEN, TOOL_CLOSE, FN_OPEN, FENCE):
                p = s.find(m, i)
                while p != -1 and p + len(m) <= start:   # wholly in the carried tail: seen before
                    p = s.find(m, p + 1)
                if p != -1:
                    hits.append((p, m))
            if not hits:
                return
            p, m = min(hits)
            i = p + len(m)
            if m == TOOL_OPEN:
                self.in_call, self.shell_call = True, False
            elif m == TOOL_CLOSE:
                self.in_call = self.shell_call = False
                self.fn_pending = None
            elif m == FN_OPEN:
                if self.in_call:
                    rest = s[i:]
                    j = rest.find('>')
                    if j >= 0:
                        self.shell_call = rest[:j].strip().lower() in SHELL_TOOLS
                        self.fn_pending = None
                    else:
                        self.fn_pending = rest
                        return
            elif not self.in_call:      # a fence outside a tool call
                if self.in_fence:
                    self.in_fence, self.fence_lang, self.lang_pending = False, '', None
                else:
                    self.in_fence = True
                    rest = s[i:]
                    j = rest.find('\n')
                    if j >= 0:
                        self.fence_lang, self.lang_pending = rest[:j].strip().lower(), None
                    else:
                        self.fence_lang, self.lang_pending = '', rest
                        return


# ------------------------------------------------------------------------------------------------ harnesses
HARNESSES = ('qwen-code', 'claude-code', 'hermes', 'opencode', 'codex', 'aider', 'llm-top', 'anthropic-sdk',
             'openai-sdk', 'curl', 'python', 'other', 'none')
_HARNESS_RULES = (
    ('qwen-code', re.compile(r'qwen[-_ ]?code|qwencode', re.I)),
    ('claude-code', re.compile(r'claude-cli|claude-code|claude_code', re.I)),
    ('hermes', re.compile(r'hermes', re.I)),
    ('opencode', re.compile(r'opencode', re.I)),
    ('codex', re.compile(r'codex', re.I)),
    ('aider', re.compile(r'aider', re.I)),
    ('llm-top', re.compile(r'llm-top|llmtop', re.I)),
    ('anthropic-sdk', re.compile(r'^anthropic/', re.I)),
    ('openai-sdk', re.compile(r'^openai/', re.I)),
    ('curl', re.compile(r'^curl/', re.I)),
    ('python', re.compile(r'^python-(urllib|requests|httpx)|^python/', re.I)),
)


def harness_of(user_agent):
    """a User-Agent -> one of HARNESSES (a fixed set: the raw string never becomes a label)"""
    if not user_agent:
        return 'none'
    ua = user_agent[:512]
    for name, rx in _HARNESS_RULES:
        if rx.search(ua):
            return name
    return 'other'


# ------------------------------------------------------------------------------------------------ tool calls
_FAIL_NOTES = (('undeclared tool', 'undeclared'), ('without a call', 'malformed'), ('JSON without a name', 'malformed'),
               ('unterminated <parameter=', 'malformed'), ('parameter name too long', 'malformed'),
               ('text inside <function> dropped', 'malformed'))


def tool_call_report(msg, notes):
    """(usable calls, failures {kind: n}, lenient notes) of a parsed reply (toolparse.ParsedMessage and the parser's
    notes). Failures: 'cut' (a call the end cut inside a value: never reported), 'unclosed' (a call that lost its
    closing tags), 'undeclared' (a tool the request did not declare), 'malformed' (tag text that was not a call). The
    other notes are leniency the parser applied (aliases, coerced values, missing newlines)."""
    fails = {'cut': 0, 'unclosed': 0, 'undeclared': 0, 'malformed': 0}
    ok = 0
    for c in msg.tool_calls:
        if c.cut:
            fails['cut'] += 1
        else:
            ok += 1
            if not c.complete:
                fails['unclosed'] += 1
    lenient = 0
    for n in notes:
        for pat, kind in _FAIL_NOTES:
            if pat in n:
                fails[kind] += 1
                break
        else:
            if 'cut off' not in n and 'missing </' not in n:     # counted from the calls above
                lenient += 1
    return ok, fails, lenient


# ------------------------------------------------------------------------------------------------ loops
class LoopDetector:
    MIN_LINE = 12
    REPEAT = 3
    MAX_LINES = 4096

    def __init__(self, thinking, on_live=None):
        self.reasoning = bool(thinking)
        self.buf = ''
        self.counts = {}
        self.max_repeat = 0
        self.repeated = 0           # line occurrences beyond the first
        self.on_live = on_live

    @property
    def flagged(self):
        return self.max_repeat >= self.REPEAT

    def feed(self, text):
        if not self.reasoning or not text:
            return
        s = self.buf + text
        j = s.find(THINK_END)
        if j >= 0:
            s = s[:j] + '\n'
            self.reasoning = False
        lines = s.split('\n')
        self.buf = lines.pop()
        if len(self.buf) > 4096:
            self.buf = self.buf[-4096:]
        for line in lines:
            self._line(line)

    def _line(self, line):
        line = line.strip()
        if len(line) < self.MIN_LINE:
            return
        h = hash(line)
        c = self.counts.get(h)
        if c is None:
            if len(self.counts) < self.MAX_LINES:
                self.counts[h] = 1
            return
        c += 1
        self.counts[h] = c
        self.repeated += 1
        if c > self.max_repeat:
            self.max_repeat = c
            if self.on_live is not None:
                self.on_live(c)

    def finish(self):
        if self.reasoning and self.buf:
            self._line(self.buf)
        self.buf = ''
        return self.flagged, self.repeated
