"""Agent-behaviour signals for the metrics (docs/specs/spec-metrics.md, 'agent behaviour' and 'DFlash'): the content
class of the text being generated, the harness behind a User-Agent, the tool-call outcome of a parsed reply, and the
loop detector over the reasoning and the answer-loop guard. Only bounded partial lines are held while streaming;
monitoring records classes, counts and hashes, never request text.

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
THINK_OPEN = '<think>'
_MARKERS = (TOOL_OPEN, TOOL_CLOSE, FN_OPEN, FENCE)
_MARKER_RE = re.compile('|'.join(re.escape(m) for m in _MARKERS))
_MARKER_PREFIXES = frozenset(m[:n] for m in _MARKERS for n in range(1, len(m)))


class ContentClass:
    """the class of the text being generated (see the module docstring); feed(text) after each pass"""
    __slots__ = ('in_call', 'shell_call', 'fn_pending', 'fn_long', 'in_fence', 'fence_lang', 'lang_pending', 'lang_long', 'tail')
    MAX_FN, MAX_LANG = 128, 64
    MAX_TAIL = max(map(len, _MARKERS)) - 1

    def __init__(self):
        self.in_call = False
        self.shell_call = False
        self.fn_pending = None      # the text after '<function=' while its '>' has not arrived
        self.fn_long = False
        self.in_fence = False
        self.fence_lang = ''
        self.lang_pending = None    # the text after an opening fence while its newline has not arrived
        self.lang_long = False
        self.tail = ''              # only a proper marker prefix, at most MAX_TAIL characters

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
        self.tail = ''
        # Hold only a marker prefix, never re-scan a completed marker. This
        # handles tool/function tags split anywhere, as well as split fences.
        start = 0
        for hit in _MARKER_RE.finditer(s):
            self._plain(s[start:hit.start()])
            self._marker(hit.group())
            start = hit.end()
        rest = s[start:]
        for n in range(min(len(rest), self.MAX_TAIL), 0, -1):
            if rest[-n:] in _MARKER_PREFIXES:
                self.tail, rest = rest[-n:], rest[:-n]
                break
        self._plain(rest)
        return self.cls

    def _plain(self, text):
        """Finish a bounded function name or fence language between markers."""
        if self.fn_pending is not None:
            j = text.find('>')
            part = text if j < 0 else text[:j]
            self.fn_long |= len(self.fn_pending) + len(part) > self.MAX_FN
            self.fn_pending = (self.fn_pending + part[:self.MAX_FN])[:self.MAX_FN]
            if j >= 0:
                self.shell_call = not self.fn_long and self.fn_pending.strip().lower() in SHELL_TOOLS
                self.fn_pending = None
        if self.lang_pending is not None:
            j = text.find('\n')
            part = text if j < 0 else text[:j]
            self.lang_long |= len(self.lang_pending) + len(part) > self.MAX_LANG
            self.lang_pending = (self.lang_pending + part[:self.MAX_LANG])[:self.MAX_LANG]
            if j >= 0:
                self.fence_lang = '' if self.lang_long else self.lang_pending.strip().lower()
                self.lang_pending = None

    def _marker(self, marker):
        if marker == TOOL_OPEN:
            self.in_call, self.shell_call = True, False
            self.fn_pending = None
        elif marker == TOOL_CLOSE:
            self.in_call = self.shell_call = False
            self.fn_pending = None
        elif marker == FN_OPEN:
            if self.in_call:
                self.fn_pending, self.fn_long = '', False
        elif not self.in_call:      # a fence outside a tool call
            if self.in_fence:
                self.in_fence, self.fence_lang, self.lang_pending = False, '', None
            else:
                self.in_fence = True
                self.fence_lang, self.lang_pending, self.lang_long = '', '', False


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


class AnswerLoopGuard:
    """Stop a sustained run of identical prose sentences, outside thinking, fenced code and tool payloads.

    Eight consecutive nonempty lines of at least 24 characters is deliberately stricter than the reasoning metric.
    A real runaway acknowledgement reached 78 copies while that reasoning-only metric stayed clear. This contains
    that failure; it is not a general quality judgement. State is bounded: one line, its hash and a content classifier.
    """
    REPEAT = 8
    MIN_LINE = 24
    MAX_LINE = 4096

    def __init__(self, thinking, think_open=True):
        self.thinking = bool(thinking)
        self.open_pending = '' if thinking and not think_open else None
        self.tail = ''
        self.buf = ''
        self.discard = False
        self.content = ContentClass()
        self.previous = None
        self.run = self.max_repeat = self.repeated = 0

    @property
    def flagged(self):
        return self.max_repeat >= self.REPEAT

    def _reset_run(self):
        self.previous, self.run = None, 0

    def _line(self, line):
        before = self.content.cls
        after = self.content.feed(line + '\n')
        if before != 'prose' or after != 'prose' or any(m in line for m in (TOOL_OPEN, TOOL_CLOSE, FENCE)):
            self._reset_run()
            return
        line = line.strip()
        if not line:                            # a repeated paragraph can have blank lines between its sentences
            return
        plain = line.strip('*#>_` ')
        if len(line) < self.MIN_LINE or not plain or not plain[0].isalpha() or plain[-1] not in '.!?。！？':
            self._reset_run()
            return
        h = hash(line)
        if h == self.previous:
            self.run += 1
            self.repeated += 1
        else:
            self.previous, self.run = h, 1
        self.max_repeat = max(self.max_repeat, self.run)

    def feed(self, text):
        if self.open_pending is not None:
            s = (self.open_pending + text).lstrip(' \t\n\v\f\r')
            if THINK_OPEN.startswith(s) and len(s) < len(THINK_OPEN):
                self.open_pending = s
                return False
            self.open_pending = None
            if s.startswith(THINK_OPEN):
                text = s[len(THINK_OPEN):]
            else:
                self.thinking, text = False, s
        if self.thinking:
            s = self.tail + text
            ends = [(p, marker) for marker in (THINK_END, TOOL_OPEN) if (p := s.find(marker)) >= 0]
            if not ends:
                self.tail = s[-(max(len(THINK_END), len(TOOL_OPEN)) - 1):]
                return False
            end, marker = min(ends)
            self.thinking, self.tail = False, ''
            text = s[end + len(marker):] if marker == THINK_END else s[end:]
        for part in text.splitlines(keepends=True):
            complete = part.endswith('\n')
            if self.discard:
                self.content.feed(part)
                if complete:
                    self.discard = False
                continue
            self.buf += part
            if len(self.buf) > self.MAX_LINE:
                self.content.feed(self.buf)
                self.buf, self.discard = '', not complete
                self._reset_run()
            elif complete:
                self._line(self.buf[:-1])
                self.buf = ''
            if self.flagged:
                return True
        return self.flagged

    def finish(self):
        if self.open_pending is not None:
            self.thinking = False
            pending, self.open_pending = self.open_pending, None
            self.feed(pending)
        if not self.thinking and not self.discard and self.buf:
            self._line(self.buf)
        self.buf, self.tail = '', ''
        return self.flagged
