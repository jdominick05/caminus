"""The owner's language, named in the prompt and held with one nudge (issue #354).

A reply that switches language mid-goal ("分支已推送。现在开 PR (base ember-next)。" while the owner works in
English) is not usable: the chat is what the owner reads and the done judge grades. Two halves, both cheap
enough for every turn:

- the owner's language is a session setting (`caminus ember agent language --to LANGUAGE`, key `language` in
  config.json). Without one it is derived from the owner's own prompts (the first eight goals, prose only, a
  letter census): when one script holds DERIVE_SHARE of their letters it names the language, else English.
  Latin is the exception: English, Spanish and French are one script, so Latin-script goals name no language
  and the session keeps English until `--to` names it.
  `prompt.py` names it, so the brain is told which language to keep to in replies, notes, files, commit
  messages and GitHub text.
- after every stored assistant reply, `check` censuses the prose's units by script (a unit, not a letter: the
  definition below is the one the floors are in). When a majority of the
  reply's units is in a script other than the owner's, the drift is an event (kind `language_drift`, with the
  brain that wrote it, so a drift rate per brain is visible in the event log) and, at most once every
  DRIFT_EVERY assistant turns, one short nudge goes into the goal's inbox: the next turn reads
  "[nudge #N, keeps your plan: ... Reply in English ...]".

What a unit is: one letter in a script that does not space its words (Han, Kana) and one run of letters
otherwise, so a comma, a digit or a hyphen ends a run: `don't` and `ember-next` are two units each, and
"OK, let me check and rerun" is six. The floors below are in units, not words (review P2).
Counting Han characters is what catches a reply like the one above, where raw letters are mostly Latin
identifiers ("PR", "base", "ember-next"); counting words keeps an English reply that quotes a few CJK
characters far under the threshold. Fenced code and inline `code` spans are not prose and are left out, so a
code block does not drown the narration around it.
"""
import bisect
import json
from pathlib import Path
import re
import sqlite3
import urllib.parse

from . import sim

DEFAULT = 'English'
MIN_UNITS = 6          # fewer units than this cannot name a language: "ok", a path, a one-word reply
WHOLE_MIN_UNITS = 3    # a reply with none of the owner's units at all is another language from a word, not a pair
DRIFT_EVERY = 3        # assistant turns between two nudges on one goal (the issue's "at most once every N turns")
OTHER_SHARE = .5       # "most letters of a reply are in another script": a strict majority
DERIVE_SHARE = .6      # your goals name a language only when one script holds this share of their letters
QUOTE_MIN_UNITS = 3    # a straight-quoted span shorter than this is a term (`"on"`, `"开"`), not a citation

# (first, last, script), sorted by first; Unicode blocks, letters only where a block mixes letters and digits
_RANGES = sorted([
    (0x0041, 0x005A, 'Latin'), (0x0061, 0x007A, 'Latin'), (0x00AA, 0x00AA, 'Latin'), (0x00BA, 0x00BA, 'Latin'),
    (0x00C0, 0x00D6, 'Latin'), (0x00D8, 0x00F6, 'Latin'), (0x00F8, 0x02AF, 'Latin'),
    (0x0370, 0x0373, 'Greek'), (0x0376, 0x0377, 'Greek'), (0x037B, 0x037D, 'Greek'), (0x037F, 0x037F, 'Greek'),
    (0x0386, 0x0386, 'Greek'), (0x0388, 0x03FF, 'Greek'),
    (0x0400, 0x052F, 'Cyrillic'), (0x1C80, 0x1C8F, 'Cyrillic'), (0x2DE0, 0x2DFF, 'Cyrillic'), (0xA640, 0xA69F, 'Cyrillic'),
    (0x05D0, 0x05EA, 'Hebrew'), (0x05EF, 0x05F2, 'Hebrew'), (0xFB1D, 0xFB4F, 'Hebrew'),
    (0x0620, 0x065F, 'Arabic'), (0x066E, 0x06D3, 'Arabic'), (0x06D5, 0x06D5, 'Arabic'), (0x06E5, 0x06E6, 'Arabic'),
    (0x06EE, 0x06FF, 'Arabic'), (0x0750, 0x077F, 'Arabic'), (0x08A0, 0x08FF, 'Arabic'),
    (0xFB50, 0xFDFF, 'Arabic'), (0xFE70, 0xFEFF, 'Arabic'),
    (0x0904, 0x0939, 'Devanagari'), (0x093D, 0x093D, 'Devanagari'), (0x0950, 0x0950, 'Devanagari'),
    (0x0958, 0x0961, 'Devanagari'), (0x0972, 0x097F, 'Devanagari'), (0xA8E0, 0xA8FF, 'Devanagari'),
    (0x0E01, 0x0E3A, 'Thai'), (0x0E40, 0x0E5B, 'Thai'),
    (0x1100, 0x11FF, 'Hangul'), (0x3130, 0x318F, 'Hangul'), (0xA960, 0xA97F, 'Hangul'),
    (0xAC00, 0xD7AF, 'Hangul'), (0xD7B0, 0xD7FF, 'Hangul'),
    (0x3041, 0x3096, 'Kana'), (0x309B, 0x309F, 'Kana'), (0x30A1, 0x30FA, 'Kana'), (0x30FD, 0x30FF, 'Kana'),
    (0x31F0, 0x31FF, 'Kana'), (0xFF66, 0xFF9F, 'Kana'),
    (0x3400, 0x4DBF, 'Han'), (0x4E00, 0x9FFF, 'Han'), (0xF900, 0xFAFF, 'Han'),
    (0x20000, 0x2A6DF, 'Han'), (0x2A700, 0x2EBEF, 'Han'), (0x30000, 0x3134F, 'Han'),
    (0x1D00, 0x1D7F, 'Latin'), (0x1E00, 0x1EFF, 'Latin'), (0x1F00, 0x1FFF, 'Greek'),
    (0x2C60, 0x2C7F, 'Latin'), (0xA720, 0xA7FF, 'Latin'), (0xAB30, 0xAB6F, 'Latin'), (0xFB00, 0xFB06, 'Latin'),
    (0xFF21, 0xFF3A, 'Latin'), (0xFF41, 0xFF5A, 'Latin'),
])
_STARTS = [r[0] for r in _RANGES]
SPACELESS = ('Han', 'Kana')

# A language name reads as its script, so `agent language NAME Chinese` knows what to census without a model.
# Latin is the default and the widest family; an unusual language can still be named (the prompt follows it),
# the census just cannot judge it.
_SCRIPTS_FOR = {}
for _script, _names in {
        'Latin': 'english french german spanish portuguese italian dutch swedish norwegian danish finnish polish '
                 'czech slovak hungarian turkish indonesian malay vietnamese romanian catalan basque irish welsh '
                 'estonian latvian lithuanian slovenian croatian albanian azerbaijani uzbek tagalog filipino hausa '
                 'yoruba igbo zulu xhosa afrikaans swahili',
        'Cyrillic': 'russian ukrainian bulgarian serbian belarusian macedonian kazakh kyrgyz tajik tatar chechen mongolian',
        'Greek': 'greek', 'Arabic': 'arabic persian farsi urdu pashto kurdish sindhi uyghur',
        'Hebrew': 'hebrew yiddish', 'Devanagari': 'hindi marathi nepali sanskrit konkani bhojpuri',
        'Thai': 'thai', 'Hangul': 'korean', 'Han': 'chinese mandarin cantonese', 'Kana': 'japanese'}.items():
    _SCRIPTS_FOR[_script.lower()] = (_script,)
    for _name in _names.split():
        _SCRIPTS_FOR[_name] = (_script,)
# Japanese writes in two scripts, and a Chinese owner reading kanji is still a Han script
_SCRIPTS_FOR['japanese'] = ('Han', 'Kana')
_SCRIPTS_FOR['日本語'] = ('Han', 'Kana')
_SCRIPTS_FOR['中文'] = ('Han',)
_SCRIPTS_FOR['汉语'] = ('Han',)
_SCRIPTS_FOR['한국어'] = ('Hangul',)

_LANG_OF = {'Cyrillic': 'Russian', 'Greek': 'Greek', 'Arabic': 'Arabic', 'Hebrew': 'Hebrew',
            'Devanagari': 'Hindi', 'Thai': 'Thai', 'Hangul': 'Korean', 'Kana': 'Japanese', 'Han': 'Chinese'}

_FENCE = re.compile(r'```.*?(?:```|\Z)', re.S)      # an unterminated fence runs to the end of the reply
_TICK = re.compile(r'`[^`\n]*`')
# a quotation is a citation, not the reply's language: a quoted foreign sentence must not make an English reply a
# drift. Straight quotes need guards so an apostrophe (`don't`, `it's`) is not read as an opening quote: the opening
# one follows a non-letter and the closing one precedes a non-letter (#354 review P2).
_QUOTED = re.compile(r'「[^」\n]*」|『[^』\n]*』|“[^”\n]*”|‘[^’\n]*’|«[^»\n]*»|"[^"\n]*"'
                     r"|(?<![\w'])'[^'\n]*'(?![\w'])")


def _strip_quoted(text):
    """`text` with each citation removed, but a short quoted span kept: it is a term, not a quotation.

    A quote carries no signal that its author meant a citation — a markdown attribute, a shell snippet and a label
    all use one — so the pair alone is not enough. A narration that labels flags (`set the flag "开" and the label
    "关"`) held a reply's only foreign-script units inside two same-line pairs, and dropping them censused the reply
    as pure English: a drift that never fired (#390 review P2, r23). The floor is on the span's units, so a one-word
    label stays in the census and a sentence in quotes is still a citation. It applies to every quote form: `“开”`
    is the same label as `"开"`, and keeping only the straight form's floor censused a curly-quoted label away.

    The pair is bounded to one line in every form, for the reason the straight form already was: the typographic
    forms used to cross lines, so a quotation opened in one paragraph and closed in another swallowed the prose
    between them (73 characters in, 6 out, measured) and a reply could never drift there (#390 review P2-2). A
    quotation that spans lines now stays in the census, the same documented cost as the straight form.
    """
    def one(match):
        span = match.group(0)
        if _census(prose(span[1:-1]))[1] < QUOTE_MIN_UNITS:
            return span
        return ' '
    return _QUOTED.sub(one, text)


def prose(text):
    """The text without code and quotations: a code block or a cited sentence is not a reply's language.

    An unterminated fence (a reply cut off mid-block, or a ` ```markdown ` that never closes) truncates the rest:
    leaving its body in the census let a code block's CJK literals outweigh the narration around it (#354 review P2).

    A straight-quote pair is bounded to one line, for the opposite reason: a single unpaired `"` (a markdown
    attribute such as `href="x`, a mis-escaped snippet) used to pair with the next quote or with nothing and drop
    everything after it, so a genuine drift later in the reply was never detected (#390 review P2). The cost is
    that a quotation spanning lines stays in the census; a term-sized one cannot move the majority on its own.

    And a *short* quoted span is not a citation at all: a label (`set the flag "开"`) is the reply's own
    word, and stripping it censused an otherwise-foreign reply as English and missed the drift (#390 review P2,
    r23). `_strip_quoted` applies the floor to every quote form, typographic ones included (#390 review P2-2).
    """
    return _TICK.sub(' ', _strip_quoted(_FENCE.sub(' ', text or '')))


def script_of(ch):
    """The script of one character, '' for anything that is not a letter in the table."""
    o = ord(ch)
    i = bisect.bisect_right(_STARTS, o) - 1
    if i >= 0:
        first, last, script = _RANGES[i]
        if o <= last:
            return script
    return ''


def units(text):
    """Units per script of a reply (see the module docstring), and their total: prose in, one census out."""
    return _census(prose(text))


def _census(text):
    """Units per script of text that is already prose: `units` is the one wrapper that strips code and quotations.

    Counting the text it is given (not `prose(text)` again) keeps one pass per reply (rev-muse P2 on `58d39b8`):
    `units()` used to hand `prose(text)` to a counter that applied `prose()` a second time, and the quote floor
    reached the same counter, so the reply the checker recorded and the span the floor measured were counted by
    two passes whose agreement nothing pinned. The floor still prose's its own span (`_strip_quoted`), because a
    code block inside a quotation is not the citation's units; that is a sub-span, not a second pass over the
    reply.
    """
    per = {}
    script, run = '', 0
    for ch in text:
        s = script_of(ch)
        if s == script:
            run += 1
            continue
        if script:
            per[script] = per.get(script, 0) + (run if script in SPACELESS else 1)
        script, run = s, (1 if s else 0)
    if script:
        per[script] = per.get(script, 0) + (run if script in SPACELESS else 1)
    return per, sum(per.values())


def drift(text, scripts):
    """(script, share) when a strict majority of the reply's units is in one script the owner does not write in.

    A reply in a spaceless script borrows Latin terms — `git push origin pr-390`, `pytest tests/x.py` — inside
    the clauses that also carry the owner's script; those are terms, not the reply's language, and counting
    them would nudge a correct Chinese reply back to Chinese. A clause that is entirely in another script is
    still the reply switching, so a real drift is caught.

    The floor depends on how many of the owner's own units the reply has, not on the reply's total size: one
    unit of the owner's script in an otherwise-foreign reply is a borrowed term (`ok`, a `git push`), so the
    whole-reply floor WHOLE_MIN_UNITS applies and two of them do not buy the larger one; two or more are
    content, so a word or a path in another script is not a language ("ok", `文件 missing`) and MIN_UNITS holds
    it. A reply with none of the owner's units at all is another language from WHOLE_MIN_UNITS on
    ("好的。完成了。" is 5, and a bare two-character path like `文件` is a fragment, not a reply: review P2).
    One floor for both shapes let an English owner's short Chinese confirmations pass unheld forever, however
    many arrived (review P2); keying the mixed floor on the total let a reply like "文件在 ok" (three Han units
    and one borrowed `ok`) pass both (review P2).

    A citation is not the reply's language, and `prose()` strips it in every quoted or code form, so the same
    foreign sentence in quotes is not censused. Bare, it is: a two-word English preamble followed by an unquoted
    Han phrase ("Good point — 谢谢你不错不错不错.") is 82 % Han by units and fires (review P2, r21). No floor can
    separate that from a real switch, since by units it is one, and the replies the floors were tuned on (ten or
    more English units around a few CJK terms) stay far under it; the phrase has to be quoted to read as cited.
    """
    hit = drift_detail(text, scripts)
    return (hit['script'], hit['share']) if hit else None


def drift_detail(text, scripts):
    """The census behind `drift`: the drift's script, every foreign script's units, own units and total.

    `drift` returns one script and a share, and the share is the SUM of the non-owner scripts while the name
    is only the largest of them, so a reply split near evenly between two foreign scripts is described by a
    name that holds half its share (review P2, r22). The caller that records the event has to show the split;
    the caller that only needs a yes/no keeps the pair `drift` returns.
    """
    if not scripts:
        return None
    if any(s in SPACELESS for s in scripts):
        per, total = _census(_without_borrowed(text, scripts))   # that helper returns prose: one pass, not two
    else:
        per, total = units(text)
    foreign = {s: n for s, n in per.items() if s not in scripts}
    other = sum(foreign.values())
    own = total - other
    if not other or other < WHOLE_MIN_UNITS or (own >= 2 and total < MIN_UNITS):
        return None
    if other / total <= OTHER_SHARE:
        return None
    return dict(script=max(foreign, key=foreign.get),
                scripts=dict(sorted(foreign.items(), key=lambda kv: (-kv[1], kv[0]))),
                own=own, total=total, share=other / total)


_CLAUSE = re.compile(r'[.!?;,\u3002\uff01\uff1f\uff1b\uff0c\n]')


def _without_borrowed(text, own):
    """`text` with the borrowed runs dropped from clauses the owner's own script out-measures.

    A spaceless-script owner (Han, Kana) writes Latin terms inside their own clauses — `git push origin pr-390`,
    `pytest tests/x.py` — and those are terms, not the reply switching language. Dropping every Latin run of
    every clause that has one own-script unit went too far: a Han owner whose reply is mostly English with one
    Han term per clause ("We pushed 分支 the branch. The pull 请求 request is open.") had all its Latin dropped
    and measured as a Han reply, so the drift was never caught (review P2, r15). The clause decides now: the
    scripts the owner does not write are the clause's language only when, together, they out-measure the owner's
    own units in that clause; otherwise they are terms inside it.

    It was Latin only at first, so a Han reply that quoted Greek or Cyrillic names kept every one of those units
    and could be nudged on a correct reply (review P2, r20). The rule is about the clause, not about which script
    the borrowed term happens to be written in, so it now drops every non-owner script the clause's own script
    out-measures. The cost is the same one the Latin rule already accepted, in the other direction: a reply that
    really did switch to another script while planting one owner-script term per clause reads as the owner's.
    """
    kept = []
    for clause in _CLAUSE.split(prose(text)):
        if any(script_of(ch) in own for ch in clause):
            per, _ = _census(clause)          # clauses come from `prose`, so the counter takes them as they are
            borrowed = [s for s in per if s not in own]
            if sum(per[s] for s in borrowed) <= sum(per.get(s, 0) for s in own):
                clause = ''.join(ch for ch in clause if script_of(ch) not in borrowed)
        kept.append(clause)
    return ' '.join(kept)


def named(name, source):
    """The owner's language as the prompt and the check use it: its name, its scripts, where it came from.

    The table knows the languages people usually name. For one it does not, a configured name (`--to पाली`) is
    read as the script it is written in, when its own letters are in one: leaving the drift check off for such
    a name let a session run for days unheld with one log line as the only sign of it (#390 review P2). A name
    written in Latin letters (`Esperanto`, `Pali`) therefore censuses Latin, which still holds a reply that
    leaves Latin script altogether; the cost is the other direction, a Pali session in Devanagari that is
    nudged for it, which is why the CLI says where the scripts came from.

    Only config names fall back: a derived name is ours to keep in the table (`_LANG_OF`), and a derived name
    with no table must stay visible as `scripts == ()` so the test that guards `_LANG_OF` keeps its teeth.
    """
    name = str(name or '').strip() or DEFAULT
    scripts = _SCRIPTS_FOR.get(name.lower(), ())
    if not scripts and source == 'config' and census_of(name) == 'name':
        per, _total = letters(name)
        scripts = (max(per, key=per.get),)
    return dict(name=name, scripts=scripts, source=source)


def census_of(name):
    """Where a configured name's scripts come from: `table`, its own spelling (`name`), or nothing (``).

    Split out of `named` so that dict keeps its three keys: `named`'s shape is read by the prompt, the CLI and
    a dozen assertions, and the one caller that must tell the owner which case it got is the CLI.
    """
    name = str(name or '').strip() or DEFAULT
    if _SCRIPTS_FOR.get(name.lower()):
        return 'table'
    per, total = letters(name)
    if per and per[max(per, key=per.get)] * 2 > total:
        return 'name'
    return ''


def letters(text):
    """Units per script counting every letter of a spaceless script separately from every letter elsewhere.

    The drift check counts a Han run as its characters (a Chinese sentence has few spaces, so one word would
    hide it), but that same count over a *goal* lets a pasted log line outvote the owner: "Fix the failing
    push step. Log: 测试失败了，分支已推送，请检查。" is 13 Han units against 6 English words. A letter
    census sees 26 Latin letters against 13 Han and keeps English, which is what the owner wrote.
    """
    per = {}
    for ch in prose(text):
        s = script_of(ch)
        if s:
            per[s] = per.get(s, 0) + 1
    return per, sum(per.values())


def _decisive(per, total):
    """The language a letter census names, else None: one script at DERIVE_SHARE of at least MIN_UNITS letters."""
    if total < MIN_UNITS:
        return None
    script = max(per, key=per.get)
    if per[script] / total < DERIVE_SHARE:
        return None
    if script == 'Han' and per.get('Kana', 0) >= .15 * (per['Han'] + per.get('Kana', 0)):
        script = 'Kana'                            # kanji and kana together: Japanese, not Chinese
    if script == 'Latin':
        return None                                # one script, many languages: not a derivation
    if (language := _LANG_OF.get(script)):
        return named(language, 'your goals')
    return None


def derive(prompts):
    """The owner's language from their own prompts, when one script is clear enough to name, else None.

    Read as letters over prose, and the FIRST prefix that names a language names it: a later goal cannot
    rename the session, so the prompt (a cached prefix) changes only until then. That rule is a property of the
    corpus, not of the running process, so `owner` (a turn), `from_home` (the CLI) and a restart all land on
    the same answer for the same goals. Reading the whole window and deciding once let a second goal flip a
    first one that had already named the session, and a pin kept on the runtime hid that from every other
    reader (review P2).

    Read as letters over prose: an owner who writes English around a pasted line in another script is English,
    and one excerpt cannot name the session (a wrong name changes the prompt and nudges every reply of it).
    Fenced code is not the owner's prose either. Japanese is kanji and kana together, so Kana shares the count
    with Han instead of splitting it.

    Latin is the exception: English, Spanish, French and German are one script, so a Latin-majority corpus
    names no language (None, and the caller keeps the documented default). Naming it English would tell a
    Spanish owner "your human works in English" and steer their replies there; `--to LANGUAGE` is the setting
    for an owner who writes a Latin-script language other than English.
    """
    per, total = {}, 0
    for p in prompts:
        more, n = letters(p or '')
        if not n:
            continue
        total += n
        for script, count in more.items():
            per[script] = per.get(script, 0) + count
        if (language := _decisive(per, total)):
            return language
    return None


# Created order, never `position`: a queue reorder rewrites `position` (store.queue_move), and the corpus
# a session derives its language from must not change under the owner, the CLI or a restart for that reason.
_GOALS = 'SELECT id, prompt FROM goals ORDER BY created, id LIMIT 8'


def owner(rt):
    """The owner's language for this session: config `language`, else the letters of their goals, else English.

    `derive` is a pure function of the goal window and the window is created order, so this answer cannot drift
    between a turn, the CLI (`from_home`) and a restart; the cache below is an optimisation, not a second
    opinion, and is keyed by the window's own goal ids, not its size, so a short first goal does not pin the
    session before one names it and a goal deleted and another added keeps the cache honest (review P2: a count
    key served a language derived from a window that no longer exists).
    Neither cache lives in config: the setting is `--to`, and a derived name that were written there would be
    a pin `--auto` could not drop.
    """
    name = str((getattr(rt, 'config', None) or {}).get('language') or '').strip()
    if name:
        return named(name, 'config')
    try:
        rows = list(rt.store.db.execute(_GOALS))
        key = tuple(r['id'] for r in rows)
    except sqlite3.Error:
        return named(DEFAULT, 'default')            # a store error is not an answer to cache
    cache = getattr(rt, '_language_owner', None)
    if cache and cache[0] == key:
        return cache[1]
    who = derive([r['prompt'] for r in rows]) or named(DEFAULT, 'default')
    rt._language_owner = (key, who)
    return who


def from_home(home, config=None):
    """`owner` for a caller with no runtime (the CLI): read config.json and, if it exists, the store read-only.

    It must not create either: `agent prompt --home NOT-INITIALIZED` prints a prompt before `init` and leaves
    the directory absent.

    `config` is for an op that has already read config.json and must not read it again: `agent language --to`
    writes the setting and then asks this function what it set, and a second read in between would let another
    writer's value be reported as the one this op just wrote (#390 review P2). Passing it changes only where the
    `language` key comes from; the store read below still happens when the config names no language.
    """
    home = Path(home)
    if config is None:
        config = {}
        try:
            config = json.loads((home / 'config.json').read_text())
        except (OSError, ValueError):
            pass
    name = str(config.get('language') or '').strip() if isinstance(config, dict) else ''
    if name:
        return named(name, 'config')
    prompts = []
    store = home / 'agent.sqlite3'
    if store.is_file():
        try:
            db = sqlite3.connect('file:%s?mode=ro' % urllib.parse.quote(str(store)), uri=True)
            try:
                db.row_factory = sqlite3.Row
                prompts = [r['prompt'] for r in db.execute(_GOALS)]
            finally:
                db.close()
        except sqlite3.Error:
            pass
    return derive(prompts) or named(DEFAULT, 'default')


def _reply_in(name, detail, source):
    """The nudge's text: one line, naming the owner's language and the script(s) the reply slid into.

    `share` is the sum of the foreign scripts, so a reply split near evenly between two of them is nudged
    naming only the larger one and reads as if the other were the owner's (review P2, r22). Name the runner-up
    too when it is at least half the winner.

    A `default` source names no language, as the prompt does not (#390 review P1-2): `derive` refuses Latin
    script because English, Spanish and French share it, so "Reply in English" told a Spanish owner who never
    ran `--to` to answer in a language that is not theirs. The nudge describes the language instead, and the
    brain reads it from the owner's own goals, which are in its history.
    """
    per, script = detail['scripts'], detail['script']
    runner = next((s for s in per if s != script), None)
    if runner and per[runner] * 2 >= per[script]:
        script = '%s and %s' % (script, runner)
    owner = ("%s, the owner's language" % name if source != 'default'
             else "the language your human writes their goals in")
    return "Reply in %s: your last reply came out mostly in another script (%s)." % (owner, script)


def check(rt, gid, text, cfg=None, due=True):
    """Record a drift in `text` and, when the goal is due for one, return the nudge for its next turn.

    Every detection is an event (kind `language_drift`, the brain in its data, so a rate per brain is a count
    of that event); the nudge comes at most once every DRIFT_EVERY assistant turns of the goal, counted from
    the last turn that was nudged (not the last detection: a reply that keeps drifting is detected every turn,
    and counting from the detection would push the next nudge back on every one of them).

    `due=False` is for a reply that ends the goal on this tick (no tool call): the drift is still an event, but
    no nudge is queued for a turn that never comes, and `nudged` stays false so the cadence is not advanced by
    it (review P2). A free-time goal (`free`, `dream`) is `due=False` for the same reason and has no next
    graded turn: a drift there is recorded, and the prompt's language is the only thing holding it (review P2).
    Both queries run over the goal's own rows, so the per-turn cost is bounded by this goal's history, not by the
    whole log. `EXPLAIN QUERY PLAN` on this schema shows `SEARCH events USING INDEX events_goal (goal=? AND kind=?)`
    for the first and the same index as a covering one with `rowid>?` for the second. The `json_extract` predicate
    in the first is a filter over this goal's drift rows, not an index term (review P2: measured, and the earlier
    wording claimed more than the plan shows).
    The event log's per-brain drift rate is therefore a count of DETECTIONS, not of nudges, and a free-time goal
    contributes detections that were never followed by a nudge: a rate that pools free and work goals mixes the
    two populations and should be split by goal kind before it is compared (review P2, r23).
    """
    who = owner(rt)
    scripts = who['scripts']
    if not scripts:
        # A configured name whose letters name no single script leaves the drift check switched off, in one of
        # two shapes: its own letters are in no script the census knows (`--to ᚠᚢᚦ`), or they are split between
        # known scripts with no one in the majority (`--to ab 中文`). Either way the CLI prints a `check: false`
        # note, but a session started before or elsewhere had nothing in its log to say why no drift was ever
        # detected (#390 review P2). A name the table does not know but that is written in a known script
        # (`--to Pali`, Latin letters) censuses that script and is NOT this case. One event per process per
        # resolved name, not per turn: this runs on every reply, and the answer cannot change while the process
        # holds the same config.
        seen = rt.__dict__.setdefault('_language_unchecked', set())
        if who['source'] == 'config' and who['name'] not in seen:
            seen.add(who['name'])
            try:
                rt.store.event('language_setting_unchecked',
                               dict(language=who['name'], reason='no script table for this name'), gid)
            except sqlite3.Error:
                pass        # (the same best effort as the drift event below: never a dead turn for a log line)
        return ''
    detail = drift_detail(text, scripts)
    if not detail:
        return ''
    script, share = detail['script'], detail['share']
    # The last nudged drift, from the whole log: a LIMIT 32 window of the newest drifts forgot the nudge as soon
    # as a goal drifted 32 more times, and the cadence then fired again early (review P2). Ask SQLite for the
    # maximum seq with json_extract, which is how other event data is read here; a runtime attribute would be
    # lost on the daemon's restart, and the log is already the record.
    #
    # `since` counts the reply being censused: `store.model_reply` writes the `assistant` event before `_tick`
    # calls `check` (store.py:480, runtime.py:2424), so a nudge at reply N reads `since == 3` at reply N+3 and
    # fires there, which is what DRIFT_EVERY means. Recording the reply after the check would silently stretch
    # the cadence by one reply per nudge, so the count is pinned by `test_the_cadence_counts_the_reply_being_censused`
    # rather than left to the call order.
    try:
        last = rt.store.db.execute("SELECT MAX(seq) FROM events WHERE goal=? AND kind='language_drift' AND "
                                   "json_extract(data,'$.nudged')=1", (gid,)).fetchone()[0]
        since = rt.store.db.execute("SELECT COUNT(*) FROM events WHERE goal=? AND kind='assistant' AND seq>?",
                                    (gid, last or 0)).fetchone()[0]
        nudge = due and (last is None or since >= DRIFT_EVERY)
        # `script` is the largest foreign script and `share` the sum of all of them, so the split is recorded
        # beside them: without it the event described a reply as more Han than it was (review P2, r22).
        rt.store.event('language_drift', dict(script=script, scripts=dict(detail['scripts']),
                                              language=who['name'], share=round(share, 3),
                                              brain=sim.brain(sim.serving(rt, cfg)), nudged=nudge), gid)
    except sqlite3.Error as e:
        # A locked db or a schema drift here raises inside runtime._tick right after the reply was recorded and
        # before the inbox write, so the turn died mid-tool-call; owner() above already treats an unreadable store
        # as tolerable, and a drift event lost this turn is re-detected on the next one - not worth a dead turn
        # (review P2). Losing it silently is its own cost, though: the event log's per-brain drift rate is what
        # this feature is measured by, so the failure itself is recorded (review P2). If the store is the thing
        # that failed, the record fails the same way and is lost too: best effort, never a second exception.
        try:
            rt.store.event('drift_check_error', dict(error='%s: %s' % (type(e).__name__, str(e)[:200]),
                                                     script=script, language=who['name']), gid)
        except sqlite3.Error:
            pass
        # The cadence is unknown now, but the drift is not: a turn that is due gets the nudge anyway. The caller
        # writes the inbox row in its own transaction, so one failed read or insert need not cost the nudge too,
        # and an unnudged real drift is the silent failure this check exists to prevent (review P2 r18). A failure
        # that repeats nudges again: a repeated correct nudge is visible and costs a turn, a lost one costs none.
        return _reply_in(who['name'], detail, who['source']) if due else ''
    if not nudge:
        return ''
    return _reply_in(who['name'], detail, who['source'])
