"""/delegate (issue #342, full version): a child goal run through this harness's own goal loop.

The parent states a task and what done looks like; the child is an ordinary goal of the same store - its own
conversation, todo list, done check, gate, verify and spend row - with limits that make it safe to run from
inside another goal:

- permissions, chosen per call by the parent's brain: `read` (the default) may not change anything - write, edit,
  checkpoint, rollback are refused and terminal runs only a read-only command; `write` may change files, but only
  inside the worktree the delegate creates for it (a branch off HEAD), so the parent's tree is untouched until it
  merges or discards. A parent in plan mode may not grant write: a child never gets more than its parent has. The
  sandbox stays available to both: the folder it sees is a read-only bind at /work and /box dies with the box.
  Residual risk, stated plainly: a writing child is trusted the way the parent is - the worktree keeps its changes
  apart from the parent's tree, it is not a jail for a hostile process. A program that computes a path at run time
  can still write anywhere: `delegate._escaping` refuses the spellings that hand a program to a shell or an
  interpreter on the command line (`sh -c`, `python3 -c`, `perl -e`, `ruby -e`, `node -e`, `php -r`, `lua -e`,
  `Rscript -e`, a here-doc), but a script file in the worktree, `awk 'BEGIN{system(...)}'` and a `make` recipe are
  not seen. It is a best-effort tripwire that stops the obvious spellings and the worktree's shared git metadata
  (`git config`, `git -c`, `.git`), not a boundary. The worktree cannot be moved into a sandbox box as it stands:
  its `.git` is a file naming `<parent>/.git/worktrees/<name>`, so a box binding only the worktree cannot run git;
- depth 1: a child cannot delegate, so the tree is one level deep by construction. The human-facing and outward
  tools (ask_user, ask_root, github_handoff, send_message, report, remember, sim, schedule, lease, browser,
  propose_plan, advisor) and MCP tools belong to the parent: a child reports what it found and the parent acts;
- caps: a step limit, a time limit and a dollar cap per child. The cap is config `delegate_usd` (0.50 by default)
  and a call's `cap_usd` may lower it, never raise it; reaching it pauses the child and the parent gets a partial
  report. Spend is usd + the OpenCode plan's allowance, because inside a Go allowance a turn costs $0 of money and
  real allowance;
- a short structured report back: outcome (completed / capped / failed / incomplete / orphaned), summary, what it
  verified, what it skipped, steps, spend, allowance and its cap; a writing child's report also names its worktree,
  branch, base commit, the files it changed, a diffstat and the merge / discard commands;
- a SIM record per child: a `recorded` prediction at the start (task, role, permissions, caps, the done check it
  was given) and its `observed` outcome when the child closes, in the parent's goal.

Parallel children: children are ordinary queued goals, so the one loop runs them interleaved without threads. A
delegate call with `wait=true` (the default) runs the child through Runtime.run(target=child) inside that call and
replaces its result with the report; `wait=false` returns a handle, parks the parent (state waiting_children) once
its turn's other actions ran, and the loop's `settle` closes each child as it finishes and wakes the parent with
their reports. `collect` reads the reports of finished children and names the running ones. A card per child is
the panel's (`workers` in the status a UI reads).
"""
import json
import os
import shlex
import subprocess
import time

ROLES = ('worker', 'reviewer')
PERMISSIONS = ('read', 'write')
DEFAULTS = dict(cap_usd=0.50, max_steps=12, max_seconds=600, max_children=4)
# A wake names at most this many children in its text; the rest are a count plus the `collect` pointer (#384
# review: the text was unbounded, ~300 chars per child). Every id still goes in the event, so nothing is dropped.
WAKE_LINES = 12
# Every child, whatever its permissions: the outward and human-facing tools, and a nested delegate (depth 1).
ALWAYS_REFUSED = ('ask_user', 'ask_root', 'github_handoff', 'send_message', 'report', 'remember', 'sim',
                  'schedule', 'lease', 'browser', 'propose_plan', 'delegate', 'advisor')
# A read-only child may not change anything; a writing child may, inside its worktree.
WRITE_ONLY = ('write', 'edit', 'checkpoint', 'rollback')
REFUSED = ALWAYS_REFUSED + WRITE_ONLY   # (every child-refused name, for callers that do not know the permissions)
CHILD_PROMPT = '''[delegate child of goal {parent}] You are a {perms} child goal, run by that goal's delegate tool.

TASK
{task}

WHAT DONE LOOKS LIKE (call define_done with this, then verify and finish)
{done}

Rules that differ from a top-level goal: you cannot ask your human, message a peer, report, remember, write a SIM
record, schedule, hand off to GitHub, browse, or delegate further - those belong to your parent. {perms_text} Your
parent reads only your structured report (summary, verified, skipped, spend, and for a writing child the files it
changed), so the answer belongs in your finish summary: say plainly what you checked and what you could not. Name
files by path and line and quote the few lines that matter.'''
READ_TEXT = '''You cannot write or edit a file or run a command that changes anything: you read, search, run read-only
commands, use the sandbox, the web and machine, then define_done, verify and finish with what you found.'''
WRITE_TEXT = '''You may write and edit files and run commands, but only inside the worktree this delegate created for you
(call checkpoint before your first change to it, and name files relative to it): everything outside that worktree is
your parent's to change, and a read of it is refused the same way. A terminal command starts with
`cd <your worktree> && ` or passes `cwd` inside it, and every path it names must stay inside.
The worktree keeps your changes apart from your parent's tree; it is not a jail, so treat it as your own checkout and
do not try to reach past it. Two things are off limits because your worktree shares them with the parent's
repository: `git config`, `git -c`, `--config-env`, `--exec-path`, and any path naming `.git` - they are refused.
The checks above read the command text, so a program that computes a path at run time is not seen: a script file
running `python3 script.py`, `awk`, a `make` recipe, or a here-doc body. Handing a program to a shell or an
interpreter on the command line is refused (`sh -c`, `python3 -c`, `perl -e`, `ruby -e`, `node -e`, `php -r`,
`lua -e`, `Rscript -e`, and `python3 - <<EOF`): write the program to a file in your worktree and run that, so your
parent's diff shows it. Your parent reviews the worktree's diff, and a write outside it is a change your parent
would not review. Keep every write where you were told to.'''


def switched_on(rt):
    """Whether the delegate tool is offered at all: config `delegate` is not `false`.

    The key is the off switch *and* the role-model dict, and it used to be read as neither where it matters:
    `tools()` listed the tool under `false` (a spending tool the prompt says is off), and `true` reached
    `True.get(...)` in _brain - an AttributeError `_action` does not catch, so `tick()` marked the *parent* goal
    `error` (#382 review, Opus 5.5 P1-2)."""
    return rt.config.get('delegate', True) is not False


def role_models(rt):
    """The role models: the `delegate` key when it is a dict, {} when the key is absent or null.

    Anything else is a typo - `true`, a string, a list - and is refused here naming the key, rather than
    crashing later with an AttributeError from a call on a bool (#382 review, Opus 5.5 P1-2)."""
    cfg = rt.config.get('delegate')
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError('delegate is %r: it is either false (no delegate tool at all) or an object naming the '
                         'role models, e.g. {"worker_model": "..."}. Set it to one of those, or remove the key'
                         % (cfg,))
    return cfg


def read_only(cmd):
    """runtime.read_only_command without the circular import (runtime imports this module)."""
    from .runtime import read_only_command
    return bool(cmd) and read_only_command(cmd)


def _inside(wt, path):
    """True when `path` stays inside `wt` once the OS resolves it.

    A plain `path.startswith(wt + '/')` accepted `<wt>/../x`, which the OS then resolves outside the worktree:
    a writing child's `write(path='../x')` and `cd <wt> && echo x > ../x` both left it.
    realpath() normalises `..`, `.`, doubled slashes and symlinks on both sides, so neither a `..` path nor a
    link inside the worktree can name a file outside it.
    """
    wt = os.path.realpath(wt)
    p = path if os.path.isabs(path) else os.path.join(wt, path)
    p = os.path.realpath(p)
    return p == wt or p.startswith(wt + os.sep)


# Programs that run another command from a string, or start a shell of their own: a text check cannot see what
# their argument does (a nested `sh -c "echo x > /tmp/evil"` hides its path inside one word), so a writing child
# may not call them. This is a name deny-list and it cannot hold on its own: `awk
# 'BEGIN{system(...)}'`, `make` and `git -c alias.x='!cmd' x` all run a command without naming one of these, and
# the module docstring says so. It narrows the obvious spellings; the worktree is not a jail.
NESTED = frozenset('sh bash dash zsh ksh csh tcsh fish eval exec source xargs env sudo doas su chroot '
                   'unshare nsenter script'.split())

# An interpreter handed its program on the command line is the same spelling with a different name: `python3 -c
# 'open("../p/secret","w").write(1)'` hides the path inside one quoted word, exactly as `sh -c` does. The
# interpreters themselves are not refused, because a writing child's real work needs them (`python3 -m pytest -q`
# and `perl Makefile.PL` are pinned as allowed), so only the flag that hands one its program is refused, and the
# child writes the program to a file in its worktree instead - where the parent's diff shows it. Each entry is
# (inline flag letters, letters that take a value, letters that end the scan, inline long options): a value
# letter may be attached (`-Wignore`) or the next word, and it must be listed or that word would look like a
# script and end the scan before a later `-e` (#384 review P2). `-m` (python) ends the scan because the program
# is the module, so `python3 -m pytest -c cfg` keeps its meaning for pytest. A word shaped like an invocation is
# refused wherever it stands, like NESTED: `grep python3 -c f` is a false positive (write `grep -c -e python3 f`),
# the direction a tripwire should err in.
#
# This list is bounded and stays bounded on purpose: a program-string runner it leaves out - `deno eval`,
# `osascript -e`, `gnuplot -e`, `sqlite3 -cmd '.shell sh'`, `awk 'BEGIN{system("...")}'` - runs a program this
# check cannot see, and adding each one the next review names would be an endless race against a deny-list that
# the module docstring already declares is not a jail (the child can always `make`, `git -c alias`, or write its
# program to a file first). The entries here are the spellings a child reaches for first; everything else is
# refused by the writing child's own contract - it writes files with `write`, and a command it runs that the
# deny-list does not name is the parent's diff to review.
INLINE = {'python': ('c', 'WX', 'm', ()), 'python3': ('c', 'WX', 'm', ()),
          'pypy': ('c', 'WX', 'm', ()), 'pypy3': ('c', 'WX', 'm', ()),
          'perl': ('eE', 'IMm', '', ()),
          'ruby': ('e', 'ErI', '', ()),
          'node': ('ep', 'r', '', ('--eval', '--print')), 'nodejs': ('ep', 'r', '', ('--eval', '--print')),
          'php': ('r', 'fcdz', '', ()),
          'lua': ('e', 'l', '', ()),
          'Rscript': ('e', '', '', ()), 'R': ('e', '', '', ())}


def _inline_program(prog, args):
    """The word that hands `prog` a program on its command line, or ''. args: the words after the program.

    A python minor version (`python3.12`) is python. The scan stops at `--`, at an operand (the script) and at a
    stop letter (`-m`), which is where the program stops coming from the command line."""
    spec = INLINE.get(prog)
    if spec is None and prog.startswith('python') and all(c.isdigit() or c == '.' for c in prog[6:]):
        spec = INLINE['python']
    if spec is None:
        return ''
    inline, valued, stop, long = spec
    i = 0
    while i < len(args):
        w = args[i]
        if w == '--' or not w:
            return ''
        if w == '-':
            return w                           # `python3 -` reads the program from stdin
        if not w.startswith('-'):
            # a redirect word ends the scan, not the command: its target is still a path word this walk checks, so
            # `python3 < /dev/null` stays allowed and `python3 < ../evil.py` is refused by the `..` rule. A
            # here-doc is the exception: `python3 <<'EOF'` supplies the program in a body no path check reads,
            # which is `-c` by another spelling
            return w if '<<' in w else ''
        if w.startswith('--'):
            head = w.split('=', 1)[0]
            if head in long:
                return head
            # an option this file does not know: keep scanning so a later `-e`/`-c` is still seen (`node --inspect
            # -e x`). The cost is a miss for a long option that takes its value as the next word and is followed by
            # a program flag (`--check-hash-based-pycs always -c x`); the check errs toward catching the flag,
            # which is the spelling a child would use on purpose, and the module docstring calls it best effort.
            i += 1
            continue
        cluster = w[1:]
        for k, c in enumerate(cluster):
            if c in inline:
                return w
            if c in stop:
                return ''                      # `-m pytest`: the program is the module, not the command line
            if c in valued:
                if k == len(cluster) - 1:
                    i += 1                     # the value is the next word: `-W ignore`, `-I lib`
                break                          # `-Wignore`: the rest of this word is the value
        i += 1
    return ''

# Redirect targets that are streams, not files: a command may write its errors to one of these and change nothing
# outside the worktree, so they are not "climbs out of the worktree" (#384 review: `ls 2>/dev/null` was refused).
DEVICES = frozenset(('/dev/null', '/dev/stdout', '/dev/stderr'))

# git options that make a git command run or read something of the caller's choosing, with no shell in sight:
# `-c alias.x='!cmd' x` runs cmd through git's own alias expansion, `-c core.pager='sh -c ...' log` runs it as a
# pager, `-c diff.external`, `core.fsmonitor` and `remote.uploadpack` alike; `--config-env` reads a value from the
# environment; `--exec-path` moves the programs git runs. A child can set a setting it needs in its own worktree
# with `git config`, so refusing these costs it nothing.
GIT_INJECT = frozenset(('-c', '--config-env', '--exec-path'))

# git's global options that take a value as the next word, so the walk over the option prefix must step over both
# (`git -C . config ...`). The injecting ones are here AND in GIT_INJECT (one fact, one place).
_GIT_VALUE_OPTS = GIT_INJECT | frozenset(('-C', '--git-dir', '--work-tree'))


def _git_words(rest):
    """(git's global options, the words from its subcommand on) from the words after `git`.

    `-c`, `--config-env` and `--exec-path` are options only BEFORE the subcommand: `git switch -c x` and
    `git commit -c HEAD` use `-c` for something else entirely (create a branch, re-edit a message), and a scan
    over every word refused them. A command separator ends the git command, and an `--opt=value` word is one
    word."""
    i = 0
    while i < len(rest):
        w = rest[i]
        if w and all(c in ';&|()<>`' for c in w):
            break
        if w.startswith('-'):
            i += 2 if w in _GIT_VALUE_OPTS else 1
            continue
        break
    return rest[:i], rest[i:]


def _unquoted_brace(text):
    """True when a `{` or `}` stands outside quotes in `text`, where the shell expands it.

    Brace expansion makes a word this check cannot read: `cd W && touch .{.,}/evil` runs as `touch ../evil
    ./evil`, two paths nobody wrote down, and each `.` passes every path check (#384 review P2). Inside any
    quote the character is literal, so `awk 'BEGIN{print $1}'` is not this case - a first version of this check
    looked at the parsed words, where shlex has already dropped the quotes, and refused every awk program (CI
    caught it). A backslash outside quotes escapes the next character: `\\{` is literal too.
    """
    quote, i = '', 0
    while i < len(text):
        c = text[i]
        if quote == "'":
            quote = '' if c == "'" else quote
        elif c == '\\':
            i += 1                        # the next character is literal, `\{` and (in double quotes) `\"`
        elif c in '\'"':
            quote = '' if quote == c else (quote or c)
        elif not quote and c in '{}':
            return True
        i += 1
    return False


def _escaping(cmd, wt, writing=True, base=''):
    """Why a writing child's command could reach outside its worktree, or ''.

    Checked for every command of a writing child, read-only or not (#384 review): `cd W && cat ../parent_secret`
    reads what the parent did not hand the child, which is exactly the path the `read` tool refuses. `writing`
    says the command changes something: only then do the checks that guard against a *change* through git's
    shared metadata apply (`git config` / `git -c` / `--config-env` / `--exec-path`, a `GIT_` variable, a word
    naming `.git`); a read-only command may read that metadata and changes nothing.

    A best-effort tripwire on the command text, not a boundary (see the module docstring): a program that computes
    a path at run time (`python3 script.py`, `awk 'BEGIN{system(...)}'`, a `make` recipe) can still write
    anywhere, and no text check can see it. What is refused is the spelling that hands a program over the command
    line - `sh -c`, `python3 -c`, `perl -e`, `ruby -e`, `node -e`, `php -r`, `lua -e`, `Rscript -e`, `python3 -`,
    a here-doc - because the path it writes then lives inside one quoted word. What holds exactly is the file
    tools' path check (`_inside`).

    The command must start with `cd <worktree> &&` unless `base` names a directory inside the worktree the command
    already starts in (a read-only command with a `cwd`); every path word is then resolved against that directory,
    and a later `cd` moves it. No word may name a program that runs another command (`sh -c`, `eval`, `xargs`), an
    absolute path outside the worktree, a `..` component, a `$` expansion or a `~` home; the value of
    `--opt=path` is checked as a path too. Words are split with punctuation, so a redirect glued to its target
    (`echo x >../out`) yields `../out` as a word. A redirect to /dev/null (or /dev/stdout, /dev/stderr) changes
    nothing and is allowed.

    That punctuation split is load-bearing, not a convenience (#384 review P2): a separator glued to the next word
    (`cd W &&git config core.fsmonitor x`, `echo hi&&git -c ...`) is only seen as a separator because shlex has
    `punctuation_chars` and `whitespace_split`. Without it the word is the single token `&&git`, which is neither a
    separator nor a program this file knows, so the path walk would pass it and the git guard would never run -
    the tests pin both the glued spellings and the spaced ones so that edit cannot pass quietly. Wrappers do not
    launder the program word either: `command`, `builtin`, `nohup`, `time`, `nice`, `timeout`, `setsid`, `stdbuf`,
    `if`/`while`/`for`/`{ }`/`!` all leave `git` in command position for this walk, and `env`/`exec` are refused
    by name as nested commands.
    """
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=';&|()<>`')
        lex.whitespace_split = True
        words = list(lex)
    except ValueError:
        return 'it cannot be parsed'
    if _unquoted_brace(cmd):
        # checked on the raw text, where the quotes still are: shlex drops them, so a word test refused every awk
        # program (`awk '{print $1}'`) as well as the real case below. `cd W && touch .{.,}/evil` runs as
        # `touch ../evil ./evil`, two paths nobody wrote down, and each `.` passes every path check.
        return ('brace expansion makes paths this check cannot read (`.{.,}/evil` is `../evil ./evil`): quote the '
                'word or write every path out')
    if base:
        if not _inside(wt, base):
            return '%s climbs out of the worktree' % base
        start = 0
    else:
        if len(words) < 3 or words[0] != 'cd' or words[1] != wt or words[2] != '&&':
            return 'it must start with `cd %s && `' % wt
        base = os.path.realpath(wt)
        start = 3
    at_cmd = True                                  # the next word starts a command (after `&&`, `;`, `|`, `(`)
    cwd = os.path.realpath(base or wt)             # where a relative path word resolves
    j = start
    while j < len(words):
        w = words[j]
        if w and all(c in ';&|()<>`' for c in w):
            at_cmd = True
            j += 1
            continue
        if writing and '=' in w and w.split('=', 1)[0].startswith('GIT_'):
            # `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.fsmonitor GIT_CONFIG_VALUE_0='<cmd>' git status` is
            # `git -c` by another spelling: it runs <cmd> from a worktree. The rule is the GIT_ namespace, not a
            # list of names - GIT_EXEC_PATH, GIT_PAGER, GIT_SSH_COMMAND and GIT_CONFIG_GLOBAL reach outside the
            # worktree the same way. Refused wherever the word stands, not only in command position: `export
            # GIT_CONFIG_COUNT=1 ...; git status` and `declare -x GIT_EXEC_PATH=...` set the same variable one
            # word earlier, and a command string runs in one shell, so the export lives as long as the bypass.
            # A bare `GIT_` name without `=` is not refused, so `grep GIT_ f` still reads a file.
            return ('%s sets a git environment variable, and git reads or runs what it names: the worktree shares '
                    'its repository with the parent' % w)
        if '=' in w and w.split('=', 1)[1].startswith('~') and (
                at_cmd or (j and words[j - 1].rsplit('/', 1)[-1]
                           in ('export', 'declare', 'local', 'typeset', 'readonly'))):
            # an assignment's value is a value, but the shell still expands it: `X=~/notes` is $HOME/notes, so
            # `X=~` reached outside where `~` alone was refused (#384 review). Assignment position is a command
            # word, or the word after the builtin that declares one (`export X=~/x` expands the same way), so
            # `grep X=~ f` still reads a file. A value holding `$` is left alone: an expansion cannot be
            # resolved here, and `PATH=./bin:$PATH make` is an ordinary build command.
            return '%s could resolve outside the worktree, because the shell expands its ~' % w
        if at_cmd and '=' in w:
            # a command-position assignment's value is path-checked like any other word: `X=../x make` names a
            # directory outside the worktree the command then runs in (#384 review)
            v = w.split('=', 1)[1]
            if v and not _inside(wt, v if os.path.isabs(v) else os.path.join(cwd, v)):
                return '%s climbs out of the worktree' % w
            j += 1
            continue                               # another plain env assignment: still before the command word
        at_cmd = False
        prog = w.rsplit('/', 1)[-1]
        if prog == 'cd':
            # `cd` moves where the next relative path resolves. Its directory word is checked like any other path
            # and then skipped, so `cd sub && cat ../ok.txt` reads a file in the worktree while `cd ..` leaves it.
            k = j + 1
            while k < len(words) and words[k].startswith('-'):
                k += 1                             # cd -P / -L / -e: an option, then the directory
            if k >= len(words) or not words[k] or all(c in ';&|()<>`' for c in words[k]) or words[k] == '-':
                return 'cd without a directory goes to your home, outside the worktree'
            d = words[k]
            if d.startswith('~') or '$' in d:
                return '%s could resolve outside the worktree' % d
            target = d if os.path.isabs(d) else os.path.join(cwd, d)
            if not _inside(wt, target):
                return '%s climbs out of the worktree' % d
            cwd = os.path.realpath(target)
            j = k + 1
            continue
        if writing and prog == 'git':
            # git's own config is the parent repository's: a worktree's .git is a FILE naming
            # <parent>/.git/worktrees/<name>, so `git config core.fsmonitor '<cmd>'` here writes the parent's config
            # and every later `git status` in the parent runs <cmd> (reproduced). `-c`, `--config-env` and
            # `--exec-path` do the same per command (`git -c alias.x='!cmd' x`). All are refused; a child that
            # needs a setting uses a file in its worktree. Only the options BEFORE the subcommand count:
            # `git switch -c x` and `git commit -c HEAD` mean something else by `-c`.
            opts, after = _git_words(words[j + 1:])
            for v in opts:
                if v in GIT_INJECT or v.split('=', 1)[0] in GIT_INJECT:
                    return ('%s lets git read or run something the worktree shares with the parent repository' % v)
            if after and after[0] == 'config':
                return ('git config writes a config the worktree shares with the parent repository, so a later '
                        'git command there would run what it names')
        if prog in NESTED or prog.lstrip('-') in NESTED:
            return '%s runs another command, which this check cannot see' % w
        inline = _inline_program(prog, words[j + 1:])
        if inline:
            # `python3 -c '<program>'` is `sh -c` by another name: the path lives inside one quoted word, so no
            # path check can see it. The interpreter stays allowed (`-m pytest`, `script.py`), only the flag that
            # hands it a program from the command line is refused, wherever the word stands (like NESTED).
            return ('%s hands %s a program on its command line, and a program this check cannot read can write '
                    'anywhere. Write it to a file in the worktree and run that instead' % (inline, prog))
        if w.startswith('~') or '$' in w:
            return '%s could resolve outside the worktree' % w
        if w in DEVICES:
            j += 1
            continue                               # a redirect to a stream writes nothing: `ls 2>/dev/null`
        for p in ([w, w.split('=', 1)[1]] if '=' in w else [w]):
            if writing and '.git' in p.split('/'):
                # the worktree's .git is a FILE naming <parent>/.git/worktrees/<name>: rewriting it points every
                # later git command in the worktree at another repository (the parent's own tree, or a bare repo
                # whose hooks then run), which is a change outside the worktree by another name
                return ('%s names the worktree\'s .git, which is a pointer into the parent repository' % w)
            if not _inside(wt, p if os.path.isabs(p) else os.path.join(cwd, p)):
                return '%s climbs out of the worktree' % w
        j += 1
    return ''


def gate_refusal(gate, row=None):
    """Why a delegate child may not use this done-check gate, or ''. A child's gate runs in a shell when the goal
    finishes (and after a change turn), so it must not change anything the parent would not review.

    A read-only command is always fine. A writing child's natural gate is rooted in its worktree
    (`cd $WT && pytest -q`, `cd $WT && git diff --quiet`), which the global read-only allow-list does not know:
    holding it to that list refused the very check the child can prove its work with, so the child had to leave
    the gate empty (#384 review P2). Such a gate is held to the same worktree grammar as the child's terminal
    commands (`_escaping`): it starts with `cd <worktree> && `, every path word stays inside, no nested command,
    no `$`/`~`, and the git-metadata checks a change owes apply (`writing=True`: a gate may write in the tree).
    """
    row = row or {}
    if not gate:
        return ''
    ro = read_only(gate)
    write = (row.get('permissions') or 'read') == 'write'
    wt = (row.get('worktree') or '').rstrip('/')
    if write and wt:
        # read-only is not enough on its own: the allow-list is text, so `cd <wt> && cat ../parent_secret` is a
        # "read-only" command that reads what the `read` tool refuses, exactly as in the terminal branch. A
        # read-only gate is path-checked from the worktree root; a gate that can change something must be rooted
        # there (`cd <wt> && ...`), like every command of a writing child.
        why = _escaping(gate, wt, writing=not ro, base=wt if ro else '')
        if not why:
            return ''
        return ('a writing child\'s gate runs in a shell when the goal finishes, so it may change or read only '
                'its own worktree (%s): %s. Fix the command, or leave gate empty and report what you verified'
                % (wt, why))
    if ro:
        return ''
    return ('a delegate child\'s gate must be a read-only command, because it runs in a shell when the goal '
            'finishes. Check with a command that only reads (grep, test -f, ls), or leave gate empty and report '
            'what you verified')


def refused(name, args, row=None):
    """Why a delegate child may not run this call, or None. Checked by the runtime before the tool runs.
    row: the child's delegates row - its permissions ('read' when absent) and its worktree when it may write."""
    row = row or {}
    write = (row.get('permissions') or 'read') == 'write'
    wt = (row.get('worktree') or '').rstrip('/')
    if name in ALWAYS_REFUSED or name.startswith('mcp__'):
        return ('a delegate child cannot %s: the human-facing and outward tools belong to the parent, a child is '
                'one level deep, and an MCP tool can change things this harness does not see. Report what you '
                'found; your parent acts.' % name)
    if name in WRITE_ONLY and not write:
        return ('a read-only delegate child cannot %s: report what you found; your parent makes the change' % name)
    if name == 'define_done' and args.get('gate') and gate_refusal(args['gate'], row):
        # a gate runs in a shell at finish (and after a change turn): `define_done(gate='touch x && ...')` created
        # the file and still reported completed (#382 review). A read-only gate proves as much and cannot run one;
        # a writing child may root its gate in its own worktree, under the same grammar as its terminal (#384 review)
        return gate_refusal(args['gate'], row)
    if name == 'terminal':
        if args.get('stdin') is not None or args.get('signal'):
            # typing into a shell (or signalling one) reaches whatever that session is, possibly another goal's,
            # and bypasses both the read-only rule and a writing child's worktree rule: every child is refused
            # (#382 review kept this for a read-only child; the full version applies it to both)
            return ('a delegate child cannot type into a running shell or signal a session%s: that session may '
                    'belong to another goal, and typing bypasses the read-only and worktree checks. Run a command '
                    'of its own, or report what is missing'
                    % (' (a read-only child reads, it does not interact)' if not write else ''))
        cmd = str(args.get('command') or '')
        cwd = str(args.get('cwd') or '')
        if write and wt and cwd and not (os.path.isabs(cwd) and _inside(wt, cwd)):
            return ('a writing child starts a terminal command inside its worktree (%s), not %s'
                    % (wt, cwd))
        ro = read_only(cmd)
        if not write and cmd and not ro:
            return ('a read-only delegate child runs only a read-only command here. Report what you found; your '
                    'parent makes the change')
        if write and wt and cmd:
            # every command, read-only or not (#384 review): `cd W && cat ../parent_secret.txt` reads what the
            # `read` tool refuses, so a read-only command is path-checked as well. `writing` adds only the checks
            # against a change through git's shared metadata; the path rule is the same. A command that already
            # starts inside the worktree (its `cwd`) needs no `cd W && ` prefix.
            why = _escaping(cmd, wt, writing=not ro, base=cwd)
            if why:
                return ('a writing child %s only inside its worktree (%s): %s. Start the command with `cd %s && ` '
                        '(or set cwd inside it) and keep every path inside it'
                        % ('reads' if ro else 'changes files', wt, why, wt))
    if write and wt and name in ('write', 'edit'):
        path = str(in_worktree(row, name, args).get('path') or '')
        if path.startswith('/') and not _inside(wt, path):
            return ('a writing child changes files inside its worktree (%s): name the file relative to it; the '
                    'files outside are your parent\'s to change' % wt)
        if '.git' in path.split(os.sep):
            # the worktree's .git is a FILE naming <parent>/.git/worktrees/<name>: rewriting it points the next
            # git command in the worktree at another repository (or a bare repo whose hooks then run), so it is a
            # change outside the worktree by another name, the same class as `git config`
            return ('a writing child cannot rewrite .git (%s): in a worktree it is a pointer into the parent '
                    'repository, not a file of yours. Change your files and let git update its own metadata' % path)
    if write and wt and name in ('read', 'search'):
        # a writing child's world is its worktree: reading the parent's tree (or a home) from it leaks what the
        # parent did not hand it, and the module docstring says such a read is refused. The path checked is the one
        # the tool will actually resolve (in_worktree's), so a relative path and search's own home default are
        # covered as well as a caller-supplied absolute (#384 review P1).
        path = str(in_worktree(row, name, args).get('path') or '')
        if path and not _inside(wt, path):
            return ('a writing child reads inside its worktree (%s): name the file relative to it' % wt)
    if write and wt and name == 'checkpoint':
        outside = [str(p) for p in (in_worktree(row, name, args).get('paths') or [])
                   if str(p).startswith('/') and not _inside(wt, str(p))]
        if outside:
            return ('a writing child checkpoints paths inside its worktree (%s): %s is outside it'
                    % (wt, ', '.join(outside[:3])))
    if write and wt and name == 'sandbox':
        # the box mounts `folder` read-only at /work, and a named folder is resolved against the agent home (a
        # writing child's world is its worktree): `sandbox(folder='work')` read the parent's tree, the same read
        # `read` refuses, and the box's own /box (no folder named) is the child's scratch, so only a named folder
        # outside the worktree is refused (#384 review P2).
        folder = str(args.get('folder') or '')
        if folder and not (os.path.isabs(folder) and _inside(wt, folder)):
            return ('a writing child sandboxes in its worktree (%s): name the folder inside it, or omit the '
                    'folder and use the box\'s own /box' % wt)
    return None


def in_worktree(row, name, args):
    """A writing child's view of its world: a relative path means inside its worktree, not the agent home.

    files.read / write / edit / search and checkpoint resolve a relative path against the home directory; for a
    child with a worktree the worktree is its home, so every path is made absolute here before the tool runs.
    Other calls (and other children) are left exactly as the model wrote them.
    """
    wt = (row or {}).get('worktree') or ''
    if not wt or (row.get('permissions') or 'read') != 'write':
        return args
    args = dict(args or {})
    if name in ('write', 'edit', 'read', 'search'):
        path = str(args.get('path') or '')
        if not path and name == 'search':
            # search's own default is the agent home (files.search roots at `.`), so a child that names no path
            # would list and grep the parent's whole tree - exactly the read `read(path='../x')` is refused for
            # (#384 review P1). The child's home is its worktree, so that is the default here too.
            args['path'] = wt
        elif path and not os.path.isabs(path):
            args['path'] = os.path.join(wt, path)
    elif name == 'checkpoint':
        args['paths'] = [p if os.path.isabs(str(p)) else os.path.join(wt, str(p)) for p in (args.get('paths') or [])]
    return args


def _caps(rt, args):
    """(cap_usd, max_steps, max_seconds): the config's defaults, lowered by the call, never raised.

    A cap is a positive number; a missing or null config key takes the default. Zero or less is refused rather
    than read as "unlimited": `cap_usd=0` used to disable a child's cap altogether (#382 review), and a cap
    nobody can reach is not a cap."""
    cfg = rt.config
    cap = cfg.get('delegate_usd', DEFAULTS['cap_usd'])
    cap = DEFAULTS['cap_usd'] if cap is None else float(cap)
    if cap <= 0:
        raise ValueError('delegate_usd is %s: a cap of 0 does not mean unlimited, so no child can run under it. '
                         'Set a positive cap (the default is %s)' % (cfg.get('delegate_usd'), DEFAULTS['cap_usd']))
    if args.get('cap_usd') is not None:
        their = float(args['cap_usd'])
        if their <= 0:
            raise ValueError('cap_usd must be more than 0: it is what the child may spend, and 0 used to mean '
                             '"uncapped". Omit it to use delegate_usd ($%s)' % cap)
        cap = min(cap, their)
    steps = int(cfg.get('delegate_max_steps', DEFAULTS['max_steps']))
    if args.get('max_steps') is not None:
        their = int(args['max_steps'])
        if their <= 0:
            raise ValueError('max_steps must be at least 1: a child that may take no step cannot do anything')
        steps = min(steps, their)
    secs = int(cfg.get('delegate_max_seconds', DEFAULTS['max_seconds']))
    if args.get('max_seconds') is not None:
        their = int(args['max_seconds'])
        if their <= 0:
            raise ValueError('max_seconds must be at least 1: a child with no time cannot do anything')
        secs = min(secs, their)
    return cap, max(1, steps), max(1, secs)


def _brain(rt, args):
    """(Model, {model, provider}) the child runs: a call's model/provider from the catalog, the role's
    configured delegate.<role>_model, or the parent's own brain (the default)."""
    from .model import Model, price
    from . import catalog
    role = args.get('role') or 'worker'
    name = str(args.get('model') or '').strip()
    provider = str(args.get('provider') or '').strip() or None
    if not name:
        name = str(role_models(rt).get(role + '_model') or '').strip()
    if not name:
        return Model(dict(rt.config)), dict(model=rt.config.get('model'),
                                            provider=rt.config.get('provider') or rt.config.get('provider_kind'))
    cat = catalog.load(rt.home)
    entry = catalog.find(cat, name, provider)
    if entry is None:
        entry = catalog.find(catalog.refresh(rt.home, rt.config), name, provider)
    if entry is None:
        raise ValueError(f'no model {name} in the catalog: `models --refresh` lists what the providers serve')
    spec = catalog.providers(rt.config).get(entry['provider'])
    if not spec:
        raise ValueError('the provider of %s is no longer configured' % entry['id'])
    endpoint, api = catalog.transport(spec, entry)
    conf = dict(rt.config)
    conf.update(model=entry['id'], provider=entry['provider'], provider_kind=spec['kind'], endpoint=endpoint, api=api)
    # the entry's own facts, by the same normalization /model uses (an OpenCode plan charges what its reply says, so
    # its price tables are not needed; any other paid brain must be priceable or its cap can never be read)
    catalog.apply_entry(conf, entry, spec)
    brain = Model(conf)
    if (brain.remote and spec['kind'] not in catalog.OPENCODE and price(entry['id'], {}) is None):
        raise ValueError('no price for %s: a paid remote child must be priceable, or its dollar cap can never trip '
                         '(the catalog has no prices for it, and only an OpenCode plan reports its own charge)'
                         % entry['id'])
    return brain, dict(model=entry['id'], provider=entry['provider'])


def prepare(rt, args):
    """What start() needs that may wait or reach the network, resolved before the runtime opens its write
    transaction: the caps, the role, the permissions, the repository and the child's brain (a model from the
    catalog may need a refresh, which is a network call, and the store has one writer: #382 review)."""
    if not switched_on(rt):
        # the off switch, checked where the call lands as well as in tools(): a model replaying an old action, or a
        # goal whose tool list was built before the owner flipped the key, must not spend through it (#382 review).
        raise ValueError('delegate is off in config (delegate: false): this goal cannot start a child. Ask your '
                         'human to set delegate to a role-model object or remove the key')
    task = str(args.get('task') or '').strip()
    done = str(args.get('done_check') or '').strip()
    if not task:
        raise ValueError('delegate needs a task: what the child should find out or check')
    if not done:
        raise ValueError('delegate needs done_check: what done looks like for the child, as something it can '
                         'check (it becomes the child\'s define_done)')
    role = args.get('role') or 'worker'
    if role not in ROLES:
        raise ValueError('role must be one of ' + ', '.join(ROLES))
    perms = str(args.get('permissions') or 'read')
    if perms not in PERMISSIONS:
        raise ValueError('permissions must be one of ' + ', '.join(PERMISSIONS))
    wait = args.get('wait')
    wait = True if wait is None else bool(wait)
    repo = ''
    if perms == 'write':
        repo = str(args.get('repo') or '.').strip() or '.'
        if not os.path.isabs(repo):
            repo = str(rt.home / repo)
        repo = _git(['-C', repo, 'rev-parse', '--show-toplevel'])
    cap, steps, secs = _caps(rt, args)
    model, who = _brain(rt, args)
    return dict(task=task, done=done, role=role, permissions=perms, repo=repo, wait=wait,
                cap_usd=cap, max_steps=steps, max_seconds=secs, model=model, who=who)


def _git(args, cwd=None):
    """Run git (list argv, no shell) and return its stdout, or raise ValueError naming what it said."""
    p = subprocess.run(['git'] + args, cwd=cwd, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise ValueError('git %s failed: %s' % (' '.join(args), (p.stderr or p.stdout).strip()[:300]))
    return p.stdout.strip()


def _worktree(rt, child, repo):
    """(repo top, base commit, worktree path, branch) for a writing child; creates the worktree and its branch.
    The parent's tree is never touched: the child works in its own checkout, on its own branch off HEAD.

    The branch and the checkout carry the child's whole id: an eight-character prefix collided for two children
    whose ids share it, and `git worktree add -b` fails on a branch name that already exists, so a branch left
    un-merged by an earlier child made every later collision lethal (#384 review)."""
    top = _git(['-C', str(repo), 'rev-parse', '--show-toplevel'])
    base = _git(['-C', top, 'rev-parse', 'HEAD'])
    branch = 'delegate/' + child
    wt = rt.home / 'worktrees' / child
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git(['-C', top, 'worktree', 'add', '-b', branch, str(wt), 'HEAD'])
    return top, base, str(wt), branch


def _remove_worktree(repo, worktree, branch):
    """Best effort: a writing child that failed validation must not leave a checkout or branch behind."""
    subprocess.run(['git', '-C', str(repo), 'worktree', 'remove', '--force', str(worktree)],
                   capture_output=True, timeout=60)
    subprocess.run(['git', '-C', str(repo), 'branch', '-D', branch], capture_output=True, timeout=60)


def _commit(row):
    """A writing child's changes belong on its branch: commit them in its worktree before the parent reads the
    report, so `git merge <branch>` merges them and the diffstat names the files. A child that only writes leaves
    them untracked, and merging a branch with no commit merges nothing (a test of the full version caught it).
    Best effort: a missing worktree still leaves the files and the commands in the report, and a failed commit
    returns why, which _survey puts in the report - the parent must not read a merge command whose branch lacks
    the changes as if it were the child's work (#384 review)."""
    wt = row.get('worktree') or ''
    if not wt or not os.path.isdir(wt):
        return 'the worktree %s is gone, so nothing was committed' % wt if wt else ''
    try:
        if not _git(['-C', wt, 'status', '--porcelain']).strip():
            return ''                 # nothing changed: the branch already points at the base commit
        _git(['-C', wt, 'add', '-A'])
        # `-c` here stamps the commit object's identity and writes no config file, which is why the harness may use
        # it while a child may not (GIT_INJECT): the harness is trusted, a child's command line is not. Do not copy
        # this spelling into a child-facing path or into `refused()`'s allow-list (#384 review P2).
        _git(['-C', wt, '-c', 'user.name=Caminus delegate', '-c', 'user.email=delegate@caminus.invalid',
              'commit', '-q', '-m', 'delegate %s: what its report describes' % (row.get('child') or '')[:8]])
    except ValueError as e:
        return str(e)[:400]
    return ''


def _survey(row):
    """What a writing child changed: its worktree's status, its diffstat against the base commit, and the two
    commands the parent runs to keep or drop it. Best effort: a worktree already removed says so."""
    wt, repo, branch = row.get('worktree') or '', row.get('repo') or '.', row.get('branch') or ''
    out = dict(worktree=wt, branch=branch, base=row.get('base') or '')
    if failed := _commit(row):
        # The branch may be the base commit while the files sit uncommitted: say so next to the merge command, so
        # the parent does not read the merge as the child's work (#384 review).
        out['commit_failed'] = failed
    base = row.get('base') or 'HEAD'
    try:
        # the commit's files and whatever is still uncommitted; a rename is its new name
        changed = {ln.strip() for ln in _git(['-C', wt, 'diff', '--name-only', base]).splitlines() if ln.strip()}
        changed |= {ln[3:].split(' -> ')[-1] for ln in _git(['-C', wt, 'status', '--porcelain']).splitlines() if ln.strip()}
        out['files'] = sorted(changed)[:50]
        out['diffstat'] = _git(['-C', wt, 'diff', '--stat', base])[:2000]
    except ValueError as e:
        out['files'], out['diffstat'] = [], str(e)[:200]
    out['merge'] = 'git -C %s merge %s' % (repo, branch)
    try:
        # `merge` runs in the parent's own tree, so a parent with uncommitted work can stop the merge half-applied
        # on a file the child also changed. The report says so next to the command rather than leaving the parent to
        # find out mid-merge (#384 review F3).
        if _git(['-C', repo, 'status', '--porcelain']).strip():
            out['merge_note'] = ('the parent tree has uncommitted changes: commit or stash them first, or the merge '
                                 'can stop half-applied on a file the child also changed')
    except ValueError:
        pass        # no repo to read (already gone): the merge command is best-effort advice, as the docstring says
    out['discard'] = ('git -C %s worktree remove --force %s && git -C %s branch -D %s'
                      % (repo, wt, repo, branch))
    return out


def start(rt, parent, args, prepared=None):
    """Create the child goal and register its brain; the caller runs it after its transaction commits.
    (Creating the goal is part of the parent's action, atomically; running it is not - the child needs turns of
    its own, and a nested loop inside the write transaction would deadlock on BEGIN.) `prepared` is prepare()'s
    result, which the runtime resolves before BEGIN (#382 review)."""
    p = prepared or prepare(rt, args)
    if p['permissions'] == 'write' and rt.store.goal(parent).get('plan') == 1:
        raise ValueError('a plan-first goal has no approved plan yet: it may not grant a child write '
                         'permissions (a child never gets more than its parent has). Explore with a read-only '
                         'child, or finish the plan first')
    limit = int(rt.config.get('delegate_max_children', DEFAULTS['max_children']))
    used = rt.store.db.execute("SELECT count(*) FROM delegates WHERE parent=? AND outcome=''",
                               (parent,)).fetchone()[0]
    if used >= limit:
        raise ValueError('this goal has %d delegate children still running (delegate_max_children): their '
                         'reports arrive first; do the rest yourself, or delegate again once one finishes' % limit)
    child = rt.store._enqueue('', max_steps=p['max_steps'], max_seconds=None, kind='delegate')
    top = base = worktree = branch = ''
    try:
        if p['permissions'] == 'write':
            top, base, worktree, branch = _worktree(rt, child, p['repo'])
        prompt = CHILD_PROMPT.format(parent=parent[:8], task=p['task'], done=p['done'],
                                     perms=p['permissions'],
                                     perms_text=WRITE_TEXT if p['permissions'] == 'write' else READ_TEXT)
        if worktree:
            prompt += '\nYour worktree: %s (branch %s, off %s). Name files relative to it.' % (worktree, branch, base[:12])
        # the goals.prompt column too (a card and a restart read it); store.update refuses it on purpose - the
        # generic API never rewrites a goal's prompt, which for a top-level goal is the human's own words
        rt.store.db.execute('UPDATE goals SET prompt=? WHERE id=?', (prompt, child))
        rt.store.delegate_add(child, parent, p['who'].get('model') or '', p['who'].get('provider') or '',
                              p['role'], p['cap_usd'], permissions=p['permissions'], worktree=worktree,
                              branch=branch, base=base, repo=top, max_seconds=p['max_seconds'])
        rid = ''
        try:
            from . import sim
            if sim.enabled(rt):
                rec = sim.record(rt, parent, dict(
                    problem_group='delegate child',
                    change='delegate %s a %s child: %s' % (p['role'], p['permissions'], p['task']),
                    rationale='a child goal is bounded by its caps ($%.2f, %d steps, %ds) and reports back; %s'
                              % (p['cap_usd'], p['max_steps'], p['max_seconds'],
                                 'it writes only in its own worktree' if worktree else 'it is read-only'),
                    predicted='the child finishes within its caps and its report says what it verified: "%s"'
                              % p['done'][:400]),
                    eval_context='delegate %s' % p['role'])
                rid = rec['id']
                rt.store.db.execute('UPDATE delegates SET sim=? WHERE child=?', (rid, child))
        except Exception:
            rid = ''                      # a SIM record is a record, never a reason a child cannot run
    except BaseException:
        # nothing of a refused call may remain: the child goal, and (a writing child's) worktree and branch.
        # The caller (the runtime's dispatch) already holds the store's transaction, so this must not open its own
        # BEGIN: a nested BEGIN raises and would mask the real error with 'cannot start a transaction within a
        # transaction' (the failure that hid a bad field name in the first run of the delegate tests).
        if rt.store.db.in_transaction:
            rt.store.db.execute('DELETE FROM delegates WHERE child=?', (child,))
            rt.store.db.execute('DELETE FROM goals WHERE id=?', (child,))
        else:
            with rt.store.transaction():
                rt.store.db.execute('DELETE FROM delegates WHERE child=?', (child,))
                rt.store.db.execute('DELETE FROM goals WHERE id=?', (child,))
        if worktree:
            _remove_worktree(top, worktree, branch)
        raise
    rt.delegate_models[child] = p['model']
    rt.delegate_caps[child] = p['cap_usd']
    rt.store.event('delegate_started', dict(child=child, model=p['who'].get('model'),
                                           provider=p['who'].get('provider'), role=p['role'],
                                           permissions=p['permissions'], worktree=worktree, branch=branch,
                                           cap_usd=p['cap_usd'], max_steps=p['max_steps'],
                                           max_seconds=p['max_seconds'], wait=p['wait'],
                                           sim=rid), parent)
    out = dict(child=child, model=p['who'].get('model'), provider=p['who'].get('provider'), role=p['role'],
               permissions=p['permissions'], cap_usd=p['cap_usd'], max_steps=p['max_steps'],
               max_seconds=p['max_seconds'], outcome='running', sim=rid, wait=p['wait'],
               note=('the child runs now; this result is replaced by its report' if p['wait'] else
                     'the child runs in the background; this goal waits for it and collect reads its report'))
    if worktree:
        out.update(worktree=worktree, branch=branch, base=base)
    if p['wait']:
        out['_delegate_run'] = True
    else:
        out['_delegate_async'] = True
    return out


# The three errors a child's own caps leave when they pause it: delegate.capped() for the dollar cap, and the
# runtime for the step limit and the deadline. _report reads an outcome from these exact messages, never from the
# words 'delegate' or 'budget' appearing somewhere in an error (#384 review).
CAPPED_ERRORS = ('delegate spend cap reached', 'oneshot time budget reached', 'oneshot model-step budget reached')


def _capped_error(err):
    """True when an error text is one of the caps' own messages: the message itself, or the message followed by
    ': detail' (the way capped() writes its spend). Not `startswith` on the bare prefix: a future message that
    merely begins with one of these strings would be read as a cap (#384 review)."""
    return any(err == m or err.startswith(m + ':') for m in CAPPED_ERRORS)


def _report(rt, row, g, outcome=None):
    """The short structured report a parent reads: the child's goal, what it verified, what it skipped, its spend
    and (a writing child) what it changed and the commands that keep or drop it."""
    # The child id comes from the row, or from the goal when the row is gone: `run_child` closes with
    # `rt.store.delegate(child) or {}`, and a vanished delegates row made `row['child']` raise before `_close`
    # could file anything, so `settle` repeated the same KeyError on every pass (#384 review P2). The goal row
    # survives the delegates row, and the report is the only record left when the row is gone.
    child, state = row.get('child') or g['id'], g['state']
    if outcome is None:
        err = g.get('error') or ''
        if state == 'completed':
            outcome = 'completed'
        elif state == 'paused' and _capped_error(err):
            outcome = 'capped'       # the dollar cap, the step budget or the time budget ended it
        elif state == 'cancelled':
            outcome = 'incomplete'   # its parent was cancelled: it did not finish its own work
        else:
            outcome = 'failed'
    report = dict(child=child, role=row.get('role'), model=row.get('model'), provider=row.get('provider'),
                  outcome=outcome, state=state, permissions=row.get('permissions') or 'read',
                  summary=(g.get('summary') or g.get('error') or '')[:2000],
                  done_check=(g.get('done_check') or '')[:1000],
                  verified=state == 'completed' and rt.store.verified_since_steer(child),
                  skipped=[it['text'] for it in rt.store.todo(child)['items'] if it['state'] != 'done'][:20],
                  steps=int(g.get('steps') or 0), spend_usd=round(row.get('spend_usd') or 0.0, 6),
                  allowance_usd=round(row.get('allowance_usd') or 0.0, 6), cap_usd=row.get('cap_usd') or 0.0)
    if row.get('worktree'):
        report.update(_survey(row))
    if outcome != 'completed':
        report['note'] = ('the child did not complete: %s. Use what it found, or do the rest yourself; a new '
                          'delegate gets a fresh child.' % (report['summary'][:300] or state))
    return report


def _report_gone(rt, row):
    """The report for a row whose child goal is gone: the row still records the brain, the role, what the child
    may do and what it spent, so its parent reads that much instead of the child vanishing from its list. `state`
    is `gone` because there is no goal left to read a state from, and the outcome is orphaned (#384 review)."""
    return _report(rt, row, dict(state='gone'), 'orphaned')


def _unchanged(row):
    """True when a writing child's checkout holds nothing of its own: a clean status and HEAD still at the base.

    _survey runs _commit first, so whatever the child left uncommitted is already on its branch; a clean checkout
    whose HEAD is the base commit is a child that changed nothing, and its checkout and branch can go."""
    wt = row.get('worktree') or ''
    if not wt or not os.path.isdir(wt):
        return False                  # already gone: nothing to remove, and the branch is not ours to guess about
    try:
        if _git(['-C', wt, 'status', '--porcelain']).strip():
            return False
        return _git(['-C', wt, 'rev-parse', 'HEAD']) == (row.get('base') or '')
    except ValueError:
        return False


def _close(rt, parent, report):
    """File a child's report once: its delegates row, the event the parent and a UI read, the SIM observation,
    and the brain registration that must not outlive the run. The first close wins (settle and run_child may both
    arrive at the same child)."""
    child = report['child']
    row = rt.store.delegate(child) or {}
    if not row or row.get('outcome'):
        return report
    if row.get('repo') and row.get('worktree') and report['outcome'] != 'completed' and _unchanged(row):
        # A stopped parent used to leave its child's checkout and branch in the parent's repo, and the report's
        # discard command was the only pointer to them (#384 review). A child that changed nothing has nothing
        # to keep, so both go now and the report says so; a child that did change something keeps its checkout
        # and its merge/discard commands. A row with no repo is skipped whole: `git -C ''` runs at the daemon's
        # own cwd, so a `delegate/<child>` branch there could be deleted from whatever repository that happens
        # to be, and the report keeps its discard command as the pointer to a checkout that still exists.
        _remove_worktree(row['repo'], row['worktree'], row.get('branch') or '')
        # `worktree`, `branch` and `base` go with the commands: `_survey` set them from the row, and a report that
        # said `worktree_removed` while still naming the branch `git branch -D` just deleted sent the parent to a
        # merge of a ref that is not there (#384 review P2).
        for k in ('merge', 'merge_note', 'discard', 'files', 'diffstat', 'commit_failed',
                  'worktree', 'branch', 'base'):
            report.pop(k, None)
        report['worktree_removed'] = ('the child changed nothing, so its checkout and branch are gone: there is '
                                      'nothing to merge or discard')
    rt.store.delegate_done(child, report['outcome'], report)
    try:
        rt.store.goal(parent)
    except ValueError:
        # The parent's goal is gone (its session was removed): the delegates row above still closes, so the child
        # stops being listed, but the event and the SIM observation are for a goal nothing can read. The worktree
        # cleanup above already ran, and the row keeps the report's merge/discard commands. The brain registration
        # still has to go with the run.
        rt.delegate_models.pop(child, None)
        rt.delegate_caps.pop(child, None)
        return report
    rt.store.event('delegate_done', {k: v for k, v in report.items() if k not in ('summary', 'skipped')}, parent)
    rid = row.get('sim') or ''
    if rid:
        try:
            from . import sim
            sim.observe(rt, parent, rid, observed='%s in %d steps, $%.4f of $%.2f%s%s' % (
                report['outcome'], report['steps'], report['spend_usd'] + report['allowance_usd'],
                report['cap_usd'] or 0.0,
                ', verified' if report['verified'] else '',
                (', changed ' + ', '.join(report['files'][:5])) if report.get('files') else ''),
                eval_context='delegate child (%s)' % (row.get('role') or 'worker'))
        except Exception:
            pass                          # the record is a record, never a gate on the report
    rt.delegate_models.pop(child, None)
    rt.delegate_caps.pop(child, None)
    return report


def park(rt, parent, handle):
    """After a delegate(wait=false) call: the goal has nothing to do until its children report, so it waits
    rather than spending turns asking whether they are done. The loop's settle wakes it.

    Only a working goal is parked. A pause that landed between the action's dispatch and this call (a person's, or
    the watchdog's) outranks the park and must stay a pause: moving it to `waiting_children` let settle wake it, so
    the goal a human had just stopped resumed (#384 review)."""
    with rt.store.transaction():
        parked = rt.store.goal(parent)['state'] == 'working'
        if parked:
            rt.store.update(parent, state='waiting_children')
    if parked:
        rt.store.event('delegate_parked', dict(child=handle.get('child')), parent)


def waiting(rt, parent):
    """This goal's children that are still running, oldest first."""
    return [r['child'] for r in rt.store.delegates(parent=parent, open_only=True)]


def _parent_state(rt, parent):
    """The parent goal's state, or None when its goal is gone (its session was removed). A child of a deleted
    parent is an orphan, and settle must close it rather than raise `unknown goal` out of the loop on every pass
    (#384 review)."""
    try:
        return rt.store.goal(parent)['state']
    except ValueError:
        return None


def settle(rt):
    """Close the children that finished, cancel those of a stopped parent, and wake a parent whose children are
    all done. The loop calls this once a pass, whichever goal it just ticked: children are ordinary queued goals,
    so they run on their own. Returns whether anything moved."""
    moved = False
    for row in rt.store.delegates(open_only=True):
        parent = row['parent']
        pstate = _parent_state(rt, parent)
        try:
            g = rt.store.goal(row['child'])
        except ValueError:
            # The child's own goal is gone (a hand edit, or a restore that kept the delegates row): this read raised
            # `unknown goal` out of the loop on every pass and the row stayed open forever. Nothing can run for a
            # goal that is not there, so the row closes from what it records itself (#384 review).
            _close(rt, parent, _report_gone(rt, row))
            moved = True
            continue
        if pstate is None:
            # The parent's goal is gone: nothing will read this child's report, and a child still runnable must
            # not run on this process's registrations. Close the row as orphaned, the way sweep closes a
            # leftover, without filing into a goal that no longer exists (#384 review).
            if g['state'] in ('queued', 'working', 'waiting_model'):
                with rt.store.transaction():
                    rt.store.update(g['id'], state='paused',
                                    error='its parent goal was deleted mid-delegation')
                g = rt.store.goal(g['id'])
            if g['state'] in ('completed', 'cancelled', 'paused', 'error'):
                _close(rt, parent, _report(rt, row, g, None if g['state'] == 'completed' else 'orphaned'))
                moved = True
            continue
        if pstate in ('cancelled', 'paused', 'error'):
            if g['state'] in ('queued', 'working', 'waiting_model'):
                with rt.store.transaction():
                    rt.store.update(g['id'], state='cancelled',
                                    error='its parent stopped (%s) while it ran' % pstate)
                g = rt.store.goal(g['id'])
            if g['state'] in ('completed', 'cancelled', 'paused', 'error'):
                _close(rt, parent, _report(rt, row, g, 'incomplete' if g['state'] != 'completed' else None))
                moved = True
            continue
        if g['state'] in ('completed', 'cancelled', 'paused', 'error'):
            _close(rt, parent, _report(rt, row, g))
            moved = True
    # A parent parked on its children wakes once none is running: the reports arrive as one nudge, the way peer
    # mail does, so its next turn has them in the conversation it reads.
    for parent in {r['parent'] for r in rt.store.delegates()}:
        if _parent_state(rt, parent) != 'waiting_children':
            continue
        if rt.store.delegates(parent=parent, open_only=True):
            continue
        rows = rt.store.delegates(parent=parent)
        # Every report this parent has not been named yet: the delegate_reports events are the record of what each
        # earlier wake delivered. A fixed window of the last eight rows dropped a generation's earliest reports,
        # which no later nudge ever named again (#384 review); a parent with more children than the window now
        # hears about all of them. An empty set (a parked parent with no new child) keeps the old window, so the
        # wake still carries something.
        shown = set()
        for e in rt.store.db.execute("SELECT data FROM events WHERE goal=? AND kind='delegate_reports'", (parent,)):
            try:
                shown.update(json.loads(e['data']).get('children') or [])
            except (ValueError, TypeError, AttributeError):
                # An unreadable row adds nothing to `shown`, so the children it named stay out of the set and
                # this wake names them again. The set only ever grows, so a parse failure can duplicate a report
                # the parent already read (bounded by WAKE_LINES), never hide one (#384 review P2 read this bare
                # `pass` as counting an unreadable row's children as shown, which is the opposite of what it does).
                pass
        lines, ids = [], []
        # The fallback window obeys the same filter: a second wake that repeats a report the parent already read
        # used to print it again with the row's older JSON (#384 review), so only a wake with nothing new at all
        # falls back to the newest rows whole.
        fresh = [r for r in rows if r['child'] not in shown]
        for r in fresh or [r for r in rows[-8:] if r['child'] not in shown] or rows[-8:]:
            try:
                rep = json.loads(r.get('report') or '{}')
            except ValueError:
                rep = {}
            # Every id is named in the event, so a truncated wake's reports are still shown=next time; the text
            # stops at WAKE_LINES lines (a turn's conversation is not a fair place to paste 30 reports) and says
            # where the rest are (#384 review).
            ids.append(r['child'])
            if len(lines) < WAKE_LINES:
                lines.append('- %s %s · %s steps · %s' % (r['child'][:8], r['outcome'] or '?', rep.get('steps', '?'),
                                                         ' '.join(str(rep.get('summary') or '').split())[:300]))
        more = len(ids) - len(lines)
        if more > 0:
            lines.append('- and %d more report%s: collect reads them' % (more, '' if more == 1 else 's'))
        text = ('[delegate] your children finished. Their reports are below; collect reads the structured ones. '
                'Verify what you act on.\n' + '\n'.join(lines))
        with rt.store.transaction():
            if rt.store.goal(parent)['state'] != 'waiting_children':
                continue
            rt.store.update(parent, state='working')
            rt.store.db.execute('INSERT INTO inbox(goal,text,kind) VALUES(?,?,?)', (parent, text, 'nudge'))
            # The event is what marks these reports shown, so it commits with the wake it belongs to: written
            # after the transaction, a death in between left the parent awake with an empty `shown` ledger, and
            # the next settle printed the same reports again (#384 review).
            rt.store.event('delegate_reports', dict(children=ids), parent)
        moved = True
    return moved


def collect(rt, parent, args=None):
    """This goal's children: the reports of the finished ones (the same dict the report event carries) and the
    ids of the ones still running. `settle` first, so a child that finished a moment ago is not reported running."""
    settle(rt)
    done, running = [], []
    for r in rt.store.delegates(parent=parent):
        if r.get('outcome'):
            try:
                rep = json.loads(r.get('report') or '{}')
            except ValueError:
                rep = dict(child=r['child'])
            rep.setdefault('child', r['child'])
            rep.setdefault('outcome', r['outcome'])
            done.append(rep)
        else:
            running.append(dict(child=r['child'], role=r.get('role'), permissions=r.get('permissions') or 'read',
                                cap_usd=r.get('cap_usd'), started=r.get('started')))
    return dict(children=done, running=running,
                note=('no children yet' if not done and not running else
                      '%d finished, %d running' % (len(done), len(running))))


def run_child(rt, parent, job):
    """Run one child through the loop, in this call (wait=true), and turn its goal into the report the parent
    reads. The child's stop also watches the parent: cancelling, pausing or failing the parent ends the child at
    its next check (#382 review: _stop alone is only the daemon's stop, so a cancelled parent's child ran on)."""
    child = job['child']
    daemon_stop = getattr(rt, '_stop', None)
    # The child's turn puts its own brain in rt.model (runtime._tick). Put the resident's back when the run ends, or
    # the parent's next turn - and every later goal - would run on the child's model: with one brain configured,
    # Brains.choose has nothing to switch back to (#384 review).
    resident_model = getattr(rt, 'model', None)

    def parent_gone():
        """True when the daemon is stopping, or the parent was cancelled, paused or failed while the child ran."""
        if daemon_stop and daemon_stop():
            return True
        # _parent_state answers None for a parent goal that is gone (its session was removed): that is gone too,
        # and reading it directly raised `unknown goal` out of the child's own run, before `_close` could file the
        # report, leaving the child runnable until the next settle (#384 review P2).
        return _parent_state(rt, parent) in ('cancelled', 'paused', 'error', None)

    try:
        rt.run(target=child, stop=parent_gone)
        g = rt.store.goal(child)
        # A parent goal that is gone reads like a stopped parent here: `_parent_state` answers None for it, where
        # the direct read raised `unknown goal` out of this call before `_close` filed anything, so the wait=true
        # call aborted and left the child runnable until a later settle (#384 review P2).
        pstate = _parent_state(rt, parent)
        outcome = None
        if pstate in (None, 'cancelled', 'paused', 'error') and g['state'] in ('working', 'queued', 'waiting_model'):
            # parent_gone() ended the loop, so the child's own state does not say why it stopped. Named here, and
            # reported incomplete: the report used to read 'capped' off the word 'delegate' in the generic error
            # text below, telling the parent a spend limit ended a child its own pause had ended (#384 review).
            state = 'cancelled' if pstate == 'cancelled' else 'paused'
            error = ('its parent was cancelled while it ran' if pstate == 'cancelled' else
                     'its parent goal was deleted while it ran' if pstate is None else
                     'its parent stopped (%s) while it ran' % pstate)
            with rt.store.transaction():
                rt.store.update(child, state=state, error=error)
            g, outcome = rt.store.goal(child), 'incomplete'
        if g['state'] in ('working', 'queued', 'waiting_model'):
            # it must not be left runnable: the next turn would pick the parent's brain (its model registration
            # is this process's), and no other runtime can serve it either. Not a cap either: the daemon stopped,
            # or the model went away, and neither is a limit the child spent.
            with rt.store.transaction():
                rt.store.update(child, state='paused', error='delegate child did not finish: its model was '
                                                            'unreachable or it ran out of its budget')
            g, outcome = rt.store.goal(child), 'incomplete'
        return _close(rt, parent, _report(rt, rt.store.delegate(child) or {}, g, outcome))
    finally:
        rt._stop = daemon_stop
        rt.model = resident_model


def clock(rt, gid, g):
    """Start a child's time cap at its first turn, and return the goal row to read on.

    The enqueue sets no deadline for a child: a child queued behind another goal may wait minutes before it first
    runs, and a deadline counted from the enqueue would pause it before its first turn and report `capped` - a
    spend cap its parent never set (#384 review, P1). The cap the parent asked for lives on the delegates row
    (max_seconds); here, at the first tick, the child's deadline becomes now + max_seconds, lowered to its
    parent's own remaining deadline (a child never outlives its parent's budget)."""
    row = rt.store.delegate(gid) or {}
    secs, deadline = row.get('max_seconds'), g.get('deadline')
    if not secs or deadline is not None:
        return g
    with rt.store.transaction():
        now = rt.store.goal(gid)               # re-read inside the transaction: another writer settled it first?
        if now['deadline'] is not None:
            return now
        parent = rt.store.goal(row['parent']) if row.get('parent') else None
        end = time.time() + float(secs)
        if parent and parent.get('deadline'):
            end = min(end, parent['deadline'])
        rt.store.update(gid, deadline=end)
    rt.store.event('delegate_clock', dict(max_seconds=float(secs), deadline=end), gid)
    return rt.store.goal(gid)


def capped(rt, gid):
    """Pause a child whose spend (money + plan allowance) reached its cap; True when it was paused.

    A pre-check: a turn cannot be priced before it is taken, and `_tick` runs this before the model call (so a cap
    stops a turn rather than discovering the overspend after paying for it). The ledger is written after the turn
    (`note_spend`, from `_meter`), so the check can lag by exactly one turn's cost - the smallest lag a pre-check
    can have. `_tick` checks again at the top of every tick, so the overshoot is seen before the next turn and
    named in the child's report instead of repeating (#384 review, P2)."""
    row = rt.store.delegate(gid)
    if not row:
        return False
    cap = row.get('cap_usd')
    if cap is None:
        return False                      # only a missing cap is no cap: a stored 0 is a $0 cap, and it pauses
    cap = float(cap)
    spend, allowance = float(row.get('spend_usd') or 0.0), float(row.get('allowance_usd') or 0.0)
    if spend + allowance < cap:
        return False
    with rt.store.transaction():
        rt.store.update(gid, state='paused', error='delegate spend cap reached: $%.4f of $%.4f' % (spend + allowance, cap))
    rt.store.event('delegate_capped', dict(spend_usd=spend, allowance_usd=allowance, cap_usd=cap), gid)
    return True


def note_spend(rt, gid, usd, allowance):
    """A child's turn: the per-subagent ledger (delegates.spend_usd / allowance_usd), the metric the cap reads."""
    if gid not in (getattr(rt, 'delegate_caps', None) or ()):
        return
    rt.store.delegate_spend(gid, usd or 0.0, allowance or 0.0)


def sweep(rt):
    """At a runtime start, a child left runnable belongs to a run that ended (the store has one writer, and the
    runtime lock proved no other process holds it): pause it so it cannot run later on the wrong brain, and close
    its report as orphaned - its parent, if it is still waiting, wakes from `settle`.

    A child this process registered (delegate_caps, filled by start()) is live here: its model and cap are in this
    process's memory, so it is not an orphan, and pausing it would end the very child a delegate(wait=false) call
    just queued (#384 review: the daemon swept on every pass, so every background child died with zero steps)."""
    for row in rt.store.delegates(open_only=True):
        if row['child'] in (getattr(rt, 'delegate_caps', None) or ()):
            continue                      # running in this process now
        try:
            g = rt.store.goal(row['child'])
        except ValueError:
            # The child's own goal is gone (a hand edit, or a restore that kept the delegates row): this read
            # raised `unknown goal` out of Runtime.run before any pass, so the daemon failed to start on every
            # restart and never reached `settle`, which is what closes this row. Close it from the row itself
            # (#384 review P1).
            _close(rt, row['parent'], _report_gone(rt, row))
            continue
        orphaned = False
        if g['state'] in ('queued', 'working', 'waiting_model'):
            with rt.store.transaction():
                rt.store.update(g['id'], state='paused', error='its runtime ended mid-delegation')
            g = rt.store.goal(g['id'])
            orphaned = True
        if g['state'] in ('completed', 'cancelled', 'paused', 'error'):
            _close(rt, row['parent'], _report(rt, row, g, 'orphaned' if orphaned else None))
