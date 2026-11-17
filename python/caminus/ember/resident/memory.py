"""The resident's long memory: a small notes index, one-fact notes, STATE.md and a search over its own history.

The agent lives in its account across goals, so what it learns must outlive one conversation:

- MEMORY.md is the notes index: one line per note, always loaded, capped at INDEX_CAP bytes. It is frozen per goal
  (the first turn of a goal takes a snapshot; a write during the goal shows from the next goal), so a mid-goal edit
  cannot shift the agent's picture of itself halfway through a task. Past INDEX_WARN bytes the goal's context and
  remember's result say how full it is and how to tidy it: remember action tidy takes the rewritten index and refuses
  one that leaves a note reachable from nowhere (named neither in the new index nor in the archive) or drops a pinned
  note; a pinned line keeps its mark.
- notes/<name>.md hold one fact each, read on demand. The remember tool writes a note and its index line together.
  When the index would pass its cap, the least recently used unpinned lines move to notes/index-archive.md (still read
  on demand; history_search returns each archive line that matches, with the note it names) and the tool says
  which, so a wrong choice is put back pinned at once. A
  note is used when it is saved or read (notes/.used.json); a pinned line (remember pin) never moves. A full index
  refused every remember for four goals of a real resident, which never tidied it by hand; then oldest-first moved
  its notes on the machine itself before a week of free-time notes.
- STATE.md is the agent's live working state (doing / verified / assumed / next). It is re-read every turn and loaded
  first, so after a compaction or a daemon restart it is the first thing the agent sees.
- history_search is a full-text search over its own past goals, replies, tool results, summaries, the index archive,
  notes and journal; each hit names its source. No model call.
"""
import json
import re
import time
from pathlib import Path

INDEX_CAP = 4096          # bytes of MEMORY.md loaded into every turn
INDEX_WARN = INDEX_CAP * 8 // 10   # past this, the goal's context and remember's result say to tidy the index
STATE_CAP = 8192          # bytes of STATE.md loaded into every turn
NOTE_CAP = 8192           # bytes of one note written by remember
LINE_CAP = 160            # bytes of one index line: a long summary is cut (the note holds the detail)
ARCHIVE = 'notes/index-archive.md'
BEFORE_INDEX = 'notes/memory-before-index.md'   # an index from before the cap, kept whole
USED = 'notes/.used.json'
PINNED = ' · pinned'
LINK = re.compile(r'\]\((notes/[^)]+)\)')        # an index line's note: - [name](notes/x.md) — summary
HISTORY_KINDS = {'goal_queued': ('prompt',), 'user_message': ('text',), 'peer_message': ('text',),
                 'assistant': ('content',), 'finished': ('summary',), 'tool_result': ('result',)}


def _clip(text, cap, name, hint):
    raw = text.encode()
    if len(raw) <= cap:
        return text
    return raw[:cap].decode(errors='ignore') + f'\n[{name} is {len(raw)} bytes, over its {cap}-byte cap: {hint}]'


def history_text(kind, data):
    """The searchable text of one event, or '' for kinds that are not history."""
    keys = HISTORY_KINDS.get(kind)
    if not keys:
        return ''
    parts = []
    for k in keys:
        v = data.get(k)
        if k == 'result' and isinstance(v, dict):
            v = v.get('output') or v.get('error') or v.get('summary') or ''
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return '\n'.join(parts)[:4000]


def slug(name):
    s = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:60]
    if not s:
        raise ValueError('a note needs a name with letters or digits')
    return s


def fill_warning(nbytes):
    """What the agent is told once its index passes INDEX_WARN bytes, or ''."""
    if nbytes <= INDEX_WARN:
        return ''
    return (f'MEMORY.md is {nbytes} of its {INDEX_CAP} bytes ({nbytes * 100 // INDEX_CAP} % full). When a save would '
            f'pass the cap, the least recently used unpinned lines move to {ARCHIVE}. Tidy it first: read MEMORY.md, '
            'then call remember with action tidy, index set to the whole rewritten index (merge duplicate lines, '
            'shorten long ones) and archive set to the lines you take out (history_search still finds them)')


class Memory:
    def __init__(self, rt):
        self.rt = rt
        self.home = Path(rt.home)

    # --- what is loaded into every turn ---------------------------------------------------------------------
    def switch_state(self, gid):
        """Make STATE.md this goal's. One file was shared by every goal and by free time: a free-time project's
        state loaded into the next goal's runtime block, and the agent spent turns setting it aside. STATE.md now
        belongs to the goal named in .state-goal; when another goal runs, the file moves to states/<goal>.md and the
        running goal's own (from an earlier turn of it) comes back, or none. The path the agent writes stays STATE.md."""
        if not gid:
            return
        p, owner_f, keep = self.home / 'STATE.md', self.home / '.state-goal', self.home / 'states'
        try:
            owner = owner_f.read_text().strip()
        except OSError:
            owner = ''
        if owner == gid:
            return
        keep.mkdir(exist_ok=True)
        if p.is_file():
            # (a file from before this rule has no owner: it is kept under 'unowned', not handed to this goal)
            p.replace(keep / ((owner or 'unowned') + '.md'))
        mine = keep / (gid + '.md')
        if mine.is_file():
            mine.replace(p)
        owner_f.write_text(gid)

    def state(self, gid=None):
        self.switch_state(gid)
        p = self.home / 'STATE.md'
        if not p.is_file():
            return ''
        return 'STATE.md (your working state; keep it current):\n' + _clip(
            p.read_text(errors='replace'), STATE_CAP, 'STATE.md', 'keep it to doing / verified / assumed / next')

    def index(self, gid=None):
        """MEMORY.md as this goal sees it: the snapshot taken at its first turn (live when there is no goal)."""
        if gid:
            snap = self.rt.store.db.execute(
                "SELECT data FROM events WHERE goal=? AND kind='memory_snapshot' ORDER BY seq DESC LIMIT 1",
                (gid,)).fetchone()
            if snap is not None:
                text = json.loads(snap['data']).get('text', '')
            else:
                text = self._read_index()
                self.rt.store.event('memory_snapshot', dict(text=text, bytes=len(text.encode())), gid)
        else:
            text = self._read_index()
        # The warning reads the live file, not the snapshot: after a tidy in this goal it clears, so it cannot ask for
        # a second one. An index already past its cap gets the clip's own message instead.
        warn = fill_warning(len(self._read_index().encode())) if len(text.encode()) <= INDEX_CAP else ''
        warn = f'\n[{warn}]' if warn else ''
        if not text.strip():
            return warn.strip()
        hint = 'move details into notes/ and keep one line per note here'
        if len(text.encode()) > INDEX_CAP and (archive := self._archive(text)):
            hint = (f'the whole file is kept in {archive} (read it, or history_search finds it); move what matters '
                    'into notes with remember and shorten MEMORY.md to one line per note')
        return 'MEMORY.md (your notes index, as of this goal\'s start; notes/ holds the details):\n' + _clip(
            text, INDEX_CAP, 'MEMORY.md', hint) + warn

    def _archive(self, text):
        """An index past its cap (a memory from before the cap, written as one long file) is copied whole into a
        note the first time it is seen, so nothing it holds is lost from sight. MEMORY.md itself is never rewritten:
        it is the agent's file. Returns the note's path, or '' when it cannot be written."""
        rel = BEFORE_INDEX
        p = self.home / rel
        try:
            if not p.exists() or len(p.read_bytes()) < len(text.encode()):
                p.parent.mkdir(exist_ok=True)
                p.write_text('# MEMORY.md as it was before the 4 KB notes index\n\n' + text)
        except OSError:
            return ''
        return rel

    def _read_index(self):
        p = self.home / 'MEMORY.md'
        return p.read_text(errors='replace') if p.is_file() else ''

    # --- the remember tool ------------------------------------------------------------------------------------
    def touch(self, rel):
        """Mark a note used (saved or read): the least recently used leave a full index first."""
        p = self.home / USED
        try:
            used = json.loads(p.read_text()) if p.is_file() else {}
        except (OSError, ValueError):
            used = {}
        used[rel] = time.time()
        try:
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(used))
        except OSError:
            pass

    def _last_used(self, line):
        """When the line's notes were last used (a merged line names several: the most recent counts)."""
        try:
            used = json.loads((self.home / USED).read_text())
        except (OSError, ValueError):
            used = {}
        last = 0.0
        for rel in LINK.findall(line):
            if rel in used:
                last = max(last, float(used[rel]))
                continue
            try:
                last = max(last, (self.home / rel).stat().st_mtime)
            except OSError:
                pass
        return last

    def _to_archive(self, lines):
        arch = self.home / ARCHIVE
        head = '' if arch.is_file() else '# Index lines moved out of MEMORY.md (oldest first)\n\n'
        arch.parent.mkdir(exist_ok=True)
        with arch.open('a') as f:
            f.write(head + '\n'.join(lines) + '\n')

    def call(self, args):
        """The remember tool: action save (the default) writes a note, action tidy rewrites the index."""
        action = args.get('action') or 'save'
        need = ('index',) if action == 'tidy' else ('name', 'text', 'summary')
        if missing := [k for k in need if not str(args.get(k) or '').strip()]:
            raise ValueError(f'remember action {action} needs {", ".join(missing)}' + (
                ' (the whole rewritten MEMORY.md: read it first)' if action == 'tidy' else ''))
        if action == 'tidy':
            return self.tidy(args['index'], args.get('archive') or '')
        return self.remember(args['name'], args['text'], args['summary'], bool(args.get('pin')))

    def remember(self, name, text, summary, pin=False):
        """Write notes/<name>.md (one fact) and add or replace its line in MEMORY.md, within the index cap."""
        if not text.strip() or not summary.strip():
            raise ValueError('remember needs the fact (text) and a one-line summary for the index')
        if len(text.encode()) > NOTE_CAP:
            raise ValueError(f'one note holds one fact: keep it under {NOTE_CAP} bytes, or split it into several notes')
        s = slug(name)
        notes = self.home / 'notes'
        notes.mkdir(exist_ok=True)
        rel = f'notes/{s}.md'
        current = self._read_index()
        pin = pin or any(f']({rel})' in ln and ln.endswith(PINNED) for ln in current.splitlines())   # pinned stays pinned
        tail = PINNED if pin else ''
        line = f'- [{name.strip()}]({rel}) — ' + ' '.join(summary.split())
        if len((line + tail).encode()) > LINE_CAP:
            line = line.encode()[:LINE_CAP - 3 - len(tail.encode())].decode(errors='ignore').rstrip() + '…'
        line += tail
        # this note's own line is replaced; a merged line that also names other notes stays, or they would be lost
        kept = [ln for ln in current.splitlines() if set(LINK.findall(ln)) != {rel}]
        moved = []
        # the least recently used unpinned note line leaves first (a heading or a plain line the agent wrote stays: it
        # is its own text; a pinned line stays)
        while len(('\n'.join([*kept, line]).strip('\n') + '\n').encode()) > INDEX_CAP:
            movable = [i for i, ln in enumerate(kept) if ln.startswith('- [') and not ln.endswith(PINNED)]
            if not movable:
                raise ValueError(f'MEMORY.md is over its {INDEX_CAP}-byte cap and every note line in it is pinned: '
                                 'merge lines or unpin one (edit MEMORY.md), then remember again')
            i = min(movable, key=lambda i: self._last_used(kept[i]))
            moved.append(kept.pop(i))
        updated = '\n'.join([*kept, line]).strip('\n') + '\n'
        (notes / f'{s}.md').write_text(text.rstrip('\n') + '\n')
        self.touch(rel)
        if moved:
            self._to_archive(moved)
        (self.home / 'MEMORY.md').write_text(updated)
        out = dict(note=rel, index_bytes=len(updated.encode()), index_cap=INDEX_CAP,
                   note_text='saved; the index line shows from your next goal (this goal keeps its snapshot)')
        if moved:
            out['moved_to_archive'] = [re.sub(r'^- \[([^\]]*)\].*', r'\1', m) for m in moved]
            out['note_text'] += (f'; the index was full, so its {len(moved)} least recently used line(s) moved to '
                                 f'{ARCHIVE} (read it or history_search finds them). If one of them still matters on '
                                 'every goal, remember it again with pin true')
        if warn := fill_warning(out['index_bytes']):
            out['index_warning'] = warn
        return out

    def tidy(self, index, archive=''):
        """Replace MEMORY.md with the agent's rewritten index (merged, shortened lines) and append the lines it takes
        out to the archive. Refused, with nothing written, when a note the old index named would be reachable from
        nowhere (in neither the new index nor the archive), when a pinned note loses its line, when a new line names a
        note that does not exist, or when the result is still over the cap. A pinned note's line keeps its mark."""
        old = self._read_index()
        lines = [ln.rstrip() for ln in index.strip('\n').splitlines()]
        out_lines = [ln.strip() for ln in archive.splitlines() if ln.strip()]
        problems, repinned = [], []
        for ln in old.splitlines():
            if not ln.endswith(PINNED):
                continue
            refs = LINK.findall(ln)
            if not refs and ln not in lines:
                problems.append(f'the pinned line "{ln[:60]}" is missing (a pinned line stays; edit MEMORY.md to unpin it)')
            for rel in refs:
                mine = [i for i, n in enumerate(lines) if f']({rel})' in n]
                if not mine:
                    problems.append(f'{rel} is pinned: keep a line for it in index (edit MEMORY.md to unpin it)')
                for i in mine:
                    if not lines[i].endswith(PINNED):
                        lines[i] += PINNED
                        repinned.append(rel)
        # a long line is refused, not cut as a save cuts it: a cut could take the link the agent wrote
        if long := [str(i + 1) for i, ln in enumerate(lines) if ln.startswith('- [') and len(ln.encode()) > LINE_CAP]:
            problems.append(f'line(s) {", ".join(long)} of index are over the {LINE_CAP}-byte line cap (a pinned line '
                            f'counts its "{PINNED.strip()}" mark): shorten them')
        try:
            in_archive = (self.home / ARCHIVE).read_text(errors='replace')
        except OSError:
            in_archive = ''
        reachable = set(LINK.findall('\n'.join([*lines, *out_lines, in_archive])))
        old_refs = list(dict.fromkeys(LINK.findall(old)))
        # a note whose file is gone has nothing to reach (a merge may have folded it into another note)
        if lost := [r for r in old_refs if r not in reachable and (self.home / r).is_file()]:
            problems.append(f'{", ".join(lost)} would be reachable from nowhere: keep a line for each in index, or put '
                            f'its old line in archive (it moves to {ARCHIVE})')
        if dead := [r for r in dict.fromkeys(LINK.findall('\n'.join(lines)))
                    if r not in old_refs and not (self.home / r).is_file()]:
            problems.append(f'index names {", ".join(dead)}, which does not exist (save it with remember first)')
        updated = '\n'.join(lines).strip('\n') + '\n'
        if len(updated.encode()) > INDEX_CAP:
            problems.append(f'the new index is {len(updated.encode())} bytes, over its {INDEX_CAP}-byte cap: merge or '
                            'shorten more lines, or move some to archive')
        if problems:
            raise ValueError('tidy refused, MEMORY.md is unchanged: ' + '; '.join(problems))
        if out_lines:
            self._to_archive(out_lines)
        (self.home / 'MEMORY.md').write_text(updated)
        out = dict(index_bytes=len(updated.encode()), was_bytes=len(old.encode()), index_cap=INDEX_CAP,
                   note_text='tidied; the new index shows from your next goal (this goal keeps its snapshot)')
        if out_lines:
            out['archived'] = len(out_lines)
            out['note_text'] += f'; {len(out_lines)} line(s) moved to {ARCHIVE} (history_search finds them)'
        if repinned:
            out['kept_pinned'] = repinned
            out['note_text'] += '; a pinned note\'s line keeps its pinned mark'
        if warn := fill_warning(out['index_bytes']):
            out['index_warning'] = warn
        return out

    # --- history_search ---------------------------------------------------------------------------------------
    def search(self, query, limit=8, gid=None):
        terms = [t for t in re.findall(r'\w+', query.lower()) if len(t) > 1]
        if not terms:
            raise ValueError('history_search needs at least one word to look for')
        limit = max(1, min(int(limit or 8), 25))
        # Prefix terms: "tokeniz" finds tokenizer and tokenizing; long words are cut to a stem-like prefix.
        fts = ' '.join('"%s"*' % t.replace('"', '')[:max(4, len(t) - 3)] if len(t) > 5 else '"%s"*' % t for t in terms)
        # The running goal (gid) is not the past: its own prompt matched every search drawn from it and came first,
        # labelled a past goal, so it is left out (the agent has it); its other hits are 'this goal' and come last.
        hits, mine = [], []
        for r in self.rt.store.db.execute(
                'SELECT f.seq, f.goal, f.kind, snippet(history_fts, 0, "[", "]", " … ", 24) AS snip, e.ts '
                'FROM history_fts f JOIN events e ON e.seq=f.seq WHERE history_fts MATCH ? '
                'AND NOT (f.goal IS ? AND f.kind=\'goal_queued\') ORDER BY f.goal IS ?, rank LIMIT ?',
                (fts, gid or '', gid or '', limit)):
            this = bool(gid) and r['goal'] == gid
            (mine if this else hits).append(dict(
                source='this goal' if this else 'past goal', where=f"goal {(r['goal'] or '')[:8]} · {r['kind']}",
                when=time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts'])), text=r['snip']))
        # The index archive, the journal and the notes live in files the agent writes; search them line by line (they
        # are small). The two index archives come first and give up to half the limit in matching lines, each naming
        # the note it points to: one line there is one note, so the first hit alone hid the rest, and the cap leaves
        # room for the journal and the notes. Any other file gives its first hit.
        archives = {ARCHIVE: 'index archive', BEFORE_INDEX: 'old index'}
        files = [self.home / rel for rel in archives]
        for folder in ('journal', 'notes'):
            if (self.home / folder).is_dir():
                files += [p for p in sorted((self.home / folder).rglob('*.md'), reverse=True)
                          if str(p.relative_to(self.home)) not in archives]
        file_hits = []
        for p in files:
            rel = str(p.relative_to(self.home))
            try:
                lines = p.read_text(errors='replace').splitlines()
            except OSError:
                continue
            source = archives.get(rel) or ('journal' if rel.startswith('journal/') else 'note')
            found = 0
            for i, ln in enumerate(lines):
                if all(t in ln.lower() for t in terms):
                    hit = dict(source=source, where=f'{rel}:{i + 1}', text=ln.strip()[:300])
                    if rel in archives and (note := LINK.search(ln)):
                        hit['note'] = note.group(1)
                    file_hits.append(hit)
                    found += 1
                    if rel not in archives or found >= max(2, limit // 2):
                        break
            if len(file_hits) >= limit * 2:
                break
        hits = hits + file_hits + mine
        return dict(query=query, hits=hits[:limit * 2], note='no matches' if not hits else '')
