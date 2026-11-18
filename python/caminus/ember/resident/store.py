"""The resident's durable goals, conversation, actions and observable event stream."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
from .memory import history_text

STAMP_EVERY_S = 600     # tool results get a time stamp at most this often while the agent works on (Store.stamp)
# Time passes in these states: the next message a goal stores after one of them carries the time (Store.stamp).
WAITING = {'paused', 'waiting_user', 'waiting_root', 'waiting_plan', 'waiting_model', 'error'}


def time_line(now):
    """the time stamp's words: local time with weekday and zone, minute resolution, e.g. [time: Wed 2026-11-18 00:41 EST]
    (Qwen Code's context note, share/harness/qwen-code/context_note.py, writes the same)"""
    return time.strftime('[time: %a %Y-%m-%d %H:%M %Z]', time.localtime(now))


def agent_home(name, home=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        raise ValueError('agent name: 1..64 letters, digits, dots, underscores or hyphens')
    return Path(home).expanduser().resolve() if home else Path.home() / 'homes' / name


def atomic_json(path, value, mode=0o600, group=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as f:
            tmp.chmod(mode)
            if group is not None:
                import os
                os.fchown(f.fileno(), -1, group)
            json.dump(value, f, ensure_ascii=False)
            f.flush()
            import os
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class Store:
    def __init__(self, home):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.home / 'agent.sqlite3', timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA busy_timeout=10000')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS goals(
            id TEXT PRIMARY KEY, prompt TEXT NOT NULL, state TEXT NOT NULL,
            messages TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '', steps INTEGER NOT NULL DEFAULT 0,
            step_limit INTEGER, deadline REAL, created REAL NOT NULL, updated REAL NOT NULL,
            position REAL, plan INTEGER NOT NULL DEFAULT 0, plan_text TEXT);
          CREATE TABLE IF NOT EXISTS actions(
            id TEXT PRIMARY KEY, goal TEXT NOT NULL, call_id TEXT NOT NULL,
            name TEXT NOT NULL, args TEXT NOT NULL, state TEXT NOT NULL,
            result TEXT, bridge_id TEXT, UNIQUE(goal,call_id));
          CREATE TABLE IF NOT EXISTS events(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
            goal TEXT, kind TEXT NOT NULL, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS inbox(
            id INTEGER PRIMARY KEY AUTOINCREMENT, goal TEXT NOT NULL,
            text TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL DEFAULT 'nudge');
          CREATE TABLE IF NOT EXISTS peer_deliveries(id TEXT PRIMARY KEY, goal TEXT, error TEXT);
          -- Every steer, verbatim and at goal level: compaction and the finish report must be able to
          -- quote what the human actually said, which a summarised transcript cannot.
          CREATE TABLE IF NOT EXISTS steers(id INTEGER PRIMARY KEY, goal TEXT, text TEXT, ts REAL,
                                            kind TEXT NOT NULL DEFAULT 'steer');
          -- Failures by signature, so "the same failure twice" is a fact about this call, not a mood.
          CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY, goal TEXT, signature TEXT,
                                              count INTEGER, last_error TEXT, ts REAL,
                                              UNIQUE(goal,signature));
          -- A request is work the user asked for that only the runtime loop may do: compaction needs the
          -- model and the single writer, so the CLI records the wish and the loop performs it.
          CREATE TABLE IF NOT EXISTS requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT, goal TEXT NOT NULL,
            kind TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'new', error TEXT NOT NULL DEFAULT '',
            created REAL NOT NULL DEFAULT 0, updated REAL NOT NULL DEFAULT 0);
          -- Every tick asks for a goal's unconsumed inbox rows, its pending actions and whether
          -- discovery already ran; without these the answer is a full scan of a table that only grows.
          CREATE INDEX IF NOT EXISTS inbox_goal ON inbox(goal,consumed);
          CREATE INDEX IF NOT EXISTS actions_goal ON actions(goal,state);
          CREATE INDEX IF NOT EXISTS events_goal ON events(goal,kind);
          CREATE INDEX IF NOT EXISTS requests_open ON requests(kind,state);
          -- A checkpoint is a tar snapshot of named paths plus its manifest: enough to put the human's
          -- workspace back the way it was, without assuming the paths are one git repository.
          CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY, goal TEXT, label TEXT, paths TEXT,
                                                 ts REAL, bytes INTEGER);
          -- A file snapshot is taken just before an edit / write runs, so /rewind can put the files of a
          -- later step back (terminal side effects are not undone). content holds the bytes as they were,
          -- or empty when skipped says the file was over the cap.
          CREATE TABLE IF NOT EXISTS file_snapshots(
            id INTEGER PRIMARY KEY AUTOINCREMENT, goal TEXT NOT NULL, step INTEGER NOT NULL,
            path TEXT NOT NULL, label TEXT NOT NULL, existed INTEGER NOT NULL,
            content BLOB, skipped INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS steers_goal ON steers(goal,id);
          CREATE INDEX IF NOT EXISTS checkpoints_goal ON checkpoints(goal,ts);
          CREATE INDEX IF NOT EXISTS failures_goal ON failures(goal,signature);
          CREATE INDEX IF NOT EXISTS file_snapshots_goal ON file_snapshots(goal,step);
          -- A grant is one exact root command the owner approved "for this goal" (`answer ID approved --grant`):
          -- the same byte-identical command is then approved without asking again, every use still audited,
          -- and the command keeps being executed by the owner's side. Grants end with their goal.
          CREATE TABLE IF NOT EXISTS grants(
            goal TEXT NOT NULL, command TEXT NOT NULL, ts REAL NOT NULL,
            UNIQUE(goal,command));
        ''')
        self._migrate()
        self.stamped = {}    # goal id -> when this process last wrote the time into its conversation (stamp)

    def stamp(self, gid, content, always=False):
        """content with the time stamp (time_line) as its last line when the goal's conversation should learn the
        time now, else content as it is. The one rule for when the agent is told the time:

        - always (the caller says so): a goal's first message (_enqueue), its human's messages and the notes that
          arrive with them (a nudge, a steer, an answer, a scheduled wake-up: the runtime's inbox), the goal restated
          after a compaction, and the result of the wait tool;
        - a tool result, or the first turn's discovery: the first one this process stores for the goal (a new goal, a
          restart), the first after the
          goal was in a WAITING state (paused and resumed, parked on its human, root, a plan approval or the model) or
          after a compaction trimmed its history, and otherwise one once STAMP_EVERY_S passed since the last stamp.

        Written once, into the newest message as it is stored, and never at send time: every later request replays
        the same bytes, so the prompt cache (Caminus serve's prefix cache, a provider's) keeps the whole history. A
        stamp made at send time would change the message on every request; in the system prompt or the tools it would
        void the cache from the first token. About a dozen tokens each."""
        now = time.time()
        last = self.stamped.get(gid)
        if not (always or last is None or now - last >= STAMP_EVERY_S):
            return content
        self.stamped[gid] = now
        return (content or '') + '\n' + time_line(now)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def event(self, kind, data, goal=None):
        cur = self.db.execute('INSERT INTO events(ts,goal,kind,data) VALUES(?,?,?,?)',
                              (time.time(), goal, kind, json.dumps(data, ensure_ascii=False)))
        if (text := history_text(kind, data)):
            # The agent's own history, searchable without a model call (memory.Memory.search).
            self.db.execute('INSERT INTO history_fts(rowid,text,goal,kind,seq) VALUES(?,?,?,?,?)',
                            (cur.lastrowid, text, goal, kind, cur.lastrowid))

    def events(self, after=0, limit=200):
        return [dict(seq=r['seq'], ts=r['ts'], goal=r['goal'], kind=r['kind'], data=json.loads(r['data']))
                for r in self.db.execute('SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?', (after, limit))]

    def enqueue(self, prompt, max_steps=None, max_seconds=None, kind='work', plan=0):
        if not prompt.strip():
            raise ValueError('goal must not be empty')
        if max_steps is not None and max_steps < 1 or max_seconds is not None and max_seconds <= 0:
            raise ValueError('test budgets must be positive')
        with self.transaction():
            gid = self._enqueue(prompt, max_steps, max_seconds, kind, plan)
        return gid

    def _enqueue(self, prompt, max_steps=None, max_seconds=None, kind='work', plan=0):
        """Caller owns the transaction, including any delivery checkpoint."""
        gid, now = uuid.uuid4().hex, time.time()
        # The queue's explicit order: newest last. next_goal runs the queued goals by it.
        row = self.db.execute('SELECT MAX(position) FROM goals').fetchone()
        position = (row[0] or 0) + 1
        # plan: 0 no, 1 unapproved plan mode, 2 plan approved (planning.py).
        self.db.execute('INSERT INTO goals(id,prompt,state,messages,step_limit,deadline,created,updated,kind,position,plan) '
                        'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (gid, prompt, 'queued', json.dumps([dict(role='user', content=self.stamp(gid, prompt, always=True))]), max_steps,
             now + max_seconds if max_seconds else None, now, now, kind, position, plan))
        self.event('goal_queued', dict(prompt=prompt, kind=kind) if kind != 'work' else dict(prompt=prompt), gid)
        return gid

    def goal(self, gid):
        r = self.db.execute('SELECT * FROM goals WHERE id=?', (gid,)).fetchone()
        if r is None:
            raise ValueError('unknown goal ' + gid)
        out = dict(r)
        out['messages'] = json.loads(out['messages'])
        return out

    def _migrate(self):
        """Add columns an older database predates. CREATE TABLE IF NOT EXISTS leaves an existing goals
        table untouched, so the loop's own state has to be added explicitly."""
        have = {r['name'] for r in self.db.execute('PRAGMA table_info(goals)')}
        for name, decl in (('done_check','TEXT'), ('steer_seq','INTEGER NOT NULL DEFAULT 0'),
                           ('approach_changes','INTEGER NOT NULL DEFAULT 0'), ('ledger','TEXT'),
                           ('kind',"TEXT NOT NULL DEFAULT 'work'"), ('wake_at','REAL'), ('wake_note','TEXT'), ('gate','TEXT'),
                           ('plan','INTEGER NOT NULL DEFAULT 0'), ('plan_text','TEXT'),
                           ('session','TEXT')):   # (session: the conversation's id an OpenCode plan is sent)
            if name not in have:
                self.db.execute('ALTER TABLE goals ADD COLUMN ' + name + ' ' + decl)
        if 'position' not in have:
            # The queue's explicit order; an older database keeps its FIFO order by filing create time as position.
            self.db.execute('ALTER TABLE goals ADD COLUMN position REAL')
            self.db.execute('UPDATE goals SET position=created WHERE position IS NULL')
        for table, name, decl in (('inbox','kind',"TEXT NOT NULL DEFAULT 'nudge'"),
                                  ('steers','kind',"TEXT NOT NULL DEFAULT 'steer'")):
            have = {r['name'] for r in self.db.execute('PRAGMA table_info(' + table + ')')}
            if have and name not in have:
                self.db.execute('ALTER TABLE ' + table + ' ADD COLUMN ' + name + ' ' + decl)
        have = {r['name'] for r in self.db.execute('PRAGMA table_info(failures)')}
        if have and 'last_error' not in have:
            self.db.execute('ALTER TABLE failures ADD COLUMN last_error TEXT')
        # history_search: a full-text index over the events that are history. A database from before it is indexed
        # once, here, from its events.
        fresh = not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='history_fts'").fetchone()
        self.db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5('
                        'text, goal UNINDEXED, kind UNINDEXED, seq UNINDEXED)')
        if fresh:
            for r in self.db.execute('SELECT seq,goal,kind,data FROM events').fetchall():
                try:
                    text = history_text(r['kind'], json.loads(r['data']))
                except ValueError:
                    continue
                if text:
                    self.db.execute('INSERT INTO history_fts(rowid,text,goal,kind,seq) VALUES(?,?,?,?,?)',
                                    (r['seq'], text, r['goal'], r['kind'], r['seq']))
        self.db.commit()

    def update(self, gid, **fields):
        allowed = {'state', 'messages', 'summary', 'error', 'steps', 'step_limit', 'deadline',
                   'done_check', 'steer_seq', 'approach_changes', 'ledger', 'wake_at', 'wake_note', 'gate',
                   'position', 'plan', 'plan_text', 'session'}
        if not fields or not set(fields) <= allowed:
            raise ValueError('invalid goal fields')
        if fields.get('state') in WAITING:
            self.stamped.pop(gid, None)     # time passes while it waits: its next message carries the time (stamp)
        if 'messages' in fields:
            fields['messages'] = json.dumps(fields['messages'], ensure_ascii=False)
        fields['updated'] = time.time()
        sql = ','.join(k + '=?' for k in fields)
        self.db.execute('UPDATE goals SET ' + sql + ' WHERE id=?', [*fields.values(), gid])

    def add_steer(self, gid, text, kind='steer'):
        """Keep the human's words verbatim at goal level, and return the id it was filed under."""
        cur = self.db.execute('INSERT INTO steers(goal,text,ts,kind) VALUES(?,?,?,?)',
                              (gid, text, time.time(), kind))
        return cur.lastrowid

    def steers(self, gid):
        return [dict(r) for r in self.db.execute(
            'SELECT id,text,ts,kind FROM steers WHERE goal=? ORDER BY id', (gid,))]

    def note_failure(self, gid, signature, error):
        """Count one more occurrence of a failure signature and return the new count."""
        self.db.execute('INSERT INTO failures(goal,signature,count,last_error,ts) VALUES(?,?,1,?,?) '
                        'ON CONFLICT(goal,signature) DO UPDATE SET count=count+1, last_error=excluded.last_error, '
                        'ts=excluded.ts', (gid, signature, str(error)[:400], time.time()))
        row = self.db.execute('SELECT count FROM failures WHERE goal=? AND signature=?', (gid, signature)).fetchone()
        return row['count'] if row else 1

    def failure(self, gid, signature):
        r = self.db.execute('SELECT count,last_error FROM failures WHERE goal=? AND signature=?',
                            (gid, signature)).fetchone()
        return dict(r) if r else None

    def verified_since_steer(self, gid):
        """Whether the done check has been exercised since the last steer (a redirect) arrived. Nudges and questions
        do not count: they do not move what done means, and the finish judge still checks every one of them. (A batch
        recorded before the field existed counts as a steer.)"""
        return bool(self.db.execute(
            'SELECT 1 FROM events WHERE goal=? AND kind=? AND seq > COALESCE('
            '(SELECT MAX(seq) FROM events WHERE goal=? AND kind=? AND data NOT LIKE ?),0) LIMIT 1',
            (gid, 'verified', gid, 'steers_applied', '%"redirect": false%')).fetchone())

    def clear_precondition_failures(self, gid, text):
        """Forget the failures that were only a precondition's refusal (text: in its error, also when the watchdog
        quoted it), once the precondition is met: retrying the same call is then the right move, not a loop."""
        self.db.execute("DELETE FROM failures WHERE goal=? AND instr(last_error, ?) > 0", (gid, text))

    def clear_failures(self, gid):
        self.db.execute('DELETE FROM failures WHERE goal=?', (gid,))

    def set_todo(self, gid, items, blockers):
        """Replace the goal's todo list (items with a state) and what blocks it now. Stored in the ledger column,
        which the todo list replaced."""
        self.db.execute('UPDATE goals SET ledger=?, updated=? WHERE id=?',
                        (json.dumps(dict(items=items, blockers=blockers), ensure_ascii=False), time.time(), gid))

    def todo(self, gid):
        """The goal's todo list. A ledger from before the todo list reads as pending items."""
        raw = self.db.execute('SELECT ledger FROM goals WHERE id=?', (gid,)).fetchone()
        try:
            out = json.loads(raw['ledger']) if raw and raw['ledger'] else {}
        except ValueError:
            out = {}
        items = out.get('items')
        if items is None:
            items = [dict(text=str(m), state='pending') for m in out.get('milestones') or []]
        return dict(items=items, blockers=out.get('blockers') or [])

    def add_checkpoint(self, cid, gid, label, paths, size):
        self.db.execute('INSERT INTO checkpoints(id,goal,label,paths,ts,bytes) VALUES(?,?,?,?,?,?)',
                        (cid, gid, label, json.dumps(paths), time.time(), size))

    def checkpoint(self, cid):
        r = self.db.execute('SELECT * FROM checkpoints WHERE id=?', (cid,)).fetchone()
        if not r:
            return None
        out = dict(r); out['paths'] = json.loads(out['paths'])
        return out

    def checkpoints(self, gid):
        return [dict(r) for r in self.db.execute(
            'SELECT id,label,ts,bytes FROM checkpoints WHERE goal=? ORDER BY ts DESC LIMIT 20', (gid,))]

    def add_snapshot(self, gid, step, path, label, existed, content, skipped=False):
        self.db.execute('INSERT INTO file_snapshots(goal,step,path,label,existed,content,skipped) '
                        'VALUES(?,?,?,?,?,?,?)', (gid, step, path, label, 1 if existed else 0,
                                                  sqlite3.Binary(content or b''), 1 if skipped else 0))

    def file_snapshots(self, gid):
        """The goal's snapshots in the order they were taken; content is not carried (rewind reads it row by row)."""
        return [dict(r) for r in self.db.execute(
            'SELECT id,step,path,label,existed,skipped FROM file_snapshots WHERE goal=? ORDER BY id', (gid,))]

    def file_snapshot_content(self, sid):
        return self.db.execute('SELECT content FROM file_snapshots WHERE id=?', (sid,)).fetchone()['content']

    def rewind_steps(self, gid):
        """One entry per step with snapshots, in order: what /rewind --list offers."""
        out = []
        for r in self.db.execute('SELECT step,label,skipped FROM file_snapshots WHERE goal=? ORDER BY id', (gid,)):
            label = ('skipped (over the cap): ' if r['skipped'] else '') + r['label']
            if out and out[-1]['step'] == r['step']:
                out[-1]['label'] += '; ' + label
            else:
                out.append(dict(step=r['step'], label=label))
        return out

    def add_grant(self, gid, command):
        self.db.execute('INSERT OR IGNORE INTO grants(goal,command,ts) VALUES(?,?,?)', (gid, command, time.time()))

    def has_grant(self, gid, command):
        """The exact command was granted for this goal, and the goal can still run (grants end with it)."""
        r = self.db.execute('SELECT 1 FROM grants g JOIN goals ON goals.id=g.goal '
                            "WHERE g.goal=? AND g.command=? AND goals.state NOT IN ('completed','cancelled')",
                            (gid, command)).fetchone()
        return r is not None

    def end_grants(self, gid):
        self.db.execute('DELETE FROM grants WHERE goal=?', (gid,))

    def next_goal(self):
        # The queue's explicit order (position); COALESCE keeps an older row that predates it in place.
        rows = self.db.execute("SELECT id FROM goals WHERE state IN ('queued','working','waiting_model','waiting_root','waiting_user') "
                               'ORDER BY COALESCE(position,created)')
        return [r['id'] for r in rows]

    def queue(self):
        """The queued goals in run order, positions numbered 1..n (what the CLI contract shows)."""
        rows = [dict(id=r['id'], prompt=r['prompt']) for r in
                self.db.execute("SELECT id,prompt FROM goals WHERE state='queued' ORDER BY position")]
        for i, row in enumerate(rows, 1):
            row['position'] = i
        return rows

    def _queued_goal(self, prefix):
        gid = self.resolve(prefix)
        state = self.db.execute('SELECT state FROM goals WHERE id=?', (gid,)).fetchone()[0]
        if state != 'queued':
            raise ValueError(f'goal {gid[:8]} is not queued (state {state}); only a queued goal can be moved or removed')
        return gid

    def queue_move(self, prefix, to):
        with self.transaction():
            gid = self._queued_goal(prefix)
            rows = [r['id'] for r in self.db.execute("SELECT id FROM goals WHERE state='queued' ORDER BY position")]
            if not 1 <= to <= len(rows):
                raise ValueError(f'--to is 1 to {len(rows)}, not {to}')
            rows.remove(gid)
            rows.insert(to - 1, gid)
            now = time.time()
            for i, g in enumerate(rows, 1):
                self.db.execute('UPDATE goals SET position=?,updated=? WHERE id=?', (i, now, g))
        return self.queue()

    def queue_remove(self, prefix):
        with self.transaction():
            gid = self._queued_goal(prefix)
            self.db.execute("UPDATE goals SET state='cancelled',updated=? WHERE id=?", (time.time(), gid))
            self.event('dequeued', {}, gid)
        return self.queue()

    def actions(self, gid):
        return [dict(r) for r in self.db.execute("SELECT * FROM actions WHERE goal=? AND state!='done' ORDER BY rowid", (gid,))]

    def request(self, goal, kind, data=None):
        """Record work only the runtime loop can do (compaction needs the model and the single writer) and
        return its id. The loop performs it on its next pass; the CLI never does it behind the loop's back."""
        now = time.time()
        cur = self.db.execute('INSERT INTO requests(goal,kind,data,state,created,updated) VALUES(?,?,?,?,?,?)',
                              (goal, kind, json.dumps(data or {}), 'new', now, now))
        return cur.lastrowid

    def pending_request(self, goal, kind):
        r = self.db.execute("SELECT * FROM requests WHERE goal=? AND kind=? AND state='new' ORDER BY id LIMIT 1",
                            (goal, kind)).fetchone()
        return dict(r) if r else None

    def pending_requests(self, kind):
        """Every goal with an unstarted request of this kind, oldest request first."""
        return [r['goal'] for r in self.db.execute(
            "SELECT goal, MIN(id) AS first FROM requests WHERE kind=? AND state='new' GROUP BY goal ORDER BY first", (kind,))]

    def settle_request(self, rid, state, error=''):
        self.db.execute('UPDATE requests SET state=?, error=?, updated=? WHERE id=?', (state, error, time.time(), rid))

    def unsend(self, prefix):
        """Pull back the newest inbox row the goal has not read yet (the up-arrow on an empty composer):
        remove it and return its text, or None when there is nothing to withdraw."""
        with self.transaction():
            gid = self.resolve(prefix)
            # Only the human's own words: the harness writes notes to the inbox too (gate results, answers to the
            # agent's requests, cut-off notices), all tagged with a leading [...]; pulling one back would delete it.
            row = self.db.execute("SELECT id,text FROM inbox WHERE goal=? AND consumed=0 AND substr(text,1,1)!='[' "
                                  'ORDER BY id DESC LIMIT 1', (gid,)).fetchone()
            if row is None:
                return gid, None
            self.db.execute('DELETE FROM inbox WHERE id=?', (row['id'],))
            self.event('message_withdrawn', dict(text=row['text']), gid)
            return gid, row['text']

    def resolve(self, prefix):
        """A goal by exact id or unambiguous prefix, so a person can type the eight characters a UI shows."""
        row = self.db.execute('SELECT id FROM goals WHERE id=?', (prefix,)).fetchone()
        if row:
            return row['id']
        rows = [r['id'] for r in self.db.execute('SELECT id FROM goals WHERE id LIKE ? LIMIT 2', (prefix + '%',))]
        if not rows:
            raise ValueError('unknown goal ' + prefix)
        if len(rows) > 1:
            raise ValueError('ambiguous goal ' + prefix)
        return rows[0]

    def latest_goal(self):
        r = self.db.execute('SELECT id FROM goals ORDER BY created DESC LIMIT 1').fetchone()
        if r is None:
            raise ValueError('no goals yet')
        return r['id']

    def model_reply(self, gid, message, reasoning='', tokens=0):
        """Store one reply. reasoning is displayed with the reply but never sent back to the model:
        it is not part of messages, so replay and the token budget are unchanged. tokens (the provider's count of
        what the reply wrote, 0 when unknown) goes on the event only, for the UI's per-turn count."""
        with self.transaction():
            g = self.goal(gid)
            if g['state'] == 'cancelled':
                return False
            message = self._unique_calls(gid, message)
            self.update(gid, messages=[*g['messages'], message], steps=g['steps'] + 1, state='working', error='')
            for call in message.get('tool_calls') or []:
                f = call['function']
                self.db.execute('INSERT INTO actions(id,goal,call_id,name,args,state) VALUES(?,?,?,?,?,?)',
                    (uuid.uuid4().hex, gid, call['id'], f['name'], f['arguments'], 'pending'))
            shown = dict(message, reasoning=reasoning) if reasoning else dict(message)
            if tokens:
                shown['tokens'] = tokens
            self.event('assistant', shown, gid)
        return True

    def _unique_calls(self, gid, message):
        """A backend may reuse a tool-call id across turns (llama.cpp numbers calls from zero on every
        request). The action rows keep one row per call, so a repeated id is replaced in both the stored
        message and its action; the reply itself is unchanged in meaning."""
        calls = message.get('tool_calls') or []
        if not calls:
            return message
        taken = {r[0] for r in self.db.execute('SELECT call_id FROM actions WHERE goal=?', (gid,))}
        out, changed = [], False
        for call in calls:
            if call['id'] in taken:
                fresh = uuid.uuid4().hex
                taken.add(fresh)
                call = dict(call, id=fresh)
                changed = True
            else:
                taken.add(call['id'])
            out.append(call)
        return dict(message, tool_calls=out) if changed else message

    def action_result(self, action, result, panel=None):
        """panel: what the event carries for the panel beside the result, never in the conversation."""
        with self.transaction():
            g = self.goal(action['goal'])
            # Only the runtime lock's owner calls this. Completed actions are never replayed.
            row = self.db.execute('SELECT state FROM actions WHERE id=?', (action['id'],)).fetchone()
            if row is None or row[0] == 'done':
                # Gone (a superseded or deduplicated call) or already done: recording a result for it would
                # only fail, and a missing row must never take the goal down.
                return
            self.db.execute("UPDATE actions SET state='done',result=? WHERE id=?", (json.dumps(result), action['id']))
            # the time goes into the stored message only: the action row and the event (the panel) keep the result
            msg = dict(role='tool', tool_call_id=action['call_id'],
                       content=self.stamp(g['id'], json.dumps(result, ensure_ascii=False), always=action.get('name') == 'wait'))
            from .files import image_ref
            if (ref := image_ref(action.get('name'), result)):
                msg['image'] = ref    # an image read: the transport sends the file (model.py); the store keeps no bytes
            # A stop made while the call ran (cancel, the repeat-failure watchdog's pause, an error) stands: recording
            # the result must not set the goal back to working (it did, so the watchdog could never stop a loop).
            keep = g['state'] in ('cancelled', 'paused', 'error', 'completed')
            self.update(g['id'], messages=[*g['messages'], msg], state=g['state'] if keep else 'working')
            self.event('tool_result', dict(name=action['name'], call_id=action['call_id'], result=result,
                                           **({'panel': panel} if panel else {})), g['id'])

    def aim(self, gid):
        """The goal's redirected aim, or None when it still works toward its prompt (the panel's goalAim rule)."""
        return _aim_rows(self.db.execute("SELECT kind,data FROM events WHERE goal=? AND kind IN ('user_message','done_check') "
                                         "ORDER BY seq", (gid,)))

    def status(self):
        goals = [dict(r) for r in self.db.execute('SELECT id,prompt,state,summary,error,steps,created,updated,'
                                                  'plan,plan_text,wake_at FROM goals ORDER BY created DESC LIMIT 50')]
        now = time.time()
        for g in goals:
            # parked (schedule): its state stays working in the store, so every UI showed two working goals with the
            # one-at-a-time loop; parked_until says it is waiting for its wake time instead
            w = g.pop('wake_at', None)
            if w and w > now and g['state'] in ('working', 'queued'):
                g['parked_until'] = w
        waiting = [dict(r) for r in self.db.execute("SELECT goal,name,bridge_id FROM actions WHERE state='waiting'")]
        # compactions per goal, counted from the whole history (the panel keeps only its newest events)
        compactions = {g: n for g, n in self.db.execute("SELECT goal,count(*) FROM events WHERE kind='compacted' "
                                                         "AND goal IS NOT NULL GROUP BY goal")}
        # what a redirected goal works toward now (a message after its first done check, then a new check): the panel's
        # header reads it from the events it holds; a session switcher card has only this status
        aims = {g['id']: a for g in goals[:5] if (a := self.aim(g['id']))}
        # a queued goal's place in the run order (1 runs next): the list is newest first, the queue is not
        places = {q['id']: q['position'] for q in self.queue()}
        for g in goals:
            if g['id'] in places:
                g['queue'] = places[g['id']]
        return dict(goals=goals, waiting=waiting, compactions=compactions, aims=aims,
                    last_event=self.db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0])


def _aim_rows(rows):
    text, msg, checked = None, None, False
    for kind, data in rows:
        if kind == 'user_message':
            try:
                t = (json.loads(data).get('text') or '').strip()
            except (ValueError, AttributeError):
                t = ''
            if t and checked:
                msg = t  # (a message before the first done check refines the prompt; it does not redirect it)
        else:
            if msg:
                text, msg = msg, None
            checked = True
    return text


def log_failure(home, record):
    """Append one failed or refused model call to failures.jsonl in the resident home (mode 600). Every failure is kept
    for debugging: the status, the provider's reason, the refusal category, fallback details and the request id. Never
    raises: the log is a record, not a reason to lose the turn."""
    try:
        path = Path(home) / 'failures.jsonl'
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, 'a') as f:
            f.write(json.dumps(dict(ts=time.strftime('%Y-%m-%dT%H:%M:%S'), **record), ensure_ascii=False, default=str) + '\n')
    except (OSError, TypeError, ValueError):
        pass
