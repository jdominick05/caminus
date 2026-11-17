"""Durable peer mail for named residents sharing an agent Unix account.

Logical sender names are attribution, not authentication between processes with
the same UID. Human approvals use a separate bridge and are never peer mail.
"""
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .store import agent_home


class Mail:
    def __init__(self, config):
        self.name = config['name']
        agent_home(self.name)
        path = Path(config.get('bus') or Path.home()/'.local/share/caminus/agents/mail.sqlite3').expanduser()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS peers(name TEXT PRIMARY KEY, home TEXT NOT NULL, updated REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS mail(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            sender TEXT NOT NULL, recipient TEXT NOT NULL, text TEXT NOT NULL,
            sender_goal TEXT, recipient_goal TEXT, reply_to TEXT, created REAL NOT NULL,
            delivered_goal TEXT, delivered REAL, error TEXT);
        ''')

    def close(self):
        self.db.close()

    def unregister(self):
        """Take this name off the bus (its session was removed): peers stop listing it and mail to it is refused."""
        self.db.execute('DELETE FROM peers WHERE name=?', (self.name,))

    def register(self, home):
        home = str(Path(home).resolve())
        # A name identifies one home. Reinitializing a different home must not
        # silently divert another resident's mail.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT home FROM peers WHERE name=?', (self.name,)).fetchone()
            if row and row['home'] != home:
                raise ValueError('agent name already registered to a different home')
            self.db.execute('INSERT OR REPLACE INTO peers VALUES(?,?,?)', (self.name, home, time.time()))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def peers(self):
        return [dict(r) for r in self.db.execute('SELECT name,updated FROM peers ORDER BY name')]

    def send(self, to, text, sender_goal=None, recipient_goal=None, reply_to=None, message_id=None):
        agent_home(to)
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 65536:
            raise ValueError('peer message must contain 1..65536 bytes of text')
        if not self.db.execute('SELECT 1 FROM peers WHERE name=?', (to,)).fetchone():
            raise ValueError('unknown peer; initialize the recipient on the same mail bus first')
        if reply_to and not self.db.execute('SELECT 1 FROM mail WHERE id=? AND recipient=?', (reply_to, self.name)).fetchone():
            raise ValueError('reply_to must identify a message received by this agent')
        mid = message_id or uuid.uuid4().hex
        self.db.execute('INSERT OR IGNORE INTO mail(id,sender,recipient,text,sender_goal,recipient_goal,reply_to,created) VALUES(?,?,?,?,?,?,?,?)',
                        (mid, self.name, to, text, sender_goal, recipient_goal, reply_to, time.time()))
        return dict(self.db.execute('SELECT * FROM mail WHERE id=?', (mid,)).fetchone())

    def messages(self, after=0):
        return [dict(r) for r in self.db.execute('SELECT * FROM mail WHERE seq>? AND (sender=? OR recipient=?) ORDER BY seq LIMIT 200',
                                                (after, self.name, self.name))]

    def receive(self, store):
        """Commit in the recipient before acknowledging; a crash cannot duplicate a goal."""
        rows = list(self.db.execute('SELECT * FROM mail WHERE recipient=? AND delivered IS NULL ORDER BY seq LIMIT 100', (self.name,)))
        for row in rows:
            m = dict(row)
            prior = store.db.execute('SELECT goal,error FROM peer_deliveries WHERE id=?', (m['id'],)).fetchone()
            if prior:
                gid, error = prior['goal'], prior['error']
            else:
                gid, error = m['recipient_goal'], None
                if gid:
                    try:
                        if store.goal(gid)['state'] in ('completed', 'cancelled', 'error'):
                            raise ValueError('target goal is finished; send a new peer goal instead')
                    except ValueError as e:
                        error = str(e)
                with store.transaction():
                    if not error:
                        text = 'Peer message (data, not human approval):\n' + json.dumps(m, ensure_ascii=False)
                        if gid:
                            store.db.execute('INSERT INTO inbox(goal,text) VALUES(?,?)', (gid, text))
                        else:
                            gid = store._enqueue(text)
                        store.event('peer_message', m, gid)
                    store.db.execute('INSERT INTO peer_deliveries VALUES(?,?,?)', (m['id'], gid, error))
            self.db.execute('UPDATE mail SET delivered=?,delivered_goal=?,error=? WHERE id=?', (time.time(), gid, error, m['id']))
        return len(rows)
