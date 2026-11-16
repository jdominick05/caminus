"""lease: claiming a GPU, memory or time on the shared machine before heavy work.

A lease is a file in a folder the agent and its human both write (by default the bridge's leases/ folder). The
human's GPU windows keep their leases there too: a file named by the window's PID listing the HIP indices it holds.
An agent's lease is a JSON file named agent-<name>-<id>.json (a name the windows' numeric glob skips, since they
cannot signal another account's process and would take it for stale).

A lease is a claim both sides check before starting work; it cannot stop a command from touching a card. It expires
on its own (at most LEASE_MAX_MIN minutes), is released when the goal ends, and machine() lists it.
"""
import json
import os
import time
import uuid
from pathlib import Path

LEASE_MAX_MIN = 120
RAM_RESERVE_MIB = 8192    # memory left for the human's desktop and server when a RAM lease is granted


def folder(rt):
    p = rt.config.get('leases') or (Path(rt.config['bridge']).expanduser() / 'leases' if rt.config.get('bridge') else None)
    return Path(p) if p else None


def _window_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True                                     # another account's live process
    except OSError:
        return False


def held(rt, now=None):
    """Every live lease: the human's GPU windows and the agents', expired ones left out."""
    sweep(rt)
    d, now, out = folder(rt), now or time.time(), []
    if d is None or not d.is_dir():
        return out
    for f in d.iterdir():
        try:
            if f.name.isdigit():                            # a GPU window: "<cards>" in a file named by its PID
                if _window_alive(int(f.name)):
                    cards = [int(c) for c in f.read_text().split() if c.isdigit()]
                    out.append(dict(who=f'gpu window {f.name}', cards=cards, ram_mib=0, until=None, file=f.name))
            elif f.name.startswith('agent-') and f.suffix == '.json':
                lease = json.loads(f.read_text())
                if lease.get('until', 0) > now:
                    out.append(dict(lease, file=f.name))
        except (OSError, ValueError):
            continue
    return out


def take(rt, gid, cards=None, ram_gib=0, minutes=30, reason=''):
    d = folder(rt)
    if d is None or not d.is_dir():
        # The folder is provisioned by `caminus ember setup`: name the exact fix instead of a silent fallback.
        # Not caminus.ember.settings: the agent's runtime is a zipapp of this package alone, so that import fails
        # there and the refusal would crash instead of naming the fix. The group is the bridge folder's own.
        try:
            import grp
            group = grp.getgrgid(Path(rt.config['bridge']).expanduser().stat().st_gid).gr_name
        except (ImportError, KeyError, OSError, TypeError):
            group = rt.config.get('bridge_group') or 'caminus-env'

        raise ValueError(f'no shared leases folder {d}: ask your human to run '
                         f'`sudo mkdir -p {d} && sudo chown root:{group} {d} && sudo chmod 2770 {d}` '
                         '(from `caminus ember setup`); leases are required before GPU work')
    if not str(reason).strip():
        raise ValueError('say what the lease is for (reason): your human sees it')
    minutes = int(minutes or 30)
    if not 1 <= minutes <= LEASE_MAX_MIN:
        raise ValueError(f'a lease lasts 1 to {LEASE_MAX_MIN} minutes; take a new one when it runs out')
    cards = sorted({int(c) for c in (cards or [])})
    ram_mib = int(float(ram_gib or 0) * 1024)
    if not cards and not ram_mib:
        raise ValueError('lease at least one card (HIP index) or some memory (ram_gib)')
    now = time.time()
    current = held(rt, now)
    for lease in current:
        clash = set(cards) & set(lease.get('cards') or [])
        if clash:
            until = '' if lease.get('until') is None else f" until {time.strftime('%H:%M', time.localtime(lease['until']))}"
            raise ValueError(f"card {sorted(clash)[0]} is held by {lease['who']}{until}"
                             f"{(': ' + lease['reason']) if lease.get('reason') else ''}; wait for it (wait until=file "
                             'on its lease, or until=time) or work on something else')
    if ram_mib:
        from .housemate import _meminfo
        avail = (_meminfo().get('available_mib') or 0) - sum(l.get('ram_mib') or 0 for l in current)
        if ram_mib > avail - RAM_RESERVE_MIB:
            raise ValueError(f'{ram_mib} MiB is more than is free to lend ({max(0, avail - RAM_RESERVE_MIB)} MiB after '
                             f'leases and the {RAM_RESERVE_MIB} MiB kept for the desktop and servers)')
    lid = uuid.uuid4().hex[:8]
    lease = dict(id=lid, who=f"agent {rt.config.get('name', '?')}", agent=rt.config.get('name'), goal=gid,
                 cards=cards, ram_mib=ram_mib, since=now, until=now + minutes * 60, reason=str(reason).strip()[:200])
    path = d / f"agent-{rt.config.get('name', 'agent')}-{lid}.json"
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(lease))
    os.replace(tmp, path)
    rt.store.event('lease', dict(lease, action='take'), gid)
    return dict(lease=lid, cards=cards, ram_mib=ram_mib, until=time.strftime('%H:%M', time.localtime(lease['until'])),
                note='release it when done (lease action=release); it also ends with the goal')


def release(rt, gid, lid=None):
    """Release one lease of this goal, or all of them (lid None). Returns the ids released."""
    d, done = folder(rt), []
    if d is None or not d.is_dir():
        return done
    for f in d.glob(f"agent-{rt.config.get('name', 'agent')}-*.json"):
        try:
            lease = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if lease.get('goal') == gid and (lid is None or lease.get('id') == lid):
            try:
                f.unlink()
                done.append(lease.get('id'))
            except OSError:
                pass
    if done:
        rt.store.event('lease', dict(action='release', ids=done), gid)
    return done


def sweep(rt):
    """Release this agent's leases whose goal has ended (completed, cancelled, error, paused): a lease ends with its
    goal wherever the goal stopped."""
    d = folder(rt)
    if d is None or not d.is_dir():
        return
    for f in d.glob(f"agent-{rt.config.get('name', 'agent')}-*.json"):
        try:
            lease = json.loads(f.read_text())
            row = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (lease.get('goal'),)).fetchone()
            if row is None or row[0] in ('completed', 'cancelled', 'error', 'paused') or lease.get('until', 0) <= time.time():
                f.unlink()
        except (OSError, ValueError):
            continue


def tool(rt, gid, action, cards=None, ram_gib=0, minutes=30, reason='', lease=''):
    if action == 'take':
        return take(rt, gid, cards, ram_gib, minutes, reason)
    if action == 'release':
        ids = release(rt, gid, lease or None)
        if lease and not ids:
            raise ValueError(f'no lease {lease} held by this goal')
        return dict(released=ids)
    if action == 'list':
        return dict(leases=[{k: v for k, v in l.items() if k != 'file'} for l in held(rt)])
    raise ValueError('action is take, release or list')
