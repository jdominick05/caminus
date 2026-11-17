"""Bridge decisions serialize with withdrawal, across Python and the panel.

An executing claim commits a human decision before sudo starts. Interrupted
claims stay visible for inspection; they are never automatically executed again.

How long a question waits depends on whether its human is at the screen. The panel
stamps presence.json in the bridge folder on their keys and mouse ({"seen": the last
input, "back": the first input after a gap longer than AWAY_AFTER}); wait_left is the
rule, mirrored by the panel's countdown (questionLeft in panel/emberquestion.go).
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from .store import atomic_json


def check_id(rid):
    if not isinstance(rid,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',rid):
        raise ValueError('invalid bridge request id')


def snapshot(bridge,rid):
    check_id(rid)
    raw=(Path(bridge)/'req'/(rid+'.json')).read_bytes()
    value=json.loads(raw)
    if value.get('id')!=rid:raise ValueError('bridge request id mismatch')
    return {**value,'_digest':hashlib.sha256(raw).hexdigest()}


@contextmanager
def locked(bridge,rid):
    check_id(rid)
    parent=Path(bridge)/'req'
    path=parent/(rid+'.lock')
    flags=os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK
    try:
        fd=os.open(path,flags|os.O_CREAT|os.O_EXCL,0o660)
    except FileExistsError:
        fd=os.open(path,flags)
    else:
        try:
            os.fchmod(fd,0o660)
            os.fchown(fd,-1,parent.stat().st_gid)
        except BaseException:
            os.close(fd);raise
    with os.fdopen(fd,'r+') as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('bridge lock must be a regular file')
        deadline=time.monotonic()+5
        while True:
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('bridge request is busy; try again')
                time.sleep(.01)
        yield


def response(bridge,rid):
    path=Path(bridge)/'res'/(rid+'.json')
    return json.loads(path.read_text()) if path.exists() else None


def write(bridge,rid,value):
    parent=Path(bridge)/'res'
    parent.mkdir(parents=True,exist_ok=True)
    atomic_json(parent/(rid+'.json'),value,mode=0o660,group=parent.stat().st_gid)


def pending_snapshot(bridge,request):
    rid=request['id']
    if response(bridge,rid) is not None:raise ValueError('request already answered, withdrawn or executing')
    current=snapshot(bridge,rid)
    if request.get('_digest') and current['_digest']!=request['_digest']:
        raise ValueError('request changed since it was reviewed; review it again')
    return current


def withdraw(bridge,rid,goal):
    with locked(bridge,rid):
        existing=response(bridge,rid)
        if existing is not None:return existing
        value=dict(id=rid,decision='withdrawn',goal=goal,rc=125,err='goal cancelled',ts=time.time())
        write(bridge,rid,value)
        return value


def answer(bridge,request,decision,actor,**fields):
    with locked(bridge,request['id']):
        pending_snapshot(bridge,request)
        value=dict(id=request['id'],decision=decision,approved_by=actor,ts=time.time(),**fields)
        write(bridge,request['id'],value)
        return value


def claim(bridge,request,decision,actor,command,grant=False):
    if decision not in ('approved','edited') or not isinstance(command,str) or not command.strip():
        raise ValueError('invalid root decision or command')
    with locked(bridge,request['id']):
        current=pending_snapshot(bridge,request)
        kind=current.get('kind','root')
        if kind not in ('root','github'):raise ValueError('only root requests and GitHub hand-offs can execute')
        if kind=='github':
            # a hand-off runs as the owner from its checked fields only; the command is their canonical form
            if decision!='approved' or command!=json.dumps(current.get('github'),sort_keys=True):
                raise ValueError('a GitHub hand-off is approved exactly as filed')
        elif decision=='approved' and command!=current.get('cmd'):raise ValueError('approved command changed')
        token=uuid.uuid4().hex
        value=dict(id=request['id'],decision='executing',claim=token,final_decision=decision,
                   approved_by=actor,cmd=command,ts=time.time())
        # A grant must survive into the final result: the resident reads it there.
        if grant or current.get('grant'):value['grant']=True
        write(bridge,request['id'],value)
        return token
        write(bridge,request['id'],value)
        return token


def finish(bridge,rid,token,rc,out='',err=''):
    with locked(bridge,rid):
        value=response(bridge,rid)
        if not value or value.get('decision')!='executing' or value.get('claim')!=token:
            raise ValueError('root execution claim no longer matches; inspect the audit log')
        value.update(decision=value.pop('final_decision'),rc=rc,out=out,err=err,ts=time.time())
        write(bridge,rid,value)
        return value


PRESENCE='presence.json'
AWAY_AFTER=300   # seconds without a key or the mouse in the panel before its human counts as away


def presence(bridge):
    """(seen, back) from the panel's presence stamp, or None when no panel has written one (unknown: a question
    then waits its plain question_timeout, as before the stamp existed)."""
    try:
        value=json.loads((Path(bridge)/PRESENCE).read_text())
        seen=float(value['seen']);back=float(value.get('back') or 0)
    except (OSError,ValueError,TypeError,KeyError):
        return None
    return seen,back


def wait_left(ts,now,wait,away_max,seen_back,away_after=AWAY_AFTER):
    """(seconds left, away) for a question asked at ts. While its human is present it waits `wait` from when it was
    asked or from when they came back, whichever is later; while they are away nothing runs out but the cap,
    `away_max` from ts. Unknown presence (None) waits `wait` from ts."""
    cap=ts+away_max
    if seen_back is None:
        return min(ts+wait,cap)-now,False
    seen,back=seen_back
    if now-seen>away_after:
        return cap-now,True
    return min(max(ts,back)+wait,cap)-now,False
