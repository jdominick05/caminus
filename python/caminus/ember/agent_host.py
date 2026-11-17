"""Install and address the resident over the existing SSH transport.

Only this adapter runs as the human. The uploaded stdlib zipapp, goals and OS
commands execute as the agent user. JSON travels on stdin rather than in argv.
"""
import base64
import io
import json
from pathlib import Path
import shlex
import sys
import zipfile

from . import runner


def bundle():
    root=Path(__file__).resolve().parents[1]
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in ('__init__.py','ember/__init__.py'):
            z.writestr('caminus/'+rel,(root/rel).read_bytes())
        for p in sorted((root/'ember/resident').glob('*.py')):
            z.writestr('caminus/ember/resident/'+p.name,p.read_bytes())
        z.writestr('__main__.py','from caminus.ember.resident.cli import main\nraise SystemExit(main())\n')
    return out.getvalue()


def main(args,transport):
    args=list(args)
    if '--local' in args:
        args.remove('--local')
        from .resident.cli import main as local
        return local(args)
    if args[:1]==['install']:
        data=base64.b64encode(bundle()).decode()
        script='''python3 - <<'CAMINUS_INSTALL'
import base64,hashlib,json,os
from pathlib import Path
data=base64.b64decode(PAYLOAD)
digest=hashlib.sha256(data).hexdigest()
root=Path.home()/'.local/share/caminus/resident'
root.mkdir(parents=True,exist_ok=True,mode=0o700)
p=root/(digest+'.pyz')
if not p.exists():
    tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.chmod(0o700);os.replace(tmp,p)
b=Path.home()/'bin';b.mkdir(exist_ok=True)
link=b/'caminus-agent';tmp=b/('caminus-agent.'+str(os.getpid())+'.tmp')
tmp.symlink_to(p);os.replace(tmp,link)
# every running daemon on an older runtime takes this one between goals (an upgrade left a days-old daemon running:
# install alone restarts nothing); the same request `caminus ember agent update NAME` makes
queued,skipped=[],[]
for home in sorted((Path.home()/'homes').glob('*')):
    try:
        hb=json.loads((home/'heartbeat.json').read_text())
        pid=hb.get('pid') if isinstance(hb,dict) else None
        if not isinstance(pid,int) or pid<=1:
            continue                      # (kill(0) and kill(-1) succeed: they signal groups, not a daemon)
        os.kill(pid,0)
    except (OSError,ValueError,TypeError,OverflowError):
        continue
    if hb.get('runtime')==p.name:
        continue
    u=home/'update.json'
    try:
        pending=json.loads(u.read_text()).get('program') if u.exists() else None
    except (OSError,ValueError,AttributeError):
        pending=None
    if pending==str(p):
        continue                          # already on its way to this runtime
    try:
        t=home/'update.json.tmp'
        t.write_text(json.dumps(dict(program=str(p),now=False,ts=__import__('time').time())));os.replace(t,u)
        queued.append(home.name)          # (a pending request for another runtime is replaced by this one)
    except OSError as e:
        skipped.append(dict(agent=home.name,error=str(e)[:200]))
print(json.dumps(dict(installed=str(link),artifact=str(p),sha256=digest,updates_queued=queued,updates_skipped=skipped)))
CAMINUS_INSTALL
'''.replace('PAYLOAD',repr(data))
        result=runner.run_shell(script,transport=transport)
    else:
        if args[:1] in (['--help'],['-h']) or not args:
            from .resident.cli import parser
            parser().print_help()
            print('\nRemote by default; --local selects the same runtime in a scratch home.\nagent install uploads the runtime without sudo.')
            return 0
        payload=json.dumps(dict(args=args,stdin=sys.stdin.read() if '--stdin' in args else ''),ensure_ascii=False)
        # A fresh delimiter prevents goal text from terminating the script payload.
        import uuid
        delimiter='CAMINUS_REQUEST_'+uuid.uuid4().hex
        script='exec python3 -I "$HOME/bin/caminus-agent" request <<\''+delimiter+"'\n"+payload+'\n'+delimiter+'\n'
        # A live reader must inherit stdout: `stream` and `--jsonl` keep their lines flowing over SSH
        # instead of arriving as one buffered result when the process ends.
        live='--jsonl' in args or args[:1]==['stream']
        result=runner.run_shell_stream(script) if live else runner.run_shell(script,transport=transport)
    sys.stdout.write(result.stdout or '')
    sys.stderr.write(result.stderr or '')
    return result.returncode
