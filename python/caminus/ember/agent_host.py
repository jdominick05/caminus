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
print(json.dumps(dict(installed=str(link),artifact=str(p),sha256=digest)))
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
