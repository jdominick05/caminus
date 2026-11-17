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
    """The agent runtime as a zipapp. Its bytes depend on its files alone (every entry has the same fixed date), so its
    sha256, the installed file's name, says which code a daemon runs: status compares it with this caminus's."""
    root=Path(__file__).resolve().parents[1]
    out=io.BytesIO()
    def put(z,name,data):
        info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o644<<16
        z.writestr(info,data)
    with zipfile.ZipFile(out,'w') as z:
        for rel in ('__init__.py','ember/__init__.py'):
            put(z,'caminus/'+rel,(root/rel).read_bytes())
        for p in sorted((root/'ember/resident').glob('*.py')):
            put(z,'caminus/ember/resident/'+p.name,p.read_bytes())
        put(z,'__main__.py','from caminus.ember.resident.cli import main\nraise SystemExit(main())\n')
    return out.getvalue()


def runtime_name():
    """The file name `agent install` gives this caminus's runtime (its sha256): a daemon's heartbeat names its own."""
    import hashlib
    return hashlib.sha256(bundle()).hexdigest()+'.pyz'


def main(args,transport):
    args=list(args)
    if [a for a in args if a!='--local'][:1]==['create'] and '--local-default' in args:
        # the session switcher's new session: the local model chosen for this machine's cards (the agent account
        # cannot see the cards), on the local server
        from . import serve
        args.remove('--local-default')
        model,why=serve.default_brain()
        if model is None:
            print(f'error: no default local model: {why}',file=sys.stderr)
            return 1
        args+=['--model',model,'--endpoint',f'http://127.0.0.1:{serve.PORT0}/v1']
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
        if args[:1]==['init'] and not any(a=='--bridge' or a.startswith('--bridge=') for a in args):
            # the agent account cannot see the owner's settings: a setup with its own bridge (CAMINUS_ENV_BRIDGE) is
            # named here, or the agent would wait on the default one
            from . import settings
            args+=['--bridge',settings.bridge_dir()]
        payload=json.dumps(dict(args=args,stdin=sys.stdin.read() if '--stdin' in args else ''),ensure_ascii=False)
        # A fresh delimiter prevents goal text from terminating the script payload.
        import uuid
        delimiter='CAMINUS_REQUEST_'+uuid.uuid4().hex
        script='exec python3 -I "$HOME/bin/caminus-agent" request <<\''+delimiter+"'\n"+payload+'\n'+delimiter+'\n'
        # A live reader must inherit stdout: `stream` and `--jsonl` keep their lines flowing over SSH
        # instead of arriving as one buffered result when the process ends.
        live='--jsonl' in args or args[:1]==['stream']
        result=runner.run_shell_stream(script) if live else runner.run_shell(script,transport=transport)
    out=result.stdout or ''
    if args[:1]==['status'] and '--json' in args and result.returncode==0:
        # the runtime this caminus would install, beside the daemon's own (heartbeat.runtime): the panel says when
        # they differ, in place of the /update command
        try:
            value=json.loads(out)
            if isinstance(value,dict):
                value['runtime_latest']=runtime_name()
                out=json.dumps(value)+'\n'
        except ValueError:
            pass
    if args[:1] in (['advise'],['advisor']):
        out=advisor_setup(args,out)
    sys.stdout.write(out)
    sys.stderr.write(result.stderr or '')
    return result.returncode


def advisor_setup(args,out):
    """The advisor's relay, checked on this side: the agent account cannot read the owner's relay services, so the fix
    an unavailable advisor names (`agent advise`) is made exact here from the installed unit (its cap, port and every
    model it serves now: `relay install` replaces the list), and `agent advisor NAME --model M --endpoint RELAY` warns
    when that relay would refuse M. The resident prints one JSON object: its last line with --json, indented without."""
    from . import relay
    lines=out.rstrip('\n').split('\n')
    whole=False
    try:
        value=json.loads(lines[-1])
    except ValueError:
        try:
            value,whole=json.loads(out),True           # (without --json: one indented object)
        except ValueError:
            return out
    if not isinstance(value,dict):
        return out
    un=value.get('unavailable')
    if isinstance(un,dict):
        # every word of the resident's fix comes from the agent's account (its endpoint, config and output): only a
        # command built here from the owner's own relay unit is shown as one to run
        kind,model,endpoint=un.get('kind'),str(un.get('model') or ''),str(un.get('endpoint') or '')
        unit=relay.unit_at(endpoint)
        fix=None
        if kind=='relay_model':
            fix=relay.allow_command(endpoint,model)
        elif kind=='relay_down' and unit and model in relay.PRICES:
            # installed but not answering: reinstalling restarts it, with the advisor's model allowed as well
            fix=relay.install_command('anthropic',unit['args'],sorted(set(unit['models'])|{model}))
        elif kind=='no_key' and len(args)>1:
            fix='caminus ember agent advisor %s --endpoint http://127.0.0.1:%d/v1'%(
                shlex.quote(args[1]),(relay.installed('anthropic') or {}).get('port') or relay.PROVIDERS['anthropic']['port'])
        # the reason is the agent's text too: one line, no control characters, capped, and that form is what the panel
        # shows (left as sent, a modified runtime could put a "Run this on your side" block of its own above the real one)
        reason=''.join(c for c in ' '.join(str(un.get('reason') or '').split()) if c.isprintable())[:300]
        un['reason']=reason
        un['fix']=fix or ''
        value['error']='the advisor cannot be called: %s. %s'%(reason,('Run: '+fix) if fix else
                        'caminus ember relay status shows each relay and the models it allows')
    adv=value.get('advisor')
    if args[:1]==['advisor'] and isinstance(adv,dict) and adv.get('model') and adv.get('endpoint'):
        fix=relay.allow_command(adv['endpoint'],adv['model'])
        if fix:
            value['warning']=('the key relay at %s does not allow %s, so every advisor call will be refused until you '
                              'run: %s'%(adv['endpoint'],adv['model'],fix))
            print('warning: '+value['warning'],file=sys.stderr)
    if whole:
        return json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    lines[-1]=json.dumps(value,ensure_ascii=False)
    return '\n'.join(lines)+'\n'
