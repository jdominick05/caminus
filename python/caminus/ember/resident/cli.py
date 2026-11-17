"""The scriptable resident entry point. All execution modes call Runtime.run."""
import argparse
import io
import json
import os
import urllib.parse
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from .runtime import Runtime, EXIT, PAUSED, runtime_lock
from .store import Store, agent_home, atomic_json
from .terminal import alive
from .mail import Mail
from . import approval
from .discovery import TOPICS, discover
from .prompt import render as render_prompt
from . import planning
from . import btw
from . import export as export_mod



SPEED_FRESH_S = 120   # seconds a measured decode speed stays on the status line while a goal works


def _online_s(pid):
    """how long the daemon has been running, in seconds (from /proc: its start against the boot clock); None when it
    is not running or not readable. A runtime update restarts the daemon, and the clock with it."""
    try:
        pid = int(pid)
        with open(f'/proc/{pid}/stat') as f:
            start_ticks = int(f.read().rsplit(')', 1)[1].split()[19])
        with open('/proc/uptime') as f:
            up = float(f.read().split()[0])
        return max(0.0, up - start_ticks / os.sysconf('SC_CLK_TCK'))
    except (TypeError, ValueError, OSError, IndexError):
        return None

def _effort_shown(cfg):
    """The effort in force as the status line shows it (the level that runs; '' with no effort control)."""
    from . import effort as eff
    if not eff.line(cfg):
        return ''
    return eff.runs_as(cfg,cfg.get('effort') or eff.default_level(cfg))


def _current(cfg):
    keys=('model','provider','endpoint','api','context','max_tokens','effort','thinking','relay','prices')
    return {k:cfg.get(k) for k in keys if cfg.get(k) is not None}


# files whose first line is not a headline (images, documents, archives, media)
BINARY_SUFFIXES={'.png','.jpg','.jpeg','.gif','.webp','.bmp','.ico','.tif','.tiff','.pdf','.zip','.gz','.tar','.xz',
                 '.mp3','.wav','.ogg','.flac','.mp4','.webm','.mkv','.npy','.npz','.pkl','.bin','.so','.o','.sqlite3','.db'}


def studio_op(a,home):
    """What free time made: the journal (newest first), the studio, the "for you" shelf and PROJECTS.md; --read gives
    one file, only from those folders."""
    import time as _t
    journal,studio=home/'journal',home/'studio'
    shelf=studio/'for-you'
    roots=[journal.resolve(),studio.resolve()]
    if a.read:
        f=Path(a.read).expanduser()
        f=(home/f if not f.is_absolute() else f).resolve()
        if not any(f==r or r in f.parents for r in roots) or not f.is_file():
            raise ValueError('studio --read takes a file in the journal or the studio')
        size=f.stat().st_size
        if a.raw:
            # the bytes, for the panel's image view and for opening it on the owner's desktop (24 MB at most)
            import base64
            if size>24*1024*1024:raise ValueError(f'{f.name} is {size//(1024*1024)} MB: over the 24 MB a studio read takes')
            emit(dict(path=str(f),size=size,data=base64.b64encode(f.read_bytes()).decode()),a.json);return 0
        raw=f.read_bytes()[:256*1024]
        emit(dict(path=str(f),text=raw.decode('utf-8','replace'),truncated=size>256*1024,binary=b'\0' in raw[:8192]),a.json);return 0
    def entries(folder,skip=()):
        if not folder.is_dir():return []
        out=[]
        for f in folder.rglob('*'):
            if not f.is_file() or any(s in f.parents or f==s for s in skip) or f.name.startswith('.'):continue
            st=f.stat()
            head=''
            if st.st_size<512*1024 and f.suffix.lower() not in BINARY_SUFFIXES:
                try:
                    for line in f.read_text(errors='replace').splitlines():
                        line=line.strip().lstrip('#').strip()
                        if line:head=line[:160];break
                except OSError:pass
            out.append(dict(path=str(f),name=str(f.relative_to(folder)),size=st.st_size,mtime=st.st_mtime,head=head))
        return sorted(out,key=lambda e:-e['mtime'])[:200]
    projects=(studio/'PROJECTS.md').read_text(errors='replace')[:8000] if (studio/'PROJECTS.md').is_file() else ''
    emit(dict(journal=entries(journal),studio=entries(studio,(shelf,studio/'PROJECTS.md')),shelf=entries(shelf),
              projects=projects,now=_t.time()),a.json);return 0


def model_ops(a,home):
    """provider / models / model / effort: the brains this resident can use, found by asking the providers."""
    import re as _re
    from . import catalog
    cfg=json.loads((home/'config.json').read_text())
    if a.op=='provider':
        provs=dict(cfg.get('providers') or {})
        if a.action!='list':
            if not a.provider or not _re.fullmatch(r'[A-Za-z0-9_-]{1,32}',a.provider):
                raise ValueError('name the provider: letters, digits, - and _')
            if a.action=='add':
                if not a.kind or not a.endpoint:raise ValueError('provider add NAME --kind KIND --endpoint URL')
                provs[a.provider]=dict(kind=a.kind,endpoint=a.endpoint.rstrip('/'),**(dict(relay=True) if a.relay else {}))
            elif provs.pop(a.provider,None) is None:
                raise ValueError('no provider named '+a.provider)
            if provs:cfg['providers']=provs
            else:cfg.pop('providers',None)
            atomic_json(home/'config.json',cfg)
        emit(dict(providers=provs),a.json);return 0
    if a.op=='models':
        cat=catalog.load(home)
        if a.refresh or not cat.get('models'):
            cat=catalog.refresh(home,cfg,only=a.provider)
        rows=[m for m in cat.get('models',[]) if not a.provider or m['provider']==a.provider]
        emit(dict(models=rows,providers=cat.get('providers'),current=_current(cfg)),a.json);return 0
    if a.op=='model':
        if not a.id:
            emit(dict(current=_current(cfg)),a.json);return 0
        cat=catalog.load(home)
        a.level_from_id=None
        entry=catalog.find(cat,a.id,a.provider)
        if entry is None:
            cat=catalog.refresh(home,cfg)
            entry=catalog.find(cat,a.id,a.provider)
        if entry is None:
            # a local effort id (k3-coder-xhigh): its base model at that level
            from . import effort as eff
            for lvl,suffix in eff.SUFFIX.items():
                if a.id.endswith(suffix):
                    entry=catalog.find(cat,a.id[:-len(suffix)],a.provider)
                    if entry is not None and (entry.get('variants') or {}).get(lvl)==a.id:
                        a.level_from_id=lvl
                        break
                    entry=None
        if entry is None:
            raise ValueError(f'no model {a.id} in the catalog: `models --refresh` lists what the providers serve')
        spec=catalog.providers(cfg).get(entry['provider'])
        if not spec:raise ValueError('the provider of %s is no longer configured' % entry['id'])
        cfg.update(model=entry['id'],provider=entry['provider'],provider_kind=spec['kind'],endpoint=spec['endpoint'],
                   api='anthropic' if spec['kind']=='anthropic' else 'openai')
        for k in ('relay','prices','context','thinking','model_variants','efforts'):
            cfg.pop(k,None)
        if cfg.pop('max_tokens_auto',None):
            cfg.pop('max_tokens',None)        # the previous brain's catalog maximum, not one the owner set
        # a separate compaction model belongs to the brain it was set for: kept only if the new provider serves it
        if cfg.get('compaction_model') and catalog.find(cat,cfg['compaction_model'],entry['provider']) is None:
            cfg.pop('compaction_model')
        if spec.get('relay'):cfg['relay']=True
        if entry.get('prices'):cfg['prices']=entry['prices']
        if entry.get('context'):cfg['context']=int(entry['context'])
        if entry.get('variants'):
            cfg['model_variants']=entry['variants']   # a local model's effort levels: the server's id for each
        elif spec['kind'] not in ('local','anthropic'):
            cfg['efforts']=list(entry.get('efforts') or [])   # whether this model takes reasoning_effort at all
        if entry.get('max_output') and spec['kind']!='local':
            # a paid brain bills what it writes: its catalog maximum (up to 64K); a local one's comes from its card
            cfg['max_tokens'],cfg['max_tokens_auto']=min(int(entry['max_output']),65536),True
        if spec['kind']=='anthropic' and entry.get('thinking')=='adaptive':
            cfg['thinking']='adaptive'        # current Opus and Sonnet refuse thinking off; the store keeps none anyway
        # the effort level stays when the new brain has effort control (every level runs as its nearest); with none it
        # goes back to the default
        from . import effort as eff
        note='a running resident takes it at its next turn'
        if cfg.get('effort') and not eff.line(cfg):
            cfg.pop('effort',None)
            note+='; this model has no effort control, so the effort is back to its default'
        if a.level_from_id:
            cfg['effort']=a.level_from_id
        if cfg.get('api')=='openai':
            cfg.pop('api',None)               # openai is the default transport
        atomic_json(home/'config.json',cfg)
        emit(dict(current=_current(cfg),entry=entry,note=note),a.json);return 0
    if a.op=='effort':
        # one scale on every brain (none .. max); the model runs each level as its nearest (effort.py)
        from . import effort as eff
        line=eff.line(cfg)
        levels=list(eff.LEVELS) if line else []
        before=eff.template_line(cfg,cfg.get('effort') or eff.default_level(cfg))
        if a.level:
            level=eff.normalize(a.level)
            if level=='default':
                cfg.pop('effort',None)
            elif not line:
                raise ValueError('%s has no effort control' % cfg.get('model'))
            elif level not in eff.LEVELS:
                raise ValueError('effort is one of %s, or default' % ', '.join(eff.LEVELS))
            else:
                cfg['effort']=level
            atomic_json(home/'config.json',cfg)
        current=cfg.get('effort') or eff.default_level(cfg)
        runs=eff.runs_as(cfg,current) if current else ''
        summary='effort %s%s' % (current or "the model's default",
                                  ' (the lab\'s default)' if current and not cfg.get('effort') else '')
        if runs and runs!=current:
            summary+=' · runs as '+runs
        summary+=' · '+cfg.get('model','')+': '+eff.note(cfg)
        if not a.level and (sc:=eff.scale(cfg,current)):
            summary+='\n'+sc            # (bare /effort: every level and what it runs as on this model)
        if a.level and eff.template_line(cfg,current)!=before:
            # the template's effort line is the system message's first text: the server reads the conversation again
            try:used=int(json.loads((home/'context.json').read_text()).get('tokens') or 0)
            except (OSError,ValueError,AttributeError,TypeError):used=0
            # (once, the count first: the note said it too, and at 80 columns the count was cut)
            summary+=(' · %dK tokens re-read once' % (used//1000)) if used>8000 else ' · the conversation is re-read once'
        emit(dict(effort=cfg.get('effort'),current=current,runs_as=runs,levels=levels,model=cfg.get('model'),
                  summary=summary),a.json);return 0


def parser():
    ap=argparse.ArgumentParser(prog='caminus ember agent')
    sub=ap.add_subparsers(dest='op',required=True)
    for op in ('init','create','memory','run','once','send','status','events','stream','resume','cancel','compact','unit','start','stop','peers','message','mail','discover','prompt','heartbeat','advisor','fallback','mcp','provider','models','model','effort','update','pause','studio','queue','plan','unsend','interrupt','btw','theme','export','diff','rewind','github','advise','remove'):
        p=sub.add_parser(op)
        p.add_argument('name')
        p.add_argument('--home',help='agent-side home; use an isolated directory for tests')
        p.add_argument('--json',action='store_true')
        if op=='create':
            p.add_argument('--from',dest='template',required=True,help='copy model settings from a sibling resident; memory/history start fresh')
            p.add_argument('--model',help='a local model instead of the sibling\'s brain (the session switcher passes the setup\'s default)')
            p.add_argument('--endpoint',default='http://127.0.0.1:8092/v1',help='with --model: the local server')
        if op=='remove':
            p.add_argument('--dry-run',action='store_true',help='check the session exists and say what would happen; change nothing')
        if op=='memory':
            p.add_argument('--text',help='append a note to this resident\'s MEMORY.md')
            p.add_argument('--stdin',action='store_true')
        if op=='discover':
            p.add_argument('--topic',choices=TOPICS,default='overview')
            p.add_argument('--query',default='')
        if op=='init':
            p.add_argument('--model',required=True)
            p.add_argument('--endpoint',default='http://127.0.0.1:8092/v1')
            p.add_argument('--bridge',default=os.environ.get('CAMINUS_ENV_BRIDGE','/srv/caminus-env-bridge'))
            p.add_argument('--context',type=int)
            p.add_argument('--max-tokens',type=int,help='one reply\'s token limit (default: the model\'s own, effort.reply_limit)')
            p.add_argument('--compact-threshold',type=float,default=.85)
            p.add_argument('--compact-target-ratio',type=float,default=.2)
            p.add_argument('--compaction-model',help='optional model alias for summaries on the same endpoint')
            p.add_argument('--api-key-env',default='CAMINUS_AGENT_API_KEY')
            p.add_argument('--api',choices=('openai','anthropic'),default='openai',
                           help="wire format: OpenAI chat completions (the local server) or Anthropic's Messages API")
            p.add_argument('--effort',choices=('low','medium','high','xhigh','max'),
                           help='Messages API effort (default medium; xhigh and max need thinking adaptive)')
            p.add_argument('--thinking',choices=('adaptive','disabled'),
                           help='Messages API thinking (default adaptive: the model thinks before each reply as it sees '
                                'fit; nothing is stored or replayed). disabled turns it off')
            p.add_argument('--max-model-retries',type=int,
                           help='failed model calls in a row before the goal stops (default: unbounded for a '
                                'local endpoint, 8 for a remote one)')
            p.add_argument('--bus',help='shared agent-side SQLite mail bus; defaults under ~/.local/share/caminus/agents')
            p.add_argument('--idle-timeout',type=float,default=0,
                           help='seconds without progress before the watchdog pauses the goal (0: off)')
            p.add_argument('--token-budget',type=int,default=0,
                           help='prompt tokens a goal may reach before the watchdog pauses it (0: off)')
            p.add_argument('--no-loop-gates',action='store_true',
                           help='record loop_gates false: a fixture agent that predates the done check and skips it')
        if op=='fallback':
            p.add_argument('--model',help='the fallback model while the primary is unreachable, e.g. claude-haiku-5-5')
            p.add_argument('--endpoint',help="its API root, e.g. the owner's key relay")
            p.add_argument('--api',choices=('openai','anthropic'))
            p.add_argument('--context',type=int)
            p.add_argument('--effort',choices=('low','medium','high'))
            p.add_argument('--api-key-env')
            p.add_argument('--budget-usd',type=float,help='dollars per day the fallback may spend on work (default 1.00)')
            p.add_argument('--down-seconds',type=float,help='how long a failed primary is left alone (default 60)')
            p.add_argument('--off',action='store_true')
        if op=='studio':
            p.add_argument('--read',help='one file from the journal, the studio or the shelf, as text')
            p.add_argument('--raw',action='store_true',help='with --read: the bytes, base64 (images, documents; 24 MB at most)')
        if op=='pause':
            p.add_argument('goal_id',nargs='?',help='the goal to pause; defaults to the most recent one')
        if op=='queue':
            p.add_argument('action',nargs='?',default='list',choices=('list','move','remove'),
                           help='list the queue, move a queued goal, or remove (cancel) one')
            p.add_argument('goal',nargs='?',help='the goal to move or remove (id or prefix)')
            p.add_argument('--to',type=int,help='the new 1-based position for move')
        if op=='update':
            p.add_argument('--program',help='the runtime to install (default: the newest one install uploaded, ~/bin/caminus-agent)')
            p.add_argument('--now',action='store_true',help='at the next turn boundary instead of between goals')
            p.add_argument('--wait',type=float,default=0,help='seconds to wait for the daemon to come back on it')
        if op=='provider':
            p.add_argument('action',nargs='?',default='list',choices=('list','add','remove'))
            p.add_argument('provider',nargs='?',help='a short name: local, anthropic, deepinfra, mistral, ...')
            p.add_argument('--kind',choices=('local','anthropic','deepinfra','mistral','openai'))
            p.add_argument('--endpoint',help="its API root; the owner's key relay for a paid one (it adds the key)")
            p.add_argument('--relay',action='store_true',help='the endpoint is the key relay: billed, retries bounded')
        if op=='models':
            p.add_argument('--refresh',action='store_true',help='ask the providers again (otherwise the cached list)')
            p.add_argument('--provider',help='only this provider')
        if op=='model':
            p.add_argument('id',nargs='?',help='a model id from `models` (provider/id when two serve it); none shows the current one')
            p.add_argument('--provider')
        if op=='effort':
            p.add_argument('level',nargs='?',help="a level the current model offers (see `models`), or 'default'")
        if op=='mcp':
            p.add_argument('action',nargs='?',default='list',choices=('list','add','remove','off','on'))
            p.add_argument('server',nargs='?',help='the server name (letters, digits, - and _)')
            p.add_argument('--env',action='append',default=[],help='KEY=VALUE for its environment (repeat)')
            # (after --: the command that starts the server, e.g. -- npx -y @modelcontextprotocol/server-memory;
            #  main() splits it off before parsing)
        if op=='advisor':
            p.add_argument('--model',help='the advisor model, e.g. claude-opus-5-5 or claude-sonnet-5-5')
            p.add_argument('--endpoint',help="Messages API root; the owner's own key relay keeps the advisor's key and cap separate")
            p.add_argument('--api-key-env',help='key variable, when the endpoint is not a local relay that adds the key')
            p.add_argument('--budget-usd',type=float,help='dollars per day the advisor may spend (default 1.00)')
            p.add_argument('--calls-per-goal',type=int,help='advice calls one goal may make (default 5)')
            p.add_argument('--effort',choices=('low','medium','high','xhigh','max'))
            p.add_argument('--off',action='store_true',help='remove the advisor')
        if op=='heartbeat':
            p.add_argument('state',nargs='?',choices=('on','off','status','recap','now'),default='status',
                           help='free time: the resident lives in its account when no work is queued')
            p.add_argument('--rest-minutes',type=float)
            p.add_argument('--budget-usd',type=float,help='daily free-time allowance on a priced remote model')
            p.add_argument('--afk-minutes',type=float,help='how long a free-time question waits for an answer')
            p.add_argument('--since',type=float,help='recap: stretches since this Unix time (default: the newest goal from you)')
            p.add_argument('--messages-per-day',type=int)
            p.add_argument('--stretch-minutes',type=float)
            p.add_argument('--stretch-steps',type=int,help='model steps one free-time stretch may take; 0 (the default) = no step limit, its time ends it')
        if op=='message':
            p.add_argument('to')
            p.add_argument('--text')
            p.add_argument('--stdin',action='store_true')
            p.add_argument('--recipient-goal')
            p.add_argument('--reply-to')
        if op in ('once','send'):
            p.add_argument('text',nargs='?')
            p.add_argument('--goal')
            p.add_argument('--stdin',action='store_true',help='read goal text from stdin')
            p.add_argument('--goal-id',help='resume an existing goal, or send it a steering message')
        if op in ('once','send'):
            p.add_argument('--plan',action='store_true',
                           help='plan-first: the goal starts in plan mode; the agent explores read-only and '
                                'files a plan with propose_plan, and waits for your approval (agent plan)')

        if op=='plan':
            p.add_argument('goal_id')
            p.add_argument('action',choices=('approve','reject'),help="approve the filed plan, or ask for a revision")
            p.add_argument('--note',help='a short note: what was approved, or what to change')
        if op=='send':
            p.add_argument('--continue',dest='continue_goal',action='store_true',help='allow an explicit follow-up to a completed goal, preserving its conversation')
            p.add_argument('--steer',action='store_true',
                           help='redirect the goal: the message supersedes conflicting instructions and voids the '
                                'plan. Without it a message is a nudge, which keeps the plan.')
        if op in ('once','resume'):
            p.add_argument('--max-steps',type=int,help='optional test budget; production daemon has no step limit')
            p.add_argument('--max-seconds',type=float,help='optional test budget, retained in the goal checkpoint')
        if op=='once':
            p.add_argument('--jsonl',action='store_true',help='stream events followed by a result record')
        if op=='unsend':
            p.add_argument('goal_id',help='the goal whose newest undelivered message is pulled back')
        if op=='interrupt':
            p.add_argument('goal_id',help='the goal whose reply being generated is dropped')
        if op=='btw':
            p.add_argument('goal_id',help='the goal the side question is about')
            p.add_argument('text',nargs='?',help='the question (or --stdin)')
            p.add_argument('--stdin',action='store_true',help='read the question from stdin')
        if op=='advise':
            p.add_argument('goal_id',help='the goal the advisor reviews now')
            p.add_argument('text',nargs='?',help='optional: what you want it to look at')
        if op=='github':
            p.add_argument('action',choices=('link','status','unlink'))
            p.add_argument('--token-file',help='link: a file in the agent\'s own account holding one GitHub token')
            p.add_argument('--repo',action='append',default=[],help='link: OWNER/NAME to report the token\'s rights on (repeatable)')
        if op=='theme':
            p.add_argument('text',nargs='?',help='what the theme should look like (or --stdin)')
            p.add_argument('--stdin',action='store_true',help='read the description from stdin')
        if op=='export':
            p.add_argument('goal_id',help='the goal to export')
            p.add_argument('--out',help='write here (default: exports/<goal8>-<date>.md in the agent home)')
        if op=='diff':
            p.add_argument('goal_id',nargs='?',help='the goal (default: the most recent one)')
        if op=='rewind':
            p.add_argument('goal_id',nargs='?',help='the goal (default: the most recent one)')
            p.add_argument('--to',type=int,help='cut back to this step: files changed later go back, the transcript ends there')
            p.add_argument('--list',action='store_true',help='list the steps that can be rewound to, newest work last')
        if op in ('events','mail'):
            p.add_argument('--after',type=int,default=0)
        if op=='stream':
            p.add_argument('--interval',type=float,default=.05)
            p.add_argument('--seconds',type=float,help='stop after this long (default: until the reader goes away)')
            p.add_argument('--once',action='store_true',help='print the live snapshots once and exit')
        if op in ('resume','cancel'):
            p.add_argument('goal_id')
        if op=='compact':
            p.add_argument('goal_id',nargs='?',help='goal to compact; defaults to the most recent one')
    return ap


def summary(store,gid):
    g=store.goal(gid)
    return {**{k:g[k] for k in ('id','state','summary','error','steps')},
            'pending':[dict(id=a['id'],name=a['name'],request_id=a['bridge_id']) for a in store.actions(gid)]}


def resume_goal(store,gid,max_steps=None,max_seconds=None):
    # Cancellation/completion and this check-transition must share the same
    # SQLite write lock; a stale read must never resurrect a finished goal.
    with store.transaction():
        g=store.goal(gid)
        if g['state'] in ('completed','cancelled'):
            raise ValueError('submit a new goal instead of replaying a finished one')
        store.update(gid,state='working',step_limit=g['steps']+max_steps if max_steps else None,
                     deadline=time.time()+max_seconds if max_seconds else None,error='')


def unit_text(name,home,program=None):
    program=program or str(Path(sys.argv[0]).resolve())
    cmd=shlex.join([sys.executable,'-I',program,'run',name,'--home',str(home)]).replace('%','%%')
    return (f'[Unit]\nDescription=Caminus resident agent {name}\n\n[Service]\nType=simple\n'
            f'WorkingDirectory={str(home).replace("%","%%")}\nExecStart={cmd}\n'
            'Restart=always\nRestartSec=3\nKillMode=process\nTimeoutStopSec=15\nUMask=0007\n'
            f'EnvironmentFile=-%h/.config/caminus/agents/{name}.env\n\n[Install]\nWantedBy=default.target\n')


def emit(value,json_mode=False):
    if json_mode:
        print(json.dumps(value,ensure_ascii=False),flush=True)
    elif isinstance(value,dict):
        if 'state' in value:
            print(f"{value.get('id','')}: {value['state']} {value.get('summary') or value.get('error') or ''}",flush=True)
            for p in value.get('pending',[]):
                if p.get('request_id'):print('request '+p['request_id']+' ('+p['name']+')')
        else:
            print(json.dumps(value,ensure_ascii=False,indent=2))
    else:
        print(value)


def main(argv=None,stdin=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv[:1]==['request']:
        payload=json.load(stdin or sys.stdin)
        return main(payload['args'],stdin=io.StringIO(payload.get('stdin','')))
    cmdline=[]
    if argv[:1]==['mcp'] and '--' in argv:
        # an MCP server's own command line follows --; split it off before parsing, so flags like npx -y stay its own
        cut=argv.index('--');argv,cmdline=argv[:cut],argv[cut+1:]
    a=parser().parse_args(argv)
    if a.op=='mcp':a.cmdline=cmdline
    try:
        if getattr(a,'max_steps',None) is not None and a.max_steps<1 or getattr(a,'max_seconds',None) is not None and a.max_seconds<=0:
            raise ValueError('test budgets must be positive')
        home=agent_home(a.name,a.home)
        if a.op=='create':
            agent_home(a.template)  # validate the sibling name before joining it
            source=home.parent/a.template
            cfg=json.loads((source/'config.json').read_text())
            cfg['name']=a.name
            if a.model:
                # a local brain in place of the sibling's: as /model does, drop what belonged to a paid one (its relay,
                # prices, window, thinking, effort) and keep the providers so /model can switch back
                cfg.update(model=a.model,endpoint=a.endpoint.rstrip('/'))
                cfg.pop('max_tokens',None)   # (the new brain's own limit: effort.reply_limit)
                # (and the paid helpers: a fallback brain would take over at once while the local server is down, and a
                # judge or advisor on the relay would bill every finish; the owner adds them back on purpose)
                for k in ('relay','prices','context','thinking','effort','provider','provider_kind','api','compaction_model',
                          'fallback','advisor','judge_model'):
                    cfg.pop(k,None)
            from .model import Model
            Model(cfg)
            home.mkdir(mode=0o700)  # refuse existing sessions, including partial ones
            atomic_json(home/'config.json',cfg)
            (home/'MEMORY.md').write_text('# Persistent memory\n\nKeep durable facts and links to your work here.\n')
            (home/'IDENTITY.md').write_text(f'# {a.name}\n\nYou are the resident agent whose home is this directory.\n')
            store=Store(home);store.close()
            mail=Mail(cfg)
            try:mail.register(home)
            finally:mail.close()
            emit(dict(name=a.name,home=str(home),configured=True,model=cfg['model']),True)
            return 0
        if a.op=='discover':
            emit(discover(home,a.topic,a.query),True)
            return 0
        if a.op=='prompt':
            value=render_prompt(home)
            emit(dict(prompt=value),True) if a.json else print(value)
            return 0
        if a.op=='init':
            if not 0 < a.compact_target_ratio < a.compact_threshold < 1:
                raise ValueError('compaction ratios must satisfy 0 < target < threshold < 1')
            if (a.max_tokens is not None and a.max_tokens<1) or a.context is not None and a.context<1024:
                raise ValueError('invalid model context or output budget')
            home.mkdir(parents=True,exist_ok=True,mode=0o700)
            cfg=dict(name=a.name,model=a.model,endpoint=a.endpoint.rstrip('/'),bridge=a.bridge,
                     context=a.context,max_tokens=a.max_tokens,api_key_env=a.api_key_env,bus=a.bus,
                     compact_threshold=a.compact_threshold,compact_target_ratio=a.compact_target_ratio,
                     compaction_model=a.compaction_model,loop_gates=not a.no_loop_gates,
                     idle_timeout=a.idle_timeout,token_budget=a.token_budget)
            if a.api!='openai':cfg['api']=a.api
            if a.effort:cfg['effort']=a.effort
            if getattr(a,'thinking',None):cfg['thinking']=a.thinking
            if a.max_model_retries is not None:
                if a.max_model_retries<0:raise ValueError('--max-model-retries must be 0 or more')
                cfg['max_model_retries']=a.max_model_retries
            # A bare host gets the /v1 root; a provider URL with its own path (…/v1/openai) is kept as given.
            if not urllib.parse.urlsplit(cfg['endpoint']).path.strip('/'):cfg['endpoint']+='/v1'
            from .model import Model
            Model(cfg)                     # validate before persisting
            mail=Mail(cfg)
            try:mail.register(home)
            finally:mail.close()
            atomic_json(home/'config.json',cfg)
            for name,text in (('MEMORY.md','# Persistent memory\n\nKeep durable facts and links to your work here.\n'),
                              ('IDENTITY.md',f'# {a.name}\n\nYou are the resident agent whose home is this directory.\n')):
                p=home/name
                if not p.exists():p.write_text(text)
            store=Store(home);store.close()
            emit(dict(name=a.name,home=str(home),configured=True),a.json)
            return 0
        if not (home/'config.json').is_file():
            raise ValueError('agent not initialized; run agent init first')
        if a.op=='memory':
            p=home/'MEMORY.md'
            text=(stdin or sys.stdin).read() if a.stdin else a.text
            if text is not None:
                with p.open('a') as f:f.write('\n'+text.rstrip()+'\n')
            emit(dict(name=a.name,memory=p.read_text() if p.exists() else ''),True)
            return 0
        if a.op in ('peers','message','mail'):
            mail=Mail(json.loads((home/'config.json').read_text()))
            try:
                if a.op=='peers':value=mail.peers()
                elif a.op=='mail':value=mail.messages(a.after)
                else:
                    text=(stdin or sys.stdin).read() if a.stdin else a.text
                    value=mail.send(a.to,text,recipient_goal=a.recipient_goal,reply_to=a.reply_to)
                emit(value,True)
            finally:mail.close()
            return 0
        if a.op=='remove':
            # The session switcher's destroy: its service stops and is disabled, its name leaves the mail bus, and its
            # home (config, memory, history, files) moves to removed-sessions, kept and restorable by moving it back.
            # The panel cancels a running goal first with `cancel`, which also withdraws its open requests.
            if not (home/'config.json').exists():
                raise ValueError('no session named '+a.name)
            if a.dry_run:
                emit(dict(name=a.name,removed=False,would='stop and disable its service, leave the mail bus, move its home to removed-sessions'),True)
                return 0
            unit='caminus-agent-'+a.name+'.service'
            env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()))
            subprocess.run(['systemctl','--user','disable','--now',unit],env=env,check=False,capture_output=True)
            unit_file=Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')/'systemd/user'/unit
            if unit_file.exists():
                unit_file.unlink()
                subprocess.run(['systemctl','--user','daemon-reload'],env=env,check=False,capture_output=True)
            mail=Mail(json.loads((home/'config.json').read_text()))
            try:mail.unregister()
            finally:mail.close()
            import time as _time
            root=(home.parent if a.home else Path.home()/'.local/share/caminus')/'removed-sessions'  # (a test's --home stays in its folder)
            kept=root/(a.name+'-'+_time.strftime('%Y%m%d-%H%M%S'))
            kept.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            os.replace(home,kept)
            env_file=Path.home()/'.config/caminus/agents'/(a.name+'.env')
            if env_file.exists():
                os.replace(env_file,kept/'agent.env')
            emit(dict(name=a.name,removed=True,kept=str(kept)),True)
            return 0
        if a.op in ('unit','start','stop'):
            unit='caminus-agent-'+a.name+'.service'
            if a.op=='unit':
                print(unit_text(a.name,home),end='');return 0
            env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()))
            if a.op=='start':
                d=Path(os.environ.get('XDG_CONFIG_HOME') or Path.home()/'.config')/'systemd/user';d.mkdir(parents=True,exist_ok=True)
                (d/unit).write_text(unit_text(a.name,home))
                subprocess.run(['systemctl','--user','daemon-reload'],env=env,check=True)
                subprocess.run(['systemctl','--user','enable','--now',unit],env=env,check=True)
            else:
                subprocess.run(['systemctl','--user','stop',unit],env=env,check=True)
            emit(dict(name=a.name,unit=unit,started=a.op=='start'),a.json)
            return 0
        store=Store(home)
        if a.op=='fallback':
            from .brain import DEFAULTS as FB
            from .model import Model, price
            cfg=json.loads((home/'config.json').read_text())
            fb={} if a.off else dict(cfg.get('fallback') or {})
            if not a.off:
                for flag in ('model','endpoint','api','context','effort','api_key_env','budget_usd','down_seconds'):
                    v=getattr(a,flag)
                    if v is not None:fb[flag]=v.rstrip('/') if flag=='endpoint' else v
            if fb:
                if not (fb.get('model') and fb.get('endpoint')):raise ValueError('a fallback needs --model and --endpoint')
                m=Model({k:v for k,v in fb.items() if k not in FB})          # validates the endpoint and api
                if m.remote and price(fb['model'],{}) is None:
                    raise ValueError('no known price for %s, so its spend cannot be limited' % fb['model'])
                cfg['fallback']=fb
            else:
                cfg.pop('fallback',None)
            atomic_json(home/'config.json',cfg)
            state=json.loads((home/'fallback.json').read_text()) if (home/'fallback.json').exists() else {}
            import datetime
            emit(dict(fallback={**FB,**fb} if fb else None,
                      today=state.get(datetime.date.today().isoformat(),dict(usd=0.0,turns=0))),a.json)
            return 0
        if a.op in ('provider','models','model','effort'):
            return model_ops(a,home)
        if a.op=='studio':
            return studio_op(a,home)
        if a.op=='update':
            import time as _t
            program=Path(a.program).expanduser().resolve() if a.program else (Path.home()/'bin/caminus-agent').resolve()
            if not program.is_file():raise ValueError(f'no runtime at {program}: run `caminus ember agent install` first')
            atomic_json(home/'update.json',dict(program=str(program),now=bool(a.now),ts=_t.time()))
            out=dict(name=a.name,queued=True,runtime=program.name,
                     note='the daemon installs it between goals' if not a.now else 'the daemon installs it at its next turn boundary')
            end=_t.time()+max(0,a.wait)
            while a.wait and _t.time()<end:
                try:hb=json.loads((home/'heartbeat.json').read_text())
                except (OSError,ValueError):hb={}
                if hb.get('runtime')==program.name and not (home/'update.json').exists():
                    out.update(applied=True,pid=hb.get('pid'));break
                _t.sleep(1)
            else:
                if a.wait:out['applied']=False
            emit(out,a.json);return 0
        if a.op=='mcp':
            import re as _re
            cfg=json.loads((home/'config.json').read_text())
            servers=dict(cfg.get('mcp') or {})
            if a.action!='list':
                if not a.server or not _re.fullmatch(r'[A-Za-z0-9_-]{1,40}',a.server):
                    raise ValueError('name the server: letters, digits, - and _ (at most 40)')
                if a.action=='add':
                    cmdline=a.cmdline
                    if not cmdline:raise ValueError('mcp add SERVER -- COMMAND [ARGS...]: give the command after --')
                    env={}
                    for kv in a.env:
                        k,sep,v=kv.partition('=')
                        if not sep or not k:raise ValueError('--env takes KEY=VALUE')
                        env[k]=v
                    servers[a.server]=dict(command=cmdline[0],args=cmdline[1:],env=env,enabled=True)
                elif a.server not in servers:
                    raise ValueError('no MCP server named '+a.server)
                elif a.action=='remove':
                    servers.pop(a.server)
                else:
                    servers[a.server]['enabled']=a.action=='on'
                if servers:cfg['mcp']=servers
                else:cfg.pop('mcp',None)
                atomic_json(home/'config.json',cfg)
            # an env value can be a secret: show only its keys
            emit(dict(mcp={n:dict(command=v['command'],args=v.get('args',[]),env=sorted(v.get('env') or {}),
                                  enabled=v.get('enabled',True)) for n,v in servers.items()},
                      note='restart the agent (stop, start) for a change to reach a running daemon'),a.json)
            return 0
        if a.op=='advisor':
            from .advisor import Advisor, DEFAULTS
            from .model import price
            cfg=json.loads((home/'config.json').read_text())
            adv=dict(cfg.get('advisor') or {})
            if a.off:
                adv={}
            for flag in ('model','endpoint','api_key_env','budget_usd','calls_per_goal','effort'):
                v=getattr(a,flag)
                if v is not None:
                    if isinstance(v,(int,float)) and v<0:raise ValueError('--'+flag.replace('_','-')+' must be 0 or more')
                    adv[flag]=v.rstrip('/') if flag=='endpoint' else v
            if adv.get('model') and price(adv['model'],{}) is None:
                raise ValueError('no known price for %s, so its spend cannot be limited' % adv['model'])
            if adv and not (adv.get('model') and adv.get('endpoint')):
                raise ValueError('an advisor needs --model and --endpoint')
            if adv!=(cfg.get('advisor') or {}):
                if adv:cfg['advisor']=adv
                else:cfg.pop('advisor',None)
                atomic_json(home/'config.json',cfg)
            state=json.loads((home/'advisor.json').read_text()) if (home/'advisor.json').exists() else {}
            import datetime
            emit(dict(advisor={**DEFAULTS,**adv} if adv else None,
                      today=state.get(datetime.date.today().isoformat(),dict(usd=0.0,calls=0))),a.json)
            return 0
        if a.op=='heartbeat':
            from .freetime import settings
            cfg=json.loads((home/'config.json').read_text())
            ft=dict(cfg.get('free_time') or {}) if not isinstance(cfg.get('free_time'),bool) else dict(enabled=cfg['free_time'])
            if a.state=='recap':
                from .freetime import recap
                st=Store(home)
                try:emit(dict(recap=recap(st,a.since),now=__import__('time').time()),a.json)
                finally:st.close()
                return 0
            if a.state=='now':
                # a stretch at the daemon's next pass (after any running work; within the day's budget)
                from .freetime import NOW_MARKER
                (home/NOW_MARKER).write_text(str(__import__('time').time()))
                st=Store(home)
                try:busy=[g for g in (st.goal(x) for x in st.next_goal()) if g.get('kind','work') not in ('free','dream')]
                finally:st.close()
                note='starts after the running work' if busy else 'starts at the next pass'
                # say why it will not start rather than saying it will: the owner asked again and again while the
                # day's allowance was spent, and nothing told them
                import datetime as _dt
                st8=json.loads((home/'free_time.json').read_text()) if (home/'free_time.json').exists() else {}
                spent=float((st8.get('days',{}).get(_dt.date.today().isoformat()) or {}).get('usd',0.0))
                budget=float(settings(cfg)['budget_usd'])
                if spent and spent>=budget:
                    note=(f"today's free-time allowance is spent (${spent:.2f} of ${budget:.2f}): it starts tomorrow, or "
                          f"raise it: caminus ember agent heartbeat {a.name} --budget-usd N")
                emit(dict(free_time='asked',note=note,spent_today_usd=round(spent,4),budget_usd=budget),a.json)
                return 0
            if a.state in ('on','off'):ft['enabled']=a.state=='on'
            for flag,key,scale in (('rest_minutes','rest_seconds',60),('afk_minutes','afk_seconds',60),
                                   ('stretch_minutes','stretch_seconds',60),('budget_usd','budget_usd',1),
                                   ('messages_per_day','messages_per_day',1),('stretch_steps','stretch_steps',1)):
                v=getattr(a,flag)
                if v is not None:
                    if v<0:raise ValueError('--'+flag.replace('_','-')+' must be 0 or more')
                    ft[key]=v*scale
            if ft!=(cfg.get('free_time') or {}):
                cfg['free_time']=ft;atomic_json(home/'config.json',cfg)
            state=json.loads((home/'free_time.json').read_text()) if (home/'free_time.json').exists() else {}
            import datetime
            today=state.get('days',{}).get(datetime.date.today().isoformat(),{})
            emit(dict(free_time=settings(cfg),spent_today_usd=today.get('usd',0.0),messages_today=today.get('messages',0),
                      last_start=state.get('last_start'),last_end=state.get('last_end'),last_dream=state.get('last_dream')),a.json)
            return 0
        if a.op=='status':
            heartbeat=json.loads((home/'heartbeat.json').read_text()) if (home/'heartbeat.json').exists() else {}
            cfg=json.loads((home/'config.json').read_text())
            # Measured prompt footprint from the last turn's compaction probe; absent when the
            # runtime has never measured this agent, so a UI shows capacity only, no estimate.
            measured=json.loads((home/'context.json').read_text()) if (home/'context.json').exists() else {}
            # Measured decode speed of the last streamed reply; absent until a reply reports usage.
            speed=json.loads((home/'throughput.json').read_text()) if (home/'throughput.json').exists() else {}
            # a speed is shown while it is current: the last reply's figure stayed on the status line for hours of idle
            busy=any(g['state'] in ('working','waiting_model') for g in store.status().get('goals',[]))
            if not isinstance(speed,dict) or not busy or time.time()-float(speed.get('ts') or 0)>SPEED_FRESH_S:
                speed={}
            # the thresholds in force: a paid brain (remote, or through the key relay) compacts early
            from .runtime import PAID_COMPACT
            from .model import LOOPBACK
            paid=bool(cfg.get('relay')) or urllib.parse.urlsplit(cfg.get('endpoint') or '').hostname not in LOOPBACK
            trig,targ=((cfg.get('paid_compact_threshold',PAID_COMPACT[0]),cfg.get('paid_compact_target_ratio',PAID_COMPACT[1]))
                       if paid else (cfg.get('compact_threshold',.85),cfg.get('compact_target_ratio',.2)))
            import datetime as _dt
            try:spend=json.loads((home/'spend.json').read_text())
            except (OSError,ValueError):spend={}
            today=(spend.get('days') or {}).get(_dt.date.today().isoformat()) or {}
            spend_today=today.get('usd',0.0)
            cache_hit=(today.get('cache_read',0)/today['in_tok']) if today.get('in_tok') else None
            emit(dict(name=a.name,home=str(home),model=cfg['model'],context=cfg.get('context'),
                      context_used=measured.get('tokens') if isinstance(measured,dict) else None,
                      tok_per_s=speed.get('tps') if isinstance(speed,dict) else None,
                      compact_threshold=trig,compact_target_ratio=targ,
                      effort=_effort_shown(cfg),
                      spend_today=spend_today,spend_goals=spend.get('goals') or {},cache_hit=cache_hit,
                      daemon_alive=alive(heartbeat.get('pid')),heartbeat=heartbeat,
                      online_s=_online_s(heartbeat.get('pid')),**store.status()),True)
            store.close();return 0
        if a.op=='events':
            emit(store.events(a.after),True);store.close();return 0
        if a.op=='plan':
            gid=store.resolve(a.goal_id)
            with store.transaction():
                g=store.goal(gid)
                if g.get('plan') == 0:
                    raise ValueError(f'goal {gid[:8]} is not a plan goal')
                if g['state'] in ('completed','cancelled'):
                    raise ValueError(f'goal {gid[:8]} is {g["state"]}')
                if a.action=='approve':
                    store.update(gid,plan=2,state='working',error='')
                    text=planning.approval_text(a.note)
                    verdict='approved'
                else:
                    store.update(gid,state='working',error='')
                    text=planning.rejection_text(a.note)
                    verdict='revise'
                store.db.execute('INSERT INTO inbox(goal,text,kind) VALUES(?,?,?)',(gid,text,'nudge'))
                store.event('plan_' + ('approved' if a.action=='approve' else 'rejected'),
                            dict(note=(a.note or '').strip()),gid)
            emit(dict(goal=gid,plan=verdict,state=store.goal(gid)['state']),a.json)
            store.close();return 0
        if a.op=='rewind':
            gid=store.resolve(a.goal_id) if a.goal_id else store.latest_goal()
            if a.list:
                emit(dict(goal=gid,steps=store.rewind_steps(gid)),a.json)
                store.close();return 0
            if a.to is None:raise ValueError('rewind needs --to STEP or --list')
            beat=json.loads((home/'heartbeat.json').read_text()) if (home/'heartbeat.json').exists() else {}
            if alive(beat.get('pid')):
                rid=store.request(gid,'rewind',dict(to=a.to))
                emit(dict(goal=gid,to=a.to,queued=True,request=rid),a.json)
            else:
                with runtime_lock(home):
                    runtime=Runtime(home)
                    try:info=runtime._rewind(gid,a.to)
                    finally:runtime.mail.close();runtime.store.close()
                emit(dict(goal=gid,to=a.to,**info,note='terminal side effects are not undone'),a.json)
            store.close();return 0
        if a.op=='export':
            gid=store.resolve(a.goal_id)
            path,nbytes=export_mod.export(store,gid,a.out)
            emit(dict(goal=gid,path=str(path),bytes=nbytes),a.json)
            store.close();return 0
        if a.op=='diff':
            from . import diff as diff_mod
            gid=store.resolve(a.goal_id) if a.goal_id else store.latest_goal()
            emit(dict(goal=gid,**diff_mod.diff(store,gid)),a.json)
            store.close();return 0
        if a.op=='btw':
            gid=store.resolve(a.goal_id)
            store.close()
            question=(stdin or sys.stdin).read() if a.stdin else a.text
            if not (question or '').strip():raise ValueError('btw needs a question (TEXT or --stdin)')
            rt=Runtime(home)
            try:answer=btw.ask(rt,gid,question.strip())
            finally:rt.mail.close();rt.store.close()
            emit(dict(goal=gid,answer=answer),a.json);return 0
        if a.op=='advise':
            # /advisor: the owner asks for a review now. The advisor reads the goal (its brief: the goal, the plan, the
            # recent steps) and its advice goes to the agent as a message, with the reply being generated dropped
            # (the interrupt), so the agent reads it before its next step instead of finishing what it was doing.
            gid=store.resolve(a.goal_id)
            store.close()
            rt=Runtime(home)
            try:
                focus=(' Your human asks you to look at: '+a.text.strip()) if (a.text or '').strip() else ''
                out=rt.advisor.ask(gid,'Your human asked for a review of this goal now. Where does it stand, what is '
                                       'going wrong or being missed, and what should the agent do next? Be concrete.'+focus)
                note=('[advisor review, asked for by your human] %s read your goal and says:\n%s\n'
                      'Weigh this before your next step; say what you change, or why not.' % (out.get('model'),out['advice']))
                with rt.store.transaction():
                    rt.store.db.execute('INSERT INTO inbox(goal,text,kind) VALUES(?,?,?)',(gid,note,'nudge'))
                    rt.store.request(gid,'interrupt')
                    rt.store.event('advisor_review',dict(model=out.get('model'),usd=out.get('usd'),
                                                         focus=(a.text or '')[:300],advice=(out.get('advice') or '')[:4000]),gid)
            finally:rt.mail.close();rt.store.close()
            emit(dict(goal=gid,model=out.get('model'),usd=out.get('usd'),advice=out['advice']),a.json);return 0
        if a.op=='github':
            # the agent's GitHub account (resident/github.py): config.json keeps the token file's path and the
            # account, never the token
            store.close()
            from . import github
            cfg=json.loads((home/'config.json').read_text())
            if a.action=='link':
                if not a.token_file:raise ValueError('link needs --token-file PATH (a file in the agent\'s account)')
                cfg['github']=github.link(cfg,a.token_file,a.repo)
                atomic_json(home/'config.json',cfg)
            elif a.action=='unlink':
                cfg.pop('github',None);atomic_json(home/'config.json',cfg)
            gh=cfg.get('github')
            emit(dict(linked=bool(gh),**({k:gh[k] for k in ('login','email','repos','token_file')} if gh else {}),
                      note='the agent\'s terminals use this account from their next command' if a.action=='link' else ''),a.json)
            return 0
        if a.op=='theme':
            # the panel's theme screen: the brain writes a colour theme from a description (one call, no tools, no
            # goal context); the panel validates every line before it keeps any of it
            store.close()
            desc=((stdin or sys.stdin).read() if a.stdin else a.text or '').strip()
            if not desc:raise ValueError('theme needs a description (TEXT or --stdin)')
            from . import themegen
            rt=Runtime(home)
            try:text=themegen.write(rt,desc[:2000])
            finally:rt.mail.close();rt.store.close()
            emit(dict(text=text),a.json);return 0
        if a.op=='interrupt':
            # Only a model call in flight is interrupted: its live snapshot exists from the moment the call goes out
            # (before the first token too). Anything else (a tool running, the gap between turns) has nothing to stop,
            # and the caller says so instead of claiming an interrupt.
            gid=store.resolve(a.goal_id)
            generating=(home/'live'/(gid+'.json')).exists()
            if generating:store.request(gid,'interrupt')
            emit(dict(goal=gid,queued=generating,generating=generating),a.json);store.close();return 0
        if a.op=='unsend':
            gid,text=store.unsend(a.goal_id)
            emit(dict(goal=gid,text=text),a.json);store.close();return 0
        if a.op=='queue':
            act=a.action
            if act=='move':
                if not a.goal or a.to is None:raise ValueError('queue move GOAL --to N')
                out=dict(queue=store.queue_move(a.goal,a.to))
            elif act=='remove':
                if not a.goal:raise ValueError('queue remove GOAL')
                out=dict(queue=store.queue_remove(a.goal))
            else:
                out=dict(queue=store.queue())
            emit(out,a.json);store.close();return 0
        if a.op=='pause':
            # Like /compact: a live runtime applies it between steps; with no daemon the CLI pauses it here.
            gid=store.resolve(a.goal_id) if a.goal_id else store.latest_goal()
            beat=json.loads((home/'heartbeat.json').read_text()) if (home/'heartbeat.json').exists() else {}
            if alive(beat.get('pid')):
                rid=store.request(gid,'pause')
                emit(dict(goal=gid,queued=True,request=rid,note='paused at its next step; /resume continues it'),a.json)
            else:
                with store.transaction():
                    if store.goal(gid)['state'] in ('queued','working','waiting_model','waiting_user','waiting_root'):
                        store.update(gid,state='paused',error='paused by your human')
                        store.event('paused',dict(by='owner'),gid)
                emit(dict(goal=gid,state=store.goal(gid)['state']),a.json)
            store.close();return 0
        if a.op=='compact':
            # Compaction needs the model and the single writer, so a live runtime performs it on its next
            # pass; with no daemon the CLI takes the lock and does it here.
            gid=store.resolve(a.goal_id) if a.goal_id else store.latest_goal()
            beat=json.loads((home/'heartbeat.json').read_text()) if (home/'heartbeat.json').exists() else {}
            if alive(beat.get('pid')):
                rid=store.request(gid,'compact')
                store.event('compact_requested',dict(request=rid),gid)
                if a.json:emit(dict(goal=gid,queued=True,request=rid),True)
                else:print(f"compaction queued for {gid[:8]}; the runtime performs it on its next pass",flush=True)
                store.close();return 0
            with runtime_lock(home):
                runtime=Runtime(home)
                try:
                    rid=runtime.store.request(gid,'compact')
                    runtime.compact_request(gid)
                    done=runtime.store.db.execute('SELECT state,error FROM requests WHERE id=?',(rid,)).fetchone()
                finally:runtime.mail.close();runtime.store.close()
            if done['state']=='failed':
                if a.json:emit(dict(goal=gid,failed=True,error=done['error']),True)
                else:print(f"{gid[:8]}: not compacted — {done['error']}",flush=True)
                store.close();return 1
            row=store.db.execute("SELECT data FROM events WHERE goal=? AND kind='compacted' ORDER BY seq DESC LIMIT 1",(gid,)).fetchone()
            data=json.loads(row['data']) if row else {}
            if a.json:emit(dict(goal=gid,**data),True)
            else:print(f"{gid[:8]}: compacted {data.get('before')} → {data.get('after')} messages "
                       f"({data.get('before_tokens')} → {data.get('after_tokens')} tokens)",flush=True)
            store.close();return 0
        if a.op=='stream':
            # NDJSON, one line per change of a live snapshot: a UI holds one process instead of polling the
            # agent's home over SSH. The first line of a turn carries the whole text (reset), later lines
            # carry only the suffix; a vanished snapshot is reported once as done.
            live=home/'live'
            seen={}
            deadline=time.time()+a.seconds if a.seconds else None
            store.close()
            while True:
                for p in sorted(live.glob('*.json')) if live.is_dir() else []:
                    try:data=json.loads(p.read_text())
                    except (OSError,ValueError):continue
                    if not isinstance(data,dict):continue
                    gid=str(data.get('goal') or p.stem)
                    prev=seen.get(gid) or {}
                    content=str(data.get('content') or '');reasoning=str(data.get('reasoning') or '')
                    line=dict(goal=gid,turn=str(data.get('turn') or ''),tools=data.get('tools') or [])
                    if line['turn']!=prev.get('turn'):
                        line.update(reset=True,content=content,reasoning=reasoning)
                    else:
                        line.update(content=content[len(str(prev.get('content') or '')):],
                                    reasoning=reasoning[len(str(prev.get('reasoning') or '')):])
                    if line.get('reset') or line['content'] or line['reasoning'] or line['tools']!=prev.get('tools'):
                        emit(line,True)
                    seen[gid]=data
                for gid in list(seen):
                    if not (live/(gid+'.json')).exists():
                        emit(dict(goal=gid,turn=str(seen[gid].get('turn') or ''),done=True),True)
                        seen.pop(gid)
                if a.once or (deadline and time.time()>=deadline):return 0
                time.sleep(max(.02,a.interval))
        if a.op in ('send','once'):
            text=(stdin or sys.stdin).read() if a.stdin else a.goal if a.goal is not None else a.text
            if a.op=='send' and a.goal_id:
                if not text:raise ValueError('steering text is required')
                with store.transaction():
                    state=store.goal(a.goal_id)['state']
                    if state=='cancelled' or state=='completed' and not a.continue_goal:
                        raise ValueError('goal finished; submit a new goal instead of steering it')
                    if state=='completed':
                        store.update(a.goal_id,state='working',summary='',error='',deadline=None,step_limit=None)
                        store.event('goal_continued',{},a.goal_id)
                    store.db.execute('INSERT INTO inbox(goal,text,kind) VALUES(?,?,?)',
                                     (a.goal_id,text,'steer' if a.steer else 'nudge'))
                    store.event('user_message',dict(text=text,kind='steer' if a.steer else 'nudge'),a.goal_id)
                    # A paused goal keeps the text but will not read it until someone resumes: say so
                    # rather than reporting a delivery the agent will never act on.
                    resume_required=state in ('paused','error','waiting_user','waiting_root')
                emit(dict(id=a.goal_id,queued=True,resume_required=resume_required),a.json);store.close();return 0
            if a.goal_id:
                if a.plan:raise ValueError('--plan creates a new goal; --goal-id steers an existing one')
                gid=a.goal_id
                resume_goal(store,gid,a.max_steps,a.max_seconds)
            else:
                gid=store.enqueue(text or '',getattr(a,'max_steps',None),getattr(a,'max_seconds',None),
                                  plan=1 if a.plan else 0)
            if a.op=='send':
                out=dict(id=gid,state='queued')
                try:running=alive(json.loads((home/'heartbeat.json').read_text()).get('pid'))
                except (OSError,ValueError,AttributeError):running=False
                if not running:
                    # (a goal queued on a stopped agent waited an hour unnoticed after /stop)
                    out['note']='the agent is stopped: this goal starts when you /start it'
                emit(out,a.json);store.close();return 0
        elif a.op=='resume':
            gid=a.goal_id
            resume_goal(store,gid,a.max_steps,a.max_seconds)
            emit(summary(store,gid),a.json);store.close();return 0
        elif a.op=='cancel':
            gid=a.goal_id
            in_flight=[]
            with store.transaction():
                store.goal(gid)
                bridge=Path(json.loads((home/'config.json').read_text())['bridge']).expanduser()
                # every request this goal filed, whatever its action's state: with non-blocking approvals the action
                # is already done once 'pending' is stored, and GitHub hand-offs were never withdrawn at all
                for act in store.db.execute("SELECT id,bridge_id FROM actions WHERE goal=? AND name IN ('ask_root','ask_user','github_handoff')",(gid,)):
                    rid=act['bridge_id'] or act['id'][:16]
                    if (bridge/'req'/(rid+'.json')).exists():
                        result=approval.withdraw(bridge,rid,gid)
                        if result.get('decision')=='executing':in_flight.append(rid)
                store.update(gid,state='cancelled')
                for act in store.db.execute("SELECT id,args FROM actions WHERE goal=? AND name='terminal'",(gid,)):
                    try:args=json.loads(act['args'])
                    except ValueError:continue
                    if not isinstance(args,dict):continue
                    session=args.get('session') or act['id']
                    if not isinstance(session,str) or len(session)!=32 or any(c not in '0123456789abcdef' for c in session):continue
                    job=home/'jobs'/session
                    if job.is_dir():atomic_json(job/'control'/(uuid.uuid4().hex+'.json'),dict(signal='TERM'))
                store.event('cancelled',{},gid)
            emit({**summary(store,gid),'root_already_claimed':in_flight},a.json);store.close();return 0
        else:
            gid=None
        store.close()
        stop=[False]
        last=0
        def observe(store):
            nonlocal last
            if a.op!='once' or not a.jsonl:return
            while True:
                events=store.events(last)
                for e in events:
                    last=e['seq']
                    if e['goal']==gid:print(json.dumps(dict(type='event',**e)),flush=True)
                if len(events)<200:break
        if a.op=='run':
            for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *args:stop.__setitem__(0,True))
        try:
            with runtime_lock(home):
                runtime=Runtime(home)
                try:
                    if a.op=='run':runtime.run(stop=lambda:stop[0])
                    else:runtime.run(target=gid,observe=observe)
                finally:runtime.mail.close();runtime.store.close()
        except BlockingIOError:
            if a.op=='run':raise ValueError('this agent runtime is already running')
            # A daemon already owns the loop. Oneshot submits and observes that same loop.
            watch=Store(home)
            try:
                ticks=0
                while watch.goal(gid)['state'] not in PAUSED|{'waiting_model'}:
                    observe(watch)
                    ticks+=1
                    if ticks%20==0:
                        beat=json.loads((home/'heartbeat.json').read_text()) if (home/'heartbeat.json').exists() else {}
                        if beat.get('pid') and not alive(beat['pid']):
                            # The daemon that held the loop is gone; report it instead of waiting forever.
                            failed=dict(summary(watch,gid),error='the daemon holding this goal exited; run resume')
                            emit(failed if a.json or a.jsonl else dict(error=failed['error']),a.json or a.jsonl)
                            return EXIT['error']
                    time.sleep(.1)
                observe(watch)
            finally:watch.close()
        if a.op=='run':return 0
        store=Store(home);result=summary(store,gid);store.close()
        emit(dict(type='result',**result) if a.jsonl else result,a.json or a.jsonl)
        return EXIT.get(result['state'],1)
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
        if getattr(a,'json',False):emit(dict(error=str(e)),True)
        else:print('caminus ember agent: '+str(e),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
