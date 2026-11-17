"""Bounded local capability discovery. Metadata presence is not a smoke test.

Overview never executes discovered programs, imports third-party modules, scans
user documents or queries devices. Explicit packages/containers topics may run
only the fixed read-only catalog commands below, with a timeout and output cap.
"""
import datetime
import grp
import importlib.util
import os
from pathlib import Path
import platform
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile

TOPICS = ('overview','hardware','commands','web','images','documents','network',
          'development','python','containers','packages')
COMMANDS = {
    'web': ('curl','wget','google-chrome','google-chrome-stable','chromium','chromium-browser',
            'firefox','playwright','ddgr','w3m','lynx'),
    'images': ('magick','convert','identify','ffmpeg','ffprobe','tesseract','zbarimg','qrencode','exiftool'),
    'documents': ('pdftotext','pdfinfo','qpdf','mutool','pandoc','libreoffice','unzip','7z','tar','file','strings','xxd','jq'),
    'network': ('ip','ss','dig','tcpdump','tshark','openssl','ssh','rsync','nc'),
    'development': ('python3','python','node','npm','gcc','g++','clang','make','cmake','cargo','rustc','go',
                    'java','javac','git','gdb','objdump','readelf','sqlite3'),
    'python': ('python3','python','pip','pip3','pipx','uv','conda'),
    'containers': ('docker','podman','bwrap','tmux','screen','systemctl'),
    'packages': ('apt','apt-cache','apt-get','dpkg-query'),
}
MODULES = ('requests','bs4','playwright','selenium','PIL','cv2','numpy','scipy','pandas',
           'matplotlib','Crypto','sympy','z3','scapy','pyzbar')
# The agent's own Python: `caminus ember setup` makes it and keeps this set in it (system pip refuses installs, PEP 668;
# every data task began with a venv step). (pip requirement, module it provides); lower bounds, never exact pins.
AGENT_VENV = '~/.venvs/caminus-cpu'
AGENT_PACKAGES = (('pillow>=10.0','PIL'),('requests>=2.31','requests'),('beautifulsoup4>=4.12','bs4'),
                  ('lxml>=5.0','lxml'),('scipy>=1.11','scipy'),('pandas>=2.1','pandas'),
                  ('matplotlib>=3.8','matplotlib'),('ddgs>=9.0','ddgs'),('weasyprint>=62.0','weasyprint'),
                  ('pypdf>=4.0','pypdf'))


def _read(path, limit=16384):
    try:
        with Path(path).open() as stream:return stream.read(limit)
    except OSError:return ''


def _probe(argv):
    """Local metadata only: no shell, bounded time, bounded in-memory output."""
    with tempfile.TemporaryFile() as output:
        try:
            env={**os.environ,'PAGER':'cat','LC_ALL':'C'}
            # Catalog queries stay local even when the shell selected a remote
            # Docker context or podman connection for unrelated work.
            for key in ('DOCKER_CONTEXT','DOCKER_HOST','CONTAINER_HOST','CONTAINER_CONNECTION'):
                env.pop(key,None)
            result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                                  timeout=3,env=env)
        except subprocess.TimeoutExpired:
            return dict(command=argv,status='timed_out',note='catalog availability remains unverified')
        except OSError as e:
            return dict(command=argv,status='unavailable',error=str(e))
        output.seek(0);raw=output.read(16385)
    return dict(command=argv,status='ok' if result.returncode==0 else 'query_failed',rc=result.returncode,
                output=raw[:16384].decode(errors='replace'),truncated=len(raw)>16384)


def _hardware(workspace):
    mem={}
    for line in _read('/proc/meminfo').splitlines():
        key,_,value=line.partition(':')
        if key in ('MemTotal','MemAvailable','SwapTotal','SwapFree'):mem[key]=value.strip()
    cpu=next((line.split(':',1)[1].strip() for line in _read('/proc/cpuinfo').splitlines()
              if line.startswith('model name') and ':' in line),platform.machine())
    os_release={}
    for line in _read('/etc/os-release').splitlines():
        key,_,value=line.partition('=')
        if key in ('PRETTY_NAME','ID','VERSION_ID'):os_release[key]=value.strip('"')
    gpus=[]
    for card in sorted(Path('/sys/class/drm').glob('card*')):
        if not re.fullmatch(r'card\d+',card.name):continue
        device=card/'device'
        vendor=_read(device/'vendor').strip()
        if not vendor:continue
        gpus.append(dict(card=card.name,pci=device.resolve().name,vendor=vendor,
                         device_id=_read(device/'device').strip(),
                         vram_total_bytes=_read(device/'mem_info_vram_total').strip(),
                         vram_used_bytes=_read(device/'mem_info_vram_used').strip(),
                         connected=[p.parent.name for p in card.parent.glob(card.name+'-*/status') if _read(p).strip()=='connected']))
    try:
        usage=shutil.disk_usage(workspace);disk=dict(total_bytes=usage.total,free_bytes=usage.free)
    except OSError:disk=dict(status='unverified')
    return dict(os=os_release,kernel=platform.release(),cpu=cpu,logical_cpus=os.cpu_count(),memory=mem,disk=disk,gpus=gpus)


def discover(workspace, topic='overview', query=''):
    if topic not in TOPICS:raise ValueError('unknown discovery topic')
    if not isinstance(query,str) or len(query)>128:raise ValueError('query must be at most128characters')
    query=query.strip()
    result=dict(topic=topic,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='current Unix account and interpreter; paths indicate presence, not verified operation')
    if topic in ('overview','hardware'):
        uid=os.getuid()
        try:user=pwd.getpwuid(uid).pw_name
        except KeyError:user=str(uid)
        groups=[]
        for gid in sorted(set([os.getgid(),*os.getgroups()])):
            try:groups.append(grp.getgrgid(gid).gr_name)
            except KeyError:groups.append(str(gid))
        result.update(account=dict(user=user,uid=uid,gid=os.getgid(),groups=groups,
                                   account_home=str(Path.home()),workspace=str(Path(workspace).resolve())),
                      hardware=_hardware(workspace))
    categories=COMMANDS if topic=='overview' else {topic:COMMANDS.get(topic,())}
    result['programs']={category:{name:found for name in names if (found:=shutil.which(name))}
                        for category,names in categories.items() if names}
    if topic=='commands':
        if not query:raise ValueError('commands topic needs a program-name query')
        matches={};limited=False
        for directory in dict.fromkeys(os.get_exec_path()):
            try:
                with os.scandir(directory) as entries:
                    for i,entry in enumerate(entries):
                        if i>=10000:limited=True;break
                        if query.casefold() in entry.name.casefold() and entry.name not in matches:
                            path=shutil.which(entry.name)
                            if path:matches[entry.name]=path
                        if len(matches)>=64:limited=True;break
            except OSError:continue
            if len(matches)>=64:break
        result.update(matches=matches,truncated=limited,scope='executable names on current PATH only')
    if topic in ('overview','python'):
        modules={}
        for name in MODULES:
            if query and query.casefold() not in name.casefold():continue
            try:present=importlib.util.find_spec(name) is not None
            except (ImportError,ValueError,AttributeError):present=False
            modules[name]='importable metadata found' if present else 'not found in this interpreter'
        result['python']=dict(executable=sys.executable,modules=modules,
                              note='Other virtual environments and container modules are not covered; no module was imported.')
    if topic=='containers':
        result['catalogs']=[]
        if executable:=shutil.which('docker'):
            sockets=(Path(os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))/'docker.sock',Path('/var/run/docker.sock'))
            for socket in sockets:
                try:is_socket=stat.S_ISSOCK(socket.stat().st_mode)
                except OSError:continue
                if is_socket:
                    result['catalogs'].append(_probe([executable,'--host','unix://'+str(socket),'image','ls','--format','{{.Repository}}:{{.Tag}}']))
                    break
        if executable:=shutil.which('podman'):
            result['catalogs'].append(_probe([executable,'--remote=false','image','ls','--format','{{.Repository}}:{{.Tag}}']))
        result['note']='Local cached images only. No container was started, pulled or built.'
    if topic=='packages':
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?',query):
            raise ValueError('packages topic needs an exact Debian package name')
        result['catalogs']=[]
        for name,args in (('dpkg-query',['-W','-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n','--',query]),
                          ('apt-cache',['policy','--',query])):
            if executable:=shutil.which(name):result['catalogs'].append(_probe([executable,*args]))
        result['note']='Installed package status and cached apt policy only; no index update, download or installation.'
    result['next_step']='Inspect a promising tool with --help and a small task-relevant check before building or installing an alternative.'
    return result
