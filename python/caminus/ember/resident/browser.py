"""The browser tool: Google Chrome (or a Chromium that is not a snap) in the goal's sandbox box, driven over the
DevTools protocol on --remote-debugging-pipe (fd 3 in, fd 4 out, NUL-separated JSON; standard library only).

One browser per goal, kept across calls (tabs, cookies, the page), closed when the goal ends with its box. It is a
box job (sandbox.py, terminal.py's worker): bubblewrap is the sandbox, so Chrome's own is off; its profile and
downloads live in the goal's /box (the downloads are copied to outputs/GOAL/ when it ends, the screenshots are written
there); its network is the box proxy's web mode (any public address,
logged; a LAN, link-local or loopback address, or this machine, only when a call named that host:port). The worker
runs a Driver thread that owns the pipe; the runtime hands it each action through the job's control/ and ack/
files, keyed by the action's id, so a recovered action never clicks twice.

The main view is a text snapshot of the page's accessibility tree: interactive elements with refs (e1, e2, ...)
that click, type and select take, plus headings and text, capped. It refuses to type into a password field: logins
go through ask_user. eval runs any script in the page and is not guarded.
"""
import base64
from collections import deque
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import stat
import threading
import time
import uuid
from urllib.parse import urlsplit

from . import discovery, sandbox
from .store import atomic_json
from .terminal import alive, read_state

NAMES = tuple(n for n in discovery.COMMANDS['web'] if 'chrom' in n)    # (the names discover looks for)
ACTIONS = ('open', 'snapshot', 'click', 'type', 'select', 'press', 'scroll', 'back', 'eval', 'screenshot', 'console',
           'close')
ACTS = ('click', 'type', 'select', 'press', 'eval')     # act on the page: after define_done, outside plan mode
CHANGES = ACTS + ('open', 'back')    # change the page: skipped for the rest of a turn after a refused browser call
REPEATS = ('snapshot', 'scroll', 'press', 'back', 'console', 'screenshot')   # repeated on purpose (the loop guard)
VIEWS = ('open', 'snapshot', 'click', 'type', 'select', 'press', 'scroll', 'back')   # return the page's new snapshot
NEEDS = dict(open=('url',), click=('ref',), type=('ref', 'text'), select=('ref', 'value'), press=('key',), eval=('js',))
ALLOW_MAX = 16
CALL_TIMEOUT = 90         # the runtime's wait for one action (a page load is bounded at LOAD_TIMEOUT inside it)
LOAD_TIMEOUT = 30
IDLE_CLOSE = 1800         # the default seconds without a call before the browser closes itself (config.json
                          # sandbox.browser_idle_s; 0: never, it stays open until the goal ends). No time limit
                          # otherwise: a paused goal keeps its box, but by default not a Chrome running for nobody
SNAPSHOT_LINES, SNAPSHOT_CHARS = 250, 16000
KEEP_MB = 512             # downloads copied out of the box at goal end, at most
PROFILE = '/box/browser'
FLAGS = ('--headless=new', '--no-sandbox', '--remote-debugging-pipe', f'--user-data-dir={PROFILE}/profile',
         '--proxy-server=http://127.0.0.1:%d' % sandbox.PROXY_PORT,
         '--proxy-bypass-list=<-loopback>',        # (else Chrome skips the proxy for localhost, which the box lacks)
         '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
         '--disable-sync', '--disable-component-update', '--disable-default-apps', '--disable-extensions',
         '--disable-breakpad', '--no-pings', '--metrics-recording-only', '--mute-audio', '--password-store=basic',
         '--disable-client-side-phishing-detection', '--disable-domain-reliability',
         # (one --disable-features: Chrome reads only the last; network time and autofill each connected on their own)
         '--disable-features=Translate,OptimizationHints,MediaRouter,DialMediaRouteProvider,AutofillServerCommunication,'
         'NetworkTimeServiceQuerying',
         '--window-size=1280,900', '--hide-scrollbars', '--log-level=3')
# the fresh profile's settings: no preconnecting or prefetching, no search suggestions, no Safe Browsing lookups, no
# password manager, and a search engine that is not Google (with Google, Chrome asked www.google.com about its AI
# mode at every start)
PREFS = dict(net=dict(network_prediction_options=2), search=dict(suggest_enabled=False),
             safebrowsing=dict(enabled=False), credentials_enable_service=False,
             profile=dict(password_manager_enabled=False),
             default_search_provider_data=dict(template_url_data=dict(
                 keyword='duckduckgo.com', short_name='DuckDuckGo', url='https://duckduckgo.com/?q={searchTerms}')))
# (a rule, not a limit of the browser: a model offered to send the login with curl "around the browser's guard")
PASSWORD = ('that is a password field: the browser never types passwords, and you never send a password any other way '
            '(terminal, curl, eval, a script). A login goes through your human: ask_user how they want it done (they '
            'log in themselves, or give you a token or a test account)')
SHOT_NOTE = 'look at it with read if you can see images; otherwise it is for your human'
PASSWORD_FN = ('function(){ const a = ((this.getAttribute && this.getAttribute("autocomplete")) || "").toLowerCase();'
               ' return this.tagName === "INPUT" && (this.type === "password" || /\\b(current|new)-password\\b/.test(a)); }')
# the focused element, through shadow roots and same-origin frames; focus inside a frame it cannot read (another
# origin) is refused as well ("frame"): Tab can move the focus into a login frame
FOCUSED_PASSWORD = ('(() => { let e = document.activeElement; for (;;) { if (e && e.shadowRoot && '
                    'e.shadowRoot.activeElement) { e = e.shadowRoot.activeElement; continue; } if (e && (e.tagName === '
                    '"IFRAME" || e.tagName === "FRAME")) { try { e = e.contentDocument.activeElement; continue; } '
                    'catch (x) { return "frame"; } } break; } if (!e) return false; const a = ((e.getAttribute && '
                    'e.getAttribute("autocomplete")) || "").toLowerCase(); return e.tagName === "INPUT" && (e.type === '
                    '"password" || /\\b(current|new)-password\\b/.test(a)); })()')
SELECT_ALL_FN = ('function(){ if (this.select) { this.select(); return; } const r = document.createRange(); '
                 'r.selectNodeContents(this); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }')
SELECT_FN = ('function(v){ if (this.tagName !== "SELECT") return "that is not a select element: click it instead";'
             ' const o = [...this.options].find(o => o.value === v || o.label === v || o.text.trim() === v);'
             ' if (!o) return "no option " + JSON.stringify(v) + "; the options: " + [...this.options].map(o => o.label)'
             '.join(", "); this.value = o.value; this.dispatchEvent(new Event("input", {bubbles: true}));'
             ' this.dispatchEvent(new Event("change", {bubbles: true})); return ""; }')
KEYS = dict(Enter=(13, '\r'), Tab=(9, None), Escape=(27, None), Backspace=(8, None), Delete=(46, None),
            ArrowUp=(38, None), ArrowDown=(40, None), ArrowLeft=(37, None), ArrowRight=(39, None), Home=(36, None),
            End=(35, None), PageUp=(33, None), PageDown=(34, None), Space=(32, ' '))
MODIFIERS = dict(Alt=1, Control=2, Ctrl=2, Meta=4, Shift=8)
# the snapshot: what gets a ref, what is a leaf (its name says it all), which containers show
INTERACTIVE = {'link', 'button', 'textbox', 'searchbox', 'combobox', 'listbox', 'checkbox', 'radio', 'switch',
               'slider', 'spinbutton', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'tab', 'option', 'treeitem'}
LEAVES = INTERACTIVE | {'heading', 'img', 'image'}
SHOWN = {'form', 'dialog', 'alertdialog', 'alert', 'navigation', 'main', 'table', 'search'}
NAMED_SHOWN = {'region', 'group', 'article', 'list'}
SKIPPED = {'InlineTextBox', 'LineBreak', 'ListMarker'}
VALUED = {'textbox', 'searchbox', 'combobox', 'spinbutton', 'slider'}

_found = None


def find():
    """(launcher, read-only folder for the box or '') of the first Chrome or Chromium that can run in a box, or
    (None, why). A snap (Ubuntu's chromium, and its chromium-browser shim) cannot: bubblewrap boxes cannot run snaps."""
    global _found
    if _found is None:
        path = os.environ.get('PATH', '') + ':/usr/local/bin:/usr/bin:/bin'
        _found = (None, 'no Chrome or Chromium found outside a snap (install Google Chrome\'s .deb, or a distribution '
                        'Chromium that is not a snap)')
        for name in NAMES:
            exe = shutil.which(name, path=path)
            if not exe:
                continue
            real = os.path.realpath(exe)
            try:
                with open(real, 'rb') as f:
                    head = f.read(65536)
            except OSError:
                continue
            if real.startswith('/snap/') or b'/snap/' in head:
                continue
            top = real.split('/')[1]
            _found = (real, '' if top in ('usr', 'bin', 'sbin', 'lib', 'lib64', 'etc') else os.path.dirname(real))
            break
    return _found


def available():
    """'' when the browser tool can run here, else why not (the tool is then not offered)."""
    exe, why = find()
    if exe is None:
        return why
    return sandbox.usable()


def _check(args):
    act = args.get('action')
    if act not in ACTIONS:
        raise ValueError('action is one of: ' + ', '.join(ACTIONS))
    for k in NEEDS.get(act, ()):
        if args.get(k) is None or (k != 'text' and str(args[k]).strip() == ''):
            raise ValueError(f'browser {act} needs {k}')
    if act == 'open':
        if urlsplit(str(args['url'])).scheme not in ('http', 'https'):
            raise ValueError('browser open takes an http or https URL')
        allow = args.get('allow') or []
        if not isinstance(allow, list) or len(allow) > ALLOW_MAX:
            raise ValueError(f'allow is a list of at most {ALLOW_MAX} host:port names')
        for t in allow:
            sandbox.split_target(t)
    if act == 'scroll':
        if args.get('direction', 'down') not in ('up', 'down', 'left', 'right'):
            raise ValueError('direction is up, down, left or right')
        if not 1 <= int(args.get('amount', 600)) <= 20000:
            raise ValueError('amount is 1 to 20000 pixels')


def kept_dir(home, gid):
    """The goal's folder under outputs/: its screenshots, and its downloads once the goal ends (the box goes)."""
    return Path(home) / 'outputs' / gid


def keep(home, gid):
    """Before the goal's box is removed: copy what the browser downloaded into it to the goal's outputs folder.
    Regular files only, never a link (code in the box could have left one to a file of the agent's), at most
    KEEP_MB in all. Returns the files copied."""
    src = sandbox.box_dir(home, gid) / 'downloads'
    if src.is_symlink() or not src.is_dir():
        return []
    dst, left, out = kept_dir(home, gid) / 'downloads', KEEP_MB << 20, []
    for p in sorted(src.iterdir()):
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_size > left:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as f, open(dst / p.name, 'wb') as g:
                shutil.copyfileobj(f, g)
        except OSError:
            continue
        left -= st.st_size
        out.append(str(dst / p.name))
    return out


def idle_of(config):
    """config.json sandbox.browser_idle_s: seconds without a call before the browser closes itself, 0 for never."""
    v = ((config or {}).get('sandbox') or {}).get('browser_idle_s', IDLE_CLOSE)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError('config sandbox.browser_idle_s is a whole number of seconds, or 0 for never (the browser '
                         'then stays open until the goal ends)')
    return v


def _launch(rt, gid):
    """A new browser job for the goal: Chrome in its box, with the web proxy."""
    exe, ro = find()
    box_d = sandbox.box_dir(rt.home, gid)
    box_d.mkdir(parents=True, exist_ok=True)
    limits = sandbox.limits_of(rt.config)
    limits['time_s'] = None
    box = dict(goal=gid, box=str(box_d), folder='', hosts=[], web=True, ro=[ro] if ro else [], limits=limits)
    command = (f'mkdir -p {PROFILE}/profile/Default && printf %s {shlex.quote(json.dumps(PREFS))} > '
               f'{PROFILE}/profile/Default/Preferences && HOME={PROFILE} exec ' + shlex.join([exe, *FLAGS, 'about:blank']))
    job = rt.terminal.jobs / uuid.uuid4().hex
    job.mkdir()
    atomic_json(job / 'request.json', dict(command=command, cwd=str(box_d), workspace=str(rt.home), box=box,
                                           browser=dict(launcher=exe, idle_s=idle_of(rt.config))))
    rt.terminal.launch(job)
    rt.__dict__.setdefault('_browsers', {})[gid] = job.name
    return job


def _job_of(rt, gid):
    """(the goal's newest browser job or None, whether it runs)."""
    jobs = Path(rt.terminal.jobs)
    sid = rt.__dict__.get('_browsers', {}).get(gid)
    found = [jobs / sid] if sid and (jobs / sid).is_dir() else []
    if not found:
        for job in jobs.iterdir() if jobs.is_dir() else []:
            try:
                req = json.loads((job / 'request.json').read_text())
            except (OSError, ValueError):
                continue
            if req.get('browser') and (req.get('box') or {}).get('goal') == gid:
                found.append(job)
        found.sort(key=lambda j: j.stat().st_mtime)
    if not found:
        return None, False
    job = found[-1]
    state = read_state(job)
    return job, state.get('phase') in (None, 'starting', 'running') and alive(state.get('worker'))


def _ended(job):
    """Why a browser job is not running, in a few words (its error, else Chrome's last output line)."""
    state = read_state(job)
    if state.get('error'):
        return state['error']
    try:
        return (job / 'closed').read_text()
    except OSError:
        pass
    try:
        tail = [l for l in (job / 'output.log').read_bytes()[-4000:].decode(errors='replace').splitlines()
                if l.strip() and 'dbus' not in l]
    except OSError:
        tail = []
    return 'Chrome exited' + (f' ({tail[-1].strip()[:200]})' if tail else '')


def call(rt, gid, action_id, args):
    """One browser call (the runtime's): the goal's browser, started by open when there is none."""
    if (why := available()):
        raise ValueError(why)
    _check(args)
    act = args['action']
    job, running = _job_of(rt, gid)
    note = ''
    if not running:
        if act == 'close':
            return dict(closed=True, note='no browser was open')
        if act != 'open':
            raise ValueError('no browser is open for this goal: call browser with action open and a url' +
                             (f' (the last one ended: {_ended(job)}; its tabs are gone)' if job else ''))
        if job is not None:
            note = f'the last browser ended ({_ended(job)}): this is a new one, its tabs and cookies are gone'
        job = _launch(rt, gid)
    (job / 'control').mkdir(exist_ok=True)
    ctl = job / 'control' / (action_id + '.json')
    if not ctl.exists():
        atomic_json(ctl, dict(browser=dict(args)))
    ack = job / 'ack' / (action_id + '.json')
    end = time.monotonic() + CALL_TIMEOUT
    while True:
        try:
            got = json.loads(ack.read_text())
        except (OSError, ValueError):
            got = {}
        if got.get('state') == 'done':
            result = dict(got['result'])
            break
        state = read_state(job)
        if state.get('phase') in ('finished', 'interrupted') or (state.get('worker') and not alive(state['worker'])):
            raise ValueError(f'the browser stopped during this call: {_ended(job)}')
        if time.monotonic() > end:
            raise ValueError(f'the browser did not answer in {CALL_TIMEOUT} s')
        time.sleep(.05)
    result['session'] = job.name
    if note:
        result['note'] = note
    conns = sandbox.new_connections(job)
    if conns:
        # the model gets the counts and what was refused; the panel's card gets every connection (_panel: the
        # tool_result event only)
        refused = [f"{c['target']} ({c['verdict']})" for c in conns if not c['verdict'].startswith('allowed')]
        result['network'] = dict(allowed=len(conns) - len(refused), **({'refused': refused[:10]} if refused else {}))
        result['_panel'] = dict(connections=conns[-200:])
    return result


class CDPError(Exception):
    pass


class CDP:
    """The DevTools protocol over a pipe pair: requests by id, every event to on_event (a reader thread drains the
    pipe, so a page's events can never fill it and stall Chrome)."""

    def __init__(self, w, r, on_event):
        self.w, self.r, self.on_event = w, r, on_event
        self.n, self.slots, self.closed = 0, {}, None
        self.lock, self.wlock = threading.Lock(), threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b''
        while True:
            try:
                data = os.read(self.r, 1 << 20)
            except OSError:
                data = b''
            if not data:
                self.closed = 'Chrome exited'
                for slot in list(self.slots.values()):
                    slot[0].set()
                return
            buf += data
            while (i := buf.find(b'\0')) >= 0:
                raw, buf = buf[:i], buf[i + 1:]
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if 'id' in msg:
                    slot = self.slots.get(msg['id'])
                    if slot:
                        slot[1] = msg
                        slot[0].set()
                else:
                    try:
                        self.on_event(msg.get('method'), msg.get('params') or {}, msg.get('sessionId'))
                    except Exception:
                        pass

    def _write(self, i, method, params, session):
        m = dict(id=i, method=method, params=params or {})
        if session:
            m['sessionId'] = session
        data = json.dumps(m).encode() + b'\0'
        with self.wlock:
            while data:
                data = data[os.write(self.w, data):]

    def post(self, method, params=None, session=None):
        """Send without waiting (from the reader thread: a dialog's answer)."""
        with self.lock:
            self.n += 1
            i = self.n
        self._write(i, method, params, session)

    def send(self, method, params=None, session=None, timeout=30):
        if self.closed:
            raise CDPError(self.closed)
        slot = [threading.Event(), None]
        with self.lock:
            self.n += 1
            i = self.n
            self.slots[i] = slot
        try:
            self._write(i, method, params, session)
            if not slot[0].wait(timeout):
                raise TimeoutError(f'Chrome did not answer {method} in {timeout} s')
        except BrokenPipeError:
            raise CDPError('Chrome exited')
        finally:
            self.slots.pop(i, None)
        msg = slot[1]
        if msg is None:
            raise CDPError(self.closed or 'Chrome exited')
        if 'error' in msg:
            raise CDPError(f"{method}: {msg['error'].get('message', msg['error'])}")
        return msg.get('result') or {}


def _names_itself(host):
    """A URL whose host names a local service by itself: localhost, or a loopback, LAN, link-local or this machine's
    own address. A DNS name that resolves to one does not (a page could steer the agent to such a name without it
    knowing it reaches a local service): that needs allow."""
    host = host.strip('[]').lower()
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        a = ipaddress.ip_address(host)
    except ValueError:
        return False
    return sandbox.local(a)


def _prop(n, key):
    return (n.get(key) or {}).get('value')


class Driver:
    """The worker's half of the browser: the DevTools pipe and a thread that runs the actions in order."""

    def __init__(self, job, req):
        self.job, self.box = Path(job), Path(req['box']['box'])
        self.home, self.goal = Path(req['workspace']), req['box']['goal']
        self.idle = int(req['browser'].get('idle_s', IDLE_CLOSE))
        # Chrome reads fd 3 and writes fd 4; the child's ends are copied to 10 and up first, so placing them on 3
        # and 4 can't overwrite one with the other
        a_r, self.w = os.pipe()
        self.r, b_w = os.pipe()
        self._child = (fcntl.fcntl(a_r, fcntl.F_DUPFD_CLOEXEC, 10), fcntl.fcntl(b_w, fcntl.F_DUPFD_CLOEXEC, 10))
        os.close(a_r)
        os.close(b_w)
        self.q = queue.Queue()
        self.cdp = self.proxy = self.session = self.target = None
        self.loading, self.gone, self.opened, self.notes = False, False, [], []
        self.console, self.console_n, self.console_seen = deque(maxlen=200), 0, 0
        self.refs, self.ref_of, self.labels, self.next_ref = {}, {}, {}, 1
        self.downloads = {}

    def child(self):
        """In the box's forked child, before exec."""
        os.dup2(self._child[0], 3)
        os.dup2(self._child[1], 4)

    def start(self, proxy):
        for fd in self._child:
            os.close(fd)
        self.proxy = proxy
        self.cdp = CDP(self.w, self.r, self.on_event)
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, ack, op):
        self.q.put((ack, op))

    def stop(self):
        self.q.put(None)

    def _run(self):
        try:
            self._setup()
            broken = None
        except Exception as e:
            broken = f'Chrome did not start in its box: {e}'
        while True:
            try:
                item = self.q.get(timeout=self.idle or None)     # (0: never closes on its own)
            except queue.Empty:
                # a paused or failed goal keeps its box (it can resume), not a Chrome left running for nobody
                (self.job / 'closed').write_text(f'it closed itself after {self.idle} s without a call')
                self.a_close({})
                continue
            if item is None:
                return
            ack, op = item
            try:
                result = dict(error=broken) if broken else self.do(op)
            except (CDPError, TimeoutError, ValueError, OSError, KeyError, TypeError) as e:
                result = dict(error=str(e))
            except Exception as e:             # (an ack is always written: the runtime's call never hangs on one)
                result = dict(error=f'{type(e).__name__}: {e}')
            atomic_json(ack, dict(state='done', result=result))

    def _setup(self):
        self.cdp.send('Browser.getVersion', timeout=30)
        self.cdp.send('Target.setDiscoverTargets', dict(discover=True))
        self.cdp.send('Browser.setDownloadBehavior', dict(behavior='allow', downloadPath='/box/downloads',
                                                          eventsEnabled=True))
        self._attach(self._a_page())

    def _a_page(self):
        pages = [t['targetId'] for t in self.cdp.send('Target.getTargets')['targetInfos'] if t['type'] == 'page']
        return pages[0] if pages else self.cdp.send('Target.createTarget', dict(url='about:blank'))['targetId']

    def _attach(self, tid):
        self.session = self.cdp.send('Target.attachToTarget', dict(targetId=tid, flatten=True))['sessionId']
        self.target, self.gone, self.loading = tid, False, False
        self._forget()
        for m in ('Page.enable', 'Runtime.enable', 'Log.enable'):
            self.page(m)

    def _forget(self):
        self.refs, self.ref_of, self.labels = {}, {}, {}

    def page(self, method, params=None, timeout=30):
        return self.cdp.send(method, params, self.session, timeout)

    def on_event(self, method, p, sid):
        if method == 'Target.targetCreated':
            info = p.get('targetInfo') or {}
            if info.get('type') == 'page' and info.get('openerId') == self.target:
                self.opened.append(info['targetId'])
            return
        if method == 'Target.targetDestroyed' and p.get('targetId') == self.target:
            self.gone = True
        if method == 'Browser.downloadWillBegin':
            self.downloads[p.get('guid')] = name = str(p.get('suggestedFilename') or 'a file')[:120]
            self.notes.append(f'downloading {name} to {self.box}/downloads (kept in '
                              f'{kept_dir(self.home, self.goal)}/downloads when the goal ends)')
            return
        if method == 'Browser.downloadProgress' and p.get('state') in ('completed', 'canceled'):
            self.notes.append(f"download {p['state']}: {self.downloads.get(p.get('guid'), 'a file')}")
            return
        if sid is None or sid != self.session:
            return
        if method == 'Page.frameStartedLoading' and p.get('frameId') == self.target:
            self.loading = True
        elif method == 'Page.frameStoppedLoading' and p.get('frameId') == self.target:
            self.loading = False
        elif method == 'Page.frameNavigated' and not (p.get('frame') or {}).get('parentId'):
            self._forget()                        # a new document: the old refs point at nothing
        elif method == 'Page.javascriptDialogOpening':
            accept = p.get('type') in ('alert', 'beforeunload')
            self.cdp.post('Page.handleJavaScriptDialog', dict(accept=accept), sid)
            self.notes.append(f"a {p.get('type')} dialog said {str(p.get('message', ''))[:200]!r}: "
                              + ('accepted' if accept else 'dismissed'))
        elif method == 'Runtime.consoleAPICalled':
            text = ' '.join(str(a.get('value', a.get('description', a.get('type', '')))) for a in p.get('args', []))
            self._log(p.get('type'), text)
        elif method == 'Runtime.exceptionThrown':
            d = p.get('exceptionDetails') or {}
            self._log('exception', (d.get('exception') or {}).get('description') or d.get('text', ''))
        elif method == 'Log.entryAdded':
            e = p.get('entry') or {}
            self._log(e.get('level'), e.get('text', '') + (' ' + e['url'] if e.get('url') else ''))

    def _log(self, level, text):
        self.console_n += 1
        self.console.append((self.console_n, dict(level=level, text=str(text)[:500])))

    # -- actions --------------------------------------------------------------------------------------------------

    def do(self, op):
        act = op['action']
        if self.gone:
            self._attach(self._a_page())
            self.notes.append('the tab you were on closed: now on another one')
        before = len(self.opened)
        out = getattr(self, 'a_' + act)(op)
        if len(self.opened) > before:              # a link or a script opened a tab: follow it
            self._attach(self.opened[-1])
            self._wait_load(LOAD_TIMEOUT)
            self.notes.append('that opened a new tab: now on it (back does not return to the old one)')
        if act in VIEWS:
            out.update(self._view())
        if self.notes:
            out['notes'], self.notes = self.notes, []
        return out

    def _wait_load(self, limit):
        end = time.monotonic() + limit
        time.sleep(.05)
        while self.loading and time.monotonic() < end and not self.cdp.closed:
            time.sleep(.05)
        while time.monotonic() < end and not self.cdp.closed:
            try:
                state = self.page('Runtime.evaluate', dict(expression='document.readyState', returnByValue=True),
                                  timeout=5)['result'].get('value')
            except (CDPError, TimeoutError, KeyError):
                state = None
            if state == 'complete':
                return
            time.sleep(.1)
        self.notes.append(f'the page was still loading after {limit} s')

    def _settle(self):
        time.sleep(.3)                             # (a script reacts to the input; a navigation starts)
        if self.loading:
            self._wait_load(15)

    def _node(self, ref):
        bid = self.refs.get(str(ref).strip())
        if bid is None:
            raise ValueError(f'{ref} is not on this page: use a ref from the latest snapshot')
        return bid

    def _call_on(self, bid, fn, *args):
        try:
            obj = self.page('DOM.resolveNode', dict(backendNodeId=bid))['object']['objectId']
        except CDPError:
            raise ValueError('that element is gone from the page: take a snapshot')
        r = self.page('Runtime.callFunctionOn', dict(objectId=obj, functionDeclaration=fn, returnByValue=True,
                                                    arguments=[dict(value=a) for a in args], awaitPromise=True))
        return (r.get('result') or {}).get('value')

    def _refuse_focused_password(self):
        r = self.page('Runtime.evaluate', dict(expression=FOCUSED_PASSWORD, returnByValue=True))
        if (r.get('result') or {}).get('value') == 'frame':
            raise ValueError('the focus is inside a frame from another site, where the browser cannot see whether it '
                             'is a password field, so it does not type there. ' + PASSWORD.split(': ', 1)[1])
        if (r.get('result') or {}).get('value'):
            raise ValueError(PASSWORD)

    def a_open(self, op):
        for t in op.get('allow') or []:
            self.proxy.named.add(sandbox.split_target(t))
        u = urlsplit(op['url'])
        if u.hostname and _names_itself(u.hostname):    # the call names the local service it opens
            self.proxy.named.add((u.hostname.lower(), u.port or (443 if u.scheme == 'https' else 80)))
        started = len(self.downloads)
        r = self.page('Page.navigate', dict(url=op['url']))
        end = time.monotonic() + 3
        while r.get('errorText') == 'net::ERR_ABORTED' and len(self.downloads) == started and time.monotonic() < end:
            time.sleep(.05)                        # (a download's event can follow the aborted navigation)
        if r.get('errorText') == 'net::ERR_ABORTED' and len(self.downloads) > started:
            self.loading = False                   # (the URL is a file: a download, the page stays)
            time.sleep(.5)
        elif r.get('errorText'):
            self.loading = False
            raise ValueError(f"the page did not load: {r['errorText']}" + (
                ' (the proxy refused it or could not reach it: see network)' if 'TUNNEL' in r['errorText'] else ''))
        if r.get('loaderId'):
            self._wait_load(LOAD_TIMEOUT)
        return {}

    def a_snapshot(self, op):
        return {}

    def a_click(self, op):
        bid = self._node(op['ref'])
        label = self.labels.get(op['ref'], op['ref'])
        try:
            self.page('DOM.scrollIntoViewIfNeeded', dict(backendNodeId=bid))
            quads = self.page('DOM.getContentQuads', dict(backendNodeId=bid)).get('quads') or []
        except CDPError:
            quads = []
        if quads:
            q = quads[0]
            x, y = sum(q[0::2]) / 4, sum(q[1::2]) / 4
            for kind in ('mouseMoved', 'mousePressed', 'mouseReleased'):
                self.page('Input.dispatchMouseEvent', dict(type=kind, x=x, y=y, clickCount=1,
                                                           button='none' if kind == 'mouseMoved' else 'left'))
        else:                                       # (no box on screen: the element's own click)
            self._call_on(bid, 'function(){ this.click(); }')
        self._settle()
        return dict(did=f'clicked {label}')

    def a_type(self, op):
        bid = self._node(op['ref'])
        if self._call_on(bid, PASSWORD_FN):
            raise ValueError(PASSWORD)
        self.page('DOM.focus', dict(backendNodeId=bid))
        self._refuse_focused_password()
        self._call_on(bid, SELECT_ALL_FN)
        self.page('Input.insertText', dict(text=str(op['text'])))
        self._settle()
        return dict(did=f"typed {str(op['text'])[:80]!r} into {self.labels.get(op['ref'], op['ref'])}")

    def a_select(self, op):
        bid = self._node(op['ref'])
        if (err := self._call_on(bid, SELECT_FN, str(op['value']))):
            raise ValueError(err)
        self._settle()
        return dict(did=f"selected {str(op['value'])[:80]!r} in {self.labels.get(op['ref'], op['ref'])}")

    def a_press(self, op):
        *mods, key = str(op['key']).split('+') if op['key'] != '+' else ['+']
        mask = 0
        for m in mods:
            if m not in MODIFIERS:
                raise ValueError(f'{m} is not a modifier (Alt, Control, Meta, Shift)')
            mask |= MODIFIERS[m]
        if key in KEYS:
            code, text = KEYS[key]
        elif len(key) == 1:
            code = ord(key.upper()) if key.isascii() and key.isalnum() else 0
            text = key.upper() if mask == 8 else key
        else:
            raise ValueError('key is one character or one of ' + ', '.join(KEYS) + ', with Control+, Alt+, Shift+ '
                             'or Meta+ in front if needed')
        if mask & 7:
            text = None                            # (Control+a is a command, not text)
        if (text and text != '\r') or (mask and key.lower() == 'v'):
            self._refuse_focused_password()          # (Control+V would paste into it)
        base = dict(key=' ' if key == 'Space' else key, windowsVirtualKeyCode=code, modifiers=mask)
        self.page('Input.dispatchKeyEvent', dict(base, type='keyDown' if text else 'rawKeyDown',
                                                 **({'text': text} if text else {})))
        self.page('Input.dispatchKeyEvent', dict(base, type='keyUp'))
        self._settle()
        return dict(did=f"pressed {op['key']}")

    def a_scroll(self, op):
        d, n = op.get('direction', 'down'), int(op.get('amount', 600))
        dx, dy = dict(up=(0, -n), down=(0, n), left=(-n, 0), right=(n, 0))[d]
        self.page('Input.dispatchMouseEvent', dict(type='mouseWheel', x=640, y=450, deltaX=dx, deltaY=dy))
        self._settle()
        r = self.page('Runtime.evaluate', dict(returnByValue=True, expression=(
            '[Math.round(scrollY), document.documentElement.scrollHeight, innerHeight]')))
        y, height, view = (r.get('result') or {}).get('value') or (0, 0, 0)
        return dict(did=f'scrolled {d} {n} px', position=f'{y} of {height} px (the window shows {view})')

    def a_back(self, op):
        h = self.page('Page.getNavigationHistory')
        i = h.get('currentIndex', 0)
        if i <= 0:
            raise ValueError('there is no earlier page in this tab')
        self.page('Page.navigateToHistoryEntry', dict(entryId=h['entries'][i - 1]['id']))
        self._wait_load(LOAD_TIMEOUT)
        return dict(did='went back')

    def a_eval(self, op):
        r = self.page('Runtime.evaluate', dict(expression=str(op['js']), returnByValue=True, awaitPromise=True,
                                               userGesture=True, timeout=10000), timeout=20)
        if r.get('exceptionDetails'):
            d = r['exceptionDetails']
            raise ValueError('the script threw: ' + ((d.get('exception') or {}).get('description') or d.get('text', '')
                                                     )[:1000])
        res = r.get('result') or {}
        value = res['value'] if 'value' in res else res.get('description', res.get('type'))
        text = json.dumps(value, ensure_ascii=False)
        out = dict(value=value if len(text) <= 20000 else text[:20000])
        if len(text) > 20000:
            out['more'] = f'{len(text) - 20000:,} more characters: return less (slice it in the script)'
        return out

    def a_screenshot(self, op):
        data = base64.b64decode(self.page('Page.captureScreenshot', dict(format='png'))['data'])
        # into the goal's outputs folder, outside the box: it outlives the goal, and code in the box cannot reach it
        folder = kept_dir(self.home, self.goal) / 'screenshots'
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            path = folder / f'shot-{n}.png'
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                break
            except FileExistsError:
                n += 1
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        info = self.cdp.send('Target.getTargetInfo', dict(targetId=self.target))['targetInfo']
        return dict(screenshot=str(path), title=info.get('title', ''), url=info.get('url', ''), note=SHOT_NOTE)

    def a_console(self, op):
        new = [m for n, m in self.console if n > self.console_seen]
        self.console_seen = self.console_n
        out = dict(messages=new[-50:])
        if len(new) > 50:
            out['more'] = f'{len(new) - 50} earlier messages not shown'
        return out

    def a_close(self, op):
        try:
            self.cdp.send('Browser.close', timeout=10)
        except (CDPError, TimeoutError):
            pass
        return dict(closed=True)

    # -- the snapshot ---------------------------------------------------------------------------------------------

    def _view(self):
        info = self.cdp.send('Target.getTargetInfo', dict(targetId=self.target))['targetInfo']
        return dict(title=info.get('title', ''), url=info.get('url', ''), snapshot=self._snapshot(info.get('url', '')))

    def _passwords(self):
        """The backend ids of the page's password fields (marked in the snapshot)."""
        try:
            root = self.page('DOM.getDocument', dict(depth=0))['root']['nodeId']
            ids = self.page('DOM.querySelectorAll', dict(nodeId=root, selector=(
                'input[type=password],input[autocomplete~=current-password],input[autocomplete~=new-password]')))
            return {self.page('DOM.describeNode', dict(nodeId=i))['node']['backendNodeId'] for i in ids['nodeIds']}
        except (CDPError, KeyError):
            return set()

    def _snapshot(self, url=''):
        nodes = self.page('Accessibility.getFullAXTree')['nodes']
        if not nodes:
            return '(an empty page)'
        u = urlsplit(url)
        origin = f'{u.scheme}://{u.netloc}' if u.scheme in ('http', 'https') else ''   # (links there read as paths)
        by_id = {n['nodeId']: n for n in nodes}
        secret = self._passwords()
        lines, hidden = [], 0
        size = 0

        def ref_for(bid):
            if bid not in self.ref_of:
                r = f'e{self.next_ref}'
                self.next_ref += 1
                self.ref_of[bid], self.refs[r] = r, bid
            return self.ref_of[bid]

        def add(depth, text):
            nonlocal hidden, size
            if len(lines) >= SNAPSHOT_LINES or size > SNAPSHOT_CHARS:
                hidden += 1
                return False
            lines.append('  ' * depth + '- ' + text)
            size += len(lines[-1]) + 1
            return True

        def options(n):
            out = []
            stack = list(n.get('childIds', []))
            while stack:
                c = by_id.get(stack.pop(0))
                if c is None:
                    continue
                if _prop(c, 'role') == 'option':
                    out.append(str(_prop(c, 'name') or '').strip())
                else:
                    stack[:0] = c.get('childIds', [])
            return out

        def walk(n, depth, parent_name):
            nonlocal size
            role, name = _prop(n, 'role') or '', str(_prop(n, 'name') or '').strip()
            kids = [by_id[c] for c in n.get('childIds', []) if c in by_id]
            if n.get('ignored') or role in ('generic', 'none', 'paragraph', 'RootWebArea', 'listitem', 'LabelText',
                                            'Section', 'div', 'Pre', 'strong', 'emphasis', 'superscript', 'subscript'):
                for c in kids:
                    walk(c, depth, parent_name)
                return
            if role in SKIPPED:
                return
            if role == 'StaticText':
                if name and name != parent_name:
                    add(depth, 'text ' + json.dumps(name[:200], ensure_ascii=False) + (' …' if len(name) > 200 else ''))
                return
            props = {p['name']: (p.get('value') or {}).get('value') for p in n.get('properties', [])}
            if role in LEAVES:
                if role in ('img', 'image') and not name:
                    return
                text = role + (' ' + json.dumps(name[:100], ensure_ascii=False) if name else '')
                bid = n.get('backendDOMNodeId')
                if role in INTERACTIVE and bid is not None and len(lines) < SNAPSHOT_LINES and size <= SNAPSHOT_CHARS:
                    ref = ref_for(bid)
                    self.labels[ref] = f'{role} {json.dumps(name[:60], ensure_ascii=False)}' if name else role
                    text += f' [{ref}]'
                value = _prop(n, 'value')
                if role in VALUED and value not in (None, ''):
                    text += ' = ' + json.dumps(str(value)[:100], ensure_ascii=False)
                if role == 'heading' and props.get('level'):
                    text += f" (level {props['level']})"
                flags = [f for f, on in (('checked', props.get('checked') == 'true'), ('mixed', props.get('checked') == 'mixed'),
                                         ('disabled', props.get('disabled')), ('focused', props.get('focused')),
                                         ('required', props.get('required')), ('selected', props.get('selected')),
                                         ('expanded', props.get('expanded') is True),
                                         ('collapsed', props.get('expanded') is False),
                                         ('password', bid in secret)) if on]
                if role in ('checkbox', 'radio', 'switch') and props.get('checked') == 'false':
                    flags.insert(0, 'unchecked')
                if flags:
                    text += ' (' + ', '.join(flags) + ')'
                if role == 'link' and props.get('url'):
                    href = str(props['url'])
                    text += ' -> ' + (href[len(origin):] if origin and href.startswith(origin + '/') else href)[:120]
                if name and lines and lines[-1] == '  ' * depth + '- text ' + json.dumps(name[:200], ensure_ascii=False):
                    size -= len(lines.pop()) + 1           # (a field's label, already its name)
                if role in ('combobox', 'listbox') and (opts := options(n)):
                    text += ' options: ' + ' | '.join(o[:40] for o in opts[:12]) + (
                        f' (+{len(opts) - 12} more)' if len(opts) > 12 else '')
                    add(depth, text)
                    return
                add(depth, text)
                if role in ('listbox', 'tab', 'treeitem'):       # (their children are options and panels)
                    for c in kids:
                        walk(c, depth + 1, name)
                return
            if role in SHOWN or (role in NAMED_SHOWN and name):
                add(depth, role + (' ' + json.dumps(name[:100], ensure_ascii=False) if name else ''))
                for c in kids:
                    walk(c, depth + 1, name)
                return
            for c in kids:
                walk(c, depth, name or parent_name)

        walk(nodes[0], 0, '')
        if hidden:
            lines.append(f'... {hidden} more lines not shown: scroll and take a snapshot, or eval to search the page')
        return '\n'.join(lines) or '(nothing to show: the page has no text or controls)'
