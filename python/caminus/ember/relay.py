"""caminus ember relay: a local, spend-capped relay that adds the owner's API key for the agent.

The owner runs one relay per provider, as their own user. The agent sends requests to http://127.0.0.1:PORT with no
key; the relay adds the key from a file only the owner can read and forwards the request upstream. The key never
leaves this process and is never logged.

Hiding the key does not protect the money: anything on this machine that can reach the port can spend through it. So
the relay enforces its own limits, whatever the client asks for:
  - a daily dollar cap (a ledger file, priced from each reply's usage); once spent, requests get HTTP 402
  - a path allowlist, and for Anthropic a model allowlist (--model; default claude-haiku-5-5)
  - max_tokens clamped to --max-tokens
  - it listens on 127.0.0.1 only and exits after --ttl seconds

Providers:
  anthropic  api.anthropic.com: POST /v1/messages, /v1/messages/count_tokens; GET /v1/models (the catalog). Priced from
             the transport's table (caminus.ember.resident.model.PRICES); a model without a price cannot be allowed.
  deepinfra  api.deepinfra.com: POST /v1/openai/chat/completions; GET /v1/openai/models (the catalog). Any chat model
             DeepInfra publishes a price for is allowed (the cap is the guard): prices are read from its model list at
             start and every hour. A reply that reports no usage is charged a high estimate (its whole output budget).
  mistral    api.mistral.ai (Mistral AI Studio): POST /v1/chat/completions; GET /v1/models (the catalog). Mistral's
             model list carries no prices, so the owner gives them: a JSON file (--prices, default
             ~/.config/caminus/mistral-prices.json) of {"model-id": [input, output]} in USD per million tokens, re-read
             when it changes. A model without a price there is refused (it cannot be metered).

Run a separate relay (own port, key file, cap, ledger and log) for each provider and for an advisor.

As a service that outlasts any session (a systemd --user unit for the owner; no sudo, survives logouts with lingering,
restarts on failure, no expiry: the daily cap guards spend):
  caminus ember relay install --provider deepinfra --cap-usd 3
  caminus ember relay status            (every installed relay: active or not, today's spend)
  caminus ember relay uninstall --provider deepinfra
"""
import argparse
import subprocess
import datetime
import http.client
import http.server
import json
import os
from pathlib import Path
import select
import socket
import ssl
import sys
import threading
import time

from .resident.model import FALLBACK_BETA, FALLBACK_MODELS, PRICES

PROVIDERS = {
    'anthropic': dict(host='api.anthropic.com', post=('/v1/messages', '/v1/messages/count_tokens'),
                      get=('/v1/models',), port=8765, key_file='~/.config/anthropic/api_key'),
    'deepinfra': dict(host='api.deepinfra.com', post=('/v1/openai/chat/completions',),
                      get=('/v1/openai/models',), port=8766, key_file='~/.config/caminus/deepinfra-key'),
    'mistral': dict(host='api.mistral.ai', post=('/v1/chat/completions',), get=('/v1/models',), port=8767,
                    key_file='~/.config/caminus/mistral-key', prices='~/.config/caminus/mistral-prices.json'),
}
PRICE_REFRESH = 3600


def anthropic_cost(model, usage, served=None):
    row = PRICES[model]
    if served and served != model:
        row = PRICES.get(served) or max(PRICES.values(), key=lambda r: r[0][1])
    low, high, line = row
    inp = usage.get('input_tokens') or 0
    read = usage.get('cache_read_input_tokens') or 0
    wrote = usage.get('cache_creation_input_tokens') or 0
    out = usage.get('output_tokens') or 0
    p = high if inp + read + wrote > line else low
    return (inp * p[0] + out * p[1] + read * p[2] + wrote * p[3]) / 1e6


def openai_cost(prices, usage):
    """prices: (input, output, cache_read) USD per million; usage: OpenAI-style prompt / completion tokens."""
    inp, out, read_rate = prices
    total = usage.get('prompt_tokens') or 0
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    return (max(0, total - cached) * inp + cached * read_rate + (usage.get('completion_tokens') or 0) * out) / 1e6


class Ledger:
    def __init__(self, path, cap):
        self.path, self.cap, self.lock = Path(path), cap, threading.Lock()

    def _load(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def today(self):
        return datetime.date.today().isoformat()

    def spent(self):
        with self.lock:
            return self._load().get(self.today(), {}).get('usd', 0.0)

    def add(self, usd, inp=0, out=0, read=0, wrote=0):
        with self.lock:
            data = self._load()
            day = data.setdefault(self.today(), dict(usd=0.0, requests=0, input=0, output=0, cache_read=0, cache_write=0))
            day['usd'] = round(day['usd'] + usd, 6)
            day['requests'] += 1
            day['input'] += inp
            day['output'] += out
            day['cache_read'] += read
            day['cache_write'] += wrote
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.path)
            return day['usd']


class DeepInfraPrices:
    """Per-model prices from DeepInfra's own model list, refreshed hourly; a model it publishes no price for is refused."""

    def __init__(self, key, log, fetch=None):
        self.key, self.log, self.table, self.at, self.lock = key, log, {}, 0.0, threading.Lock()
        self.fetch = fetch or self._fetch

    def _fetch(self):
        conn = http.client.HTTPSConnection(PROVIDERS['deepinfra']['host'], timeout=60,
                                           context=ssl.create_default_context())
        try:
            conn.request('GET', '/v1/openai/models', headers={'Authorization': 'Bearer ' + self.key})
            up = conn.getresponse()
            if up.status != 200:
                raise OSError('model list status %d' % up.status)
            return json.loads(up.read())
        finally:
            conn.close()

    def get(self, model):
        with self.lock:
            if time.time() - self.at > PRICE_REFRESH or (model not in self.table and time.time() - self.at > 60):
                try:
                    table = {}
                    for m in self.fetch().get('data') or []:
                        # an entry with a null price or no id is skipped, never the whole table (a TypeError there
                        # dropped the client's connection and left every model unpriced)
                        try:
                            p = (m.get('metadata') or {}).get('pricing') or {}
                            cin, cout = p.get('input_tokens'), p.get('output_tokens')
                            if cin is None or cout is None or not m.get('id'):
                                continue
                            cread = p.get('cache_read_tokens')
                            table[m['id']] = (float(cin), float(cout), float(cin if cread is None else cread))
                        except (TypeError, ValueError, AttributeError):
                            continue
                    self.table, self.at = table, time.time()
                    self.log('prices: %d priced models' % len(table))
                except (OSError, ValueError) as e:
                    self.log('price list unavailable: %s' % type(e).__name__)
            return self.table.get(model)


def _list_models(spec, key):
    """The provider's model list (GET its catalog path with the key), for resolving a model's names."""
    conn = http.client.HTTPSConnection(spec['host'], timeout=60, context=ssl.create_default_context())
    try:
        conn.request('GET', spec['get'][0], headers={'Authorization': 'Bearer ' + key, 'accept': 'application/json'})
        up = conn.getresponse()
        if up.status != 200:
            raise OSError('model list status %d' % up.status)
        return json.loads(up.read())
    finally:
        conn.close()


class OwnerPrices:
    """Per-model prices the owner wrote (a provider that publishes none): {"model": [input, output]} or
    {"model": {"input": .., "output": .., "cache_read": ..}}, USD per million tokens, re-read when the file changes.
    A model not in it, or a file that cannot be read, gets no price: the request is refused.

    A model has several names (a dated id and aliases such as mistral-large-latest): a price written under any of them
    applies to all, resolved through the provider's own model list (fetch: () -> {data: [{id, aliases}]}, refreshed
    hourly). A name that resolves to nothing priced is still refused, so a wrong name blocks a call, never meters it
    wrongly."""

    def __init__(self, path, log, fetch=None):
        self.path, self.log, self.table, self.mtime, self.lock = Path(os.path.expanduser(path)), log, {}, None, threading.Lock()
        self.fetch, self.names, self.names_at = fetch, {}, 0.0

    def get(self, model):
        with self.lock:
            try:
                mt = self.path.stat().st_mtime
                if mt != self.mtime:
                    table = {}
                    for k, v in json.loads(self.path.read_text()).items():
                        if isinstance(v, dict):
                            table[k] = (float(v['input']), float(v['output']), float(v.get('cache_read', v['input'])))
                        else:
                            table[k] = (float(v[0]), float(v[1]), float(v[2] if len(v) > 2 else v[0]))
                    self.table, self.mtime = table, mt
                    self.log('prices: %d models from %s' % (len(table), self.path))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
                if self.table or self.mtime is not None:
                    self.log('price file unreadable: %s' % type(e).__name__)
                self.table, self.mtime = {}, None
            if model in self.table or not self.fetch:
                return self.table.get(model)
            if time.time() - self.names_at > PRICE_REFRESH or (model not in self.names and time.time() - self.names_at > 60):
                try:
                    names = {}
                    for m in self.fetch().get('data') or []:
                        group = [m['id'], *(m.get('aliases') or [])]
                        for n in group:
                            names.setdefault(n, set()).update(group)
                    self.names, self.names_at = names, time.time()
                except (OSError, ValueError, KeyError, TypeError) as e:
                    self.log('model list unavailable: %s' % type(e).__name__)
            for other in sorted(self.names.get(model, ())):
                if other in self.table:
                    return self.table[other]
            return None


def make_handler(provider, key, ledger, models, max_tokens, log, prices=None, connect=None, install_args=None):
    """install_args: the `relay install` flags that would rebuild this relay (cap, port, ...; models aside), so a refused
    model's error names the exact command that allows it as well"""
    spec = PROVIDERS[provider]
    connect = connect or (lambda: http.client.HTTPSConnection(spec['host'], timeout=900,
                                                              context=ssl.create_default_context()))

    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def refuse(self, status, kind, message):
            err = dict(type='error', error=dict(type=kind, message='relay: ' + message)) if provider == 'anthropic' \
                else dict(error=dict(message='relay: ' + message, type=kind, code=status))
            raw = json.dumps(err).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            log('refused %d %s' % (status, message))

        def watch_client(self, upsock, done):
            """While a stream is relayed, notice a client that hangs up even when upstream sends nothing (a long
            prompt, a model thinking): the upstream connection is closed at once, so the provider stops generating.
            Without this the hang-up was seen only at the next write, after upstream's next line. Returns the event
            set when the client went; setting done ends the watch."""
            gone = threading.Event()

            def watch():
                while not done.is_set():
                    try:
                        r, _, _ = select.select([self.connection], [], [], 0.1)
                        if not r or self.connection.recv(1, socket.MSG_PEEK) != b'':
                            if r:
                                time.sleep(0.1)     # (bytes waiting from the client: not a hang-up)
                            continue
                    except (OSError, ValueError):
                        pass                        # (the connection is already closed: the client went)
                    if done.is_set():
                        return
                    gone.set()
                    if upsock is not None:
                        try:
                            upsock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return
            threading.Thread(target=watch, daemon=True, name='relay-client-watch').start()
            return gone

        def auth(self):
            return {'x-api-key': key} if provider == 'anthropic' else {'Authorization': 'Bearer ' + key}

        def do_GET(self):
            # the catalog's model list: no cost, no body; the reply is passed through as is
            path = self.path.split('?')[0]
            if path not in spec['get']:
                return self.refuse(404, 'not_found_error', 'path not allowed: GET ' + path)
            conn = connect()
            try:
                headers = dict(self.auth(), accept='application/json')
                if provider == 'anthropic':
                    headers['anthropic-version'] = self.headers.get('anthropic-version') or '2026-10-21'
                conn.request('GET', self.path, None, headers)
                up = conn.getresponse()
                data = up.read()
                if provider == 'mistral' and up.status == 200 and prices is not None:
                    # the owner's prices ride along in the list (the agent cannot read the price file; without them
                    # its spend meter priced every turn as unknown and counted $0)
                    try:
                        listed = json.loads(data)
                        for m in listed.get('data') or []:
                            r = prices.get(m.get('id'))
                            if r:
                                m['pricing'] = dict(input=r[0], output=r[1], cache_read=r[2])
                        data = json.dumps(listed).encode()
                    except (ValueError, AttributeError, TypeError):
                        pass
                self.send_response(up.status)
                self.send_header('Content-Type', up.getheader('Content-Type') or 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True
            except OSError as e:
                log('upstream error: %s' % type(e).__name__)
                self.refuse(502, 'api_error', 'upstream unreachable')
            finally:
                conn.close()

        def do_POST(self):
            path = self.path.split('?')[0]
            if path not in spec['post']:
                return self.refuse(404, 'not_found_error', 'path not allowed: ' + path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
            except (ValueError, TypeError):
                return self.refuse(400, 'invalid_request_error', 'body is not JSON')
            model = body.get('model')
            if provider == 'anthropic':
                if model not in models:
                    # the allowed list and the command that adds this model while keeping them: an advisor or a /model
                    # on a model the relay was not installed for failed with no hint of the fix
                    hint = ''
                    if isinstance(model, str) and model in PRICES and install_args is not None:
                        hint = '; to allow it too: ' + install_command(provider, install_args, sorted(models) + [model])
                    return self.refuse(403, 'permission_error', 'model not allowed: %r (this relay allows %s%s)' % (
                        model, ', '.join(sorted(models)), hint))
                rates = None
            else:
                rates = prices.get(model) if isinstance(model, str) else None
                if rates is None:
                    return self.refuse(403, 'permission_error', 'no price for %r: it cannot be metered%s' % (
                        model, ' (add it to the relay\'s --prices file)' if provider == 'mistral' else ''))
            spent = ledger.spent()
            if spent >= ledger.cap:
                return self.refuse(402, 'billing_error', 'daily cap $%.2f reached ($%.4f spent today)' % (ledger.cap, spent))
            counting = path.endswith('count_tokens')
            beta = None
            if not counting:
                if not isinstance(body.get('max_tokens'), int) or body['max_tokens'] > max_tokens:
                    body['max_tokens'] = max_tokens
                for banned in ('speed', 'service_tier', 'container', 'mcp_servers', 'n'):
                    body.pop(banned, None)
                if provider == 'anthropic':
                    if body.get('fallbacks') != 'default' or model not in FALLBACK_MODELS:
                        body.pop('fallbacks', None)
                    beta = FALLBACK_BETA if 'fallbacks' in body and FALLBACK_BETA in (self.headers.get('anthropic-beta') or '') else None
                    if 'fallbacks' in body and not beta:
                        body.pop('fallbacks')
                elif body.get('stream') and provider == 'deepinfra':
                    body['stream_options'] = dict(include_usage=True)   # the only way its stream reports usage
                    # (Mistral sends usage in its last stream chunk and refuses fields it does not know)
            raw = json.dumps(body).encode()
            conn = connect()
            try:
                headers = dict(self.auth(), **{'content-type': 'application/json',
                                               'accept': self.headers.get('Accept') or 'application/json'})
                if provider == 'anthropic':
                    headers['anthropic-version'] = self.headers.get('anthropic-version') or '2026-10-21'
                    if beta:
                        headers['anthropic-beta'] = beta
                conn.request('POST', path, raw, headers)
                # (taken now: http.client drops conn.sock in getresponse when upstream will close the connection)
                upsock = getattr(conn, 'sock', None)
                up = conn.getresponse()
                # A client that hangs up (an interrupt, a steer and /advisor end a reply this way) never skips the
                # ledger, on any path: before the headers, mid-stream or on a whole reply. Upstream already billed it.
                client_gone, streamed, final = False, 0, False   # streamed: output characters sent; final: the count came
                try:
                    self.send_response(up.status)
                    for name in ('Content-Type', 'retry-after', 'request-id', 'x-request-id'):
                        if up.getheader(name):
                            self.send_header(name, up.getheader(name))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                except OSError:
                    client_gone = True
                usage, served, stop, category, fallback, errtext = {}, model, None, None, False, ''
                rid = up.getheader('request-id') or up.getheader('x-request-id')
                if 'text/event-stream' in (up.getheader('Content-Type') or ''):
                    watched, ended = threading.Event(), False   # ended: the reply's last event came
                    gone = self.watch_client(upsock, watched)
                    while not client_gone:
                        try:
                            line = up.readline()
                        except (OSError, ValueError, http.client.HTTPException):
                            # (a chunked upstream shut mid-chunk raises IncompleteRead, an HTTPException: the hang-up is
                            #  still metered, as for a plain socket error)
                            if not gone.is_set():
                                raise
                            line = b''
                        if not line:
                            # (a client that closes right after the last event is not a hang-up)
                            client_gone = client_gone or (gone.is_set() and not ended)
                            break
                        if line[5:].strip() == b'[DONE]' or b'"message_stop"' in line:
                            ended = True
                        try:
                            self.wfile.write(line)
                            self.wfile.flush()
                        except OSError:
                            # stop after this line (closing upstream ends the generation); the line is still read
                            # below, so a final usage event that failed to send is not lost
                            client_gone = True
                        if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except ValueError:
                            continue
                        if provider != 'anthropic':
                            if ev.get('usage'):
                                usage = ev['usage']
                            for ch in ev.get('choices') or []:
                                d = (ch or {}).get('delta') or {}
                                streamed += len(d.get('content') or '') + len(d.get('reasoning_content') or '')
                                for tc in d.get('tool_calls') or []:
                                    streamed += len(((tc or {}).get('function') or {}).get('arguments') or '')
                            continue
                        t = ev.get('type')
                        if t == 'message_start':
                            usage.update((ev.get('message') or {}).get('usage') or {})
                            served = (ev.get('message') or {}).get('model') or served
                        elif t == 'content_block_start' and (ev.get('content_block') or {}).get('type') == 'fallback':
                            fallback = True
                            served = ((ev['content_block'].get('to') or {}).get('model')) or served
                        elif t == 'message_delta':
                            usage.update(ev.get('usage') or {})
                            final = True
                            stop = (ev.get('delta') or {}).get('stop_reason') or stop
                            category = ((ev.get('delta') or {}).get('stop_details') or {}).get('category') or category
                        elif t == 'error':
                            errtext = json.dumps(ev.get('error'))[:500]
                        elif t == 'content_block_delta':
                            d = ev.get('delta') or {}
                            streamed += sum(len(d.get(k) or '') for k in ('text', 'thinking', 'partial_json'))
                    watched.set()
                else:
                    data = up.read()
                    try:
                        self.wfile.write(data)
                    except OSError:
                        client_gone = True      # the whole reply was billed: its usage below is exact
                    if up.status != 200:
                        errtext = data[:500].decode('utf-8', 'replace')
                    if not counting and up.status == 200:
                        try:
                            usage = json.loads(data).get('usage') or {}
                        except ValueError:
                            usage = {}
                extra = ''.join(s for s in (
                    ' served=%s' % served if served != model else '', ' FALLBACK' if fallback else '',
                    ' stop=%s' % stop if stop and stop not in ('end_turn', 'tool_use') else '',
                    ' category=%s' % category if category else '',
                    ' rid=%s' % rid if (rid and (errtext or stop == 'refusal' or fallback)) else ''))
                if client_gone:
                    extra += ' CLIENT-GONE'
                    if provider == 'anthropic' and not final and 'text/event-stream' in (up.getheader('Content-Type') or ''):
                        if not usage:          # gone before message_start: the prompt was still read
                            usage = {'input_tokens': len(raw) // 4}
                        # the final count never came (a hang-up: an interrupt, a steer, /advisor end a reply this way):
                        # charge what was streamed, about 3.5 characters a token rounded up, never the whole allowance
                        # (that put phantom spend on every interrupt and could hit the daily cap early)
                        # (message_start's usage says output_tokens 1: the estimate replaces it unless larger)
                        usage['output_tokens'] = max(int(usage.get('output_tokens') or 0),
                                                     min(streamed * 2 // 7 + 1, int(body.get('max_tokens') or 0) or 10**9))
                        extra += ' (output estimated from %d streamed characters)' % streamed
                if not counting and provider == 'anthropic' and usage:
                    usd = anthropic_cost(model, usage, served)
                    total = ledger.add(usd, usage.get('input_tokens') or 0, usage.get('output_tokens') or 0,
                                       usage.get('cache_read_input_tokens') or 0, usage.get('cache_creation_input_tokens') or 0)
                    log('%s in=%s read=%s write=%s out=%s $%.5f today $%.4f%s' % (
                        model, usage.get('input_tokens'), usage.get('cache_read_input_tokens'),
                        usage.get('cache_creation_input_tokens'), usage.get('output_tokens'), usd, total, extra))
                elif not counting and provider != 'anthropic' and up.status == 200:
                    if usage:
                        usd = openai_cost(rates, usage)
                    elif client_gone and 'text/event-stream' in (up.getheader('Content-Type') or ''):
                        # a hang-up before the usage chunk: the prompt plus what was streamed, not the whole allowance
                        usd = (len(raw) / 4 * rates[0] + min(streamed * 2 // 7 + 1, body['max_tokens']) * rates[1]) / 1e6
                        extra += ' (estimated: prompt + %d streamed characters)' % streamed
                    else:
                        # no usage reported (a cut stream): charge the request as if every output token was spent
                        usd = (len(raw) / 4 * rates[0] + body['max_tokens'] * rates[1]) / 1e6
                        extra += ' NO-USAGE (charged the estimate)'
                    total = ledger.add(usd, usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0,
                                       (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
                    log('%s in=%s out=%s $%.5f today $%.4f%s' % (model, usage.get('prompt_tokens'),
                                                                  usage.get('completion_tokens'), usd, total, extra))
                if errtext or (provider == 'anthropic' and not usage and not counting) or (counting and up.status != 200):
                    log('%s %s status %d%s error=%s' % (model, path, up.status, extra, errtext or '(no usage)'))
                self.close_connection = True
            except OSError as e:
                log('upstream error: %s' % type(e).__name__)
                try:
                    self.refuse(502, 'api_error', 'upstream unreachable')
                except OSError:
                    pass
            finally:
                conn.close()
    return Relay


def install_command(provider, args, models):
    """The `caminus ember relay install` line for these flags and exactly these models (install replaces the list, so
    every model the relay should keep is named)."""
    import shlex
    return ' '.join(shlex.quote(x) for x in ['caminus', 'ember', 'relay', 'install', '--provider', provider, *args,
                                              *[a for m in models for a in ('--model', m)]])


def _flags(argv):
    """a relay's own flags (from its unit's ExecStart): the install flags apart from --model, and the models"""
    ap = argparse.ArgumentParser(add_help=False)
    for f in ('--cap-usd', '--port', '--max-tokens', '--key-file', '--prices'):
        ap.add_argument(f)
    ap.add_argument('--model', action='append')
    a, _ = ap.parse_known_args(argv)
    args = []
    for f in ('cap_usd', 'port', 'max_tokens', 'key_file', 'prices'):
        if getattr(a, f) is not None:
            args += ['--' + f.replace('_', '-'), getattr(a, f)]
    return args, a


def installed(provider):
    """The installed relay service for a provider, read from its unit: dict(port, cap_usd, models, args, active?) or
    None. models is None for a provider that allows every priced model."""
    import shlex
    try:
        text = (unit_dir() / unit_name(provider)).read_text()
    except OSError:
        return None
    line = next((l for l in text.splitlines() if l.startswith('ExecStart=')), None)
    if line is None:
        return None
    try:
        argv = shlex.split(line[len('ExecStart='):])
    except ValueError:
        return None
    if '--provider' not in argv:
        return None
    args, a = _flags(argv[argv.index('--provider'):])       # (the relay's own flags, after the interpreter's)
    try:
        port = int(a.port or PROVIDERS[provider]['port'])
        cap = float(a.cap_usd) if a.cap_usd is not None else None
    except ValueError:
        return None
    models = (sorted(set(a.model or ['claude-haiku-5-5'])) if provider == 'anthropic' else None)
    return dict(port=port, cap_usd=cap, models=models, args=args)


def unit_at(endpoint):
    """The installed Anthropic relay (installed()) that answers at this endpoint on this machine, or None."""
    import urllib.parse
    try:
        u = urllib.parse.urlsplit(endpoint or '')
        port = u.port
    except ValueError:
        return None
    if u.hostname not in ('127.0.0.1', 'localhost', '::1') or port is None:
        return None
    r = installed('anthropic')
    return r if r and r['port'] == port else None


def allow_command(endpoint, model):
    """For an Anthropic endpoint on this machine served by an installed relay that does not allow `model`: the exact
    command that allows it as well (keeping the cap, port and the models it serves now). None when no installed relay
    answers there, or it allows the model already."""
    r = unit_at(endpoint)
    if not r or model in r['models'] or model not in PRICES:
        return None
    return install_command('anthropic', r['args'], r['models'] + [model])


def unit_name(provider):
    return f'caminus-relay-{provider}.service'


def unit_dir():
    return Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'systemd' / 'user'


def freeze():
    """A copy of the code the relay runs (caminus/ember/relay.py and the resident package it prices from), in
    $XDG_DATA_HOME/caminus/relay-code/<content digest>/, so the service never runs from a checkout: editing, moving or
    archiving the source tree cannot stop or change a running relay. Named by content, so the same code reuses its
    copy and an older copy stays for any service still on it. Returns the directory that holds the caminus package."""
    import hashlib
    import shutil
    src = Path(__file__).resolve().parents[2]
    files = [Path('caminus/__init__.py'), Path('caminus/ember/__init__.py'), Path('caminus/ember/relay.py'),
             *sorted(p.relative_to(src) for p in (src / 'caminus/ember/resident').glob('*.py'))]
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f).encode() + b'\0' + (src / f).read_bytes())
    root = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local/share') / 'caminus' / 'relay-code'
    dest = root / digest.hexdigest()[:12]
    if not (dest / 'caminus/ember/relay.py').exists():
        tmp = root / (dest.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        for f in files:
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, tmp / f)
        tmp.replace(dest)
    return dest


def unit_text(provider, args, pkg_root=None):
    """The service: this interpreter, the frozen relay code (freeze), the relay's own flags; no --ttl (it runs until
    stopped)."""
    pkg_root = pkg_root or Path(__file__).resolve().parents[2]
    cmd = [sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, sys.argv[1]); from caminus.ember import relay; '
           'relay.main(sys.argv[2:])', str(pkg_root), '--provider', provider, '--ttl', '0', *args]
    quote = lambda a: '"' + a.replace('\\', '\\\\').replace('"', '\\"') + '"' if any(c in a for c in ' ";') else a
    return ('[Unit]\nDescription=Caminus key relay (%s), spend-capped\nAfter=network-online.target\n\n'
            '[Service]\nType=simple\nExecStart=%s\nRestart=on-failure\nRestartSec=10\n'
            'NoNewPrivileges=yes\nPrivateTmp=yes\n\n[Install]\nWantedBy=default.target\n'
            % (provider, ' '.join(quote(a) for a in cmd)))


def service(argv, run=subprocess.run):
    """install / uninstall / status of the relay services."""
    op, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f'caminus ember relay {op}')
    ap.add_argument('--provider', choices=sorted(PROVIDERS), required=op != 'status')
    ap.add_argument('--cap-usd', type=float)
    ap.add_argument('--model', action='append')
    ap.add_argument('--port', type=int)
    ap.add_argument('--max-tokens', type=int)
    ap.add_argument('--key-file')
    ap.add_argument('--prices', help='mistral: the price file')
    a = ap.parse_args(rest)
    ctl = lambda *x: run(['systemctl', '--user', *x], capture_output=True, text=True)
    if op == 'install':
        if a.cap_usd is None:
            sys.exit('install needs --cap-usd: the daily cap is what guards the spend')
        key_file = Path(os.path.expanduser(a.key_file or PROVIDERS[a.provider]['key_file']))
        if not key_file.exists():
            sys.exit(f'no key file at {key_file}: save the key there (mode 600) first')
        args = ['--cap-usd', str(a.cap_usd)]
        for flag, val in (('--port', a.port), ('--max-tokens', a.max_tokens), ('--key-file', a.key_file),
                          ('--prices', a.prices)):
            if val is not None:
                args += [flag, str(val)]
        for m in a.model or []:
            args += ['--model', m]
        d = unit_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / unit_name(a.provider)).write_text(unit_text(a.provider, args, freeze()))
        ctl('daemon-reload')
        r = ctl('enable', '--now', unit_name(a.provider))
        if r.returncode:
            sys.exit('systemctl --user enable failed: ' + (r.stderr or r.stdout).strip()[:300])
        # enable --now leaves a running relay on its old settings: rerunning install to change the cap did nothing
        # until the next restart. Restart it (today's spend is in the ledger file and carries over).
        ctl('restart', unit_name(a.provider))
        print(f'installed {unit_name(a.provider)}: cap ${a.cap_usd:.2f}/day, no expiry, restarts on failure. '
              'For it to run while you are logged out: loginctl enable-linger')
        return 0
    if op == 'uninstall':
        ctl('disable', '--now', unit_name(a.provider))
        try:
            (unit_dir() / unit_name(a.provider)).unlink()
        except FileNotFoundError:
            pass
        ctl('daemon-reload')
        print('removed ' + unit_name(a.provider))
        return 0
    state = Path.home() / '.local/state/caminus'
    for prov in ([a.provider] if a.provider else sorted(PROVIDERS)):
        unit = installed(prov)
        is_on = (unit_dir() / unit_name(prov)).exists()
        active = ctl('is-active', unit_name(prov)).stdout.strip() if is_on else 'not installed'
        ledger = state / ('key-relay-ledger.json' if prov == 'anthropic' else f'{prov}-relay-ledger.json')
        spent = Ledger(ledger, 0).spent() if ledger.exists() else 0.0
        line = f'{prov:10s} {active:14s} today ${spent:.4f}'
        if unit:
            # the cap, the port and (Anthropic) the models it allows: an advisor or brain on another model is refused
            line += (f' of ${unit["cap_usd"]:.2f}' if unit['cap_usd'] is not None else '') + f'  port {unit["port"]}'
            if unit['models'] is not None:
                line += '  models ' + ', '.join(unit['models'])
        print(line)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] and argv[0] in ('install', 'uninstall', 'status'):
        return service(argv)
    ap = argparse.ArgumentParser(prog='caminus ember relay', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--provider', choices=sorted(PROVIDERS), default='anthropic')
    ap.add_argument('--port', type=int, help='default 8765 (anthropic), 8766 (deepinfra), 8767 (mistral)')
    ap.add_argument('--prices', help='mistral: the owner\'s price file (default ~/.config/caminus/mistral-prices.json)')
    ap.add_argument('--cap-usd', type=float, default=1.0, help='daily spend cap in USD')
    ap.add_argument('--ttl', type=float, default=1800, help='seconds before the relay exits on its own; 0 = never')
    # a request's output cap; billing is per token written and the dollar caps hold, so 64K costs nothing unused
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--model', action='append', help='anthropic: allowed model (repeatable); default claude-haiku-5-5')
    ap.add_argument('--key-file', help='default ~/.config/anthropic/api_key, ~/.config/caminus/deepinfra-key or '
                                       '~/.config/caminus/mistral-key')
    ap.add_argument('--ledger', help='default ~/.local/state/caminus/<provider>-relay-ledger.json')
    ap.add_argument('--log', help='default ~/.local/state/caminus/<provider>-relay.log')
    a = ap.parse_args(argv)
    spec = PROVIDERS[a.provider]
    state = Path.home() / '.local/state/caminus'
    legacy = a.provider == 'anthropic'    # the first relay's file names, so its ledger keeps counting today
    ledger_path = a.ledger or str(state / ('key-relay-ledger.json' if legacy else f'{a.provider}-relay-ledger.json'))
    log_path = a.log or str(state / ('key-relay.log' if legacy else f'{a.provider}-relay.log'))
    models = set(a.model or ['claude-haiku-5-5'])
    if a.provider == 'anthropic':
        unknown = models - set(PRICES)
        if unknown:
            sys.exit('no price for %s: add it to PRICES first' % ', '.join(sorted(unknown)))
    elif a.model:
        sys.exit('--model is for anthropic: deepinfra and mistral allow every model with a price (the cap is the guard)')
    key_file = Path(os.path.expanduser(a.key_file or spec['key_file']))
    try:
        if key_file.stat().st_mode & 0o077:
            sys.exit('%s is readable by others: chmod 600 it first' % key_file)
        key = key_file.read_text().strip()
    except OSError as e:
        sys.exit('cannot read the key file %s: %s' % (key_file, type(e).__name__))
    if not key:
        sys.exit('key file is empty')
    state.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, 'a', buffering=1)

    def log(msg):
        logf.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))

    ledger = Ledger(ledger_path, a.cap_usd)
    prices = (DeepInfraPrices(key, log) if a.provider == 'deepinfra' else
              OwnerPrices(a.prices or spec['prices'], log, fetch=lambda: _list_models(spec, key))
              if a.provider == 'mistral' else None)
    port = a.port or spec['port']
    # (no --key-file or --prices: the hint reaches the agent, which has no business with the owner's file paths; the
    # owner's side adds them from the installed unit, and an install without them stops at a missing default key file)
    install_args = ['--cap-usd', str(a.cap_usd)]
    for flag, val, default in (('--port', a.port, None), ('--max-tokens', a.max_tokens, 65536)):
        if val is not None and val != default:
            install_args += [flag, str(val)]
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port),
                                          make_handler(a.provider, key, ledger, models, a.max_tokens, log, prices,
                                                       install_args=install_args))
    if a.ttl > 0:
        threading.Timer(a.ttl, srv.shutdown).start()
    root = '/v1/openai' if a.provider == 'deepinfra' else '/v1'
    log('start %s port %d cap $%.2f ttl %ds%s (spent today $%.4f)' % (
        a.provider, port, a.cap_usd, a.ttl, ' models %s' % sorted(models) if a.provider == 'anthropic' else '',
        ledger.spent()))
    print('relay %s on http://127.0.0.1:%d%s  cap $%.2f/day  ttl %ds  spent today $%.4f' % (
        a.provider, port, root, a.cap_usd, a.ttl, ledger.spent()), flush=True)
    try:
        srv.serve_forever()
    finally:
        log('stop (spent today $%.4f)' % ledger.spent())


if __name__ == '__main__':
    main()
