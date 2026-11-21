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
  meta       api.meta.ai (the Meta Model API: the Muse models): POST /v1/chat/completions; GET /v1/models (the
             catalog). Its model list carries no prices; Meta's published table is the transport's
             (caminus.ember.resident.model.META_PRICES, with the long-prompt tier of Muse Ember (Meta)), so only the Muse chat
             models in it are allowed (Standard mode: the Contributor-mode ids, where Meta trains on the prompts, are not).
  opencode-go  opencode.ai (OpenCode Go, the subscription), under /inference/go: POST its Chat Completions and Messages
             endpoints (/openai/v1/chat/completions, /anthropic/v1/messages); GET /v1/models (the catalog, with prices).
             The models Ember offers are allowed (catalog.opencode_offered: chat models that call tools, none whose
             prompts train a model), priced from that list (refreshed hourly). Each request must name its conversation
             (x-opencode-session, which Go requires; refused here with a 400 before it goes out) and goes out as Ember
             (its User-Agent). A reply says what it charged (usage.cost: 0 inside the Go allowance, dollars from the
             balance past it): that is what the cap counts; the allowance it used is kept beside it. A reply cut
             before its usage came is booked at the list price times the largest peak-hour multiplier in force.

Run a separate relay (own port, key file, cap, ledger and log) for each provider and for an advisor.

As a service that outlasts any session (a systemd --user unit for the owner; no sudo, survives logouts with lingering,
restarts on failure, no expiry: the daily cap guards spend):
  caminus ember relay install --provider deepinfra --cap-usd 3
  caminus ember relay status            (every installed relay: active or not, today's spend)
  caminus ember relay uninstall --provider deepinfra
"""
import argparse
import base64
import math
import re
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

from .resident.files import IMAGE_HEAD, IMAGE_MAX_SIDE, image_info
from .resident.catalog import OPENCODE_PATHS, opencode_charge, opencode_multiplier, opencode_offered, opencode_row
from .resident.model import (FALLBACK_BETA, FALLBACK_MODELS, META_PRICES, OPENCODE_ANTHROPIC_VERSION, PRICES, USER_AGENT,
                             flat_row, image_tokens, row_cost)

GO = '/inference/go'
PROVIDERS = {
    'anthropic': dict(host='api.anthropic.com', post=('/v1/messages', '/v1/messages/count_tokens'),
                      get=('/v1/models',), port=8765, key_file='~/.config/anthropic/api_key'),
    'deepinfra': dict(host='api.deepinfra.com', post=('/v1/openai/chat/completions',),
                      get=('/v1/openai/models',), port=8766, key_file='~/.config/caminus/deepinfra-key'),
    'mistral': dict(host='api.mistral.ai', post=('/v1/chat/completions',), get=('/v1/models',), port=8767,
                    key_file='~/.config/caminus/mistral-key', prices='~/.config/caminus/mistral-prices.json'),
    'meta': dict(host='api.meta.ai', post=('/v1/chat/completions',), get=('/v1/models',), port=8768,
                 key_file='~/.config/caminus/meta-key', hidden_reasoning=True),   # (reasoning_content comes empty)
    # root: the plan's prefix (OpenCode Zen is the same API under another); session: every POST names its conversation
    'opencode-go': dict(host='opencode.ai', root=GO, post=(GO + OPENCODE_PATHS['openai'] + '/chat/completions',
                                                           GO + OPENCODE_PATHS['anthropic'] + '/messages'),
                        get=(GO + '/v1/models',), port=8769, key_file='~/.config/caminus/opencode-key', session=True),
}
SESSION_ID = re.compile(r'[A-Za-z0-9_-]{8,128}')   # an x-opencode-session the relay forwards (Ember sends uuid4 hex)
PRICE_REFRESH = 3600
# Seconds between attempts after a price list fails to load, then the last one again until it loads: a relay that
# starts before the network does (a systemd unit 12 s after boot) is refused by nothing while it waits, and does not
# hammer a provider that is down. The last delay is not PRICE_REFRESH: an hour is the window this bug cost.
LOAD_BACKOFF = (10, 30, 60, 300)
# The wait a refusal names while the owner's price file cannot be read: the file is re-read on every request, so
# this only tells the agent how long to wait before the retry that may find the file written.
PRICE_FILE_RETRY = 60
# Seconds a last good price table outlives the price file becoming unreadable. A file caught mid-rewrite (truncate,
# then write) must keep pricing the models it priced a moment ago, so the table is not dropped on the first failed
# read; but a file the owner DELETED must not meter at its old rates forever. After this grace the relay drops the
# table and stops answering 503 for the file: a request then gets the same 403 as any unpriced model, and the first
# request that can read the file prices again (the file is re-read on every request).
PRICE_FILE_GRACE = 900
# A decode-rate ceiling for output a stream does not show (Meta bills its models' reasoning but sends it empty), used
# only to bound a reply cut before its usage came. Meta publishes no speed: an assumed ceiling (unverified), above the
# decode rates hosted APIs usually report, so a hang-up errs high rather than booking one token.
HIDDEN_TPS = 400


def anthropic_cost(model, usage, served=None):
    row = PRICES[model]
    if served and served != model:
        row = PRICES.get(served) or max(PRICES.values(), key=lambda r: r[0][1])
    return row_cost(row, usage)


def provider_image_tokens(provider, width, height):
    """An image's input tokens where the provider's count never came. Anthropic: Claude's rule (model.image_tokens, 28 px
    patches after its downscale: about w*h/750). Others: 28 px patches with no downscale. Mistral's docs give no rule;
    Ministral 3B / 8B / 14B billed four screenshots (640-860 px wide) at exactly that count plus the request's text
    (measured 2026-11), and assuming no downscale errs high for a large image."""
    if provider == 'anthropic':
        return image_tokens(width, height)
    return math.ceil(width / 28) * math.ceil(height / 28)


def prompt_tokens(body, provider):
    """A request's input tokens, estimated for a reply cut before its usage arrived: the JSON text at 4 bytes a token,
    each base64 image at its image-token count (by its bytes, one 1 MB screenshot booked about 350K phantom tokens). An
    image whose header can't be read counts as the largest read sends (files.IMAGE_MAX_SIDE a side)."""
    images = []

    def strip(x):
        if isinstance(x, list):
            return [strip(v) for v in x]
        if not isinstance(x, dict):
            return x
        src = x.get('source')
        if x.get('type') == 'image' and isinstance(src, dict) and isinstance(src.get('data'), str):
            images.append(src['data'])                          # (Messages API image block)
            return dict(x, source=dict(src, data=''))
        url = x.get('image_url')
        url = url.get('url') if isinstance(url, dict) else url
        if x.get('type') == 'image_url' and isinstance(url, str) and url.startswith('data:'):
            images.append(url.split(',', 1)[-1])                # (Chat Completions data URI)
            return dict(x, image_url='')
        return {k: strip(v) for k, v in x.items()}

    total = len(json.dumps(strip(body)).encode()) // 4
    for data in images:
        size = None
        try:
            found = image_info(base64.b64decode(data[:(IMAGE_HEAD // 3) * 4]))
            size = found[1:] if found else None
        except ValueError:                                      # (binascii.Error is a ValueError)
            pass
        total += provider_image_tokens(provider, *(size or (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE)))
    return total


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

    def day(self):
        """today's row: usd, requests, tokens; an OpenCode plan's too: allowance and list_usd (USD), billing
        ({source: requests})"""
        with self.lock:
            return dict(self._load().get(self.today(), {}))

    def add(self, usd, inp=0, out=0, read=0, wrote=0, plan=None):
        """plan: an OpenCode reply's usage.opencode (billing_source, allowance, list_cost), kept beside the dollars so
        the Go allowance a day used shows apart from what the balance paid"""
        with self.lock:
            data = self._load()
            day = data.setdefault(self.today(), dict(usd=0.0, requests=0, input=0, output=0, cache_read=0, cache_write=0))
            day['usd'] = round(day['usd'] + usd, 6)
            day['requests'] += 1
            day['input'] += inp
            day['output'] += out
            day['cache_read'] += read
            day['cache_write'] += wrote
            if isinstance(plan, dict):
                for field, key in (('allowance', 'allowance'), ('list_cost', 'list_usd')):
                    v = plan.get(field)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        day[key] = round(day.get(key, 0.0) + float(v), 8)
                source = str(plan.get('billing_source') or 'unknown')[:16]
                day.setdefault('billing', {})[source] = day.get('billing', {}).get(source, 0) + 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.path)
            return day['usd']


def deepinfra_rates(m):
    """A DeepInfra model row's (input, output, cache read) in USD per million tokens, or None (a null price)."""
    p = (m.get('metadata') or {}).get('pricing') or {}
    cin, cout = p.get('input_tokens'), p.get('output_tokens')
    if cin is None or cout is None:
        return None
    cread = p.get('cache_read_tokens')
    return (float(cin), float(cout), float(cin if cread is None else cread))


def opencode_rates(m):
    """An OpenCode model row's PRICES-form row (with its long-context tier) if Ember offers the model, else None."""
    return opencode_row(m) if opencode_offered(m) else None


def _cannot_price(what, model, wait, tail='the relay tries again in %ds'):
    """The one refusal a price this relay cannot know YET gets: (status, kind, message, retry_after, local), a 503
    the agent waits out. `what` names the cause and never carries the owner's path (this text reaches the agent).
    local marks it X-Relay-Refusal on the wire, so a resident never reads it as the provider being down (a 503 from
    an outage is what moves a goal to its fallback brain; brain.Brains.on_failure)."""
    return (503, 'api_error', '%s, so %r cannot be priced yet; %s' % (what, model, tail % wait), wait, True)


class ListPrices:
    """Per-model prices from the provider's own model list (DeepInfra's, OpenCode's), refreshed hourly; a model it
    publishes no price for, or one rates() leaves out, is refused. fetch: () -> the list; rates: a row -> its rates
    or None; rows keeps each priced row by id (OpenCode's peak-hour multipliers are read from it).

    A load that fails is retried on LOAD_BACKOFF's schedule (a relay that starts before the network does must not
    refuse every model until someone restarts it), and until the list has loaded once a model is not refused as
    unpriced: refusal() answers 503 with the next try instead, so the agent waits rather than stopping its goal."""

    def __init__(self, log, fetch, rates=deepinfra_rates, backoff=LOAD_BACKOFF):
        self.log, self.fetch, self.rates, self.backoff = log, fetch, rates, tuple(backoff)
        self.table, self.rows, self.at, self.lock = {}, {}, 0.0, threading.Lock()
        self.tries, self.next, self.error = 0, 0.0, ''      # (the failed loads in a row, when the next may run, why)
        self.loading = False                                # (one fetch at a time, with the lock released)

    def _due(self, model, now):
        if now < self.next:                                 # a load failed: its backoff, whatever made it due
            return False                                    # (a refresh that fails must not run on every request)
        if not self.at:                                     # never loaded: the first request loads it
            return True
        if now - self.at > PRICE_REFRESH:
            return True
        return model not in self.table and now - self.at > 60

    def _fetch(self):
        """The provider's list parsed to (table, rows), with no lock held: the fetch is the slow part.

        A 200 whose data is a number or a bool raised a TypeError out of get() (the client's connection dropped
        and next/tries never advanced, so every later request fetched again); a row whose id was a list or a dict
        put an unhashable key in the table, the same TypeError by another route; a list that prices nothing is a
        truncated or empty reply, not a list. All three reach _load's handler as a failed load.
        """
        table, rows = {}, {}
        data = self.fetch().get('data') or []
        if not isinstance(data, list):
            raise ValueError('the model list is not a list: %s' % type(data).__name__)
        for m in data:
            # an entry with a null price or no id is skipped, never the whole table. The id must be a string.
            try:
                r = self.rates(m) if isinstance(m.get('id'), str) and m['id'] else None
            except (TypeError, ValueError, AttributeError):
                continue
            if r is not None:
                table[m['id']], rows[m['id']] = r, m
        if not table:
            raise ValueError('no priced model in the list')
        return table, rows

    def _load(self):
        """One load: the fetch outside the lock, the swap and the backoff under it.

        The clock is read here, after the fetch, and never taken from the caller: the backoff and `at` both start
        when the fetch ENDS (a stalled provider's delay must not be already spent). A `now` parameter read before
        the fetch would say the opposite (#389 review P2). The lock used to be held across the fetch, and `prime()`
        starts the first load exactly when the relay's port starts answering, so a provider that accepted and
        stalled kept the first request blocked for the whole fetch instead of getting the prompt 503 it waits out.
        One loader at a time: a request that arrives mid-fetch is answered from the table as it is, and refusal()
        says it is loading.
        """
        with self.lock:
            if self.loading:
                return
            self.loading = True
        try:
            try:
                table, rows = self._fetch()
                err = None
            except (OSError, ValueError, AttributeError, http.client.HTTPException) as e:
                # http.client.HTTPException too: BadStatusLine and IncompleteRead are not OSErrors, and one
                # escaping here dropped the client's connection (the price list was simply never loaded)
                err = e
            with self.lock:
                # the clock starts when the fetch ENDS, not when it began: a provider that accepted and stalled for
                # longer than the delay left `next` (and `at`) in the past, so the very next request fetched again and
                # the backoff this change exists for lapsed exactly when the provider was slow (#389 review P2)
                now = time.time()
                if err is None:
                    self.table, self.rows, self.at = table, rows, now
                    self.tries, self.next, self.error = 0, 0.0, ''
                    self.log('prices: %d priced models' % len(table))
                else:
                    self.tries += 1
                    delay = self.backoff[min(self.tries, len(self.backoff)) - 1]
                    self.next = now + delay
                    self.error = '%s: %s' % (type(err).__name__, str(err)[:200])
                    self.log('price list unavailable (%d in a row): %s; next try in %ds'
                             % (self.tries, self.error, round(delay)))
        finally:
            with self.lock:
                self.loading = False

    def get(self, model):
        with self.lock:
            due = self._due(model, time.time())
        if due:
            self._load()                     # (a fetch already in flight returns at once)
        with self.lock:
            return self.table.get(model)

    def refusal(self, model):
        """(status, kind, message, retry_after, local) while the list has never loaded, or None once it has: a model
        a good list does not price is a fact about the model, a list that never loaded is not. A first load still in
        flight is neither: it is a 503 with the short wait, not the 403 that stops the agent's goal.

        A refresh that failed is not a fact about the model either: a name the last good table does not price may be
        one the retried refresh will price, so within its backoff it gets the same 503 and wait (#389 review P2).
        Only a loaded table with no failed load behind it answers None for an unpriced name (the caller's 403)."""
        with self.lock:
            if not isinstance(model, str):
                return None                  # (not a model name: the caller refuses it; `model in self.table` needs a key)
            if self.at:
                if self.tries and self.error and model not in self.table:
                    wait = max(0, math.ceil(self.next - time.time()))
                    return _cannot_price('the price list refresh failed (%s) and the last good list does not price '
                                         'it' % self.error, model, wait)
                return None
            if self.loading and not self.error:
                return _cannot_price('the price list is still loading', model, LOAD_BACKOFF[0])
            if not self.error:
                return None
            wait = max(0, math.ceil(self.next - time.time()))
            return _cannot_price('the price list has not loaded yet (%s)' % self.error, model, wait)

    def prime(self):
        """Load once now, off the request path: a relay that starts before the network does logs the failure and
        retries on the schedule while its port answers."""
        threading.Thread(target=self.get, args=(None,), daemon=True, name='relay-price-prime').start()

    def multiplier(self, model):
        """the largest price multiplier the model's rules apply now (catalog.opencode_multiplier); 1 without rules"""
        with self.lock:
            row = self.rows.get(model)
        return opencode_multiplier(row) if row else 1.0


def _list_models(spec, key):
    """The provider's model list (GET its catalog path with the key), for resolving a model's names."""
    conn = http.client.HTTPSConnection(spec['host'], timeout=60, context=ssl.create_default_context())
    try:
        conn.request('GET', spec['get'][0], headers={'Authorization': 'Bearer ' + key, 'accept': 'application/json',
                                                     'User-Agent': USER_AGENT})
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
        self.error = ''                                    # why the file last could not be read ('' once it could)
        self.error_at = 0.0                                # when this unreadable streak began (0.0 = no failure)
        self.names_error = ''                              # why the model list (for aliases) last could not be read
        self.names_loading = False                         # (one alias-list fetch at a time, with the lock released)

    def _names_due(self, model, now):
        """Whether the provider's alias list is due for a fetch: hourly, and 60 s after a name it does not know (the
        alias may have been added since). A failed fetch still moves `names_at`, so that rule is the retry, not every
        request."""
        if now - self.names_at > PRICE_REFRESH:
            return True
        return model not in self.names and now - self.names_at > 60

    def _load_names(self):
        """One alias-list fetch: the provider's HTTPS GET outside the lock, the swap and the clock under it.

        The fetch used to run inside get()'s `with self.lock`, and this class is shared by every request thread, so a
        provider that accepted and stalled made every price lookup wait for it (#389 review P2). As in
        ListPrices._load, one loader runs at a time and a request that arrives mid-fetch answers from the names as
        they are; refusal() says the list is still loading. The clock is read after the fetch, so a stalled
        provider's delay is not already spent when the retry rule is asked again.
        """
        with self.lock:
            if self.names_loading:
                return
            self.names_loading = True
        try:
            names, err = None, None
            try:
                names = {}
                for m in self.fetch().get('data') or []:
                    group = [m['id'], *(m.get('aliases') or [])]
                    for n in group:
                        names.setdefault(n, set()).update(group)
            except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as e:
                err = e
            with self.lock:
                now = time.time()
                if err is None:
                    self.names, self.names_at, self.names_error = names, now, ''
                else:
                    why = '%s: %s' % (type(err).__name__, str(err)[:200])
                    if why != self.names_error:
                        self.log('model list unavailable: %s' % why)
                    # a failed fetch still moves the clock: names_at used to stay 0, so every request fetched a list
                    # that was down (the 60 s rule in _names_due is the retry, not every call)
                    self.names_at, self.names_error = now, why
        finally:
            with self.lock:
                self.names_loading = False

    def get(self, model):
        """The price for MODEL, or None. The price file is read under the lock (a local file, fast); the provider's
        model list is fetched outside it and its names swapped in under the lock, as ListPrices._load does."""
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
                    self.table, self.mtime, self.error, self.error_at = table, mt, '', 0.0
                    self.log('prices: %d models from %s' % (len(table), self.path))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
                # logged on the first failure too: a price file missing at start (a relay that starts before the owner
                # wrote it) refused every model with no line in the log, and the log is all the owner has. The message
                # keeps no path (FileNotFoundError's text carries it): this text reaches the agent, the log does not.
                # The last good table stays: a file caught mid-rewrite (truncate, then write) must not un-price the
                # models it priced a moment ago; mtime None re-reads it on the next request either way.
                why = '%s: %s' % (type(e).__name__, e.strerror if isinstance(e, OSError) else str(e)[:200])
                if why != self.error:
                    self.log('price file unreadable: %s (%s)' % (why, self.path))
                now = time.time()
                if not self.error_at:
                    self.error_at = now                     # the first failed read of this streak
                elif self.table and now - self.error_at > PRICE_FILE_GRACE:
                    # the file has been unreadable long enough to be gone, not caught mid-rewrite: the last good
                    # table is dropped, so nothing is metered at rates whose file no longer exists (before this,
                    # a deleted file priced every model at its old rates forever). The refusal stays the 503 wait:
                    # the owner may restore the file, and a wait spends nothing.
                    self.log('price file unreadable for %ds: dropping the last good table (%d models)'
                             % (round(now - self.error_at), len(self.table)))
                    self.table, self.error_at = {}, now     # one line per grace, not one per request
                self.error, self.mtime = why, None
            if model in self.table or not self.fetch:
                return self.table.get(model)
            due = self._names_due(model, time.time())
        if due:
            self._load_names()               # (a fetch already in flight returns at once)
        with self.lock:
            for other in sorted(self.names.get(model, ())):
                if other in self.table:
                    return self.table[other]
            return None

    def refusal(self, model):
        """(status, kind, message, retry_after, local) for a model this relay cannot price, or None when it has no
        better account than 'no price for MODEL'. Two causes are not about the model: a price file that cannot be
        read (the message names the file, never its path: this text reaches the agent) and a model list that has not
        loaded, which may be what prices the alias the call used. Both are a 503 the agent waits on, never the 403
        that stops its goal: a file missing at boot (the owner writes it later) and a file caught mid-rewrite are as
        transient as a list that has not loaded. Only a model that is not priced is asked at all (do_POST asks this
        after get() answered None), and the file is re-read on the next request, so the wait named is the agent's.

        A non-string model (a JSON list or object, a number) is not a model name and is refused as one: `model in
        self.names` raised TypeError for an unhashable one, out of the handler, and the client's connection dropped
        with no logged refusal (#389 review P2).

        A name the provider's list knows it does not carry is a typo whatever the file is doing, so it is refused at
        once even when the last refresh failed (a list that loaded once keeps its names). With the list never loaded
        there is nothing to compare against: the name waits with the file (the PR body states this trade-off)."""
        with self.lock:
            if not isinstance(model, str):
                return None                  # (the caller's own 403 names the model; do_POST guards every table lookup)
            if self.names and model not in self.names:
                # the file's absence says nothing about a NAME, the provider's own list does: a name that list does
                # not know is a typo, and waiting an hour on a 503 for it hides that. Refused at once (the caller's
                # 403 "no price for"), whatever the file is doing. A name it does know still waits below.
                return None
            if self.error:
                what = 'the relay cannot read its price file (%s)' % self.error
                tail = 'the owner re-reads it on change, no restart, so retry in %ds'
                if not self.table and self.error_at:
                    # the grace has passed: the last good table is gone. Say so, or the 503 reads as a file that
                    # was just caught mid-rewrite however long the relay has been refusing (#389 review P2).
                    what += ('; the last good table was dropped after %d s unreadable'
                             % PRICE_FILE_GRACE)
                    tail = 'nothing is priced until the file is back; retry in %ds'
                return _cannot_price(what, model, PRICE_FILE_RETRY, tail)
            if self.fetch and self.names_loading and not self.names and not self.names_error:
                # the first fetch is still in flight (it runs outside the lock, so get() answered None from the names
                # as they are): the name may be an alias of a priced model. A 503 with the short wait, as
                # ListPrices.refusal does, never the 403 that stops a goal (#389 review P2).
                return _cannot_price('the price list is still loading', model, LOAD_BACKOFF[0])
            if self.fetch and self.names_error and model not in self.names:
                wait = max(0, math.ceil(60 - (time.time() - self.names_at)))
                return _cannot_price('the price list has not loaded yet (%s), so the name cannot be resolved to a '
                                     'priced model' % self.names_error, model, wait)
            return None


def make_handler(provider, key, ledger, models, max_tokens, log, prices=None, connect=None, install_args=None):
    """install_args: the `relay install` flags that would rebuild this relay (cap, port, ...; models aside), so a refused
    model's error names the exact command that allows it as well"""
    spec = PROVIDERS[provider]
    # the Messages wire (its events, usage keys and error shape): Anthropic's relay, and an OpenCode plan's Messages path
    messages_wire = lambda path: provider == 'anthropic' or path.endswith('/messages') or path.endswith('/count_tokens')
    connect = connect or (lambda: http.client.HTTPSConnection(spec['host'], timeout=900,
                                                              context=ssl.create_default_context()))

    class Relay(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def refuse(self, status, kind, message, retry_after=None, local=False):
            err = dict(type='error', error=dict(type=kind, message='relay: ' + message)) \
                if messages_wire(self.path.split('?')[0]) else dict(error=dict(message='relay: ' + message, type=kind, code=status))
            raw = json.dumps(err).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            if retry_after is not None:
                self.send_header('Retry-After', str(retry_after))
            if local:
                # a 503 this relay made itself (a price it cannot know yet), never the provider being down: the
                # resident's fallback logic reads a bare 503 as an outage (brain.Brains.on_failure)
                self.send_header('X-Relay-Refusal', '1')
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
                if spec.get('session'):
                    headers['User-Agent'] = USER_AGENT
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
            anth = messages_wire(path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
            except (ValueError, TypeError):
                return self.refuse(400, 'invalid_request_error', 'body is not JSON')
            session = self.headers.get('x-opencode-session') or ''
            if spec.get('session') and not SESSION_ID.fullmatch(session):
                # Go serves only clients that name each conversation (its gateway answers 400 MissingSessionID too);
                # refused here, so it never reaches opencode: Ember's transport sends one per conversation
                return self.refuse(400, 'invalid_request_error', 'no x-opencode-session header (8 to 128 letters, digits, '
                                   '- or _): OpenCode Go serves only clients that send a session id per conversation, '
                                   "as Ember's transport does for a brain of kind opencode-go")
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
                row = None
            else:
                # (both the table lookup and refusal() read the model as a key: a JSON list or object in `model` is
                # a TypeError out of the handler, and the client's connection dropped with no refusal logged)
                rates = prices.get(model) if isinstance(model, str) else None
                if rates is None:
                    # a price table that could not be read, or a list that has not loaded, is not a fact about the
                    # model: it says so, and whether to wait, instead of the 'no price' that stops the agent's goal
                    why = getattr(prices, 'refusal', None)
                    better = why(model) if why and isinstance(model, str) else None
                    if better:
                        return self.refuse(*better)
                    return self.refuse(403, 'permission_error', 'no price for %r: it cannot be metered%s' % (
                        # (no file path: this text reaches the agent, which has no business with the owner's files)
                        model, ' (add it to the relay\'s --prices file as "%s": [input, output] USD per million tokens; '
                               're-read on change)' % model if provider == 'mistral' else
                        ' (Caminus has Meta\'s published prices for the Muse chat models in Standard mode only)'
                        if provider == 'meta' else
                        ' (Ember allows the chat models in the plan\'s list that call tools and do not train on your prompts)'
                        if spec.get('session') else ''))
                # (input, output, cache_read), or a PRICES-form row with a long-prompt tier (Meta's table)
                row = rates if isinstance(rates[0], tuple) else flat_row(*rates)
            spent = ledger.spent()
            if spent >= ledger.cap:
                return self.refuse(402, 'billing_error', 'daily cap $%.2f reached ($%.4f spent today)' % (ledger.cap, spent))
            counting = path.endswith('count_tokens')
            beta = None
            if not counting:
                if not anth and 'max_completion_tokens' in body:
                    # the newer name of the same cap (Meta takes both): folded into max_tokens, so the clamp below holds
                    newer = body.pop('max_completion_tokens')
                    if not isinstance(body.get('max_tokens'), int):
                        body['max_tokens'] = newer
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
                elif body.get('stream') and not anth and (provider in ('deepinfra', 'meta') or spec.get('session')):
                    body['stream_options'] = dict(include_usage=True)   # the only way their streams report usage
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
                if spec.get('session'):
                    headers.update({'x-opencode-session': session, 'User-Agent': USER_AGENT})
                    if anth:
                        headers['anthropic-version'] = OPENCODE_ANTHROPIC_VERSION   # (the version its gateway was checked at)
                t_sent, t_seen = time.monotonic(), None   # t_seen: the first visible output piece (hidden_reasoning)
                conn.request('POST', path, raw, headers)
                # (taken now: http.client drops conn.sock in getresponse when upstream will close the connection)
                upsock = getattr(conn, 'sock', None)
                up = conn.getresponse()
                # A client that hangs up (an interrupt, a steer and /advisor end a reply this way) never skips the
                # ledger, on any path: before the headers, mid-stream or on a whole reply. Upstream already billed it.
                client_gone, streamed, final = False, 0, False   # streamed: output characters sent; final: the count came
                try:
                    self.send_response(up.status)
                    for name in ('Content-Type', 'retry-after', 'request-id', 'x-request-id', 'x-opencode-log-id'):
                        if up.getheader(name):
                            self.send_header(name, up.getheader(name))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                except OSError:
                    client_gone = True
                usage, served, stop, category, fallback, errtext = {}, model, None, None, False, ''
                rid = up.getheader('request-id') or up.getheader('x-request-id') or up.getheader('x-opencode-log-id')
                cut, ended = None, False      # cut: an upstream read error after the headers (its name); ended below
                if 'text/event-stream' in (up.getheader('Content-Type') or ''):
                    watched = threading.Event()   # ended: the reply's last event came
                    gone = self.watch_client(upsock, watched)
                    while not client_gone:
                        try:
                            line = up.readline()
                        except (OSError, ValueError, http.client.HTTPException) as e:
                            # an upstream read error once the headers went out is the end of the stream (a chunked
                            # stream shut mid-chunk raises IncompleteRead, an HTTPException; a reset or the read timeout
                            # an OSError): it is booked below like any cut reply, an estimate where no usage came.
                            # Raising here skipped the ledger, though upstream billed what it had sent
                            if not gone.is_set():
                                cut = type(e).__name__
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
                        if not anth:
                            if ev.get('usage'):
                                usage = ev['usage']
                            for ch in ev.get('choices') or []:
                                d = (ch or {}).get('delta') or {}
                                streamed += len(d.get('content') or '') + len(d.get('reasoning_content') or '')
                                for tc in d.get('tool_calls') or []:
                                    streamed += len(((tc or {}).get('function') or {}).get('arguments') or '')
                            if streamed and t_seen is None:
                                t_seen = time.monotonic()
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
                    try:
                        data = up.read()
                    except (OSError, http.client.HTTPException) as e:
                        data, cut = getattr(e, 'partial', b'') or b'', type(e).__name__   # (booked as a cut reply)
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
                    # (an OpenCode reply's log id on every line: what opencode asks for when an issue is reported)
                    ' rid=%s' % rid if (rid and (errtext or stop == 'refusal' or fallback or spec.get('session'))) else ''))
                if client_gone:
                    extra += ' CLIENT-GONE'
                if cut:
                    extra += ' UPSTREAM-CUT (%s)' % cut
                # (a Messages stream that upstream cut, or any OpenCode one that ended early, is estimated the same way:
                #  nothing else says its output)
                if (client_gone or cut or spec.get('session')) and up.status == 200:
                    if anth and not final and 'text/event-stream' in (up.getheader('Content-Type') or ''):
                        if not usage:          # gone before message_start: the prompt was still read
                            usage = {'input_tokens': prompt_tokens(body, provider)}
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
                    # an estimate errs high at the price's peak-hour multiplier in force (OpenCode lists them; 1 else)
                    mult = prices.multiplier(model) if hasattr(prices, 'multiplier') else 1.0
                    plan = usage.get('opencode') if spec.get('session') and isinstance(usage.get('opencode'), dict) else None
                    # the reply's own charge (catalog.opencode_charge: finite, at least 0, a $0 only from the Go
                    # allowance or a free model), and only from a reply that ended: a stream's final usage event
                    charged = opencode_charge(usage) if spec.get('session') else None
                    streamed_reply = 'text/event-stream' in (up.getheader('Content-Type') or '')
                    finished = (final if anth else ended) if streamed_reply else not cut
                    if spec.get('session') and 'cost' in usage and (charged is None or not finished):
                        extra += ' (charge %r %s: list estimate)' % (
                            usage.get('cost'), 'refused' if charged is None else 'before the reply ended')
                        charged = None
                    if charged is not None:
                        # what the reply says it charged: 0 inside a Go allowance, dollars from the balance past it
                        usd = float(charged)
                        if plan:
                            extra += ' billing=%s allowance=%s list=%s' % (plan.get('billing_source'), plan.get('allowance'),
                                                                         plan.get('list_cost'))
                    elif usage:
                        usd = row_cost(row, usage) * mult
                        if spec.get('session'):
                            extra += ' (no charge in the reply: list price x%g)' % mult
                    elif client_gone and 'text/event-stream' in (up.getheader('Content-Type') or ''):
                        # a hang-up before the usage chunk: the prompt plus what was streamed, not the whole allowance
                        # (at the rates past a long-prompt line, where the row has one: an estimate errs high)
                        out_est = streamed * 2 // 7 + 1
                        if spec.get('hidden_reasoning'):
                            # its reasoning is billed but never streamed: the time before the first visible piece
                            # (prefill included, so it errs high) at a ceiling decode rate, so an interrupt during the
                            # reasoning is not booked as one token
                            hidden = ((t_seen or time.monotonic()) - t_sent) * HIDDEN_TPS
                            out_est += int(hidden)
                            extra += ' (+%d hidden-reasoning tokens by time)' % int(hidden)
                        usd = mult * (prompt_tokens(body, provider) * row[1][0]
                                      + min(out_est, body['max_tokens']) * row[1][1]) / 1e6
                        extra += ' (estimated: prompt + %d streamed characters%s)' % (
                            streamed, ', x%g peak' % mult if mult != 1 else '')
                    else:
                        # no usage reported (a cut stream): charge the request as if every output token was spent
                        usd = mult * (prompt_tokens(body, provider) * row[1][0] + body['max_tokens'] * row[1][1]) / 1e6
                        extra += ' NO-USAGE (charged the estimate)'
                    if anth:      # (an OpenCode plan's Messages usage: input_tokens beside its cache reads and writes)
                        read, wrote = usage.get('cache_read_input_tokens') or 0, usage.get('cache_creation_input_tokens') or 0
                        inp, out = (usage.get('input_tokens') or 0) + read + wrote, usage.get('output_tokens') or 0
                    else:
                        inp, out = usage.get('prompt_tokens') or 0, usage.get('completion_tokens') or 0
                        read, wrote = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0, 0
                    total = ledger.add(usd, inp, out, read, wrote, plan=plan)
                    log('%s in=%s out=%s $%.5f today $%.4f%s' % (model, inp, out, usd, total, extra))
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
        day = Ledger(ledger, 0).day() if ledger.exists() else {}
        line = f'{prov:11s} {active:14s} today ${day.get("usd", 0.0):.4f}'
        if 'allowance' in day:
            # an OpenCode plan: what the plan's allowance covered (in dollars of its monthly limit) is not money spent
            line += f' (Go allowance ${day["allowance"]:.4f}, list ${day.get("list_usd", 0.0):.4f})'
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
    ap.add_argument('--port', type=int, help='default 8765 (anthropic), 8766 (deepinfra), 8767 (mistral), 8768 (meta), '
                                             '8769 (opencode-go)')
    ap.add_argument('--prices', help='mistral: the owner\'s price file (default ~/.config/caminus/mistral-prices.json)')
    ap.add_argument('--cap-usd', type=float, default=1.0, help='daily spend cap in USD')
    ap.add_argument('--ttl', type=float, default=1800, help='seconds before the relay exits on its own; 0 = never')
    # a request's output cap; billing is per token written and the dollar caps hold, so 64K costs nothing unused
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--model', action='append', help='anthropic: allowed model (repeatable); default claude-haiku-5-5')
    ap.add_argument('--key-file', help='default ~/.config/anthropic/api_key, ~/.config/caminus/deepinfra-key, '
                                       '~/.config/caminus/mistral-key, ~/.config/caminus/meta-key or '
                                       '~/.config/caminus/opencode-key')
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
        sys.exit('--model is for anthropic: the other providers allow every model with a price (the cap is the guard)')
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
    listed = lambda: _list_models(spec, key)
    prices = (ListPrices(log, listed) if a.provider == 'deepinfra' else
              ListPrices(log, listed, opencode_rates) if spec.get('session') else
              OwnerPrices(a.prices or spec['prices'], log, fetch=listed)
              if a.provider == 'mistral' else META_PRICES if a.provider == 'meta' else None)
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
    if getattr(prices, 'prime', None):
        # load the price list off the request path: a relay that starts before the network does logs the failure and
        # retries on its schedule while the port answers, instead of refusing the first request for a model it prices
        prices.prime()
    root = '/v1/openai' if a.provider == 'deepinfra' else spec.get('root') or '/v1'
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
