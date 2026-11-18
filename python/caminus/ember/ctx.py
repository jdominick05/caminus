"""The context budget (docs/specs/spec-ember.md section 10): which servers run on this machine and the context
window each reports in /props (Caminus sizes it to the cards — not always 262K). The agent reads the same /props
from inside its own user; `caminus ember ctx` prints it from the harness side."""
import json
import urllib.request

from .resident.catalog import props_window

BASE_PORTS = range(8092, 8102)


def servers(ports=BASE_PORTS, opener=urllib.request.urlopen):
    """[{port, model, n_ctx}] for every server answering on 127.0.0.1 (skips silently when none runs)"""
    out = []
    for port in ports:
        try:
            with opener(f'http://127.0.0.1:{port}/props', timeout=0.5) as r:
                p = json.load(r)
        except Exception:
            continue
        out.append({'port': port, 'model': p.get('model_alias') or p.get('model'), 'n_ctx': props_window(p)})
    return out
