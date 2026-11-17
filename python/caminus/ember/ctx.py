"""The context budget (docs/specs/spec-ember.md section 10): which servers run on this machine and the context
window each reports in /props (Caminus sizes it to the cards — not always 262K). The agent reads the same /props
from inside its own user; `caminus ember ctx` prints it from the harness side."""
import json
import urllib.request

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
        n_ctx = p.get('n_ctx')
        if n_ctx is None:
            n_ctx = (p.get('default_generation_settings') or {}).get('n_ctx')
        out.append({'port': port, 'model': p.get('model_alias') or p.get('model'), 'n_ctx': n_ctx})
    return out
