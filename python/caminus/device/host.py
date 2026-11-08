"""The host's GPUs and where a model runs on them (D-292: nothing assumes which card is where).

`gpus(root)` lists every GPU the kernel shows (sysfs and the KFD topology only, never torch): the cards
caminus.device.machine reads for the guard, plus what placement needs:
- status: 'supported' (RX 7900 XTX / XT: gfx1100, tested), 'untested' (another gfx1100 card: allowed, with a request for a
  hardware report), 'unsupported' (anything else: listed, never used), with the reason in `why`;
- link: the slowest hop of the PCIe path from the root port to the GPU. Each device reports its own link, and an AMD card's
  endpoint sits behind a switch on the card itself, so the endpoint alone reads x16 even in an x4 slot (this machine's XT:
  endpoint 16 GT/s x16, its upstream port 16 GT/s x4, the OCuLink). `link_ok`: at least PCIe 4.0 x4, the tested minimum;
- hip: the card's HIP device index with HIP_VISIBLE_DEVICES unset (the KFD GPU nodes by node id, an iGPU included);
- pci: the canonical id of a card everywhere in Caminus (G8).

Placement (the rules the user settled, G1-G5):
- `pair(gpus)`: the two cards of a two-card model, in engine order. The bigger card is card 0 and takes the bigger share; on
  a tie the faster link goes first, then the PCI address. Only cards with an adequate link pair; three or more cards: the
  two first in that order, the rest stay free.
- `split(pair)`: XTX + XT keeps the tuned 10 / 12 split (the bitwise references); any other pair gets the even 8 / 10
  (the Coder's: 8 of its 16 key heads, 10 of its 20 intermediate blocks; the only engine that takes the placement split).
- `one(gpus, fits)`: a one-card model on the smallest usable card it fits, so the bigger card stays free.
- `hip_visible(cards)`: the HIP_VISIBLE_DEVICES string that makes process device i the i-th card given, so the engines keep
  addressing card 0 / card 1 by role.
A refusal is a PlacementError whose message names the card and the reason (the panel and doctor show it as is)."""
import glob
import os
import re

from . import machine

SUPPORTED = {'XTX': 'Radeon RX 7900 XTX', 'XT': 'Radeon RX 7900 XT'}
MIN_LINK_GBPS = 7.8         # PCIe 4.0 x4: 16 GT/s x 4 lanes x 128/130 / 8 = 7.88 GB/s (the XT's OCuLink, the tested minimum)
MIN_LINK_TEXT = 'PCIe 4.0 x4'
VENDORS = {'0x1002': 'AMD', '0x10de': 'NVIDIA', '0x8086': 'Intel'}
INTEGRATED_MIB = 4096       # an AMD GPU with less VRAM than this is an APU's carve-out (the desktop's), never a compute card
TUNED_SPLIT = {('XTX', 'XT'): dict(gdn_keys=10, moe_blocks=12)}   # measured on this pair (the bitwise references' split)
EVEN_SPLIT = dict(gdn_keys=8, moe_blocks=10)   # half of the Coder's 16 key heads and 20 blocks of 32 (640 / 32)


class PlacementError(RuntimeError):
    pass


def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _gts(s):
    m = re.match(r'\s*([0-9.]+)\s*GT/s', s or '')
    return float(m.group(1)) if m else None


def link_gbps(gts, width):
    """a PCIe link's payload bandwidth one way in GB/s (8b/10b below 8 GT/s, 128b/130b from 8 GT/s)"""
    return gts * width * (128 / 130 if gts >= 8 else 0.8) / 8


def pcie_gen(gts):
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(gts)


def link_text(lk):
    if not lk:
        return 'link unknown'
    g = pcie_gen(lk['gts'])
    return (f'PCIe {g}.0' if g else f"{lk['gts']:g} GT/s") + f" x{lk['width']}"


def _link_at(p):
    g, w = _gts(_read(os.path.join(p, 'current_link_speed'))), _int(_read(os.path.join(p, 'current_link_width')))
    if not g or not w:
        return None
    return dict(gts=g, width=w, gbps=round(link_gbps(g, w), 2), max_gts=_gts(_read(os.path.join(p, 'max_link_speed'))),
                max_width=_int(_read(os.path.join(p, 'max_link_width'))), at=os.path.basename(p))


def link(dev, root='/'):
    """the slowest hop (by bandwidth) of the PCIe path from the root port down to the GPU at sysfs path dev: dict(gts, width,
    gbps, max_gts, max_width, at), or None when no hop reports a link (a fixture without one: not refused, shown unknown)"""
    p = os.path.realpath(dev)
    top = os.path.realpath(os.path.join(root, 'sys', 'devices'))
    best = None
    while p.startswith(top + os.sep):
        lk = _link_at(p)
        if lk and (best is None or lk['gbps'] < best['gbps']):
            best = lk
        p = os.path.dirname(p)
    return best if best is not None else _link_at(os.path.realpath(dev))


def hip_order(root='/'):
    """{render minor: HIP device index} with HIP_VISIBLE_DEVICES unset: HIP numbers the KFD GPU nodes (simd_count > 0) by
    node id, an iGPU included"""
    nodes = []
    for p in glob.glob(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', '*', 'properties')):
        n = os.path.basename(os.path.dirname(p))
        if not n.isdigit():
            continue
        kv = {}
        for line in (_read(p) or '').splitlines():
            w = line.split()
            if len(w) == 2 and w[1].lstrip('-').isdigit():
                kv[w[0]] = int(w[1])
        if kv.get('simd_count', 0) > 0:
            nodes.append((int(n), kv.get('drm_render_minor')))
    return {minor: i for i, (_, minor) in enumerate(sorted(nodes)) if minor is not None}


def gfx_name(v):
    """gfx_target_version (major * 10000 + minor * 100 + stepping) -> 'gfx1100' / 'gfx1036' / 'gfx90a'"""
    if not v:
        return None
    return f'gfx{v // 10000}{(v // 100) % 100}{v % 100:x}'


def _status(c):
    if c['gfx'] == machine.GFX_TARGET:
        if c['label'] in SUPPORTED:
            return 'supported', SUPPORTED[c['label']]
        return 'untested', ('a gfx1100 card Caminus has not been tested on: it is allowed; please send a hardware report '
                            '(what worked, what did not)')
    if c['total_mib'] < INTEGRATED_MIB:
        return 'unsupported', 'integrated GPU: not used by Caminus'
    if c['gfx'] is None:
        return 'unsupported', 'no compute node in the KFD topology (the amdgpu compute stack does not see it)'
    return 'unsupported', f"{gfx_name(c['gfx'])}: Caminus runs on gfx1100 cards (RX 7900 XTX / XT) only"


def gpus(root='/'):
    """every GPU of the host (see the module docstring), the AMD cards in DRM order, then the other vendors' cards"""
    hip = hip_order(root)
    out = []
    for c in machine.cards(root):
        st, why = _status(c)
        lk = link(c['dev'], root)
        out.append(dict(c, vendor='AMD', status=st, why=why, integrated=c['total_mib'] < INTEGRATED_MIB, hip=hip.get(c['minor']),
                        link=lk, link_ok=lk is None or lk['gbps'] >= MIN_LINK_GBPS))
    seen = {c['card'] for c in out}
    drm = os.path.join(root, 'sys', 'class', 'drm')
    names = [n for n in os.listdir(drm) if re.fullmatch(r'card\d+', n)] if os.path.isdir(drm) else []
    for n in sorted(names, key=lambda s: int(s[4:])):
        dev = os.path.join(drm, n, 'device')
        vendor = _read(os.path.join(dev, 'vendor'))
        if n in seen or vendor in ('0x1002', None):
            continue
        pci = None
        for line in (_read(os.path.join(dev, 'uevent')) or '').splitlines():
            if line.startswith('PCI_SLOT_NAME='):
                pci = line.split('=', 1)[1]
        out.append(dict(card=n, dev=dev, pci=pci, label=n, total_mib=None, gfx=None, watched=False, minor=None, line_mib=None,
                        display=None, reserve_mib=None, vendor=VENDORS.get(vendor, vendor), status='unsupported',
                        why=f'a {VENDORS.get(vendor, vendor)} GPU: Caminus runs on AMD gfx1100 cards only', integrated=False,
                        hip=None, link=None, link_ok=None))
    return out


def usable(gs):
    return [g for g in gs if g['status'] in ('supported', 'untested')]


def _rank(g):
    return (-g['total_mib'], -(g['link']['gbps'] if g['link'] else 0.0), g['pci'] or g['card'])


def ranked(gs):
    """the usable cards in placement order: the biggest first, then the faster link, then the PCI address (brain / hands: the
    first is the brain's card)"""
    return sorted(usable(gs), key=_rank)


def describe(g):
    return (f"{g['label']} {g['card']}" if g['label'] != g['card'] else g['card']) + f" ({g['pci']})"


def pair(gs, held=()):
    """the two cards of a two-card model, card 0 first (the module docstring's rule); PlacementError names why not"""
    free = [g for g in usable(gs) if g['pci'] not in held]
    ok = sorted([g for g in free if g['link_ok']], key=_rank)
    if len(ok) >= 2:
        return ok[:2]
    bad = [g for g in free if not g['link_ok']]
    if len(free) >= 2 and bad:
        raise PlacementError('; '.join(f"{describe(g)} is on a {link_text(g['link'])} link" for g in bad)
                             + f': two-card models need {MIN_LINK_TEXT} or better (move the card to another slot)')
    n = len(usable(gs))
    if n >= 2:
        raise PlacementError(f'two-card models need two free cards: {n - len(free)} of {n} are in use')
    raise PlacementError(f'two-card models need two supported cards (RX 7900 XTX / XT); this host has {n}')


def split(cards):
    """the TP split of a pair in engine order: the tuned one for XTX + XT, else the even one"""
    return dict(TUNED_SPLIT.get(tuple(g['label'] for g in cards), EVEN_SPLIT))


def link_side(cards):
    """the engine index of the pair's card on the narrower link (the side the exchange pulls from; 1 on a tie)"""
    a, b = (c['link']['gbps'] if c['link'] else float('inf') for c in cards)
    return 0 if a < b else 1


def one(gs, fits, held=()):
    """the card of a one-card model: the smallest free usable card that fits(card) says it fits (ties: PCI address)"""
    free = sorted([g for g in usable(gs) if g['pci'] not in held], key=lambda g: (g['total_mib'], g['pci'] or g['card']))
    for g in free:
        if fits(g):
            return g
    if not usable(gs):
        raise PlacementError('no supported card (RX 7900 XTX / XT) on this host')
    if not free:
        raise PlacementError('every supported card is in use')
    raise PlacementError('the model fits none of the free cards: ' + ', '.join(f"{describe(g)} {g['total_mib']:,} MiB" for g in free))


def hip_visible(cards):
    """HIP_VISIBLE_DEVICES for these cards in this order (process device i = cards[i])"""
    if any(c['hip'] is None for c in cards):
        raise PlacementError('no HIP index for ' + ', '.join(describe(c) for c in cards if c['hip'] is None) + ' (no KFD node)')
    return ','.join(str(c['hip']) for c in cards)


def problems(gs):
    """the lines the panel and doctor show for the host's GPUs: (level, card or None, text); 'error' = Caminus cannot run
    anything (no supported card), 'warn' = a card is not used or a rule limits it, 'info' = a note"""
    out = []
    u = usable(gs)
    if not u:
        out.append(('error', None, 'no supported GPU: Caminus runs on AMD RX 7900 XTX / XT (gfx1100) cards'))
    for g in gs:
        if g['status'] == 'unsupported' and not g['integrated']:
            out.append(('warn', g['card'], f"{describe(g)}: not supported ({g['why']}); it is not used"))
        elif g['status'] == 'untested':
            out.append(('warn', g['card'], f"{describe(g)}: {g['why']}"))
        if g['status'] != 'unsupported' and not g['link_ok']:
            out.append(('warn', g['card'], f"{describe(g)} is on a {link_text(g['link'])} link: two-card models need "
                                           f"{MIN_LINK_TEXT} or better (it still runs one-card models)"))
    if len(u) == 1:
        out.append(('info', u[0]['card'], 'one supported card: the one-card models run; the two-card models need two'))
    return out


def place(n_cards, fits=None, env=None, root='/'):
    """the cards a server runs on, chosen before the GPU runtime starts (caminus serve): dict(cards, split, hip_visible,
    note). n_cards: the architecture's card count (1 or 2). fits(cards) -> bool: the model's projection at the minimum
    context on those cards (None: every card fits; the engine's own check still runs). env: HIP_VISIBLE_DEVICES, when set,
    limits the choice to those GPUs (HIP's own indices) and is rewritten so process device i is the engine's card i: a
    two-card model's pair is put in engine order whatever order it was given in. Unsupported cards are refused by name."""
    env = os.environ if env is None else env
    gs = gpus(root)
    given = (env.get('HIP_VISIBLE_DEVICES') or '').strip()
    pool = gs
    if given:
        try:
            idx = [int(x) for x in given.split(',') if x.strip()]
        except ValueError:
            raise PlacementError(f'HIP_VISIBLE_DEVICES={given!r}: not a list of HIP device indices') from None
        by_hip = {g['hip']: g for g in gs if g['hip'] is not None}
        miss = [i for i in idx if i not in by_hip]
        if miss:
            raise PlacementError(f'HIP_VISIBLE_DEVICES={given}: no GPU with HIP index {", ".join(map(str, miss))}')
        pool = [by_hip[i] for i in idx]
        bad = [g for g in pool if g['status'] == 'unsupported']
        if bad:
            raise PlacementError('; '.join(f"{describe(g)} is not supported ({g['why']})" for g in bad))
    ok = fits or (lambda cs: True)
    if given and n_cards >= 2 and len(usable(pool)) < 2:
        raise PlacementError(f'HIP_VISIBLE_DEVICES={given} gives {len(usable(pool))} supported card(s): a two-card model needs two')
    if n_cards >= 2:
        cards = pair(pool)
        sp = split(cards)
        if not ok(cards):                       # (a model may also raise PlacementError with its own reason: arch.Coder.fits)
            raise PlacementError(f"{' + '.join(describe(c) for c in cards)}: the model does not fit these two cards at its minimum "
                                 f"context (32,768 positions) with a working prompt cache, with what they hold now (caminus ps: another "
                                 f"server or program may be using them)")
    else:
        cards = [one(pool, lambda g: ok([g]))]
        sp = None
    return dict(cards=cards, split=sp, hip_visible=hip_visible(cards),
                note=' + '.join(describe(c) for c in cards) + (f", split {sp['gdn_keys']} / {sp['moe_blocks']}" if sp else ''))
