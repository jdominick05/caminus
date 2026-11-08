"""CPU tests of caminus.device.host (D-292): the card inventory and placement on synthetic sysfs trees, one per host layout
the user asked to support (one card, two of any mix, three, unsupported cards, a slow link, headless, two displays, other
vendors), plus this machine read only. Nothing touches a GPU.

  python -I tests/device/test_host.py --scratch DIR      RESULT PASS"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
FAILS = []
GIB = 1 << 30


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


# a card: (drm name, pci slot, PCI device id, VRAM GiB, gfx_target_version, render minor, display, slot link (GT/s, width),
#          vendor). The card's own switch (upstream + downstream port, x16) sits between the slot and its endpoint, as on
#          a real RX 7900: the endpoint reads x16 whatever the slot is.
XTX = ('0x744c', 24, 110000)
XT = ('0x744c', 20, 110000)


def card(name, slot, kind, minor, display=False, link=(16.0, 16), vendor='0x1002'):
    did, gib, gfx = kind
    return (name, slot, did, gib, gfx, minor, display, link, vendor)


def tree(root, cards, kfd_extra=()):
    """sysfs under root: /sys/devices/pci0000:00/<root port>/<card upstream>/<card downstream>/<endpoint> with each hop's
    link, /sys/class/drm/cardN/device -> the endpoint, the KFD topology (CPU node 0, then one GPU node per card, in the order
    given, plus kfd_extra (render minor, gfx) nodes)"""
    if os.path.isdir(root): shutil.rmtree(root)
    node = 1
    w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', '0', 'properties'),
      'cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\ndrm_render_minor 0\n')
    for i, (name, slot, did, gib, gfx, minor, display, (gts, width), vendor) in enumerate(cards):
        bus = int(slot.split(':')[1], 16)
        rp = os.path.join(root, 'sys', 'devices', 'pci0000:00', f'0000:00:01.{i + 1}')
        up = os.path.join(rp, f'0000:{bus - 2:02x}:00.0'); dn = os.path.join(up, f'0000:{bus - 1:02x}:00.0'); ep = os.path.join(dn, slot)
        for p, (g, wd) in ((rp, (gts, width)), (up, (gts, width)), (dn, (16.0, 16)), (ep, (16.0, 16))):
            w(os.path.join(p, 'current_link_speed'), f'{g:.1f} GT/s PCIe\n'); w(os.path.join(p, 'current_link_width'), f'{wd}\n')
            w(os.path.join(p, 'max_link_speed'), '16.0 GT/s PCIe\n'); w(os.path.join(p, 'max_link_width'), '16\n')
        w(os.path.join(ep, 'vendor'), vendor + '\n'); w(os.path.join(ep, 'device'), did + '\n')
        w(os.path.join(ep, 'uevent'), f'DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n')
        if vendor == '0x1002':
            w(os.path.join(ep, 'mem_info_vram_total'), f'{int(gib * GIB)}\n'); w(os.path.join(ep, 'mem_info_vram_used'), f'{100 << 20}\n')
            w(os.path.join(ep, 'boot_vga'), '0\n')
            os.makedirs(os.path.join(ep, 'drm', f'renderD{minor}'), exist_ok=True)
        cd = os.path.join(root, 'sys', 'class', 'drm', name)
        os.makedirs(cd, exist_ok=True)
        os.symlink(ep, os.path.join(cd, 'device'))
        w(os.path.join(root, 'sys', 'class', 'drm', f'{name}-DP-1', 'status'), ('connected' if display else 'disconnected') + '\n')
        if vendor == '0x1002' and gfx:
            w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', str(node), 'properties'),
              f'cpu_cores_count 0\nsimd_count {96 if gib >= 4 else 4}\ngfx_target_version {gfx}\ndrm_render_minor {minor}\n')
            node += 1
    for minor, gfx in kfd_extra:
        w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', str(node), 'properties'),
          f'cpu_cores_count 0\nsimd_count 96\ngfx_target_version {gfx}\ndrm_render_minor {minor}\n')
        node += 1
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True, help='a scratch directory (fixture trees are written under it)')
    a = ap.parse_args()
    from caminus.device import host as H
    S = os.path.abspath(a.scratch); os.makedirs(S, exist_ok=True)
    IGPU = ('0x164e', 0.5, 100306)
    W7900 = ('0x7448', 48, 110000)
    RX7800 = ('0x7480', 16, 110001)

    def labels(cs): return [c['label'] for c in cs]
    def refuses(fn, *words):
        try: fn(); return False, 'no refusal'
        except H.PlacementError as e: return all(x in str(e) for x in words), str(e)

    # -- this machine's layout: XTX (display, x16), XT (x4 behind its own x16 switch), the iGPU ---------------------------
    r = tree(os.path.join(S, 'this'), [card('card1', '0000:03:00.0', XTX, 128, display=True), card('card2', '0000:06:00.0', XT, 129, link=(16.0, 4)),
                                       card('card3', '0000:76:00.0', IGPU, 130)])
    g = H.gpus(r)
    check([(c['label'], c['status'], c['hip']) for c in g] == [('XTX', 'supported', 0), ('XT', 'supported', 1), ('card3', 'unsupported', 2)],
          f'this layout: XTX / XT supported, the iGPU unsupported, HIP 0 / 1 / 2 ({[(c["label"], c["status"], c["hip"]) for c in g]})')
    check(g[1]['link']['width'] == 4 and g[1]['link']['at'] == '0000:04:00.0' and g[1]['link_ok'] and g[0]['link']['width'] == 16,
          f'this layout: the XT\'s link is the slot\'s x4 (its upstream hop), not the endpoint\'s x16 ({H.link_text(g[1]["link"])} at {g[1]["link"]["at"]})')
    p = H.pair(g)
    check(labels(p) == ['XTX', 'XT'] and H.split(p) == dict(gdn_keys=10, moe_blocks=12) and H.hip_visible(p) == '0,1' and H.link_side(p) == 1,
          'this layout: the pair XTX, XT at the tuned 10 / 12, HIP_VISIBLE_DEVICES 0,1, the exchange pulls on the XT')
    check(g[0]['display'] and not g[1]['display'] and g[2]['integrated'], 'this layout: the display on the XTX, the iGPU integrated')
    check(H.problems(g) == [], f'this layout: nothing to report ({H.problems(g)})')
    check(H.one(g, lambda c: c['total_mib'] >= 11 * 1024)['label'] == 'XT' and H.one(g, lambda c: c['total_mib'] >= 21 * 1024)['label'] == 'XTX',
          'this layout: a one-card model on the smallest card it fits (11 GiB -> XT, 21 GiB -> XTX)')
    check(H.one(g, lambda c: True, held={'0000:06:00.0'})['label'] == 'XTX', 'this layout: a held card is skipped')

    # -- the DRM order swapped (the XT is card1 / HIP 0): the engine order still puts the XTX first ------------------------
    r = tree(os.path.join(S, 'swapped'), [card('card1', '0000:03:00.0', XT, 128), card('card2', '0000:06:00.0', XTX, 129, display=True)])
    p = H.pair(H.gpus(r))
    check(labels(p) == ['XTX', 'XT'] and H.hip_visible(p) == '1,0' and H.split(p)['gdn_keys'] == 10,
          f'swapped DRM order: the pair XTX, XT, HIP_VISIBLE_DEVICES 1,0 ({H.hip_visible(p)})')

    # -- one card each ------------------------------------------------------------------------------------------------------
    for k, lab in ((XT, 'XT'), (XTX, 'XTX')):
        r = tree(os.path.join(S, 'one' + lab), [card('card0', '0000:03:00.0', k, 128)])
        g = H.gpus(r)
        ok, msg = refuses(lambda: H.pair(g), 'two supported cards', 'has 1')
        check(ok, f'one {lab}: a two-card model refused ({msg})')
        check(H.one(g, lambda c: True)['label'] == lab and H.problems(g) == [('info', 'card0', 'one supported card: the one-card models run; the two-card models need two')],
              f'one {lab}: a one-card model runs; the panel notes it')

    # -- equal pairs: the even split, the faster link first, then the PCI address ------------------------------------------
    r = tree(os.path.join(S, 'xtxt'), [card('card1', '0000:03:00.0', XT, 128, link=(16.0, 8)), card('card2', '0000:06:00.0', XT, 129)])
    p = H.pair(H.gpus(r))
    check(labels(p) == ['XT', 'XT'] and p[0]['pci'] == '0000:06:00.0' and H.split(p) == H.EVEN_SPLIT == dict(gdn_keys=8, moe_blocks=10) and H.link_side(p) == 1,
          f'XT + XT: the even 8 / 10 split, the x16 card first ({[c["pci"] for c in p]})')
    r = tree(os.path.join(S, 'xtxxtx'), [card('card1', '0000:06:00.0', XTX, 128), card('card2', '0000:03:00.0', XTX, 129)])
    p = H.pair(H.gpus(r))
    check(labels(p) == ['XTX', 'XTX'] and p[0]['pci'] == '0000:03:00.0' and H.split(p)['moe_blocks'] == 10,
          'XTX + XTX: the even split, a tie broken by the PCI address')

    # -- three cards: the two first in rank, the third free for a one-card server ------------------------------------------
    r = tree(os.path.join(S, 'three'), [card('card1', '0000:03:00.0', XT, 128), card('card2', '0000:06:00.0', XTX, 129), card('card3', '0000:09:00.0', XT, 130)])
    g = H.gpus(r); p = H.pair(g)
    check(labels(p) == ['XTX', 'XT'] and p[1]['pci'] == '0000:03:00.0', f'three cards: the pair XTX + the first XT ({[c["pci"] for c in p]})')
    check(H.one(g, lambda c: True, held={c['pci'] for c in p})['pci'] == '0000:09:00.0', 'three cards: the third hosts a one-card model')

    # -- a slow link: two-card models refused, naming the card and the rule; one-card models still run ---------------------
    r = tree(os.path.join(S, 'x1'), [card('card1', '0000:03:00.0', XTX, 128), card('card2', '0000:06:00.0', XT, 129, link=(8.0, 1))])
    g = H.gpus(r)
    ok, msg = refuses(lambda: H.pair(g), 'XT card2 (0000:06:00.0)', 'PCIe 3.0 x1', 'PCIe 4.0 x4 or better')
    check(ok, f'an x1 riser: the pair refused ({msg})')
    check(any(lv == 'warn' and 'PCIe 3.0 x1' in t for lv, _, t in H.problems(g)) and H.one(g, lambda c: c['total_mib'] < 22000)['label'] == 'XT',
          'an x1 riser: the panel warns; a one-card model still runs on that card')
    r = tree(os.path.join(S, 'x1three'), [card('card1', '0000:03:00.0', XTX, 128), card('card2', '0000:06:00.0', XT, 129, link=(8.0, 1)),
                                          card('card3', '0000:09:00.0', XT, 130)])
    check(labels(H.pair(H.gpus(r))) == ['XTX', 'XT'] and H.pair(H.gpus(r))[1]['pci'] == '0000:09:00.0', 'three cards, one on x1: the pair skips it')

    # -- unsupported cards: listed with the reason, never used; no supported card is an error ------------------------------
    r = tree(os.path.join(S, 'mixed'), [card('card1', '0000:03:00.0', RX7800, 128, display=True), card('card2', '0000:06:00.0', XTX, 129)])
    g = H.gpus(r)
    check(g[0]['status'] == 'unsupported' and 'gfx1101' in g[0]['why'] and [c['label'] for c in H.usable(g)] == ['XTX'],
          f'a 7800 XT (gfx1101) + an XTX: the 7800 XT unsupported ({g[0]["why"]})')
    ok, msg = refuses(lambda: H.pair(g), 'two supported cards', 'has 1')
    check(ok and any(lv == 'warn' and 'gfx1101' in t for lv, _, t in H.problems(g)), f'a 7800 XT + an XTX: two-card refused, the panel warns ({msg})')
    r = tree(os.path.join(S, 'nvidia'), [card('card0', '0000:01:00.0', ('0x2684', 24, None), 0, display=True, vendor='0x10de'),
                                         card('card1', '0000:03:00.0', XTX, 128)])
    g = H.gpus(r)
    check([(c['card'], c['vendor'], c['status']) for c in g] == [('card1', 'AMD', 'supported'), ('card0', 'NVIDIA', 'unsupported')]
          and any('NVIDIA' in t for _, _, t in H.problems(g)), 'an NVIDIA card: listed, unsupported, the panel warns')
    r = tree(os.path.join(S, 'none'), [card('card0', '0000:76:00.0', IGPU, 128)])
    g = H.gpus(r)
    check(H.problems(g)[0][0] == 'error' and not H.usable(g), f'only an iGPU: the error line ({H.problems(g)[0][2]})')
    ok, msg = refuses(lambda: H.one(g, lambda c: True), 'no supported card')
    check(ok, f'only an iGPU: a one-card model refused ({msg})')

    # -- an untested gfx1100 card: allowed, with the hardware-report request ----------------------------------------------
    r = tree(os.path.join(S, 'w7900'), [card('card1', '0000:03:00.0', XTX, 128), card('card2', '0000:06:00.0', W7900, 129)])
    g = H.gpus(r); p = H.pair(g)
    check(g[1]['status'] == 'untested' and labels(p) == ['card2', 'XTX'] and H.split(p) == H.EVEN_SPLIT == dict(gdn_keys=8, moe_blocks=10)
          and any(lv == 'warn' and 'hardware report' in t for lv, _, t in H.problems(g)),
          'a W7900 (48 GB gfx1100) + an XTX: untested but used, the W7900 first, the even split, the report request shown')

    # -- headless and two displays: the flags follow the connectors --------------------------------------------------------
    r = tree(os.path.join(S, 'headless'), [card('card1', '0000:03:00.0', XTX, 128), card('card2', '0000:06:00.0', XT, 129)])
    g = H.gpus(r)
    check(not any(c['display'] for c in g) and [c['reserve_mib'] for c in g] == [0, 0], 'headless: no display card, no reserve')
    r = tree(os.path.join(S, 'twodisp'), [card('card1', '0000:03:00.0', XTX, 128, display=True), card('card2', '0000:06:00.0', XT, 129, display=True)])
    g = H.gpus(r)
    check(all(c['display'] for c in g) and [c['reserve_mib'] for c in g] == [512, 512], 'two displays: both cards keep the reserve')

    # -- a tree without /sys/devices (the older fixtures): the endpoint's own link, or none (unknown, not refused) ---------
    from caminus.device import machine
    r = os.path.join(S, 'flat')
    if os.path.isdir(r): shutil.rmtree(r)
    for n, slot, did, gib, minor in (('card1', '0000:03:00.0', '0x744c', 24, 128), ('card2', '0000:06:00.0', '0x744c', 20, 129)):
        d = os.path.join(r, 'sys', 'class', 'drm', n, 'device')
        w(os.path.join(d, 'mem_info_vram_total'), f'{gib * GIB}\n'); w(os.path.join(d, 'vendor'), '0x1002\n'); w(os.path.join(d, 'device'), did + '\n')
        w(os.path.join(d, 'uevent'), f'PCI_SLOT_NAME={slot}\n'); os.makedirs(os.path.join(d, 'drm', f'renderD{minor}'), exist_ok=True)
        w(os.path.join(r, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', str(minor - 127), 'properties'),
          f'simd_count 96\ngfx_target_version 110000\ndrm_render_minor {minor}\n')
    w(os.path.join(r, 'sys', 'class', 'drm', 'card2', 'device', 'current_link_speed'), '16.0 GT/s PCIe\n')
    w(os.path.join(r, 'sys', 'class', 'drm', 'card2', 'device', 'current_link_width'), '4\n')
    g = H.gpus(r)
    check(g[0]['link'] is None and g[0]['link_ok'] and g[1]['link']['width'] == 4 and labels(H.pair(g)) == ['XTX', 'XT'],
          'a flat tree: no link file = unknown (not refused), the endpoint\'s own file read')
    check([c['card'] for c in machine.watched(r)] == ['card1', 'card2'], 'machine.watched unchanged on a flat tree')

    # -- place(): caminus serve's choice before HIP starts; a given HIP_VISIBLE_DEVICES limits it and is put in engine order --
    r = tree(os.path.join(S, 'place'), [card('card1', '0000:03:00.0', XTX, 128, display=True), card('card2', '0000:06:00.0', XT, 129, link=(16.0, 4)),
                                        card('card3', '0000:76:00.0', IGPU, 130)])
    P = lambda n, env, fits=None: H.place(n, fits=fits, env=env, root=r)
    check(P(2, {})['hip_visible'] == '0,1' and P(2, {})['split']['gdn_keys'] == 10, 'place: a two-card model, nothing given: 0,1 at 10 / 12')
    check(P(2, {'HIP_VISIBLE_DEVICES': '1,0'})['hip_visible'] == '0,1', 'place: HIP_VISIBLE_DEVICES=1,0 is rewritten into engine order (0,1)')
    check(P(1, {'HIP_VISIBLE_DEVICES': '1'})['hip_visible'] == '1' and P(1, {})['hip_visible'] == '1',
          'place: a one-card model on the given card, or on the smallest card it fits (the XT)')
    check(P(1, {}, fits=lambda cs: cs[0]['total_mib'] >= 21 * 1024)['hip_visible'] == '0', 'place: a model that fits only the XTX goes there')
    for env, words, what in (({'HIP_VISIBLE_DEVICES': '2'}, ('card3 (0000:76:00.0) is not supported', 'integrated'), 'the iGPU given'),
                             ({'HIP_VISIBLE_DEVICES': '5'}, ('no GPU with HIP index 5',), 'an index that does not exist'),
                             ({'HIP_VISIBLE_DEVICES': 'x'}, ('not a list',), 'a malformed list'),
                             ({'HIP_VISIBLE_DEVICES': '1'}, ('HIP_VISIBLE_DEVICES=1 gives 1', 'needs two'), 'one card given to a two-card model')):
        ok, msg = refuses(lambda: P(2 if 'two-card' in what else 1, env), *words)
        check(ok, f'place: refused: {what} ({msg})')
    rs = tree(os.path.join(S, 'place_sw'), [card('card1', '0000:03:00.0', XT, 128), card('card2', '0000:06:00.0', XTX, 129, display=True)])
    check(H.place(2, env={}, root=rs)['hip_visible'] == '1,0', 'place: the swapped layout puts the XTX (HIP 1) first')
    ok, msg = refuses(lambda: H.place(1, fits=lambda cs: False, env={}, root=r), 'fits none of the free cards')
    check(ok, f'place: a model too big for every card refused ({msg})')

    # -- gfx names ----------------------------------------------------------------------------------------------------------
    check([H.gfx_name(v) for v in (110000, 110001, 100306, 90010)] == ['gfx1100', 'gfx1101', 'gfx1036', 'gfx90a'], 'gfx names')

    # -- this machine, read only ------------------------------------------------------------------------------------------
    live = H.gpus('/')
    print('NOTE this machine: ' + '; '.join(f"{c['label']} {c['card']} {c['pci']} {c['status']} hip {c['hip']} {H.link_text(c['link'])}"
                                             for c in live), flush=True)
    if [c['label'] for c in H.usable(live)] == ['XTX', 'XT']:
        p = H.pair(live)
        check(labels(p) == ['XTX', 'XT'] and H.split(p)['gdn_keys'] == 10, 'this machine: the pair XTX, XT at 10 / 12')
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'), flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
