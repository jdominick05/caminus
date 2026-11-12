"""The GPUs the guard watches and their lines, read from sysfs (docs/dev/packaging.md, "Guard"). caminus/share/guard.sh computes
the same in bash (`guard.sh --lines [ROOT]` prints it); caminus.device.capacity reads it from here, so the runtime's VRAM checks and the
guard's kill lines agree by construction (tests/packaging/test_packaging.py compares both on fixture trees and on this machine).

Watched: every amdgpu DRM card whose KFD node (matched by its render minor) is gfx1100 (gfx_target_version 110000, the target
the package ships), in DRM order. An iGPU or another target is not watched (its VRAM is a small carve-out the desktop uses: a
0.97 line would kill for nothing).
Kill line (MiB of sysfs mem_info_vram_used): floor(floor(total_MiB * 97 / 100) / 50) * 50, CLAUDE.md's 0.97 rounded down to
50 MiB: 24,560 -> 23,800 (the RX 7900 XTX), 20,464 -> 19,850 (the RX 7900 XT), the old fixed lines exactly.
Display-driving: a connected connector (cardN-*/status == 'connected') or boot_vga == 1; such a card keeps the prompt cache's
512 MiB reserve below its line (capacity.DISPLAY_RESERVE_MIB), the others the usual margin. Nothing here needs torch."""
import glob
import os
import re

GFX_TARGET = 110000         # gfx1100
DISPLAY_RESERVE_MIB = 512   # kept free below the line of a card that drives a display (the desktop's own VRAM grows)
CAP_MARGIN_MIB = 256        # kept free below every card's guard line (caminus.device.capacity.MARGIN_MIB: allocator rounding,
                            # HIP's own growth); here so that a process without torch applies the same cap
NAMES = {(0x744c, 23): 'XTX', (0x744c, 24): 'XTX', (0x744c, 19): 'XT', (0x744c, 20): 'XT'}   # (PCI device, total GiB) -> label


def _read(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def line_mib(total_mib):
    """the guard's kill line for a card of total_mib MiB (integers, as guard.sh computes it)"""
    return (int(total_mib) * 97 // 100) // 50 * 50


def _kfd_targets(root):
    """{render minor: gfx_target_version} from the KFD topology"""
    out = {}
    for p in glob.glob(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'topology', 'nodes', '*', 'properties')):
        kv = {}
        for line in (_read(p) or '').splitlines():
            w = line.split()
            if len(w) == 2 and w[1].lstrip('-').isdigit():
                kv[w[0]] = int(w[1])
        if kv.get('drm_render_minor') and kv.get('gfx_target_version'):
            out[kv['drm_render_minor']] = kv['gfx_target_version']
    return out


def cards(root='/'):
    """every amdgpu DRM card under root: dicts (card, dev, pci, label, total_mib, gfx, watched, minor, line_mib, display, reserve_mib)
    in DRM order. line_mib / reserve_mib are None for a card the guard does not watch"""
    tgt = _kfd_targets(root)
    out = []
    drm = os.path.join(root, 'sys', 'class', 'drm')
    names = [n for n in os.listdir(drm) if re.fullmatch(r'card\d+', n)] if os.path.isdir(drm) else []
    for n in sorted(names, key=lambda s: int(s[4:])):
        dev = os.path.join(drm, n, 'device')
        total = _read(os.path.join(dev, 'mem_info_vram_total'))
        if total is None or _read(os.path.join(dev, 'vendor')) not in ('0x1002', None):
            continue
        total_mib = int(total) >> 20
        minor = None
        for r in glob.glob(os.path.join(dev, 'drm', 'renderD*')):
            m = re.search(r'renderD(\d+)$', r)
            if m: minor = int(m.group(1))
        gfx = tgt.get(minor)
        pci = None
        for line in (_read(os.path.join(dev, 'uevent')) or '').splitlines():
            if line.startswith('PCI_SLOT_NAME='):
                pci = line.split('=', 1)[1]
        conn = any(_read(s) == 'connected' for s in glob.glob(os.path.join(drm, n + '-*', 'status')))
        display = conn or _read(os.path.join(dev, 'boot_vga')) == '1'
        did = _read(os.path.join(dev, 'device'))
        label = NAMES.get((int(did, 16) if did else None, total_mib // 1024), n)
        watched = gfx == GFX_TARGET
        out.append(dict(card=n, dev=dev, pci=pci, label=label, total_mib=total_mib, gfx=gfx, watched=watched, minor=minor,
                        line_mib=line_mib(total_mib) if watched else None, display=display,
                        reserve_mib=(DISPLAY_RESERVE_MIB if display else 0) if watched else None))
    return out


def watched(root='/'):
    return [c for c in cards(root) if c['watched']]


def guard_lines(root='/'):
    """{DRM card name: kill line MiB} of the watched cards (capacity.GUARD_LINE_MIB)"""
    return {c['card']: c['line_mib'] for c in watched(root)}


def by_card(card_dev, root='/'):
    """the cards() entry of a /sys/class/drm/cardN/device path (None when not found)"""
    if card_dev is None:
        return None
    n = os.path.basename(os.path.dirname(card_dev.rstrip('/')))
    for c in cards(root):
        if c['card'] == n:
            return c
    return None
