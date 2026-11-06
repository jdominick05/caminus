"""Hardware readings for the metrics (docs/agentic/spec-metrics.md, 'TP / hardware'): VRAM, power, clocks and their
state per card from amdgpu's sysfs, read-only, under a root directory so tests run on fixture trees (write_fixture).

    HwReader(root='/', cards=None, min_interval=1.0, phase=None)
        cards: PCI slot names in engine device order ('0000:03:00.0', ...); None = every amdgpu card with VRAM of at
        least 4 GiB and a hwmon power1_average (the iGPU has neither), the larger VRAM first (the XTX, device 0, has 24
        GiB; the XT 20 GiB: both are PCI id 1002:744c, so the id cannot tell them apart).
    read()      -> [per-card dict]; cached for min_interval seconds (amdgpu answers power and clocks from the SMU)
    sample()    one reading now, and the energy integral: per card, joules += power x dt (trapezoid between readings),
                attributed to the engine phase at the reading (phase(): 'idle' | 'prefill' | 'decode')
    start(interval) / stop()
                a daemon sampler thread (the server's; never the engine thread)

Files per card (/sys/class/drm/cardN/device): mem_info_vram_used / _total (bytes), gpu_busy_percent,
power_dpm_force_performance_level ('high' is what gpu-high sets), pp_dpm_sclk / pp_dpm_mclk (the '*' line),
current_link_speed / width, uevent (PCI_SLOT_NAME), power/control and power/runtime_status (the PCI device's runtime PM:
'on' never suspends, 'auto' may), hwmon/hwmonN/{power1_average (uW), freq1_input (sclk Hz), freq2_input (mclk Hz),
temp1_input / temp2_input (edge / junction, m°C)}. A missing or unreadable file is None.

Host counters (M7 ops; read by the sampler thread, host only): this process's KFD queue evictions
(/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms per node, caminus.passpolice.KfdEvictions: pread at offset 0) and the
system's page migrations (/proc/vmstat pgmigrate_success: system-wide, not per process). Each sample hands the rises since
the previous one to on_host(evicted_ms, pgmigrate, busy) (the server's EngineStats.host_sample): the eviction as the largest
per-node rise (one eviction stops every node at once: passpolice's rule, never the sum), busy = the engine phase was not
idle at either end of the interval.

clock_check(cards): the startup check of the clock state (reported, never changed): per card the DPM level and the runtime
PM control; 'high' with runtime PM 'on' is what gpu-high sets (tools/guard.sh runs every command under it). At
'auto' decode is ~13 % slower; 'high' with runtime PM 'auto' is not held: after a runtime suspend / resume the SMU runs at
auto clocks while the level still reads 'high' (the gpu-high note)."""
import glob
import os
import threading
import time

from ..passpolice import KfdEvictions

GIB = 1 << 30
MIB = 1 << 20
PHASES = ('idle', 'prefill', 'decode')


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _num(path, scale=1.0):
    s = _read(path)
    if s is None:
        return None
    try:
        return float(s.split()[0]) * scale
    except (ValueError, IndexError):
        return None


def _dpm_current(path):
    """the MHz of the '*' line of pp_dpm_sclk / pp_dpm_mclk ('1: 2304Mhz *')"""
    s = _read(path)
    if not s:
        return None
    for line in s.splitlines():
        if line.rstrip().endswith('*'):
            for tok in line.split():
                t = tok.lower()
                if t.endswith('mhz'):
                    try:
                        return float(t[:-3])
                    except ValueError:
                        return None
    return None


class Card:
    __slots__ = ('index', 'name', 'dev', 'hwmon', 'pci', 'vram_total')

    def __init__(self, index, name, dev, hwmon, pci, vram_total):
        self.index, self.name, self.dev, self.hwmon, self.pci, self.vram_total = index, name, dev, hwmon, pci, vram_total


def discover(root='/', pci=None, names=None):
    """the engine's cards under root (see the module docstring)"""
    found = []
    for dev in sorted(glob.glob(os.path.join(root, 'sys', 'class', 'drm', 'card[0-9]*', 'device'))):
        if not os.path.exists(os.path.join(dev, 'mem_info_vram_total')):
            continue
        slot = None
        ue = _read(os.path.join(dev, 'uevent')) or ''
        for line in ue.splitlines():
            if line.startswith('PCI_SLOT_NAME='):
                slot = line.split('=', 1)[1].strip()
        if slot is not None and any(slot == c[4] for c in found):
            continue            # one device, several card nodes
        hw = sorted(glob.glob(os.path.join(dev, 'hwmon', 'hwmon*')))
        hw = hw[0] if hw else None
        total = _num(os.path.join(dev, 'mem_info_vram_total')) or 0
        found.append((total, dev, hw, hw is not None and os.path.exists(os.path.join(hw, 'power1_average')), slot))
    if pci is not None:
        by = {c[4]: c for c in found}
        pick = [by[p] for p in pci if p in by]
    else:
        pick = [c for c in found if c[0] >= 4 * GIB and c[3]]
        pick.sort(key=lambda c: -c[0])
    names = names or ()
    return [Card(i, names[i] if i < len(names) else f'card{i}', c[1], c[2], c[4], c[0]) for i, c in enumerate(pick)]


class HostCounters:
    """this process's KFD evictions per node and the system's page migrations, as rises between samples"""
    def __init__(self, root='/', pid=None):
        self.kfd = KfdEvictions(pid, root=os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'proc'))
        self.vmstat = os.path.join(root, 'proc', 'vmstat')
        self._prev = None           # (evicted per node, pgmigrate)

    def read_pgmigrate(self):
        try:
            with open(self.vmstat) as f:
                for line in f:
                    if line.startswith('pgmigrate_success '):
                        return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            return None
        return None

    def sample(self):
        """(evicted ms since the previous sample: the largest per-node rise, pgmigrate rise); None for a value that cannot
        be read (no KFD proc directory yet, no /proc/vmstat) or on the first sample"""
        ev, pg = self.kfd.sample(), self.read_pgmigrate()
        prev, self._prev = self._prev, (ev, pg)
        if prev is None:
            return None, None
        d_ev = None
        if ev is not None and prev[0] is not None and len(ev) == len(prev[0]):
            d_ev = max([max(0, b - a) for a, b in zip(prev[0], ev)] or [0])
        d_pg = None if pg is None or prev[1] is None else max(0, pg - prev[1])
        return d_ev, d_pg


class HwReader:
    def __init__(self, root='/', cards=None, names=('XTX', 'XT'), min_interval=1.0, clock=time.monotonic,
                 phase=None, host=True, pid=None, on_host=None):
        self.root = root
        self.cards = discover(root, cards, names)
        self.min_interval = min_interval
        self.clock = clock
        self.phase = phase or (lambda: 'idle')
        self.joules = [dict.fromkeys(PHASES, 0.0) for _ in self.cards]
        self.samples = 0
        self._last = None           # (t, [card dicts])
        self._prev_power = None     # (t, [watts])
        self._prev_phase = 'idle'
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.host = HostCounters(root, pid) if host else None
        self.on_host = on_host      # on_host(evicted_ms, pgmigrate, busy) per sample (the server's EngineStats.host_sample)

    def read_card(self, c):
        d, h = c.dev, c.hwmon
        used = _num(os.path.join(d, 'mem_info_vram_used'))
        total = _num(os.path.join(d, 'mem_info_vram_total'))
        level = _read(os.path.join(d, 'power_dpm_force_performance_level'))
        sclk = _num(os.path.join(h, 'freq1_input'), 1e-6) if h else None
        if sclk is None:
            sclk = _dpm_current(os.path.join(d, 'pp_dpm_sclk'))
        mclk = _num(os.path.join(h, 'freq2_input'), 1e-6) if h else None
        if mclk is None:
            mclk = _dpm_current(os.path.join(d, 'pp_dpm_mclk'))
        power = _num(os.path.join(h, 'power1_average'), 1e-6) if h else None
        if power is None and h:
            power = _num(os.path.join(h, 'power1_input'), 1e-6)
        return {'index': c.index, 'name': c.name, 'pci': c.pci,
                'vram_used_mib': None if used is None else round(used / MIB, 1),
                'vram_total_mib': None if total is None else round(total / MIB, 1),
                'power_w': None if power is None else round(power, 3),
                'sclk_mhz': None if sclk is None else round(sclk, 1),
                'mclk_mhz': None if mclk is None else round(mclk, 1),
                'perf_level': level, 'forced_high': None if level is None else level == 'high',
                'busy_pct': _num(os.path.join(d, 'gpu_busy_percent')),
                'temp_edge_c': _num(os.path.join(h, 'temp1_input'), 1e-3) if h else None,
                'temp_junction_c': _num(os.path.join(h, 'temp2_input'), 1e-3) if h else None,
                'link_speed': _read(os.path.join(d, 'current_link_speed')),
                'link_width': _read(os.path.join(d, 'current_link_width')),
                'runtime_pm': _read(os.path.join(d, 'power', 'control')),
                'runtime_status': _read(os.path.join(d, 'power', 'runtime_status'))}

    def sample(self):
        """one reading now; integrates energy since the previous one; the host counters' rises go to on_host"""
        t = self.clock()
        cards = [self.read_card(c) for c in self.cards]
        ph = self.phase()
        ph = ph if ph in PHASES else 'idle'
        if self.host is not None:
            try:
                ev, pg = self.host.sample()
            except Exception:           # noqa: BLE001 - a host reading never fails a sample
                ev = pg = None
            busy = ph != 'idle' or self._prev_phase != 'idle'
            self._prev_phase = ph
            cb = self.on_host
            if cb is not None and (ev is not None or pg is not None):
                cb(ev, pg, busy)
        with self._lock:
            watts = [c['power_w'] for c in cards]
            if self._prev_power is not None:
                t0, w0 = self._prev_power
                dt = t - t0
                if 0 < dt < 30:             # a gap (a stopped sampler) is not integrated
                    for i, (a, b) in enumerate(zip(w0, watts)):
                        if a is not None and b is not None:
                            self.joules[i][ph] += (a + b) / 2 * dt
            self._prev_power = (t, watts)
            self._last = (t, cards)
            self.samples += 1
        return cards

    def read(self):
        """the latest reading, refreshed when older than min_interval (a sampler thread keeps it fresh)"""
        last = self._last
        if last is None or (self._thread is None and self.clock() - last[0] >= self.min_interval):
            return self.sample()
        return last[1]

    def energy(self):
        with self._lock:
            return [dict(j) for j in self.joules]

    def start(self, interval=1.0):
        if self._thread is not None or not self.cards:
            return self

        def run():
            while not self._stop.wait(interval):
                try:
                    self.sample()
                except Exception:       # noqa: BLE001 - a reading must never take the server down
                    pass
        self.sample()
        self._thread = threading.Thread(target=run, name='caminus-hw', daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def clock_check(self):
        """the clock state now (a fresh reading), reported, never changed: clock_check() of the module"""
        return clock_check(self.sample() if self.cards else [])


def clock_check(cards):
    """the clock / runtime-suspend check of the engine's cards (read only). Per card: ok when the DPM level is 'high' and
    runtime PM is 'on' (what gpu-high sets; tools/guard.sh runs every command under it). Notes: 'auto' (decode ~13 %
    slower: the bench note); 'high' with runtime PM 'auto' (a runtime suspend / resume drops the SMU to auto clocks while the
    level still reads 'high': not held); a suspended card; an unreadable file (unknown, not ok). Returns {ok, cards: [...],
    summary}; ok is None without cards"""
    out = []
    for c in cards:
        lvl, pm, rs = c.get('perf_level'), c.get('runtime_pm'), c.get('runtime_status')
        notes = []
        if lvl is None:
            notes.append('power_dpm_force_performance_level unreadable')
        elif lvl != 'high':
            notes.append(f"DPM level '{lvl}', not 'high': decode is ~13 % slower at auto (run under tools/guard.sh, which wraps "
                         f'gpu-high, or set the level to high)')
        if pm is None:
            notes.append('runtime PM control (power/control) unreadable')
        elif pm != 'on':
            notes.append(f"runtime PM '{pm}': the card may runtime-suspend" + (", and after a resume the SMU runs at auto clocks while "
                         "the level still reads 'high' (gpu-high sets power/control=on)" if lvl == 'high' else ''))
        if rs == 'suspended':
            notes.append('the card is runtime-suspended now')
        ok = lvl == 'high' and pm == 'on' and rs != 'suspended'
        out.append({'index': c.get('index'), 'name': c.get('name'), 'pci': c.get('pci'), 'perf_level': lvl, 'forced_high': lvl == 'high' if lvl else None,
                    'runtime_pm': pm, 'runtime_status': rs, 'sclk_mhz': c.get('sclk_mhz'), 'ok': ok, 'notes': notes})
    ok = all(x['ok'] for x in out) if out else None
    if ok is None:
        summary = 'clock check: no engine cards read'
    elif ok:
        summary = 'clock check: every engine card at DPM high with runtime PM on'
    else:
        summary = 'clock check: ' + '; '.join(f"{x['name']} ({x['pci']}): " + '; '.join(x['notes']) for x in out if not x['ok'])
    return {'ok': ok, 'cards': out, 'summary': summary}


# ------------------------------------------------------------------------------------------------ fixtures
FIXTURE_CARDS = (
    # (card node, PCI slot, VRAM total GiB, VRAM used MiB, power W, sclk MHz, mclk MHz, perf level, busy %)
    ('card1', '0000:03:00.0', 24, 20205, 286.0, 2482, 1249, 'high', 97),
    ('card2', '0000:06:00.0', 20, 15513, 231.0, 2394, 1249, 'high', 95),
    ('card3', '0000:76:00.0', 0.5, 160, 9.0, 600, 2400, 'auto', 3),     # the iGPU: no power1_average
)


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def write_host_fixture(root, pid, evicted=(0, 0, 0), pgmigrate=0, nodes=('stats_49594', 'stats_61432', 'stats_8237')):
    """the host counters' files under root (synthetic): KFD's per-node evicted_ms of process pid, /proc/vmstat's
    pgmigrate_success (rewritten in place by later calls: a test moves them between samples)"""
    for n, v in zip(nodes, evicted):
        _w(os.path.join(root, 'sys', 'class', 'kfd', 'kfd', 'proc', str(pid), n, 'evicted_ms'), f'{int(v)}\n')
    _w(os.path.join(root, 'proc', 'vmstat'), f'nr_free_pages 1000\npgmigrate_success {int(pgmigrate)}\npgmigrate_fail 0\n')
    return root


def write_fixture(root, cards=FIXTURE_CARDS, runtime_pm=None):
    """a sysfs-like tree under root with the files HwReader reads (synthetic values: the mock server and the tests).
    runtime_pm: {card node: (control, runtime_status)} (default: 'on' / 'active' at DPM high, else 'auto' / 'active')"""
    w = _w
    for node, pci, gib, used, watts, sclk, mclk, level, busy in cards:
        d = os.path.join(root, 'sys', 'class', 'drm', node, 'device')
        h = os.path.join(d, 'hwmon', 'hwmon' + node[4:])
        w(os.path.join(d, 'uevent'), f'DRIVER=amdgpu\nPCI_CLASS=30000\nPCI_ID=1002:744C\nPCI_SLOT_NAME={pci}\n')
        w(os.path.join(d, 'mem_info_vram_total'), f'{int(gib * GIB)}\n')
        w(os.path.join(d, 'mem_info_vram_used'), f'{int(used * MIB)}\n')
        w(os.path.join(d, 'gpu_busy_percent'), f'{busy}\n')
        w(os.path.join(d, 'power_dpm_force_performance_level'), level + '\n')
        pm, rs = (runtime_pm or {}).get(node, ('on' if level == 'high' else 'auto', 'active'))
        w(os.path.join(d, 'power', 'control'), pm + '\n')
        w(os.path.join(d, 'power', 'runtime_status'), rs + '\n')
        w(os.path.join(d, 'pp_dpm_sclk'), f'0: 500Mhz\n1: {sclk}Mhz *\n2: 2700Mhz\n')
        w(os.path.join(d, 'pp_dpm_mclk'), f'0: 96Mhz\n1: {mclk}Mhz *\n')
        w(os.path.join(d, 'current_link_speed'), '16.0 GT/s PCIe\n')
        w(os.path.join(d, 'current_link_width'), '16\n' if node != 'card2' else '4\n')
        w(os.path.join(h, 'name'), 'amdgpu\n')
        if gib >= 4:
            w(os.path.join(h, 'power1_average'), f'{int(watts * 1e6)}\n')
        else:
            w(os.path.join(h, 'power1_input'), f'{int(watts * 1e6)}\n')
        w(os.path.join(h, 'freq1_input'), f'{int(sclk * 1e6)}\n')
        w(os.path.join(h, 'freq2_input'), f'{int(mclk * 1e6)}\n')
        w(os.path.join(h, 'temp1_input'), '61000\n')
        w(os.path.join(h, 'temp2_input'), '78000\n')
    return root
