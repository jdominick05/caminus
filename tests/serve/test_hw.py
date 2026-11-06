"""The hardware reader (caminus.serve.hw) on fixture sysfs trees: discovery (the iGPU left out, the XTX first by VRAM,
an explicit PCI order), every reading and its units, the fallbacks (pp_dpm_* when hwmon has no freq, power1_input),
missing files as None, the reading cache, and the energy integral by engine phase. Nothing reads the real /sys.
CPU only:  HIP_VISIBLE_DEVICES= python -I tests/serve/run_all.py"""
import os
import tempfile
import unittest

import common  # noqa: F401  (sys.path)
from caminus.serve.hw import FIXTURE_CARDS, HwReader, clock_check, discover, write_fixture, write_host_fixture


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def fixture(cards=FIXTURE_CARDS):
    return write_fixture(tempfile.mkdtemp(prefix='k3-hw-fixture-'), cards)


def card_dir(root, node):
    return os.path.join(root, 'sys', 'class', 'drm', node, 'device')


def put(path, text):
    with open(path, 'w') as f:
        f.write(text)


class TestDiscovery(unittest.TestCase):
    def test_order_and_igpu(self):
        root = fixture()
        cards = discover(root, names=('XTX', 'XT'))
        self.assertEqual([(c.index, c.name, c.pci) for c in cards],
                         [(0, 'XTX', '0000:03:00.0'), (1, 'XT', '0000:06:00.0')])
        # reversed node order on disk: still the larger card first
        root2 = fixture(((FIXTURE_CARDS[1][0].replace('2', '1'),) + FIXTURE_CARDS[1][1:],
                         (FIXTURE_CARDS[0][0].replace('1', '2'),) + FIXTURE_CARDS[0][1:]))
        self.assertEqual([c.pci for c in discover(root2)], ['0000:03:00.0', '0000:06:00.0'])

    def test_explicit_pci_order(self):
        root = fixture()
        cards = discover(root, pci=['0000:06:00.0', '0000:03:00.0', '0000:99:00.0'], names=('A', 'B'))
        self.assertEqual([(c.name, c.pci) for c in cards], [('A', '0000:06:00.0'), ('B', '0000:03:00.0')])

    def test_empty_root(self):
        self.assertEqual(discover(tempfile.mkdtemp(prefix='k3-hw-empty-')), [])
        hw = HwReader(root=tempfile.mkdtemp(prefix='k3-hw-empty-'))
        self.assertEqual(hw.read(), [])
        self.assertEqual(hw.energy(), [])


class TestReadings(unittest.TestCase):
    def test_values_and_units(self):
        root = fixture()
        hw = HwReader(root=root, clock=FakeClock())
        x, t = hw.read()
        self.assertEqual((x['name'], x['vram_used_mib'], x['vram_total_mib']), ('XTX', 20205.0, 24576.0))
        self.assertEqual((x['power_w'], x['sclk_mhz'], x['mclk_mhz']), (286.0, 2482.0, 1249.0))
        self.assertEqual((x['perf_level'], x['forced_high'], x['busy_pct']), ('high', True, 97.0))
        self.assertEqual((x['temp_edge_c'], x['temp_junction_c']), (61.0, 78.0))
        self.assertEqual((t['name'], t['vram_used_mib'], t['vram_total_mib'], t['power_w']), ('XT', 15513.0, 20480.0,
                                                                                               231.0))
        self.assertEqual((t['link_speed'], t['link_width']), ('16.0 GT/s PCIe', '4'))

    def test_fallbacks_and_missing(self):
        root = fixture()
        d = card_dir(root, 'card1')
        h = os.path.join(d, 'hwmon', 'hwmon1')
        os.rename(os.path.join(h, 'freq1_input'), os.path.join(h, 'freq1_input.gone'))     # -> pp_dpm_sclk '*'
        os.rename(os.path.join(h, 'power1_average'), os.path.join(h, 'power1_input'))      # older kernels
        os.rename(os.path.join(d, 'gpu_busy_percent'), os.path.join(d, 'gpu_busy_percent.gone'))
        put(os.path.join(d, 'power_dpm_force_performance_level'), 'auto\n')
        put(os.path.join(h, 'temp2_input'), 'garbage\n')
        # discovery needs power1_average: name the cards explicitly
        hw = HwReader(root=root, cards=['0000:03:00.0'], clock=FakeClock())
        x = hw.read()[0]
        self.assertEqual((x['sclk_mhz'], x['power_w'], x['busy_pct']), (2482.0, 286.0, None))
        self.assertEqual((x['perf_level'], x['forced_high'], x['temp_junction_c']), ('auto', False, None))

    def test_read_cache(self):
        root = fixture()
        clk = FakeClock()
        hw = HwReader(root=root, clock=clk, min_interval=1.0)
        self.assertEqual(hw.read()[0]['vram_used_mib'], 20205.0)
        put(os.path.join(card_dir(root, 'card1'), 'mem_info_vram_used'), str(21000 << 20))
        clk.t += 0.5
        self.assertEqual(hw.read()[0]['vram_used_mib'], 20205.0, 'cached for min_interval')
        clk.t += 0.6
        self.assertEqual(hw.read()[0]['vram_used_mib'], 21000.0)
        self.assertEqual(hw.samples, 2)


class TestEnergy(unittest.TestCase):
    def test_trapezoid_by_phase(self):
        root = fixture()
        clk = FakeClock()
        phase = ['idle']
        hw = HwReader(root=root, clock=clk, phase=lambda: phase[0])
        hw.sample()                                     # 286 / 231 W
        h0 = os.path.join(card_dir(root, 'card1'), 'hwmon', 'hwmon1', 'power1_average')
        put(h0, str(int(300e6)))
        clk.t += 2.0
        phase[0] = 'decode'
        hw.sample()                                     # XTX: (286 + 300) / 2 * 2 = 586 J in decode
        clk.t += 1.0
        phase[0] = 'prefill'
        hw.sample()                                     # XTX: 300 J; XT: 231 J
        clk.t += 31.0                                   # a gap: not integrated
        hw.sample()
        e = hw.energy()
        self.assertAlmostEqual(e[0]['decode'], 586.0, places=6)
        self.assertAlmostEqual(e[0]['prefill'], 300.0, places=6)
        self.assertAlmostEqual(e[1]['decode'], 462.0, places=6)
        self.assertAlmostEqual(e[1]['prefill'], 231.0, places=6)
        self.assertEqual(e[0]['idle'], 0.0)
        phase[0] = 'bogus'
        clk.t += 1.0
        hw.sample()                                     # an unknown phase counts as idle
        self.assertAlmostEqual(hw.energy()[0]['idle'], 300.0, places=6)

    def test_sampler_thread(self):
        root = fixture()
        hw = HwReader(root=root).start(0.02)
        try:
            import time
            t_end = time.time() + 5
            while hw.samples < 4 and time.time() < t_end:
                time.sleep(0.01)
            self.assertGreaterEqual(hw.samples, 4)
            self.assertEqual(len(hw.read()), 2)
        finally:
            hw.stop()


class TestM7Ops(unittest.TestCase):
    """impl:m7-ops: the runtime PM readings, the startup clock check's verdicts (read only), the host counters (KFD evictions
    as the largest per-node rise, page migrations by engine phase) on fixture trees"""

    def test_runtime_pm_and_clock_check(self):
        root = fixture()
        r = HwReader(root=root, min_interval=0)
        c = r.read()
        self.assertEqual([(x['runtime_pm'], x['runtime_status']) for x in c], [('on', 'active'), ('on', 'active')])
        ck = r.clock_check()
        self.assertEqual((ck['ok'], [x['ok'] for x in ck['cards']]), (True, [True, True]))
        # high + runtime PM auto: not held (the gpu-high note); auto level; a suspended card; an unreadable level
        cases = {'card1': ('auto', 'active'), 'card2': ('on', 'suspended')}
        root2 = write_fixture(tempfile.mkdtemp(prefix='k3-hw-fixture-'), runtime_pm=cases)
        ck = HwReader(root=root2, min_interval=0).clock_check()
        self.assertFalse(ck['ok'])
        n0, n1 = ck['cards'][0]['notes'], ck['cards'][1]['notes']
        self.assertTrue(any("runtime PM 'auto'" in n and 'auto clocks' in n for n in n0), n0)
        self.assertTrue(any('runtime-suspended' in n for n in n1), n1)
        cards = list(FIXTURE_CARDS)
        cards[1] = cards[1][:7] + ('auto',) + cards[1][8:]
        root3 = write_fixture(tempfile.mkdtemp(prefix='k3-hw-fixture-'), cards)
        os.remove(os.path.join(card_dir(root3, 'card1'), 'power_dpm_force_performance_level'))
        ck = HwReader(root=root3, min_interval=0).clock_check()
        self.assertEqual([x['ok'] for x in ck['cards']], [False, False])
        self.assertTrue(any('unreadable' in n for n in ck['cards'][0]['notes']))
        self.assertTrue(any("DPM level 'auto'" in n for n in ck['cards'][1]['notes']))
        self.assertEqual(clock_check([])['ok'], None)
        # the check never writes: the fixture's files are unchanged
        self.assertEqual(open(os.path.join(card_dir(root2, 'card1'), 'power', 'control')).read(), 'auto\n')

    def test_host_counters(self):
        root, pid = fixture(), 4242
        write_host_fixture(root, pid, evicted=(10, 12, 9), pgmigrate=1000)
        got, phase = [], ['idle']
        r = HwReader(root=root, min_interval=0, pid=pid, on_host=lambda ev, pg, busy: got.append((ev, pg, busy)), phase=lambda: phase[0])
        r.sample()                                          # the first sample: no rise yet
        self.assertEqual(got, [])
        write_host_fixture(root, pid, evicted=(40, 57, 21), pgmigrate=1300)
        r.sample()                                          # idle at both ends: migrations not 'busy'
        phase[0] = 'decode'
        write_host_fixture(root, pid, evicted=(40, 57, 21), pgmigrate=1350)
        r.sample()
        phase[0] = 'idle'
        r.sample()                                          # decode at the interval's start: still busy
        self.assertEqual(got, [(45, 300, False), (0, 50, True), (0, 0, True)])
        # no KFD directory for the process / no vmstat: nothing reported, nothing raised
        r2 = HwReader(root=fixture(), min_interval=0, pid=pid, on_host=lambda *a: got.append(a))
        r2.sample(); r2.sample()
        self.assertEqual(len(got), 3)


if __name__ == '__main__':
    unittest.main()
