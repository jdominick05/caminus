"""CPU checks of placement, auto context and the shared fit report."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from caminus import arch
from caminus.device import capacity
from caminus.device.context_fit import ContextFit as Plan
from caminus.device import host
from caminus.serve import cli


class Recipe(arch.Arch):
    cards = 1

    def auto_need(self, opt, slots, cards=None):
        return 4096, [0], 4.0, lambda c: [1000.0]


class ContextFit(unittest.TestCase):
    def test_placement_cannot_admit_more_than_runtime_auto(self):
        dev = SCRATCH / 'card'
        dev.mkdir(parents=True)
        (dev / 'mem_info_vram_used').write_text('0')
        # One MiB below the runtime's exact budget, including both distinct
        # reserves: allocator margin below the guard, then display headroom.
        line = 1000 + capacity.MARGIN_MIB + capacity.DISPLAY_RESERVE_MIB + capacity.CACHE_WORK_MIB[0] + arch.AUTO_SLACK_MIB - 1
        card = dict(dev=str(dev), line_mib=line, display=True)
        A, opt = Recipe('test', 'test'), dict(max_ctx=4096)
        with patch.object(capacity, 'caps_mib', return_value=[(line - capacity.MARGIN_MIB, 0)]), \
             patch.object(capacity, 'cache_reserve_mib', return_value=[capacity.DISPLAY_RESERVE_MIB]):
            with self.assertRaises(capacity.CapacityError):
                A.auto_max_ctx(opt, log=lambda *a, **kw: None)
            self.assertFalse(A.fits(opt, [card]))

    def test_maximum_includes_native_endpoint_and_checks_both_cards(self):
        p = Plan(10000, [0, 1], 4, [100, 80], lambda c: [c / 100, c / 125], 4096)
        self.assertEqual(p.maximum(), 10000)
        self.assertEqual(p.report()['need_mib'], [100, 80])
        p = Plan(10000, [0, 1], 4, [100, 70], lambda c: [c / 100, c / 125], 4096)
        self.assertEqual(p.maximum(), 8192)
        self.assertFalse(p.fits(10000))
        small = Plan(1000, [0], 0, [1000], lambda c: [c], 4096)
        self.assertEqual(small.maximum(), 1000)

    def test_failed_fit_is_not_a_numeric_maximum(self):
        p = Plan(8192, [0], 0, [-1], lambda c: [c], 4096)
        self.assertIsNone(p.maximum())
        self.assertIsNone(p.report()['max_ctx'])
        self.assertIsNone(p.report()['need_mib'])

    def test_bad_projections_and_device_counts_fail_closed(self):
        for projected in ([], [float('nan')], [float('inf')], [-1], [0, 0]):
            with self.subTest(projected=projected):
                p = Plan(8192, [0], 0, [100], lambda c: projected, 4096)
                with self.assertRaises(ValueError): p.maximum()
        for devices, room in (([], []), ([0, 1], [100]), ([0], [float('nan')])):
            with self.assertRaises(ValueError): Plan(8192, devices, 0, room, lambda c: [0], 4096)
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            Recipe('test', 'test').context_fit({}, [])

    def test_host_and_runtime_use_identical_budget_for_card_and_drafter_choices(self):
        class Projected(arch.Arch):
            def auto_need(self, opt, slots, cards=None):
                n = opt['cards']
                return 32769, list(range(n)), 8, lambda c: [900 + 10 * i + c / 64 + slots * 8 for i in range(n)]

            def extras_mib(self, opt, devs, cards=None):
                return [opt['drafter_mib']] * len(devs)

        A = Projected('test', 'test')
        for n in (1, 2):
            for drafter in (0, 128, 384):
                with self.subTest(cards=n, drafter=drafter):
                    opt = dict(cards=n, drafter_mib=drafter, auto_cache_slots=4, max_ctx=16384)
                    cards, caps, reserves = [], [], []
                    for i in range(n):
                        dev = SCRATCH / f'budget-{n}-{drafter}-{i}'
                        dev.mkdir(parents=True)
                        used = (i + 1) * 0.25
                        (dev / 'mem_info_vram_used').write_text(str(int(used * 2 ** 20)))
                        reserve = capacity.DISPLAY_RESERVE_MIB if i == 0 else capacity.MARGIN_MIB
                        need = A.auto_need(opt, 4)[3](16384)[i]
                        line = need + used + capacity.MARGIN_MIB + reserve + capacity.CACHE_WORK_MIB[0] + arch.AUTO_SLACK_MIB + drafter
                        cards.append(dict(dev=str(dev), display=i == 0, line_mib=line))
                        caps.append((line - capacity.MARGIN_MIB, used))
                        reserves.append(reserve)
                    # The picker/placement route must never ask the GPU runtime.
                    with patch.object(capacity, 'caps_mib', side_effect=AssertionError('GPU query forbidden')), \
                         patch.object(capacity, 'cache_reserve_mib', side_effect=AssertionError('GPU query forbidden')):
                        host = A.context_fit(opt, cards)
                        self.assertTrue(A.fits(opt, cards))
                        self.assertFalse(A.fits(dict(opt, max_ctx=20480), cards))
                    with patch.object(capacity, 'caps_mib', return_value=caps), \
                         patch.object(capacity, 'cache_reserve_mib', return_value=reserves):
                        runtime = A.context_fit(opt)
                        self.assertEqual(A.auto_max_ctx(opt, log=lambda *a, **kw: None), 16384)
                    self.assertEqual(host.report(), runtime.report())
                    self.assertEqual(host.maximum(), 16384)

    def test_real_gguf_command_uses_serve_recipe_and_placement_without_gpu_initialization(self):
        # Reuse the repository's small GGUF writer without its fixture-only
        # paused-model environment setting escaping the import.
        spec = importlib.util.spec_from_file_location('fit_gguf_fixture', Path(__file__).with_name('test_arch.py'))
        fixture = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ): spec.loader.exec_module(fixture)
        path = SCRATCH / 'fit-q3moe.gguf'
        hp = dict(context_length=40960, block_count=2, embedding_length=512,
                  expert_count=4, expert_used_count=2, expert_feed_forward_length=256)
        hp.update({'attention.head_count': 4, 'attention.head_count_kv': 2, 'attention.key_length': 128})
        fixture.write_gguf(str(path), 'qwen3moe', {'qwen3moe.' + k: v for k, v in hp.items()},
                           [('token_embd.weight', 512, 16), ('output.weight', 512, 16)])
        argv = ['--model', str(path), '--draft', 'off', '--auto-cache-slots', '3']
        opt = cli.parse(argv)
        A = cli.recipe(opt, log=lambda *a, **kw: None)
        _, _, _, need = A.auto_need(opt, 3)
        dev = SCRATCH / 'report-card'
        dev.mkdir()
        (dev / 'mem_info_vram_used').write_text(str(1 << 18))
        reserve = capacity.DISPLAY_RESERVE_MIB
        line = need(32768)[0] + 0.25 + capacity.MARGIN_MIB + reserve + capacity.CACHE_WORK_MIB[0] + A.slack_mib(opt)
        card = dict(dev=str(dev), line_mib=line, display=True, pci='0000:01:00.0', hip=0,
                    status='supported', total_mib=24576, label='XTX', card='card0')
        before = os.environ.get('HIP_VISIBLE_DEVICES')
        stdout = io.StringIO()
        import torch
        from caminus import __main__
        from caminus.cmd import guarded
        with patch.object(host, 'gpus', return_value=[card]), \
             patch.object(torch.cuda, 'init', side_effect=AssertionError('GPU init forbidden')), \
             patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU init forbidden')), \
             patch.object(capacity, 'caps_mib', side_effect=AssertionError('GPU query forbidden')), \
             patch.object(A, 'build', side_effect=AssertionError('engine build forbidden')), \
             patch.object(guarded, 'exec_guarded', side_effect=AssertionError('guard execution forbidden')), \
             contextlib.redirect_stdout(stdout):
            code = __main__.main(['fit', *argv, '--json'])
            selected = host.place(A.cards, fits=lambda cs: A.fits(opt, cs), split_of=A.split)
            self.assertEqual(selected['cards'], [card])
        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(report['ok'])
        self.assertEqual(report['max_ctx'], 32768)
        self.assertEqual(report['cards'], [card['pci']])
        self.assertEqual(report['draft'], opt['draft'])
        self.assertEqual(report['need_mib'], need(32768))
        self.assertEqual(os.environ.get('HIP_VISIBLE_DEVICES'), before)
        stdout = io.StringIO()
        with patch.object(host, 'gpus', return_value=[card]), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report([*argv, '--kv-host', '--json']), 1)
        self.assertIsNone(json.loads(stdout.getvalue())['max_ctx'])

    def test_query_errors_have_no_invented_maximum_and_tp_drafting_stays_refused(self):
        for argv in (['--model', str(SCRATCH / 'missing.gguf')],):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(host, 'place') as place:
                self.assertEqual(cli.fit_report([*argv, '--json']), 1)
            report = json.loads(stdout.getvalue())
            self.assertFalse(report['ok'])
            self.assertIsNone(report['max_ctx'])
            place.assert_not_called()
        model = SCRATCH / 'refused.gguf'
        model.write_bytes(b'fixture')
        for draft in ('mtp', 'dflash'):
            with patch.object(arch, 'of', return_value=arch.ARCHS['qwen35-own']), \
                 patch.object(host, 'place') as place:
                with self.assertRaises(ValueError):
                    cli.recipe(cli.parse(['--model', str(model), '--cards', '2', '--draft', draft]))
                place.assert_not_called()

    def test_unknown_current_usage_cannot_produce_a_numeric_fit(self):
        model = SCRATCH / 'usage-model.gguf'
        model.write_bytes(b'fixture')
        dev = SCRATCH / 'usage-card'
        dev.mkdir()
        usage = dev / 'mem_info_vram_used'
        card = dict(dev=str(dev), line_mib=24000, display=True, pci='0000:01:00.0', hip=0,
                    status='supported', total_mib=24576, label='XTX', card='card0')
        A = Recipe('test', 'test')
        for value in (None, '', 'invalid', '-1'):
            with self.subTest(usage=value):
                if value is not None: usage.write_text(value)
                stdout = io.StringIO()
                with patch.object(arch, 'of', return_value=A), \
                     patch.object(host, 'gpus', return_value=[card]), contextlib.redirect_stdout(stdout):
                    code = cli.fit_report(['--model', str(model), '--draft', 'off', '--json'])
                report = json.loads(stdout.getvalue())
                self.assertEqual(code, 1)
                self.assertFalse(report['ok'])
                self.assertIsNone(report['max_ctx'])
        usage.write_text('0')
        stdout = io.StringIO()
        with patch.object(arch, 'of', return_value=A), \
             patch.object(host, 'gpus', return_value=[card]), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report(['--model', str(model), '--draft', 'off', '--json']), 0)
        self.assertEqual(json.loads(stdout.getvalue())['max_ctx'], 4096)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    SCRATCH = Path(args.scratch)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    result = unittest.main(argv=[sys.argv[0]], exit=False).result
    print('RESULT ' + ('PASS' if result.wasSuccessful() else 'FAIL'))
    raise SystemExit(not result.wasSuccessful())
