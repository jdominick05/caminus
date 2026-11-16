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
        self.assertEqual(report['max_ctx'], {'f16': 32768, 'q8': None})   # (#82: the per-format shape; Q8 is #81's)
        self.assertEqual(report['cards'], [card['pci']])
        self.assertEqual(report['draft'], opt['draft'])
        self.assertEqual(report['need_mib'], need(32768))
        self.assertEqual(os.environ.get('HIP_VISIBLE_DEVICES'), before)
        stdout = io.StringIO()
        with patch.object(host, 'gpus', return_value=[card]), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report([*argv, '--kv-host', '--json']), 1)
        self.assertIsNone(json.loads(stdout.getvalue())['max_ctx']['f16'])

    def test_qwen3moe_two_card_fit_uses_serve_recipe_and_caps_at_the_file_context(self):
        # caminus fit --cards 2 for a qwen3moe file: the two-card entry (Q3MoETP + Q3MPrefillTP), the pair placed bigger
        # card first, its per-card projection at the placed split, max_ctx capped at the file's context_length (40,960);
        # sysfs only (no GPU runtime), and the same plan as max_ctx auto's runtime route
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'engine'))
        import tiny_q3m
        path = SCRATCH / 'fit-q3moe-tp.gguf'
        tiny_q3m.write(str(path), tp=True, n_ctx=40960)
        argv = ['--model', str(path), '--cards', '2', '--draft', 'off']
        opt = cli.parse(argv)
        A = cli.recipe(opt, log=lambda *a, **kw: None)
        self.assertIs(A, arch.ARCHS['qwen3moe-tp'])
        self.assertIs(cli.recipe(cli.parse(['--model', str(path), '--draft', 'off']), log=lambda *a, **kw: None), arch.ARCHS['qwen3moe'])
        link = dict(gbps=7.9, gts=16.0, width=4)
        cards, caps, reserves = [], [], []
        for i, (label, total) in enumerate((('XT', 20464), ('XTX', 24560))):   # (sysfs order: the XT first)
            dev = SCRATCH / f'q3tp-card{i}'
            dev.mkdir()
            (dev / 'mem_info_vram_used').write_text(str(1 << 20))
            cards.append(dict(dev=str(dev), line_mib=0, display=label == 'XTX', pci=f'0000:0{3 + 3 * i}:00.0', hip=i,
                              status='supported', total_mib=total, label=label, card=f'card{i + 1}', link=link, link_ok=True))
        pair = [cards[1], cards[0]]                                             # (engine order: the bigger card first)
        _, _, _, need = A.auto_need(opt, int(opt['auto_cache_slots']), cards=pair)
        for c in pair:
            reserve = capacity.DISPLAY_RESERVE_MIB if c['display'] else capacity.MARGIN_MIB
            c['line_mib'] = need(40960)[pair.index(c)] + 1 + capacity.MARGIN_MIB + reserve + capacity.CACHE_WORK_MIB[0] + A.slack_mib(opt) + 64
            caps.append((c['line_mib'] - capacity.MARGIN_MIB, 1.0)); reserves.append(reserve)
        import torch
        stdout = io.StringIO()
        with patch.object(host, 'gpus', return_value=cards), \
             patch.object(torch.cuda, 'init', side_effect=AssertionError('GPU init forbidden')), \
             patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU init forbidden')), \
             patch.object(capacity, 'caps_mib', side_effect=AssertionError('GPU query forbidden')), \
             patch.object(A, 'build', side_effect=AssertionError('engine build forbidden')), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report([*argv, '--json']), 0)
            placed = host.place(A.cards, fits=lambda cs: A.fits(opt, cs), split_of=A.split)
        report = json.loads(stdout.getvalue())
        self.assertTrue(report['ok'])
        self.assertEqual(report['arch'], 'qwen3moe-tp')
        self.assertEqual(report['max_ctx']['f16'], 40960)
        self.assertEqual(report['cards'], [cards[1]['pci'], cards[0]['pci']])
        self.assertEqual(placed['split'], {'exp_rows': 512, 'vocab_share': 0.532})
        self.assertEqual(report['need_mib'], need(40960))
        # the runtime route (max_ctx auto in serve, the placement's split stored) reads the same plan
        ropt = dict(opt, placement=dict(split=placed['split']))
        with patch.object(capacity, 'caps_mib', return_value=caps), patch.object(capacity, 'cache_reserve_mib', return_value=reserves):
            self.assertEqual(A.context_fit(ropt).report(), A.context_fit(opt, pair).report())
            self.assertEqual(A.auto_max_ctx(ropt, log=lambda *a, **kw: None), 40960)
        # past the file's context_length: no native fit reported
        stdout = io.StringIO()
        with patch.object(host, 'gpus', return_value=cards), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report([*argv, '--max-ctx', '65536', '--json']), 1)
        self.assertIsNone(json.loads(stdout.getvalue())['max_ctx']['f16'])

    def test_qwen3moe_two_card_fit_below_the_file_context_takes_the_largest_step_that_fits(self):
        # each card's room between need(32768) and need(40960): caminus fit --cards 2 and max_ctx auto report the largest
        # AUTO_STEP multiple whose projection fits on both cards (below the file's 40,960), not the cap
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'engine'))
        import tiny_q3m
        path = SCRATCH / 'fit-q3moe-tp-step.gguf'
        tiny_q3m.write(str(path), tp=True, n_ctx=40960)
        argv = ['--model', str(path), '--cards', '2', '--draft', 'off']
        opt = cli.parse(argv)
        A = cli.recipe(opt, log=lambda *a, **kw: None)
        self.assertIs(A, arch.ARCHS['qwen3moe-tp'])
        link = dict(gbps=7.9, gts=16.0, width=4)
        for want in (32768, 36864):
            with self.subTest(max_ctx=want):
                self.assertEqual(want % arch.AUTO_STEP, 0)
                cards = []
                for i, (label, total) in enumerate((('XT', 20464), ('XTX', 24560))):   # (sysfs order: the XT first)
                    dev = SCRATCH / f'q3tp-step{want}-card{i}'
                    dev.mkdir()
                    (dev / 'mem_info_vram_used').write_text(str(1 << 20))
                    cards.append(dict(dev=str(dev), line_mib=0, display=label == 'XTX', pci=f'0000:0{3 + 3 * i}:00.0', hip=i,
                                      status='supported', total_mib=total, label=label, card=f'card{i + 1}', link=link, link_ok=True))
                pair = [cards[1], cards[0]]                                             # (engine order: the bigger card first)
                _, _, _, need = A.auto_need(opt, int(opt['auto_cache_slots']), cards=pair)
                caps, reserves = [], []
                for k, c in enumerate(pair):
                    # room halfway between need(want) and need(want + AUTO_STEP) on this card (the per-step gap of the tiny
                    # file's projection is what the room is placed inside; no fixed pad)
                    room = (need(want)[k] + need(want + arch.AUTO_STEP)[k]) / 2
                    reserve = capacity.DISPLAY_RESERVE_MIB if c['display'] else capacity.MARGIN_MIB
                    c['line_mib'] = room + 1 + capacity.MARGIN_MIB + reserve + capacity.CACHE_WORK_MIB[0] + A.slack_mib(opt)
                    caps.append((c['line_mib'] - capacity.MARGIN_MIB, 1.0)); reserves.append(reserve)
                plan = A.context_fit(opt, pair)
                for k in range(2):
                    self.assertLess(need(32768)[k], plan.room_mib[k])
                    self.assertLess(plan.room_mib[k], need(40960)[k])
                    self.assertLessEqual(need(want)[k], plan.room_mib[k])
                    self.assertLess(plan.room_mib[k], need(want + arch.AUTO_STEP)[k])
                stdout = io.StringIO()
                with patch.object(host, 'gpus', return_value=cards), \
                     patch.object(capacity, 'caps_mib', side_effect=AssertionError('GPU query forbidden')), \
                     patch.object(A, 'build', side_effect=AssertionError('engine build forbidden')), \
                     contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.fit_report([*argv, '--json']), 0)
                    placed = host.place(A.cards, fits=lambda cs: A.fits(opt, cs), split_of=A.split)
                report = json.loads(stdout.getvalue())
                self.assertTrue(report['ok'])
                self.assertEqual(report['cards'], [cards[1]['pci'], cards[0]['pci']])
                self.assertEqual(report['max_ctx']['f16'], want)
                self.assertLess(report['max_ctx']['f16'], 40960)
                self.assertEqual(report['need_mib'], list(need(want)))
                # the runtime route (max_ctx auto in serve, the placement's split stored) takes the same step
                ropt = dict(opt, placement=dict(split=placed['split']))
                with patch.object(capacity, 'caps_mib', return_value=caps), \
                     patch.object(capacity, 'cache_reserve_mib', return_value=reserves):
                    self.assertEqual(A.auto_max_ctx(ropt, log=lambda *a, **kw: None), want)

    def test_query_errors_have_no_invented_maximum_and_tp_drafting_needs_a_two_card_verify_object(self):
        for argv in (['--model', str(SCRATCH / 'missing.gguf')],):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(host, 'place') as place:
                self.assertEqual(cli.fit_report([*argv, '--json']), 1)
            report = json.loads(stdout.getvalue())
            self.assertFalse(report['ok'])
            self.assertIsNone(report['max_ctx']['f16'])
            place.assert_not_called()
        # (two-card drafting is served for Swift 1.5 and ThinkingCap; a file whose decode module has no two-card verify
        # object, the plain 27B's GSQ formats, is refused it before any placement)
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'engine'))
        import tiny_q35
        model = SCRATCH / 'refused-gsq.gguf'
        tiny_q35.write(str(model), variant='gsq', tp=True)
        for draft in ('mtp', 'dflash'):
            with patch.object(arch, 'of', return_value=arch.ARCHS['qwen35-own']), \
                 patch.object(host, 'place') as place:
                with self.assertRaisesRegex(ValueError, 'two-card verify object'):
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
                self.assertIsNone(report['max_ctx']['f16'])
        usage.write_text('0')
        stdout = io.StringIO()
        with patch.object(arch, 'of', return_value=A), \
             patch.object(host, 'gpus', return_value=[card]), contextlib.redirect_stdout(stdout):
            self.assertEqual(cli.fit_report(['--model', str(model), '--draft', 'off', '--json']), 0)
        self.assertEqual(json.loads(stdout.getvalue())['max_ctx']['f16'], 4096)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    SCRATCH = Path(args.scratch)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    result = unittest.main(argv=[sys.argv[0]], exit=False).result
    print('RESULT ' + ('PASS' if result.wasSuccessful() else 'FAIL'))
    raise SystemExit(not result.wasSuccessful())
