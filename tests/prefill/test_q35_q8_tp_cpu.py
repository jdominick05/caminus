"""TP Q8 prefill routing, both-card failure priority and drafter export, CPU only.

Construct the real prefill and exercise its operation/driver methods on CPU
views. Launchers and peer exchange are recorders; no device arithmetic runs.
"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_q35_q8_cpu as fixtures
from caminus.errors import InvalidKVEncoding
from caminus.prefill import q35own_tp as TP


def tp_target(line, q8, card):
    m = fixtures.target(line, q8)
    m.devs, m.feature_card = [0, 1], card
    m.NK *= 2; m.NV *= 2; m.nh *= 2; m.nkv *= 2
    m.f0, m.local_only = m.ffn // 2, False
    m.module = f'k3_q35_{line}_tp' + ('_kvq8' if q8 else '')
    m.gsplit = [(0, 1, [0, 1]), (1, 1, [2, 3])]
    m.L = [[copy.deepcopy(row), copy.deepcopy(row)] for row in m.L]
    m.emb_dev = [m.emb_dev.clone(), m.emb_dev.clone()]
    m.out_norm = [m.out_norm.clone(), m.out_norm.clone()]
    return m


class PrefillTP(unittest.TestCase):
    def setUp(self):
        fixtures.DenseQ8Prefill.setUp(self)
        self.stack.enter_context(patch.object(torch.cuda, 'current_stream',
            side_effect=lambda d: NS(device=d, cuda_stream=d.index, synchronize=Mock())))
        self.stack.enter_context(patch.object(TP, 'Exchange', side_effect=lambda *a, **kw: Mock()))

    def prefill(self, line='swift', q8=True, card=0, export_h=True, model=None):
        return TP.Q35OwnPrefillTP(model or tp_target(line, q8, card), chunk=16,
                                 export_h=export_h, feature_layers=(1,))

    @staticmethod
    def record(pf):
        calls = []
        for attr in ('O', 'A', 'Gl', 'Gd', 'KV'):
            group = getattr(pf, attr)
            for c in (0, 1):
                if group[c] is not None: group[c] = fixtures.Record(attr + str(c), calls, group[c])
        return calls

    def draft(self, kind, pf, order=None):
        action = (lambda *a: order.append(kind)) if order is not None else None
        return NS(kind=kind, dx=pf.m.devs[pf.feature_card], d=NS(layers_tgt=[1]),
                  fill_chunk=Mock(side_effect=action), inject_chunk=Mock(side_effect=action))

    def test_both_cards_keep_weight_modules_and_count_existing_buffers(self):
        for line in ('swift', 'tcap'):
            for card in (0, 1):
                with self.subTest(line=line, feature_card=card):
                    fixtures.Module.loaded = []
                    pf = self.prefill(line, card=card)
                    self.assertEqual([o.module for o in pf.O], ['q35_prefill_tcap' if line == 'tcap' else 'q35_prefill'] * 2)
                    self.assertEqual([o.mod.name for o in pf.KV], ['q35_prefill_kvq8'] * 2)
                    self.assertEqual(fixtures.Module.loaded.count('q35_prefill_kvq8'), 2)
                    self.assertEqual(self.cap.call_args.args[:2], ([0, 1], pf.need_mib))
                    f16 = self.prefill(line, q8=False, card=card)
                    self.assertEqual(f16.KV, [None, None])
                    self.assertEqual(pf.need_mib, f16.need_mib)
                    self.assertNotIn('dense_kv', f16.numerics())
                    self.assertIn('row-fp32', pf.numerics()['dense_kv'])
                    self.assertEqual(pf.features.shape, (1, 16, 5120))

    def test_capacity_refusal_precedes_loading_on_either_card(self):
        self.cap.side_effect = ValueError('synthetic no headroom on card 1')
        with self.assertRaisesRegex(ValueError, 'card 1'): self.prefill()
        self.assertEqual(fixtures.Module.loaded, [])

    def test_real_mix_rotates_each_card_before_append_and_before_gate(self):
        for card in (0, 1):
            pf = self.prefill(card=card)
            calls = self.record(pf)
            self.assertTrue(pf._mix(card, 0, 3, 0))
            names = [n for n, _, _ in calls]
            rotations = [(i, a[1:]) for i, (n, a, _) in enumerate(calls) if n == f'A{card}.hadamard']
            self.assertEqual([a for _, a in rotations], [(256, 3), (64, 12), (256, 18), (64, 72)])
            app, gate = names.index(f'KV{card}.append'), names.index(f'O{card}.gate_mul16')
            self.assertLess(rotations[1][0], app)
            self.assertLess(app, rotations[2][0]); self.assertLess(rotations[3][0], gate)
            self.assertIs(calls[app][1][4], pf.m.L[0][card]['cache'])
            self.assertNotIn(f'O{card}.kv_append16', names)
            self.assertNotIn(f'O{card}.attn_wm', names)

    def test_last_cache_only_layer_still_rotates_and_encodes(self):
        for card in (0, 1):
            pf = self.prefill(export_h=False)
            calls = self.record(pf)
            self.assertFalse(pf._mix(card, 1, 2, 0))
            self.assertEqual(sum(n == f'A{card}.hadamard' for n, _, _ in calls), 2)
            self.assertEqual(sum(n == f'KV{card}.append' for n, _, _ in calls), 1)
            self.assertFalse(any(n == f'KV{card}.attn_wm' for n, _, _ in calls))

    def test_numerical_error_on_either_card_wins_over_bounds_and_blocks_export(self):
        for card in (0, 1):
            for kind in ('mtp', 'dflash35'):
                with self.subTest(card=card, kind=kind):
                    pf = self.prefill(card=card)
                    pf._chunk, pf._handoff = Mock(), Mock()
                    dr = self.draft(kind, pf)
                    pf.check_drafter(dr)
                    pf.B[1 - card]['err'].fill_(1)
                    pf.B[card]['err'].fill_(2)
                    with self.assertRaises(InvalidKVEncoding): pf._run_chunk([1] * 4, 0, 0, 0, 4, [], [])
                    self.assertEqual(int(pf.B[card]['err']), 2)
                    self.assertEqual(int(pf.B[1 - card]['err']), 1)
                    dr.fill_chunk.assert_not_called(); dr.inject_chunk.assert_not_called()
                    self.assertTrue(pf.m.broken)
                    self.assertIsNone(pf.m.poisoned)
                    with self.assertRaises(InvalidKVEncoding): pf.check_flags()

    def test_both_streams_settle_before_successful_drafter_export(self):
        for kind in ('mtp', 'dflash35'):
            pf = self.prefill(card=1)
            order = []
            pf._chunk = lambda *a: order.append('chunk')
            self.sync.side_effect = lambda d: order.append('wait' + str(d.index))
            dr = self.draft(kind, pf, order)
            pf.check_drafter(dr)
            pf._run_chunk([10, 11, 12, 13], 0, 0, 0, 4, [], [])
            self.assertEqual(order, ['chunk', 'wait0', 'wait1', kind])
            if kind == 'mtp': self.assertEqual(dr.fill_chunk.call_args.args[1:], ([11, 12, 13, 13], 1))

    def test_error_found_during_host_failure_drain_is_terminal(self):
        pf = self.prefill()
        def host_error(*a):
            pf.B[1]['err'].fill_(2)
            raise OSError('synthetic host failure after launch')
        pf._chunk, pf._handoff = host_error, Mock()
        with self.assertRaises(InvalidKVEncoding): pf.run([1] * 4, 0, 0, 4)
        self.assertIsNone(pf.m.poisoned)
        pf._handoff.assert_not_called()


if __name__ == '__main__':
    r = unittest.main(exit=False).result
    print('RESULT ' + ('PASS' if r.wasSuccessful() else 'FAIL'))
    raise SystemExit(not r.wasSuccessful())
