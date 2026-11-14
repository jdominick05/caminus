"""Qwen provider budgets, thinking bindings and owned-route migration; no GPU, sockets or client required."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from caminus.cmd import manage, qwen_code as Q

OLD = 'http://127.0.0.1:8092/v1'
NEW = 'http://127.0.0.1:8093/v1'
EXTERNAL = 'https://example.invalid/v1'


def provider(name, url, owned=True):
    return dict(id=name, baseUrl=url, envKey='CAMINUS_API_KEY' if owned else 'CUSTOM_KEY')


def settings():
    return dict(modelProviders={'openai': [provider('thinkingcap', OLD), provider('thinkingcap-nothink', OLD)]},
                model={'name': 'thinkingcap', 'baseUrl': OLD},
                security={'auth': {'selectedType': 'openai'}},
                fastModel='thinkingcap-nothink')


PROVS = [provider('swift-1.5', NEW), provider('swift-1.5-nothink', NEW)]


class Providers(unittest.TestCase):
    def test_every_alias_has_native_thinking_binding_and_verified_window(self):
        for alias in ('k3-coder', 'swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b'):
            ids = [alias, alias + '-nothink', alias + '-low', alias + '-xhigh']
            with patch.object(Q, 'server_ids', return_value=(ids, 8192)):
                ps, got = Q.providers({'port': 8093})
            self.assertEqual(got, ids)
            for p in ps:
                self.assertEqual(p['capabilities'], {'reasoning': {'profile': 'qwen-chat-template'}})
                self.assertEqual(p['generationConfig']['contextWindowSize'], 8000)
                self.assertNotIn('max_tokens', p['generationConfig'])
                if p['id'].endswith('-nothink'):
                    self.assertIs(p['generationConfig']['reasoning'], False)
                else:
                    self.assertNotIn('reasoning', p['generationConfig'])

    def test_rounding_never_exceeds_positive_reported_budget(self):
        for n in (1, 512, 999, 1000, 8192, 32768, 262144, 1048576):
            with patch.object(Q, 'server_ids', return_value=(['coder'], n)):
                ps, _ = Q.providers({'port': 8092})
            w = ps[0]['generationConfig']['contextWindowSize']
            self.assertGreater(w, 0)
            self.assertLessEqual(w, n)
            self.assertEqual(w, (n // 1000) * 1000 if n >= 1000 else n)

    def test_unknown_or_invalid_context_refuses_before_settings_or_hook_writes(self):
        for n in (None, 0, -1, True, '8192', 8192.5):
            with self.subTest(n=n), tempfile.TemporaryDirectory() as td:
                p = Path(td) / 'settings.json'
                original = b'{"custom": "preserve"}\n'
                p.write_bytes(original)
                srv = dict(name='swift', model='swift-1.5', port=8092)
                with patch.object(Q, 'running', return_value=[srv]), patch.object(Q, 'server_ids', return_value=(['swift-1.5'], n)):
                    with self.assertRaisesRegex(manage.ManageError, 'reported its context'):
                        Q.setup(home=td, say=lambda _: None)
                self.assertEqual(p.read_bytes(), original)
                self.assertEqual(sorted(os.listdir(td)), ['settings.json'])

    def test_thinking_off_wire_shape_is_accepted_without_changing_effort_alias(self):
        from caminus.serve.normalize import openai_chat_to_template
        from caminus.serve import template
        for alias in ('k3-coder', 'swift-1.5', 'thinkingcap', 'qwen3.8-27b', 'qwen3-30b-a3b'):
            profiles = {alias: {'enable_thinking': True, 'reasoning_effort': 'medium'},
                        alias + '-nothink': {'enable_thinking': False, 'reasoning_effort': 'medium'},
                        alias + '-low': {'enable_thinking': True, 'reasoning_effort': 'low'}}
            with patch.dict(template.MODEL_PROFILES, profiles, clear=True):
                n = openai_chat_to_template(dict(model=alias + '-low', messages=[{'role': 'user', 'content': 'hi'}],
                                                chat_template_kwargs={'enable_thinking': False}))
                _, kw = template.resolve_template_kwargs(n['model'], n['client_kwargs'])
                self.assertFalse(kw['enable_thinking'])
                self.assertEqual(kw['reasoning_effort'], 'low')
                _, kw = template.resolve_template_kwargs(alias + '-nothink', {'enable_thinking': True})
                self.assertFalse(kw['enable_thinking'])


class Migration(unittest.TestCase):
    def merge(self, old, provs=PROVS):
        before = copy.deepcopy(old)
        out = Q.merged(old, provs, 'swift-1.5', 'swift-1.5-nothink')
        self.assertEqual(old, before)
        return out

    def test_main_route_moves_with_its_endpoint(self):
        out = self.merge(settings())
        self.assertEqual(out['model'], {'name': 'swift-1.5', 'baseUrl': NEW})
        self.assertEqual(out['fastModel'], 'swift-1.5-nothink')

    def test_removed_owned_bare_qualified_and_pinned_auxiliary_routes(self):
        for select in ('thinkingcap', 'openai:thinkingcap', 'openai:thinkingcap\0' + OLD,
                       ' thinkingcap\0' + OLD + ' '):
            old = settings()
            old.update(fastModel=select, compactionModel=select)
            out = self.merge(old)
            self.assertEqual(out['fastModel'], 'swift-1.5-nothink')
            self.assertEqual(out['compactionModel'], 'swift-1.5')

    def test_still_available_owned_auxiliary_choices_stay_exact(self):
        old = settings()
        chosen = 'openai:thinkingcap-nothink\0' + OLD
        old.update(fastModel=chosen, compactionModel=chosen)
        out = self.merge(old, PROVS + old['modelProviders']['openai'])
        self.assertEqual(out['fastModel'], chosen)
        self.assertEqual(out['compactionModel'], chosen)

    def test_same_id_new_endpoint_repairs_pinned_routes(self):
        old = settings()
        old['model']['name'] = 'swift-1.5'
        old['modelProviders']['openai'] = [provider('swift-1.5', OLD)]
        old.update(fastModel='openai:swift-1.5\0' + OLD, compactionModel='openai:swift-1.5\0' + OLD)
        out = self.merge(old)
        self.assertEqual(out['model']['baseUrl'], NEW)
        self.assertEqual(out['fastModel'], 'swift-1.5-nothink')
        self.assertEqual(out['compactionModel'], 'swift-1.5')

    def test_external_same_id_pins_and_other_auth_choices_are_preserved(self):
        for chosen in ('openai:thinkingcap\0' + EXTERNAL, 'anthropic:thinkingcap', 'hosted:custom', 42):
            old = settings()
            old['modelProviders']['openai'].append(provider('thinkingcap', EXTERNAL, owned=False))
            old.update(fastModel=chosen, compactionModel=chosen, mcpServers={'x': {'command': 'custom-mcp'}})
            old['hooks'] = {'PostToolBatch': [{'hooks': [{'name': 'custom'}]}]}
            out = self.merge(old)
            self.assertEqual((out['fastModel'], out['compactionModel']), (chosen, chosen))
            self.assertEqual(out['mcpServers'], old['mcpServers'])
            self.assertEqual(out['hooks'], old['hooks'])
            self.assertIn(provider('thinkingcap', EXTERNAL, owned=False), out['modelProviders']['openai'])

    def test_first_unpinned_external_match_and_external_primary_are_preserved(self):
        old = settings()
        old['modelProviders']['openai'].insert(0, provider('thinkingcap', EXTERNAL, owned=False))
        old.update(fastModel='thinkingcap', compactionModel='thinkingcap')
        old['model']['baseUrl'] = EXTERNAL
        out = self.merge(old)
        self.assertEqual(out['model'], old['model'])
        self.assertEqual(out['fastModel'], 'thinkingcap')
        self.assertEqual(out['compactionModel'], 'thinkingcap')

    def test_other_auth_bare_auxiliary_choices_are_not_reinterpreted(self):
        old = settings()
        old['security']['auth']['selectedType'] = 'anthropic'
        old['model'] = {}   # setting the default main model must not reclassify existing auxiliary choices
        old.update(fastModel='thinkingcap', compactionModel='thinkingcap')
        out = self.merge(old)
        self.assertEqual((out['fastModel'], out['compactionModel']), ('thinkingcap', 'thinkingcap'))

    def test_plain_external_entry_wins_for_unpinned_auxiliary_routes(self):
        for base_url in (None, ''):
            old = settings()
            external = dict(id='thinkingcap', envKey='CUSTOM_KEY')
            if base_url is not None: external['baseUrl'] = base_url
            old['modelProviders']['openai'].append(external)
            old.update(fastModel='openai:thinkingcap', compactionModel='thinkingcap')
            out = self.merge(old)
            self.assertEqual(out['model']['name'], 'swift-1.5')   # the primary was explicitly pinned to our old endpoint
            self.assertEqual(out['fastModel'], 'openai:thinkingcap')
            self.assertEqual(out['compactionModel'], 'thinkingcap')

    def test_custom_openai_provider_group_keeps_its_plain_route(self):
        old = settings()
        old['modelProviders']['hosted'] = [dict(id='thinkingcap', envKey='CUSTOM_KEY')]
        old['providerProtocol'] = {'hosted': 'openai'}
        old.update(fastModel='openai:thinkingcap', compactionModel='thinkingcap')
        out = self.merge(old)
        self.assertEqual(out['fastModel'], old['fastModel'])
        self.assertEqual(out['compactionModel'], old['compactionModel'])
        self.assertEqual(out['providerProtocol'], old['providerProtocol'])
        self.assertEqual(out['modelProviders']['hosted'], old['modelProviders']['hosted'])

    def test_unrelated_legacy_provider_groups_are_left_for_client_migration(self):
        for legacy in (None, {'models': []}, ['invalid-entry']):
            old = settings()
            old['modelProviders']['hosted'] = legacy
            old['providerProtocol'] = {'hosted': 'openai'}
            out = self.merge(old)
            self.assertEqual(out['modelProviders']['hosted'], legacy)
            self.assertEqual(out['fastModel'], 'swift-1.5-nothink')

    def test_setup_writes_migrated_settings_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as td:
            old = settings()
            old.update(fastModel='openai:thinkingcap\0' + OLD, compactionModel='thinkingcap')
            original = json.dumps(old).encode()
            p = Path(td) / 'settings.json'
            p.write_bytes(original)
            srv = dict(name='swift', model='swift-1.5', port=8093)
            with patch.object(Q, 'running', return_value=[srv]), patch.object(Q, 'server_ids', return_value=(['swift-1.5', 'swift-1.5-nothink'], 8192)):
                Q.setup(home=td, say=lambda _: None)
            got = json.loads(p.read_text())
            self.assertEqual(got['model']['baseUrl'], NEW)
            self.assertEqual(got['fastModel'], 'swift-1.5-nothink')
            self.assertEqual(got['compactionModel'], 'swift-1.5')
            self.assertEqual(list(Path(td).glob('settings.json.bak-*'))[0].read_bytes(), original)
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
