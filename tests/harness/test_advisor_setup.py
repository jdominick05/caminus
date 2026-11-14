"""API-first advisor setup with stubbed credentials/model discovery; no services, installs, sockets or user files."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from caminus.cmd import advisor as C

spec = importlib.util.spec_from_file_location('setup_advisor', C.SETUP)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class Wrapper(unittest.TestCase):
    def args(self, local=False):
        return SimpleNamespace(settings=['scratch-settings.json'], local_only=local, no_live_test=True, dry_run=True)

    def test_default_uses_api_only_setup_without_touching_homes(self):
        with patch.object(C, 'ensure_venv', return_value='sdk-python'), patch.object(C, 'run_setup', return_value=0) as run:
            self.assertEqual(C.cmd_setup(self.args(), lambda _: None), 0)
        py, args = run.call_args.args
        self.assertEqual(py, 'sdk-python')
        self.assertIn('--hosted-only', args)
        self.assertIn('--settings', args)
        self.assertIn('scratch-settings.json', args)
        self.assertIn('--no-live-test', args)
        self.assertIn('--dry-run', args)

    def test_explicit_local_compatibility_does_not_install_api_sdks(self):
        with patch.object(C, 'ensure_venv') as install, patch.object(C, 'run_setup', return_value=0) as run:
            self.assertEqual(C.cmd_setup(self.args(local=True), lambda _: None), 0)
        install.assert_not_called()
        self.assertIn('--local-only', run.call_args.args[1])
        self.assertNotIn('--hosted-only', run.call_args.args[1])

    def test_failed_sdk_install_does_not_fall_back_to_local_setup(self):
        messages = []
        with patch.object(C, 'ensure_venv', return_value=None), patch.object(C, 'run_setup') as run:
            self.assertEqual(C.cmd_setup(self.args(), messages.append), 1)
        run.assert_not_called()
        self.assertIn('retry setup', messages[-1])
        self.assertNotIn('--local-only', messages[-1])


class Configuration(unittest.TestCase):
    def configure(self, *, hosted=True, local=False, noninteractive=False, chain=(), responses=None):
        cfg = copy.deepcopy(S.config.DEFAULTS)
        cfg['provider'] = 'local'   # an existing local config can be deliberately reconfigured for API advice
        cfg['chain'] = list(chain)
        plans = []
        input_calls = []
        responses = iter(responses or ['none'])

        def choose(prompt, options, default=0):
            if prompt == 'advisors':
                plans.extend(options)
                return options[0]
            return options[default]

        def ask(prompt, default=''):
            input_calls.append((prompt, default))
            return default if noninteractive else next(responses)

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(S.config, 'CONFIG_PATH', str(Path(td) / 'config.toml')))
            stack.enter_context(patch.object(S.config, 'load', return_value=cfg))
            stack.enter_context(patch.object(S, 'A', SimpleNamespace(hosted_only=hosted, local_only=local, non_interactive=noninteractive)))
            stack.enter_context(patch.object(S, 'choose', choose))
            stack.enter_context(patch.object(S, 'ask', ask))
            stack.enter_context(patch.object(S, 'key_for', return_value='env:TEST_ADVISOR_KEY'))
            stack.enter_context(patch.object(S, 'anthropic_models', return_value=[('api-model', '')]))
            stack.enter_context(patch.object(S, 'mistral_models', return_value=[('api-model', 'reasoning')]))
            stack.enter_context(patch.object(S, 'pick_model', return_value='api-model'))
            local_setup = stack.enter_context(patch.object(S, 'configure_local'))
            finish = stack.enter_context(patch.object(S, 'finish', side_effect=lambda c: c))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                result = S.configure()
            except SystemExit:
                finish.assert_not_called()
                self.assertEqual(list(Path(td).iterdir()), [])
                raise
            return result, plans, input_calls, output.getvalue(), local_setup.call_count

    def test_api_plans_offer_no_local_model_or_discovery(self):
        cfg, plans, _, _, local_calls = self.configure()
        self.assertEqual(len(plans), 4)
        self.assertTrue(all('Local' not in p for p in plans))
        self.assertEqual(cfg['provider'], 'anthropic')
        self.assertEqual(cfg['fallback_provider'], '')
        self.assertEqual(cfg['chain'], [])
        self.assertEqual(local_calls, 0)

    def test_api_chain_keeps_supported_entries_and_reprompts_for_local(self):
        cfg, _, inputs, output, _ = self.configure(chain=['local:old'], responses=['local:old', 'mistral:api-model@high, anthropic:other'])
        self.assertEqual(cfg['chain'], ['mistral:api-model@high', 'anthropic:other'])
        self.assertEqual(len(inputs), 2)
        self.assertIn('API advisor setup accepts only', output)

    def test_noninteractive_rejects_local_chain_without_writing_configuration(self):
        with self.assertRaisesRegex(SystemExit, 'Existing config and Qwen settings were not changed'):
            self.configure(noninteractive=True, chain=['local:old'])

    def test_api_chain_rejects_unknown_provider_and_empty_model(self):
        for invalid in ('other:model', 'anthropic:', 'local', 'mistral:', 'anthropic:@high', 'mistral: @high'):
            with self.subTest(invalid=invalid):
                cfg, _, inputs, _, _ = self.configure(responses=[invalid, 'none'])
                self.assertEqual(cfg['chain'], [])
                self.assertEqual(len(inputs), 2)

    def test_noninteractive_api_chain_is_preserved(self):
        chain = ['mistral:api-model@high', 'anthropic:other']
        cfg, _, _, _, _ = self.configure(noninteractive=True, chain=chain)
        self.assertEqual(cfg['chain'], chain)

    def test_standalone_and_explicit_local_modes_keep_compatibility(self):
        cfg, plans, _, _, local_calls = self.configure(hosted=False, responses=['local:legacy'])
        self.assertEqual(len(plans), 7)
        self.assertEqual(cfg['chain'], ['local:legacy'])
        self.assertEqual(local_calls, 0)   # the chosen standalone plan was Anthropic
        cfg, plans, inputs, _, local_calls = self.configure(hosted=False, local=True, chain=['local:old', 'anthropic:other'])
        self.assertEqual(len(plans), 1)
        self.assertEqual(cfg['provider'], 'local')
        self.assertEqual(cfg['chain'], ['local:old'])
        self.assertEqual(local_calls, 1)
        self.assertEqual(inputs, [])


class SetupCli(unittest.TestCase):
    BOOT = '''import importlib.util, sys
path = sys.argv[1]
sys.argv = sys.argv[1:]
spec = importlib.util.spec_from_file_location('setup_advisor', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
s.key_for = lambda *a: 'env:TEST_ADVISOR_KEY'
s.anthropic_models = lambda *a: [('api-model', '')]
s.mistral_models = lambda *a: [('api-model', 'reasoning')]
s.main()
'''

    def test_actual_cli_writes_only_scratch_api_settings_and_preserves_rejected_chain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / 'qwen-home'
            home.mkdir()
            settings = home / 'settings.json'
            settings.write_text(json.dumps({'custom': 'preserve'}))
            cfg = root / 'cfg' / 'qwen-advisor' / 'config.toml'
            env = dict(os.environ, HOME=td, XDG_CONFIG_HOME=str(root / 'cfg'), XDG_DATA_HOME=str(root / 'data'),
                       XDG_STATE_HOME=str(root / 'state'), QWEN_ADVISOR_CONFIG=str(cfg), CAMINUS_QWEN_HOME=str(home))
            cmd = [sys.executable, '-I', '-c', self.BOOT, C.SETUP, '--hosted-only', '--non-interactive',
                   '--no-live-test', '--settings', str(settings)]
            first = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            got = S.config.load(str(cfg))
            self.assertEqual(got['provider'], 'anthropic')
            self.assertEqual(got['chain'], [])
            self.assertEqual(json.loads(settings.read_text())['custom'], 'preserve')
            self.assertEqual(cfg.stat().st_mode & 0o777, 0o600)
            # A local chain must be corrected explicitly; noninteractive API setup cannot silently erase it.
            got['chain'] = ['local:legacy']
            cfg.write_text(S.toml_dump(got))
            before_cfg, before_settings = cfg.read_bytes(), settings.read_bytes()
            rejected = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('API advisor setup accepts only', rejected.stderr)
            self.assertEqual(cfg.read_bytes(), before_cfg)
            self.assertEqual(settings.read_bytes(), before_settings)


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    print('RESULT', 'PASS' if result.wasSuccessful() else 'FAIL')
    sys.exit(not result.wasSuccessful())
