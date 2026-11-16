"""Small CPU fixtures: discovery inspects metadata and bounds local queries."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from caminus.ember.resident import discovery
from caminus.ember.resident.prompt import render


class Discovery(unittest.TestCase):
    def test_overview_never_executes_programs_or_exposes_environment(self):
        with tempfile.TemporaryDirectory() as td, \
             patch.object(discovery.subprocess,'run',side_effect=AssertionError('executed a program')), \
             patch.object(discovery.importlib.util,'find_spec',return_value=None), \
             patch.dict(os.environ,{'PRIVATE_TEST_TOKEN':'secret-not-an-inventory-value'}):
            result=discovery.discover(td)
        serialized=json.dumps(result)
        self.assertIn('not verified operation',result['scope'])
        self.assertNotIn('secret-not-an-inventory-value',serialized)
        self.assertNotIn('PRIVATE_TEST_TOKEN',serialized)
        self.assertEqual(result['python']['modules']['playwright'],'not found in this interpreter')
        self.assertIn('account',result);self.assertIn('hardware',result)

    def test_path_search_finds_existing_specialized_tool_without_running_it(self):
        with tempfile.TemporaryDirectory() as td:
            tool=Path(td)/'local-pdf-extractor'
            tool.write_text('#!/bin/sh\ntouch SHOULD_NOT_RUN\n');tool.chmod(0o755)
            with patch.dict(os.environ,{'PATH':td}):result=discovery.discover(td,'commands','pdf')
            self.assertEqual(result['matches'],{'local-pdf-extractor':str(tool)})
            self.assertFalse((Path(td)/'SHOULD_NOT_RUN').exists())

    def test_packages_query_is_exact_and_argv_only(self):
        calls=[]
        def probe(argv):calls.append(argv);return dict(status='ok')
        with patch.object(discovery.shutil,'which',side_effect=lambda name:'/tools/'+name), \
             patch.object(discovery,'_probe',side_effect=probe):
            result=discovery.discover('.','packages','tshark')
            for query in ('--config-file=other','pkg; touch file','*','foo bar',''):
                with self.assertRaises(ValueError):discovery.discover('.','packages',query)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1],['/tools/apt-cache','policy','--','tshark'])
        self.assertTrue(all('install' not in args and 'update' not in args for args in calls))
        self.assertIn('no index update',result['note'])

    def test_container_discovery_forces_local_socket_and_never_runs_an_image(self):
        calls=[]
        with tempfile.TemporaryDirectory() as td,socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(Path(td)/'docker.sock'))
            with patch.dict(os.environ,{'XDG_RUNTIME_DIR':td,'DOCKER_HOST':'tcp://remote.invalid:2375'}), \
                 patch.object(discovery.shutil,'which',side_effect=lambda name:'/tools/docker' if name=='docker' else None), \
                 patch.object(discovery,'_probe',side_effect=lambda argv:calls.append(argv) or dict(status='ok')):
                discovery.discover(td,'containers')
        self.assertEqual(calls,[['/tools/docker','--host','unix://'+td+'/docker.sock','image','ls','--format','{{.Repository}}:{{.Tag}}']])

    def test_probe_bounds_output_and_reports_timeout(self):
        def run(argv,**kwargs):
            self.assertEqual(kwargs['timeout'],3)
            self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
            self.assertNotIn('DOCKER_CONTEXT',kwargs['env'])
            self.assertFalse(kwargs.get('shell',False))
            kwargs['stdout'].write(b'x'*17000)
            return subprocess.CompletedProcess(argv,0)
        with patch.dict(os.environ,{'DOCKER_CONTEXT':'remote'}),patch.object(discovery.subprocess,'run',side_effect=run):
            result=discovery._probe(['apt-cache','policy','tshark'])
        self.assertTrue(result['truncated']);self.assertEqual(len(result['output']),16384)
        with patch.object(discovery.subprocess,'run',side_effect=subprocess.TimeoutExpired(['probe'],3)):
            self.assertEqual(discovery._probe(['probe'])['status'],'timed_out')

    def test_prompt_describes_actual_account_and_discovery_before_building(self):
        text=render('/agent/workspace')
        self.assertIn('/agent/workspace',text)
        self.assertIn('## Discover before building',text)
        self.assertIn('apt-get',text);self.assertIn('ask_root',text)
        self.assertIn('Task-specific restrictions',text)
        self.assertNotIn("You don't remember past sessions",text)


result=unittest.main(argv=[sys.argv[0]],exit=False).result
print('RESULT','PASS' if result.wasSuccessful() else 'FAIL')
raise SystemExit(not result.wasSuccessful())
