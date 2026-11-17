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

    def test_commands_query_matches_any_word_against_names_and_man_descriptions(self):
        # Canned `apropos --long` lines as man-db prints them; the stubs' names hold none of the query words.
        apropos=('tesseract (1)        - command-line OCR engine\n'
                 'pandoc (1)           - general markup converter\n'
                 'pdftotext (1)        - Portable Document Format (PDF) to text converter\n'
                 'grep (1)             - print lines that match patterns\n')
        calls=[]
        def probe(argv,limit=16384):
            calls.append(argv);return dict(status='ok',output=apropos)
        with tempfile.TemporaryDirectory() as td:
            for name in ('apropos','tesseract','pandoc','ffmpeg','pdftotext','pdfinfo','pdf-and-ocr-tool'):
                (Path(td)/name).write_text('#!/bin/sh\ntouch SHOULD_NOT_RUN\n');(Path(td)/name).chmod(0o755)
            with patch.dict(os.environ,{'PATH':td}),patch.object(discovery,'_probe',side_effect=probe):
                ocr=discovery.discover(td,'commands','ocr')
                multi=discovery.discover(td,'commands','search ocr pdf')
                video=discovery.discover(td,'commands','video')
                odd=discovery.discover(td,'commands','.* -- ?')
                flag=discovery.discover(td,'commands','ocr --config-file=x')
                filler=discovery.discover(td,'commands','tool to ocr a pdf')
            self.assertFalse((Path(td)/'SHOULD_NOT_RUN').exists())
        self.assertEqual(ocr['matches'],{'tesseract':td+'/tesseract','pdf-and-ocr-tool':td+'/pdf-and-ocr-tool'})
        self.assertEqual(ocr['about']['tesseract'],'command-line OCR engine')
        # Ranked by the words hit, each weighted by rarity (ocr hits 2 tools, pdf 3); grep is on no PATH here.
        self.assertEqual(list(multi['matches']),['pdf-and-ocr-tool','tesseract','pdfinfo','pdftotext'])
        self.assertIn('ffmpeg',video['matches'])          # the man page says "media converter": the short table's word
        self.assertEqual(odd['matches'],{});self.assertIn('tesseract',flag['matches'])
        # "a" and "to" hit almost every name: beside real words they are dropped, and they never reach apropos
        self.assertEqual(list(filler['matches'])[:2],['pdf-and-ocr-tool','tesseract'])
        self.assertFalse({'a','to'}&{argv[-1] for argv in calls})
        self.assertTrue(all(argv[-1]!='.*' and '--config-file=x' not in argv[argv.index('--')+1:] for argv in calls))
        with self.assertRaises(ValueError):discovery.discover('.','commands','  ')

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
