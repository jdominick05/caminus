"""The comparison runner must judge effects, isolate runs and retain failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

ap=argparse.ArgumentParser();ap.add_argument('--scratch',required=True);a=ap.parse_args()
scratch=Path(a.scratch).resolve();scratch.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]


class Comparison(unittest.TestCase):
    def setUp(self):
        self.home=scratch/uuid.uuid4().hex;self.case=self.home/'case';(self.case/'seed').mkdir(parents=True)
        (self.case/'goal.txt').write_text('Read seed.txt and create proof.txt with exactly the same text.')
        (self.case/'seed/seed.txt').write_text('verified-effect')
        (self.case/'verify.py').write_text("from pathlib import Path\nimport sys\np=Path(sys.argv[1]);assert (p/'proof.txt').read_text()==(p/'seed.txt').read_text()\n")
        self.profile=self.home/'profile.json';self.profile.write_text(json.dumps(dict(mode='CPU fixtures; no performance claim')))

    def run_compare(self,other=None,timeout=5):
        agent=[sys.executable,'-I','-c',"import sys;from pathlib import Path;p=Path('.');assert not (p/'proof.txt').exists();(p/'prompt.txt').write_text(sys.stdin.read());(p/'proof.txt').write_text((p/'seed.txt').read_text())"]
        p=subprocess.run([sys.executable,'-I',str(root/'benchmarks/resident_compare.py'),'--case',str(self.case),
                          '--out',str(self.home/'out'),'--profile',str(self.profile),'--repeats','2','--timeout',str(timeout),
                          '--qwen-argv',json.dumps(agent),'--resident-argv',json.dumps(other or agent)],capture_output=True,text=True,timeout=20)
        return p,json.loads((self.home/'out/results.json').read_text())

    def test_identical_goals_isolated_workspaces_and_alternating_order(self):
        p,r=self.run_compare();self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual([x['harness'] for x in r['runs']],['qwen','resident','resident','qwen'])
        self.assertEqual(len({x['workspace'] for x in r['runs']}),4)
        for run in r['runs']:
            self.assertTrue(run['correct']);self.assertEqual((Path(run['workspace'])/'prompt.txt').read_text(),(self.case/'goal.txt').read_text())
        self.assertEqual(r['summary']['resident']['passed'],2);self.assertIn('resident_speedup',r)

    def test_exit_zero_with_wrong_effects_fails_and_has_no_speedup(self):
        p,r=self.run_compare([sys.executable,'-I','-c',"from pathlib import Path;Path('proof.txt').write_text('wrong')"])
        self.assertEqual(p.returncode,1);self.assertEqual(r['summary']['resident']['passed'],0)
        self.assertNotIn('resident_speedup',r)

    def test_timeouts_are_failures(self):
        p,r=self.run_compare([sys.executable,'-I','-c','import time;time.sleep(10)'],timeout=.2)
        self.assertEqual(p.returncode,1)
        self.assertTrue(all(x['timed_out'] for x in r['runs'] if x['harness']=='resident'))
        self.assertNotIn('resident_speedup',r)


result=unittest.main(argv=[sys.argv[0]],exit=False).result
print('RESULT','PASS' if result.wasSuccessful() else 'FAIL')
raise SystemExit(not result.wasSuccessful())
