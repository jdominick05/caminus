"""Independent black-box acceptance; run outside the agent's working directory."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

workspace=Path(sys.argv[1]).resolve()
program=workspace/'report.py'
assert program.is_file(), 'report.py was not created'
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);source=root/'input.csv';target=root/'result.json'
    def run():
        return subprocess.run([sys.executable,'-I',str(program),str(source),str(target)],cwd=root,capture_output=True,timeout=5)
    for rows,expected in [
        ([['b','2.005'],['a','9007199254740993.01'],['b','0.005'],['a','-9007199254740993.00']],
         [dict(team='a',count=2,total='0.01'),dict(team='b',count=2,total='2.01')]),
        ([['équipe, one','1.025'],['équipe, one','0.000'],['Zulu','-0.015']],
         [dict(team='Zulu',count=1,total='-0.02'),dict(team='équipe, one',count=2,total='1.02')]),
        ([],[]),
    ]:
        with source.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['team','amount']);w.writerows(rows)
        p=run();assert p.returncode==0,p.stderr.decode(errors='replace')
        assert json.loads(target.read_text())==expected, target.read_text()
    for invalid in ['team,amount\na,nan\n','team,amount\na,Infinity\n','team,amount\na,invalid\n','team,amount\n,1.00\n','group,value\na,1.00\n']:
        source.write_text(invalid);target.write_text('previous output')
        assert run().returncode!=0,'invalid input accepted: '+invalid
        assert target.read_text()=='previous output','invalid input overwrote output'
print('RESULT PASS')
