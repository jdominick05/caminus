"""setuptools glue for pyproject.toml (docs/dev/packaging.md). Two things pyproject cannot say:

  * build/ holds ./build.sh's code objects, so setuptools builds in .pybuild/ instead;
  * a wheel is refused unless the gfx1100 runtime set is staged in python/caminus/hsaco/gfx1100 (tools/release/stage_kernels.py) and
    every object matches its provenance.json: a wheel without its code objects would install and then fail at the first
    launch."""
import hashlib
import json
import os
import sys

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(HERE, 'python', 'caminus', 'hsaco', 'gfx1100')


def check_staged():
    sys.path.insert(0, os.path.join(HERE, 'python'))
    from caminus.device import kernels
    sys.path.pop(0)
    hint = 'stage them first: ./build.sh && python -I tools/release/stage_kernels.py'
    p = os.path.join(STAGED, 'provenance.json')
    if not os.path.isfile(p):
        raise SystemExit(f'caminus: no staged code objects ({p} missing): {hint}')
    with open(p) as f:
        objs = json.load(f)['objects']
    for n in kernels.RUNTIME_SET:
        nm = n + '.hsaco'
        f = os.path.join(STAGED, nm)
        if nm not in objs or not os.path.isfile(f):
            raise SystemExit(f'caminus: staged code object {nm} missing: {hint}')
        with open(f, 'rb') as fh:
            if hashlib.sha256(fh.read()).hexdigest() != objs[nm]['sha256']:
                raise SystemExit(f'caminus: staged {nm} does not match provenance.json: {hint}')
    extra = sorted(x for x in os.listdir(STAGED) if x.endswith('.hsaco') and x[:-6] not in kernels.RUNTIME_SET)
    if extra:
        raise SystemExit(f'caminus: {STAGED} holds objects outside the runtime set {extra}: stage into an empty directory')


class Build(build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = '.pybuild'


class BuildPy(build_py):
    def run(self):
        check_staged()
        super().run()


setup(cmdclass={'build': Build, 'build_py': BuildPy})
