"""setuptools glue for pyproject.toml (docs/dev/packaging.md). Four things pyproject cannot say:

  * build/ holds ./build.sh's code objects, so setuptools builds in .pybuild/ instead;
  * a wheel is refused unless the gfx1100 runtime set is staged in python/caminus/hsaco/gfx1100 (tools/release/stage_kernels.py) and
    every object matches its provenance.json: a wheel without its code objects would install and then fail at the first
    launch;
  * the wheel is a platform wheel, WHEEL_TAG (Root-Is-Purelib: false): it carries the panel, a static x86-64 Go binary, and
    nothing in it runs elsewhere. Static, so the oldest manylinux tag is the honest one; every pip that installs Python 3.12
    reads PEP 600 tags, so no manylinux2014 alias is needed;
  * a wheel is built only on x86-64 Linux (check_host): elsewhere pip falls back to the sdist, and it never checks the tag
    of a wheel it built itself, so without the refusal it would install the x86-64 panel there. The sdist, the metadata
    and editable builds are not refused."""
import hashlib
import json
import os
import platform
import sys

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build import build
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(HERE, 'python', 'caminus', 'hsaco', 'gfx1100')
WHEEL_TAG = 'py3-none-manylinux_2_17_x86_64'   # tools/release/verify_release.py reads it from here


def check_host():
    machine = platform.machine()
    if not (sys.platform.startswith('linux') and machine == 'x86_64'):
        raise SystemExit(f'caminus: Caminus runs on x86-64 Linux with AMD gfx1100 GPUs; this machine is {sys.platform} {machine}')


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


class PlatformDistribution(Distribution):
    # bdist_wheel writes Root-Is-Purelib: false and installs the package at the wheel's root (platlib) only for a
    # distribution with extension modules; this one has none, but its wheel is as platform-bound as one that had
    def has_ext_modules(self):
        return True


class BdistWheel(bdist_wheel):
    def run(self):
        check_host()
        super().run()

    def get_tag(self):
        return tuple(WHEEL_TAG.split('-'))


setup(distclass=PlatformDistribution, cmdclass={'build': Build, 'build_py': BuildPy, 'bdist_wheel': BdistWheel})
