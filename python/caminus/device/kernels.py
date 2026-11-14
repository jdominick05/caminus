"""Where the engine's gfx1100 code objects come from, and what they were built from (docs/dev/packaging.md, "Code objects").

kernel_dir(): CAMINUS_KERNEL_DIR (or kernel_dir in ~/.config/caminus/config.toml), else a checkout's build/ (this package
imported from python/ beside build.sh and kernels/: every test and tool keeps reading ./build.sh's output), else the package's
own hsaco/gfx1100 (an installed wheel). caminus.device.hip loads from it, and the prompt cache's engine key hashes its *.hsaco.

The installed package carries RUNTIME_SET (what serving, replay and `caminus doctor --gpu` load), staged from build/ by
tools/release/stage_kernels.py together with provenance.json: the toolchain, each object's compile command as build.sh ran it, the
source commit and dirty flag, a content hash of kernels/ + build.sh, the ggml headers' commit, each object's sha256 and size.
Not shipped: the *_trace profiling builds, the *_swiglu test builds, gg_mmvf_ref and gg_mv_ref, the dormant S1 engine and training kernels,
the probes (build them from a checkout and point CAMINUS_KERNEL_DIR at build/)."""
import hashlib
import json
import os

from .. import config

ARCH = 'gfx1100'
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # (the caminus package: this module is caminus.device.kernels)
PACKAGE_DIR = os.path.join(PKG, 'hsaco', ARCH)
PROVENANCE = 'provenance.json'

K3 = ('k3_gguf_gsq', 'k3_gguf_gsq_mt', 'k3_gguf_gsq_qsa', 'k3_gguf_gsq_qsa_mt', 'k3_gguf_gsq_mtp', 'k3_gguf_gsq_mtp_mt',
      'k3_gguf_draft', 'k3_gguf', 'k3_gguf_mt', 'k3_guard',
      # the one-card engines (caminus.engine.qwen35_one: Qwen3.6 HauhauCS / GSQ, AREX-2) and Qwen3-Coder-Next (caminus.engine.qwen3next_tp),
      # T = 1 as caminus serve runs them
      'k3_gguf_q35h', 'k3_gguf_q35g', 'k3_gguf_q35d', 'k3_gguf_q3n')
MMQ_FORMATS = ('iq2_s', 'iq3_s', 'iq3_xxs', 'iq4_nl', 'iq4_xs', 'q2_0', 'q3_k', 'q4_k', 'q5_k', 'q6_k', 'q8_0', 'glue', 'glue8')   # build.sh's lists
MOE_FORMATS = ('iq2_s', 'iq3_s', 'iq3_xxs', 'iq4_xs', 'iq4_nl', 'q2_0', 'glue')
PREFILL = (('prefill_attn', 'prefill_feed', 'prefill_gdn', 'prefill_glue', 'prefill_tp', 'prefill_dense_q6_k', 'gemm',
            'prefill_glue_q35', 'prefill_glue_q35d', 'prefill_gdn_q35', 'prefill_attn_q35', 'prefill_attn_q35d')   # (caminus.prefill.q35)
           + tuple(f'prefill_mmq_{f}' for f in MMQ_FORMATS) + tuple(f'prefill_moe_{f}' for f in MOE_FORMATS))
# The own engines select these from the file's formats, including custom GGUFs.
# Keep them explicit: doctor and wheel staging must not import torch or engines.
# tests/packaging/test_runtime_kernels.py checks the real host selections against
# this list without a GPU. Adding a serving module requires extending both.
Q3_OWN = ('k3_q3m_lean', 'k3_q3m_iq', 'k3_q3m_rs4', 'k3_q3m_rs2', 'q3m_prefill', 'q3m_prefill_iq')
DENSE35_OWN = ('k3_q35_swift', 'k3_q35_swift_tp', 'k3_q35_swift_ver', 'k3_q35_swift_mtp', 'k3_q35_dflash', 'q35_prefill',
               'k3_q35_tcap', 'k3_q35_tcap_tp', 'k3_q35_tcap_ver', 'k3_q35_tcap_mtp', 'k3_q35_tcap_dflash', 'q35_prefill_tcap',
               'k3_q35_gsq', 'q35_prefill_gsq', 'k3_q35_gsq_ver', 'k3_q35_gsq_ver_m',
               'k3_q35_gsq_mtp', 'k3_q35_gsq_dflash')   # paused plain 27B: packaging does not unpause the catalog
# TP verify remains a phase-one test path until TP drafting is served. Trace,
# ablation and standalone probes are also deliberately outside the runtime set.
RUNTIME_SET = K3 + PREFILL + Q3_OWN + DENSE35_OWN + ('own_topk', 'smoke')


def checkout_build():
    """a checkout's build/ when this package is imported from <repo>/python/caminus (None otherwise)"""
    root = os.path.dirname(os.path.dirname(PKG))
    if os.path.isfile(os.path.join(root, 'build.sh')) and os.path.isdir(os.path.join(root, 'kernels')):
        return os.path.join(root, 'build')
    return None


def kernel_dir():
    """(directory, where): where 'env CAMINUS_KERNEL_DIR' / 'file ...' / 'checkout' / 'package'"""
    v, w = config.lookup('kernel_dir')
    if v:
        return os.path.abspath(v), w
    b = checkout_build()
    if b is not None:
        return b, 'checkout'
    return PACKAGE_DIR, 'package'


def missing_message(name, d, where):
    if where == 'package':
        hint = (f'{name} is not part of the installed runtime set (kernels.RUNTIME_SET; trace and test builds are not shipped): '
                'build it from a checkout with ./build.sh and set CAMINUS_KERNEL_DIR=<checkout>/build')
    elif where == 'checkout':
        hint = 'run ./build.sh in the checkout'
    else:
        hint = f'the directory comes from {where}'
    return f'code object {name}.hsaco not found in {d} ({hint})'


def provenance(d=None):
    """the directory's provenance.json (None when it has none, e.g. a checkout's build/)"""
    d = d or kernel_dir()[0]
    p = os.path.join(d, PROVENANCE)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def verify(d=None):
    """check the directory's objects against its provenance: [(name, ok, detail)] for every object provenance lists and every
    RUNTIME_SET name; [] with prov None when the directory has no provenance"""
    d = d or kernel_dir()[0]
    prov = provenance(d)
    if prov is None:
        return None, []
    out = []
    objs = prov.get('objects', {})
    for nm in sorted(set(objs) | {n + '.hsaco' for n in RUNTIME_SET}):
        p = os.path.join(d, nm)
        if nm not in objs:
            out.append((nm, False, 'not listed in provenance.json'))
        elif not os.path.isfile(p):
            out.append((nm, False, 'missing'))
        else:
            got = sha256(p)
            out.append((nm, got == objs[nm]['sha256'], 'sha256 ' + ('matches' if got == objs[nm]['sha256'] else f'{got[:12]} != {objs[nm]["sha256"][:12]}')))
    return prov, out


def describe(d=None):
    """one line: where the code objects come from and what built them"""
    if d is None:
        d, where = kernel_dir()
    else:
        where = 'given'
    prov = provenance(d)
    if prov is None:
        return f'code objects: {d} ({where}; no provenance.json: built in place)'
    s, t = prov.get('source', {}), prov.get('toolchain', {})
    return (f'code objects: {d} ({where}): {len(prov.get("objects", {}))} for {prov.get("arch")}, built {prov.get("built", "?")} '
            f'from {s.get("commit", "?")[:12]}{" (dirty)" if s.get("dirty") else ""} (kernel tree {s.get("kernel_tree_sha256", "?")[:12]}) '
            f'with {t.get("name", "?")} {t.get("version", "?")}')
