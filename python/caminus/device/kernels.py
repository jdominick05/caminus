"""Where the engine's gfx1100 code objects come from, and what they were built from (docs/dev/packaging.md, "Code objects").

kernel_dir(): CAMINUS_KERNEL_DIR (or kernel_dir in ~/.config/caminus/config.toml), else a checkout's build/ (this package
imported from python/ beside build.sh and kernels/: every test and tool keeps reading ./build.sh's output), else the package's
own hsaco/gfx1100 (an installed wheel). caminus.device.hip loads from it, and the prompt cache's engine key hashes its *.hsaco.

The installed package carries RUNTIME_SET (what serving, replay and `caminus doctor --gpu` load), staged from build/ by
tools/release/stage_kernels.py together with provenance.json: the toolchain, each object's compile command as build.sh ran it, the
source commit and dirty flag, a content hash of kernels/ + build.sh, the ggml headers' commit, each object's sha256 and size.
Not shipped: the *_trace profiling builds, ref_gg_mmvf and ref_gg_mv, the dormant S1 engine and training kernels,
the probes (build them from a checkout and point CAMINUS_KERNEL_DIR at build/)."""
import hashlib
import json
import os

from .. import config

ARCH = 'gfx1100'
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # (the caminus package: this module is caminus.device.kernels)
PACKAGE_DIR = os.path.join(PKG, 'hsaco', ARCH)
PROVENANCE = 'provenance.json'

K3 = ('dec_q4x_gsq', 'dec_q4x_gsq_ver', 'dec_q4x_gsq_qsa', 'dec_q4x_gsq_qsa_ver', 'dec_q4x_gsq_mtp', 'dec_q4x_gsq_mtp_ver',
      'dec_q4x_dflash', 'dec_q4x_gg', 'dec_q4x_gg_ver', 'dec_guard',
      # the one-card engines (caminus.engine.qwen35_one: Qwen3.6 HauhauCS / GSQ, AREX-2) and Qwen3-Coder-Next (caminus.engine.qwen3next_tp),
      # T = 1 as caminus serve runs them
      'dec_q35m_hh', 'dec_q35m_gg', 'dec_q35_arex', 'dec_q3n')
MMQ_FORMATS = ('iq2_s', 'iq3_s', 'iq3_xxs', 'iq4_nl', 'iq4_xs', 'q2_0', 'q3_k', 'q4_k', 'q5_k', 'q6_k', 'q8_0', 'glue', 'glue8')   # build.sh's lists
MOE_FORMATS = ('iq2_s', 'iq3_s', 'iq3_xxs', 'iq4_xs', 'iq4_nl', 'q2_0', 'glue')
PREFILL = (('pf_gg_attn', 'pf_feed', 'pf_gg_gdn', 'pf_gg_glue', 'pf_tp', 'pf_gg_dense_q6_k', 'pf_gemm',
            'pf_gg_glue_q35m', 'pf_gg_glue_q35', 'pf_gg_gdn_q35', 'pf_gg_attn_q35m', 'pf_gg_attn_q35')   # (caminus.prefill.q35)
           + tuple(f'pf_gg_mmq_{f}' for f in MMQ_FORMATS) + tuple(f'pf_gg_moe_{f}' for f in MOE_FORMATS))
# The own engines select these from the file's formats, including custom GGUFs.
# Keep them explicit: doctor and wheel staging must not import torch or engines.
# tests/packaging/test_runtime_kernels.py checks the real host selections against
# this list without a GPU. Adding a serving module requires extending both.
Q3_OWN = ('dec_q3m', 'dec_q3m_iq', 'dec_q3m_rs4', 'dec_q3m_rs2', 'pf_q3m', 'pf_q3m_iq',
          'pf_q3m_mm',   # (the one-card prefill's wide GEMM tiles: caminus.prefill.q3m_ops.WIDE_MODULE)
          # its two-card serve (caminus serve --cards 2: engine/q3moe_tp.py's decode module, prefill/q3m_tp.py's slot sum;
          # the exchange's add is pf_tp, in PREFILL)
          'dec_q3m_tp', 'pf_q3m_tp',
          # its 8-bit KV cache (caminus serve --kv q8): the one- and two-card decode modules and the prefill's Q8 rows and
          # attention (pf_q3m_kvq8: one object for both card counts); and its long-context mode (caminus serve
          # --long-context: YaRN x4): the YaRN builds of the decode modules and the prefill's qk_rope, F16 and Q8_0 rows
          'dec_q3m_kvq8', 'dec_q3m_tp_kvq8', 'pf_q3m_kvq8',
          'dec_q3m_yarn', 'dec_q3m_kvq8_yarn', 'dec_q3m_tp_yarn', 'dec_q3m_tp_kvq8_yarn',
          'pf_q3m_yarn', 'pf_q3m_kvq8_yarn')
DENSE35_OWN = ('dec_q35_swift', 'dec_q35_swift_tp', 'dec_q35_swift_tp_ver', 'dec_q35_swift_ver', 'dec_q35_swift_mtp', 'dec_q35_swift_dflash', 'pf_q35_swift',
               'dec_q35_tcap', 'dec_q35_tcap_tp', 'dec_q35_tcap_tp_ver', 'dec_q35_tcap_ver', 'dec_q35_tcap_mtp', 'dec_q35_tcap_dflash', 'pf_q35_tcap',
               'dec_q35_gsq', 'pf_q35_gsq', 'dec_q35_gsq_ver', 'dec_q35_gsq_ver_m',
               'dec_q35_gsq_mtp', 'dec_q35_gsq_dflash')   # paused plain 27B: packaging does not unpause the catalog
# The two-card verify objects (*_tp_ver) serve two-card drafting; its drafters are the one-card MTP / DFlash objects.
# Trace, ablation and standalone probes are deliberately outside the runtime set.
# The XTX's objects of the own decode modules: build.sh's <name>_w4 (a 4-workgroups-per-WGP grid, VGPR cap 192 instead of
# the XT's 144: kernels/dec_core.hip K3_PER_WGP); caminus.engine.k3.card_object loads one on a card whose grid is 4 per WGP
CARD_W4 = tuple(n + '_w4' for n in Q3_OWN if n.startswith('dec_q3m'))
# Qwen3.6-35B-A3B on own code (engine/q35_moe.py, prefill/q35moe.py): its decode module and the Q8 KV build (their YaRN
# mode is the same objects: the table is a buffer), the prefill's own object, its Q8 rows and attention, and the prefill
# attention object's YaRN build (pf_qk_nr's table)
Q35MOE_OWN = ('dec_q35m', 'dec_q35m_kvq8', 'pf_q35m', 'pf_q35m_kvq8', 'pf_gg_attn_q35m_yarn',
              'pf_q35m_mm')   # (the one-card prefill's wide expert tiles: caminus.prefill.q35moe.WIDE_MODULE)
RUNTIME_SET = K3 + PREFILL + Q3_OWN + DENSE35_OWN + CARD_W4 + Q35MOE_OWN + ('samp_topk', 'smoke')


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
