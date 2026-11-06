"""Paths and small helpers shared by the gate harnesses in tools/gates (CPU-only module: nothing here touches a GPU).

Every path comes from CLAUDE.md's variables (environment CODER / REAP / ARTIFACTS; unset: the placeholder '$NAME', so a
read names the variable to set) or from a command-line argument; nothing reads the ephemeral job scratch dir. The stored references and their provenance
are listed in $ARTIFACTS/refs/MANIFEST.md (sha256 list beside it)."""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.environ.get('ARTIFACTS') or '$ARTIFACTS'
CODER = os.environ.get('CODER') or '$CODER'
REAP = os.environ.get('REAP') or '$REAP'
REFS = os.path.join(ARTIFACTS, 'refs')
# the M0 llama.cpp dumps (tools/gg_golden, q8_0 KV) of the three e2e texts: the D-281 short gate's references, and their
# tokens.txt the teacher-forced token sequence of every bitwise gate (the Coder and REAP-308 tokenize these texts the same)
D281_SHORT = os.path.join(REFS, 'd281-short')
TOKENS = os.path.join(D281_SHORT, 'e2e_code_decode', 'tokens.txt')
# the stored bitwise refdumps (written by the job's gate scripts into ARTIFACTS through the job dir's symlinks; current
# set: MANIFEST.md): the Coder's trunk / QSA dumps and REAP-308's
REF_CODER = os.path.join(ARTIFACTS, 'm1', 's3', 'ref')
REF_REAP = os.path.join(ARTIFACTS, 'm1', 's6', 'verify', 'ref')
LC = os.path.join(ARTIFACTS, 'qsa', 'lc')   # the long-context probes and llama.cpp references (tools/lc_probes.py, gg_golden --lc)


def model_path(name):
    """--model: 'coder' (default), 'reap' or a GGUF path"""
    return {'coder': CODER, 'reap': REAP}.get(name, name)


def opt(a, name, default, f=str):
    """pop `name VALUE` from the argument list a (default when absent)"""
    if name in a:
        i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
    return default


def proactiveness():
    """vm.compaction_proactiveness (CLAUDE.md: recorded with every timing run; host page compaction evicts KFD queues)"""
    try: return open('/proc/sys/vm/compaction_proactiveness').read().strip()
    except OSError: return 'unreadable'


def tokens(path):
    return [int(v) for v in open(path).read().split()]


def engine_kwargs(max_ctx=None, qsa=None, tape=None):
    """FlashNextTP keyword arguments for the options given (none: the engine's defaults, as the old trunk scripts called it)"""
    kw = {}
    if max_ctx is not None: kw['max_ctx'] = int(max_ctx)
    if qsa is not None: kw['qsa'] = qsa == '1'
    if tape is not None: kw['qsa_tape'] = tape
    return kw


def setup_torch():
    """import caminus from this repo and torch, with the per-card memory fractions every gate uses (XTX 0.97: the fixed
    0.5 GB reserve for the monitors; XT 0.95)"""
    sys.path.insert(0, os.path.join(REPO, 'python'))
    import torch
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    return torch
