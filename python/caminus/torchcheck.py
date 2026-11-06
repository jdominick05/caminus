"""PyTorch with ROCm: required, never declared as a dependency (docs/RELEASE-PLAN.md). PyPI's torch is a CUDA build, so a
declared `torch` would make pip install the wrong one; the ROCm build comes from PyTorch's own index. The engine reaches the
HIP runtime through the libamdhip64.so that torch ships (caminus.hip). Importing this module imports nothing heavy."""

ROCM_INDEX = 'https://download.pytorch.org/whl/rocm7.2'
INSTALL = f'pip install torch --index-url {ROCM_INDEX}'
TESTED = '2.14.1+rocm7.2'


def status():
    """(ok, torch version or None, detail): ok when torch imports and is a ROCm (HIP) build. Imports torch, initializes no
    device (torch.version only)"""
    try:
        import torch
    except ImportError as e:
        return False, None, f'PyTorch is not installed ({e}); install the ROCm build: {INSTALL}'
    hip = getattr(torch.version, 'hip', None)
    if not hip:
        return False, torch.__version__, (f'PyTorch {torch.__version__} is not a ROCm build (torch.version.hip is None: a CUDA or CPU '
                                          f'build, e.g. from PyPI); replace it with the ROCm build: {INSTALL}')
    return True, torch.__version__, f'PyTorch {torch.__version__}, HIP {hip} (tested: {TESTED})'


def refusal(what='caminus'):
    """None when a ROCm PyTorch imports; else the one line a command prints before it exits 2 (`what`: its name), with the
    install command: never a raw ImportError or a CUDA build's later failure"""
    ok, _, detail = status()
    return None if ok else f'{what}: {detail}'


def absent(what='caminus'):
    """refusal() for a missing torch only, without importing it (importing torch starts its threads: caminus serve names its
    process first); a CUDA build is refused at the first HIP call (caminus.hip, require())"""
    import importlib.util
    try:
        found = importlib.util.find_spec('torch') is not None
    except (ImportError, ValueError):
        found = False
    return None if found else f'{what}: PyTorch is not installed; install the ROCm build: {INSTALL}'


def require(what='caminus'):
    """raise RuntimeError with the install command unless a ROCm PyTorch is importable"""
    ok, _, detail = status()
    if not ok:
        raise RuntimeError(f'{what}: {detail}')
