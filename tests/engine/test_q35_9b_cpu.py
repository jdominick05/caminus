"""9B routing, actual target/verify/F16-MTP tapes and projections, CPU only."""
import argparse
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_q35
from caminus import arch
from caminus.device import hip
from caminus.engine import k3
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify
from caminus.engine.q35_mtp import Mtp35, block_bytes
from caminus.prefill import attn, gdn, glue, q35own

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); args = ap.parse_args()
scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
path = scratch / 'nine.gguf'
tiny_q35.write(path, variant='nine', mtp=True)
with contextlib.ExitStack() as st:
    for name in ('init', '_lazy_init'):
        st.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError('GPU initialization forbidden')))
    st.enter_context(patch.object(torch.cuda, 'device', side_effect=lambda d: contextlib.nullcontext()))
    st.enter_context(patch.object(torch.cuda, 'set_device'))
    st.enter_context(patch.object(torch.cuda, 'current_device', return_value=0))
    st.enter_context(patch.object(torch.cuda, 'get_device_properties', return_value=NS(multi_processor_count=42)))
    st.enter_context(patch.object(torch.cuda, 'Event', side_effect=lambda *a, **k: NS()))
    st.enter_context(patch.object(torch.cuda, 'current_stream', return_value=NS(device=torch.device('cuda:0'))))
    for mod in (attn, gdn, glue, hip):
        st.enter_context(patch.object(mod, 'Module', side_effect=lambda name: NS(name=name)))
    st.enter_context(patch.object(k3, 'resident_grid', return_value=84))
    zeros, to = torch.zeros, torch.Tensor.to
    st.enter_context(patch.object(torch, 'zeros', side_effect=lambda *a, **kw: zeros(*a, **{k:v for k,v in kw.items() if k not in ('device','pin_memory')})))
    st.enter_context(patch.object(torch.Tensor, 'to', lambda t,*a,**kw: t if a and str(a[0]).startswith('cuda') else to(t,*a,**kw)))
    st.enter_context(patch.object(torch.Tensor, 'cuda', lambda t,*a,**kw: t))
    a = arch.of(str(path))
    assert a.name == 'qwen35-own-9b' and a.drafters == ('mtp',)
    m = Dense35Own(str(path), max_ctx=64, spec_T=4, vram_check=False)
    assert m.module == 'k3_q35_9b' and m.nh == 16 and m.nkv == 4
    assert m.E.refuse is None and not m.E.unsupported()
    for T in range(1,5):
        v = Dense35Verify(m,T,vram_check=False)
        assert v.module == 'k3_q35_9b_ver' and v.E.refuse is None
        assert v.module not in k3.ATTN_PACK_MODULES
        assert all(s.team == T for s in v.E.segs) if T > 1 else True
    d = Mtp35(m,k=3,vram_check=False)
    m.mtp35 = d
    pf = q35own.Q35OwnPrefill(m, chunk=16, check=False)
    assert pf.export_h and pf.O.attn_wm_ok(m.nh,m.nkv,m.hd)
    assert d.f16 and d.module == 'k3_q35_9b_mtp'
    assert all(w['fmt']=='F16' and w['w'].dtype==torch.float16 for w in d.W.values())
    assert all(n.dtype==torch.float32 for n in d.N.values())
    raw = sum(t.n_bytes for name,t in m.F.T.items() if name.startswith(f'blk.{d.il}.'))
    assert block_bytes(m.F,d.il) - raw == 2 * (5*m.D+2*m.hd)
    assert d.mib == Mtp35.projection_mib(str(path),64,k=3)
    for E in d.E.values():
        assert E.refuse is None and not E.unsupported()
        assert sum(s.op==k3.OP_OWN_MVF16 for s in E.segs)==8
        assert k3.ATTN_G_OF[E.module]==4
        assert all(s.dep < i and s.dep2p1-1 < i for i,s in enumerate(E.segs))
    assert all(k3.ATTN_G_OF[n]==6 for n in ('k3_q35_swift','k3_q35_swift_mtp','k3_q35_tcap'))
    try: Mtp35(m,module='k3_q35_swift_mtp',vram_check=False)
    except ValueError: pass
    else: raise AssertionError('Wrong MTP module admitted')
    m.F.close()
print('PASS actual9B target, all verify widths and F16 MTP catch/fill/chain tapes; projections and routing')
print('RESULT PASS')
