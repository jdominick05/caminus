"""Guarded non-display first gate: raw F16 MTP matvec and group4 attention.

Run only through the repository guard / GPU window. Device math against FP64,
T-token rows versus independent rows, output canaries. No model-quality claim.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from caminus.engine import k3
from caminus.device import capacity

capacity.check([0],[128],'9B op gate buffers')
rng=np.random.default_rng(871)
for rows,K in ((5,64),(17,256),(32,4096),(16,8192)):
    w=rng.normal(0,0.04,(rows,K)).astype(np.float16)
    x=rng.normal(0,0.08,(4,K)).astype(np.float16)
    W=torch.from_numpy(w.reshape(-1)).cuda(); X=torch.from_numpy(x.reshape(-1)).cuda()
    independent=[]
    for T in (1,2,3,4):
        E=k3.Engine(module='k3_q35_9b_mtp',T=T,limit_us=200000)
        buf=torch.full((T*rows+2,),12345.0,device='cuda'); out=buf[1:-1]
        E.own_matvec_f16(W,rows,K,X[:T*K],out,tokens=T)
        E.build(); E.run(check=True); torch.cuda.synchronize()
        y=out.cpu().numpy().reshape(T,rows)
        ref=x[:T].astype(np.float64)@w.astype(np.float64).T
        norm=np.abs(x[:T].astype(np.float64))@np.abs(w.astype(np.float64)).T
        assert np.max(np.abs(y-ref)/(norm+1e-30))<2e-6,(rows,K,T,'accuracy')
        assert float(buf[0])==12345 and float(buf[-1])==12345,'matvec canary'
        if T==4: full=y.copy()
    for i in range(4):
        E=k3.Engine(module='k3_q35_9b_mtp',T=1,limit_us=200000)
        out=torch.zeros(rows,device='cuda');E.own_matvec_f16(W,rows,K,X[i*K:(i+1)*K],out)
        E.build();E.run(check=True);torch.cuda.synchronize()
        assert np.array_equal(out.cpu().numpy(),full[i]),'T4 row not bitwise T1'
print('PASS F16 matvec FP64 bound, T1..4 identity, odd-row tails and output canaries',flush=True)
H,NKV,D,ROWS=16,4,256,43
cache=rng.normal(0,0.2,(ROWS,2,NKV,D)).astype(np.float16)
q=rng.normal(0,0.1,(4,H,D)).astype(np.float32)
C=torch.from_numpy(cache.reshape(-1)).cuda();Q=torch.from_numpy(q.reshape(-1)).cuda()
def run(module,T,pos,qr):
    E=k3.Engine(module=module,T=T,limit_us=200000)
    out=torch.zeros(T*H*D,device='cuda');p=torch.tensor([pos],dtype=torch.int32,device='cuda')
    E.attn_dec(qr,C,p,out,H,NKV,D,1/16,max_cells=ROWS);E.build();E.run(check=True)
    torch.cuda.synchronize();return out.cpu().numpy().reshape(T,H,D)
for start in (0,28,32,39):
    four=run('k3_q35_9b_ver',4,start,Q)
    for t in range(4):
        one=run('k3_q35_9b',1,start+t,Q[t*H*D:(t+1)*H*D])[0]
        assert np.array_equal(one,four[t]),'verify attention not bitwise decode'
        for h in range(H):
            kh=h//4;k=cache[:start+t+1,0,kh].astype(np.float64);v=cache[:start+t+1,1,kh].astype(np.float64)
            scores=k@q[t,h].astype(np.float64)/16;scores=np.exp(scores-scores.max());ref=(scores/scores.sum())@v
            assert np.max(np.abs(one[h]-ref))<1e-5*np.max(np.abs(v)),'group4 attention error'
print('PASS group4 attention FP64 bound and T4/independent decode rows across split boundaries',flush=True)
print('RESULT PASS',flush=True)
