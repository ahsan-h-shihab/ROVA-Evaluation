"""Baselines — defensible, not strawmen. Same step interface as DTCOA: reset(N); step(O,truth)->(A,ESC).
Composite evidence = nanmean(obs). Thresholds tuned on DISJOINT seeds (see experiments/baseline_tuning)."""
import numpy as np

def _composite(O, last):
    c = np.full(O.shape[0], np.nan)
    for i in range(O.shape[0]):
        oi = O[i].copy(); m = np.isnan(oi); oi[m] = last[i][m]; last[i] = oi
        c[i] = float(np.mean(oi))
    return c, last

def _to_levels(c, thr):
    return np.array([int(np.sum(ci >= thr)) for ci in c], int)

class StaticThreshold:
    def __init__(self, thr=(0.2,0.36,0.52,0.68,0.88)): self.thr=np.array(thr)
    def reset(self,N): self.N=N; self.last=np.full((N,6),0.6)
    def step(self,O,truth=None):
        c,self.last=_composite(O,self.last); self.T=c.copy(); A=_to_levels(c,self.thr); return A,(A==0)

class EWMA:
    def __init__(self, beta=0.7, thr=(0.2,0.36,0.52,0.68,0.88)): self.beta=beta; self.thr=np.array(thr)
    def reset(self,N): self.N=N; self.last=np.full((N,6),0.6); self.e=np.full(N,0.6)
    def step(self,O,truth=None):
        c,self.last=_composite(O,self.last); self.e=self.beta*self.e+(1-self.beta)*c; self.T=self.e.copy()
        A=_to_levels(self.e,self.thr); return A,(A==0)

class RuleBased:
    def __init__(self, low=0.35, mid=0.6): self.low=low; self.mid=mid
    def reset(self,N): self.N=N; self.last=np.full((N,6),0.6)
    def step(self,O,truth=None):
        c,self.last=_composite(O,self.last)
        A=np.where(c<self.low,0,np.where(c<self.mid,2,5)).astype(int); return A,(A==0)

class OracleLite:
    """Privileged REFERENCE (not a competitor): sees true reliability r, ignores hazard h.
    Maps r->autonomy by thresholds. Demonstrates the cost of hazard-blindness."""
    def __init__(self, thr=(0.2,0.36,0.52,0.68,0.88)): self.thr=np.array(thr)
    def reset(self,N): self.N=N
    def step(self,O,truth=None):
        r = truth["r"] if truth else np.full(self.N,0.6)
        A=np.array([int(np.sum(ri>=self.thr)) for ri in r],int); return A,(A==0)
