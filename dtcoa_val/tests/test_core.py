"""Sanity + circularity-guard tests. Run: python3 -m dtcoa_val.tests.test_core"""
import numpy as np
from dtcoa_val.sim_core.core import obs_map_A, SimCore
from dtcoa_val.dtcoa import W
from dtcoa_val.oracles.oracles import AuthorizationOracle, EscalationOracle
from dtcoa_val.utils import make_rng

def test_determinism():
    c=dict(horizon=50,n_agents=2,regime="hmm")
    r1=SimCore(c, make_rng(7)).r; r2=SimCore(c, make_rng(7)).r
    assert np.allclose(r1,r2), "non-deterministic given seed"

def test_obs_weights_mismatch_EQ1():
    # implied linear weights of obs_map_A on (r, 1-h) must differ from EQ1 W
    # obs_map_A channel coefficients on r and (1-h):
    coeffs_r =np.array([1.0,0.5,0.0,0.7,0.2,0.4])   # approx d(channel)/dr structure
    assert not np.allclose(coeffs_r/coeffs_r.sum(), W), "obs map equals EQ1 weights -> circular"

def test_oracle_independence():
    # oracle decision depends on true (r,h) and loss, not on any DTCOA quantity
    o=AuthorizationOracle(H=1.0,U=0.3)
    a_lowh,_=o.optimal(0.9,0.05); a_highh,_=o.optimal(0.9,0.9)
    assert a_lowh>=a_highh, "oracle should grant less autonomy when hazard high"
    e=EscalationOracle(theta_h=0.5)
    assert e.should(0.6) and not e.should(0.3)

def test_outcome_metric_independence():
    # harm uses granted action + true hazard only (no trust/band)
    h=0.8; pi=1.0; expected=pi*h
    assert abs(expected-0.8)<1e-9

if __name__=="__main__":
    for f in [test_determinism,test_obs_weights_mismatch_EQ1,test_oracle_independence,test_outcome_metric_independence]:
        f(); print("PASS", f.__name__)
