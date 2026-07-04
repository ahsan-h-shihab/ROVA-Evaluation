"""Independent oracles. Use TRUE latent (r,h) + explicit loss/severity. No DTCOA constructs/thresholds."""
import numpy as np

LEVELS = np.arange(6)
def pi_linear(a): return a/5.0           # permission fraction
def pi_convex(a): return (a/5.0)**1.5     # alternate map for robustness

class AuthorizationOracle:
    def __init__(self, H=1.0, U=0.3, pi=pi_linear):
        self.H = H; self.U = U; self.pi = pi
    def loss(self, a, r, h):
        p = self.pi(a); return p*h*self.H + (1-p)*r*self.U
    def optimal(self, r, h):
        ls = [self.loss(a, r, h) for a in LEVELS]; return int(np.argmin(ls)), float(min(ls))
    def regret(self, a, r, h):
        _, lstar = self.optimal(r, h); return float(self.loss(a, r, h) - lstar)

class EscalationOracle:
    """should-escalate G = (h >= theta_h); severity = h*H. theta_h from loss model, swept externally."""
    def __init__(self, theta_h=0.5, H=1.0):
        self.theta = theta_h; self.H = H
    def should(self, h): return bool(h >= self.theta)
    def severity(self, h): return float(h*self.H)
