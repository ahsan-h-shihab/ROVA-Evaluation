"""Randomness management + reproducible IO. Strict per-trial seed control."""
import json, os, hashlib, time
import numpy as np

def make_rng(seed):
    """Deterministic Generator from an integer seed."""
    return np.random.default_rng(int(seed))

def trial_seed(base_seed, trial_idx, tag=""):
    """Derive a stable per-trial seed from (base_seed, trial_idx, tag).
    Deterministic and collision-resistant so reruns replay exactly."""
    h = hashlib.sha256(f"{base_seed}|{trial_idx}|{tag}".encode()).hexdigest()
    return int(h[:8], 16)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonify)

def _jsonify(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    return str(o)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
