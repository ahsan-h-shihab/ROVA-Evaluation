"""Execution-environment guard (plan section 2, E-1).

Hard requirements (never overridable): the checked-out commit equals the
frozen revision tag, the working tree is clean, and every BLAS/OpenMP thread
variable equals 1. Soft requirements (hardware and software identity with the
MAIN-001 workstation) may be overridden only by an explicit written approval,
which is recorded verbatim.
"""
import os
import platform
import socket
import subprocess
import sys

EXPECTED = {
    "system": "Linux",
    "cpu_model_contains": "AMD Ryzen 9 9950X",
    "logical_cpus": 32,
    "hostname": "halyk-rtx5090-MS-7E51",
    "python": "3.13.14",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "matplotlib": "3.11.0",
}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
FROZEN_TAG = "revision-001-frozen"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def collect():
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "matplotlib": _version("matplotlib"),
        "thread_vars": {v: os.environ.get(v) for v in THREAD_VARS},
        "git_head": _git("rev-parse", "HEAD"),
        "git_frozen_tag_commit": _git("rev-parse", f"{FROZEN_TAG}^{{commit}}"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "git_describe": _git("describe", "--tags", "--always"),
    }


def check(approval=None):
    info = collect()
    soft = [
        ("system", info["system"] == EXPECTED["system"]),
        ("cpu_model", EXPECTED["cpu_model_contains"] in (info["cpu_model"] or "")),
        ("logical_cpus", info["logical_cpus"] == EXPECTED["logical_cpus"]),
        ("hostname", info["hostname"] == EXPECTED["hostname"]),
        ("python", info["python"] == EXPECTED["python"]),
        ("numpy", info["numpy"] == EXPECTED["numpy"]),
        ("scipy", info["scipy"] == EXPECTED["scipy"]),
        ("matplotlib", info["matplotlib"] == EXPECTED["matplotlib"]),
    ]
    hard = [
        ("git_commit_is_frozen_tag", info["git_head"] is not None and info["git_head"] == info["git_frozen_tag_commit"]),
        ("git_working_tree_clean", info["git_status_porcelain"] == ""),
        ("thread_vars_equal_1", all(info["thread_vars"][v] == "1" for v in THREAD_VARS)),
    ]
    soft_failed = [n for n, ok in soft if not ok]
    hard_failed = [n for n, ok in hard if not ok]
    overridden = bool(soft_failed) and bool(approval)
    passed = not hard_failed and (not soft_failed or overridden)
    return {"passed": passed, "expected": EXPECTED, "observed": info,
            "soft_checks": dict(soft), "hard_checks": dict(hard),
            "soft_failed": soft_failed, "hard_failed": hard_failed,
            "approval_text": approval, "overridden_by_approval": overridden}


if __name__ == "__main__":
    import json
    res = check(approval=(sys.argv[1] if len(sys.argv) > 1 else None))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["passed"] else 1)
