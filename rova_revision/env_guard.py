"""Execution-environment gate (plan section 2, E-1).

The reference environment is parsed from the archived execution records of
MAIN-001 and ABLATION-001 (``REFERENCE_FILES``), not typed in. The CPU model,
core and socket counts are read from ``/proc/cpuinfo``; the host name and the
interpreter path are recorded but are never taken as evidence of the hardware.

Hard requirements (never overridable): the checked-out commit equals the frozen
revision tag, the working tree is clean, every BLAS/OpenMP thread variable is 1,
and no GPU framework is loaded. Soft requirements (identity with the reference
hardware, OS, kernel, interpreter, libraries, BLAS and worker count) may be
overridden only by an explicit written approval, recorded verbatim.
"""
import json
import os
import platform
import re
import socket
import subprocess
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
GPU_MODULES = ("torch", "cupy", "jax", "tensorflow", "numba.cuda")
FROZEN_TAG = "revision-001-frozen"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_FILES = {
    "main001_config": os.path.join(REPO, "experiments", "MAIN-001_core_comparison", "config.json"),
    "main001_environment": os.path.join(REPO, "experiments", "MAIN-001_core_comparison", "raw", "environment.txt"),
    "abl001_config": os.path.join(REPO, "experiments", "ABLATION-001_mechanism_attribution", "config.json"),
}


# --------------------------------------------------------------------------- #
# Reference (archived records)
# --------------------------------------------------------------------------- #
def reference():
    with open(REFERENCE_FILES["main001_config"]) as f:
        m = json.load(f)
    with open(REFERENCE_FILES["abl001_config"]) as f:
        a = json.load(f)
    with open(REFERENCE_FILES["main001_environment"]) as f:
        env_lines = [ln.strip() for ln in f if ln.strip()]
    host = m["execution_host"]
    cpu = re.match(r"(.+?)\s*\((\d+) physical / (\d+) logical(, single socket)?\)", host["cpu"])
    uname = next(ln for ln in env_lines if ln.startswith("Linux "))
    pins = dict(ln.split("==", 1) for ln in env_lines if "==" in ln)
    blas = re.match(r"(\S+)\s+([0-9.]+)", host["blas"])
    plat = a["environment_capture"]["os"]
    return {
        "system": "Linux",
        "os_id": host["os"].split(",")[0].strip().lower(),
        "kernel": uname.split()[2],
        "glibc": re.search(r"glibc([0-9.]+)", plat).group(1),
        "cpu_model": cpu.group(1).strip(),
        "physical_cores": int(cpu.group(2)),
        "logical_cpus": int(cpu.group(3)),
        "sockets": 1 if cpu.group(4) else None,
        "hostname": uname.split()[1],
        "python": host["python"],
        "numpy": host["numpy"],
        "scipy": host["scipy"],
        "matplotlib": pins["matplotlib"],
        "blas_family": blas.group(1),
        "blas_version": blas.group(2),
        "workers": int(m["launch_config"]["num_shards"]),
        "board": a["execution_host"].get("board"),
        "interpreter": a["launch_config"].get("interpreter"),
        "sources": {k: os.path.relpath(v, REPO).replace(os.sep, "/") for k, v in REFERENCE_FILES.items()},
    }


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
def parse_cpuinfo(text):
    model, phys, cores = None, set(), set()
    cur = {}
    for line in text.splitlines() + [""]:
        if not line.strip():
            if cur:
                model = model or cur.get("model name")
                if "physical id" in cur:
                    phys.add(cur["physical id"])
                    if "core id" in cur:
                        cores.add((cur["physical id"], cur["core id"]))
            cur = {}
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            cur[k.strip()] = v.strip()
    return {"cpu_model": model, "sockets": len(phys) or None, "physical_cores": len(cores) or None}


def parse_os_release(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def _blas():
    out = {}
    for mod in ("numpy", "scipy"):
        try:
            cfg = __import__(mod).show_config(mode="dicts")
            bd = cfg.get("Build Dependencies", {})
            out[mod] = {"blas": bd.get("blas", {}), "lapack": bd.get("lapack", {})}
        except Exception as e:
            out[mod] = {"error": repr(e)}
    return out


def _meminfo_gib():
    m = re.search(r"MemTotal:\s+(\d+) kB", _read("/proc/meminfo"))
    return round(int(m.group(1)) / 2 ** 20, 1) if m else None


def collect(workers=None):
    cpu = parse_cpuinfo(_read("/proc/cpuinfo"))
    osr = parse_os_release(_read("/etc/os-release"))
    libc = platform.libc_ver()
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "kernel_version": platform.version(),
        "os_id": osr.get("ID"),
        "os_pretty_name": osr.get("PRETTY_NAME"),
        "glibc": libc[1] if libc[0] == "glibc" else None,
        "cpu_model": cpu["cpu_model"] or platform.processor(),
        "physical_cores": cpu["physical_cores"],
        "sockets": cpu["sockets"],
        "logical_cpus": os.cpu_count(),
        "memory_gib": _meminfo_gib(),
        "board": {k: _read(f"/sys/class/dmi/id/{k}").strip() or None
                  for k in ("board_vendor", "board_name", "product_name")},
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "matplotlib": _version("matplotlib"),
        "blas": _blas(),
        "thread_vars": {v: os.environ.get(v) for v in THREAD_VARS},
        "gpu_frameworks_loaded": [m for m in GPU_MODULES if m in sys.modules],
        "workers": workers,
        "git_head": _git("rev-parse", "HEAD"),
        "git_frozen_tag_commit": _git("rev-parse", f"{FROZEN_TAG}^{{commit}}"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "git_describe": _git("describe", "--tags", "--always"),
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _blas_field(info, mod, key):
    return ((info.get("blas") or {}).get(mod) or {}).get("blas", {}).get(key)


def evaluate(info, ref, approval=None):
    """Pure function of the observed and reference records (unit-testable)."""
    bn = [_blas_field(info, m, "name") for m in ("numpy", "scipy")]
    bv = [_blas_field(info, m, "version") for m in ("numpy", "scipy")]
    soft = [
        ("system", info["system"], ref["system"], info["system"] == ref["system"]),
        ("os_id", info["os_id"], ref["os_id"], (info["os_id"] or "").lower() == ref["os_id"]),
        ("kernel", info["kernel"], ref["kernel"], info["kernel"] == ref["kernel"]),
        ("glibc", info["glibc"], ref["glibc"], info["glibc"] == ref["glibc"]),
        ("cpu_model", info["cpu_model"], ref["cpu_model"], ref["cpu_model"] in (info["cpu_model"] or "")),
        ("physical_cores", info["physical_cores"], ref["physical_cores"], info["physical_cores"] == ref["physical_cores"]),
        ("logical_cpus", info["logical_cpus"], ref["logical_cpus"], info["logical_cpus"] == ref["logical_cpus"]),
        ("sockets", info["sockets"], ref["sockets"], ref["sockets"] is None or info["sockets"] == ref["sockets"]),
        ("hostname", info["hostname"], ref["hostname"], info["hostname"] == ref["hostname"]),
        ("python", info["python"], ref["python"], info["python"] == ref["python"]),
        ("numpy", info["numpy"], ref["numpy"], info["numpy"] == ref["numpy"]),
        ("scipy", info["scipy"], ref["scipy"], info["scipy"] == ref["scipy"]),
        ("matplotlib", info["matplotlib"], ref["matplotlib"], info["matplotlib"] == ref["matplotlib"]),
        ("blas_family (numpy, scipy)", bn, ref["blas_family"],
         all(isinstance(n, str) and n.startswith(ref["blas_family"]) for n in bn)),
        ("blas_version (numpy, scipy)", bv, ref["blas_version"],
         any(isinstance(v, str) and v.startswith(ref["blas_version"]) for v in bv)),
        ("workers", info["workers"], ref["workers"], info["workers"] == ref["workers"]),
    ]
    hard = [
        ("git_commit_is_frozen_tag", info["git_head"], info["git_frozen_tag_commit"],
         info["git_head"] is not None and info["git_head"] == info["git_frozen_tag_commit"]),
        ("git_working_tree_clean", info["git_status_porcelain"], "", info["git_status_porcelain"] == ""),
        ("thread_vars_equal_1", info["thread_vars"], "1", all(info["thread_vars"][v] == "1" for v in THREAD_VARS)),
        ("no_gpu_framework_loaded", info["gpu_frameworks_loaded"], [], not info["gpu_frameworks_loaded"]),
    ]
    soft_failed = [n for n, _o, _r, ok in soft if not ok]
    hard_failed = [n for n, _o, _r, ok in hard if not ok]
    overridden = bool(soft_failed) and bool(approval)
    return {"passed": not hard_failed and (not soft_failed or overridden),
            "reference": ref, "observed": info,
            "soft_checks": [{"item": n, "observed": o, "reference": r, "ok": ok} for n, o, r, ok in soft],
            "hard_checks": [{"item": n, "observed": o, "reference": r, "ok": ok} for n, o, r, ok in hard],
            "soft_failed": soft_failed, "hard_failed": hard_failed,
            "approval_text": approval, "overridden_by_approval": overridden}


def check(approval=None, workers=None):
    return evaluate(collect(workers), reference(), approval)


def format_table(res):
    info, ref = res["observed"], res["reference"]
    lines = ["EXECUTION ENVIRONMENT (reference parsed from: "
             + ", ".join(ref["sources"].values()) + ")",
             f"{'item':30s} {'status':6s} observed | reference"]
    for kind in ("hard_checks", "soft_checks"):
        for c in res[kind]:
            lines.append(f"{c['item']:30s} {'OK' if c['ok'] else 'FAIL':6s} {c['observed']} | {c['reference']}")
    lines += [f"{'platform':30s} {'info':6s} {info['platform']}",
              f"{'os_pretty_name':30s} {'info':6s} {info['os_pretty_name']}",
              f"{'board (DMI)':30s} {'info':6s} {info['board']} | {ref['board']}",
              f"{'memory_gib':30s} {'info':6s} {info['memory_gib']}",
              f"{'python_executable':30s} {'info':6s} {info['python_executable']} | {ref['interpreter']}"]
    for mod in ("numpy", "scipy"):
        b = (info["blas"].get(mod) or {}).get("blas", {})
        lines.append(f"{'blas config (' + mod + ')':30s} {'info':6s} {b.get('openblas configuration', b)}")
    verdict = "PASSED" if res["passed"] else "FAILED"
    if res["overridden_by_approval"]:
        verdict += " (soft mismatches overridden by the recorded written approval)"
    lines.append(f"GATE: {verdict}; hard failures: {res['hard_failed']}; soft failures: {res['soft_failed']}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--approval", default=None)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--json", default=None, help="write the full record to this path")
    args = ap.parse_args(argv)
    res = check(approval=args.approval, workers=args.workers)
    print(format_table(res))
    if args.json:
        with open(args.json, "w", newline="\n") as f:
            json.dump(res, f, indent=2)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
