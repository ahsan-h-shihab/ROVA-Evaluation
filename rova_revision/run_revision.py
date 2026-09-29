"""REV-001 orchestrator (plan sections 2, 7 and 8).

Usage (lab workstation, scientific run):
    python -m rova_revision.run_revision --profile full --out /path/revision_out --workers 15
Other modes:
    --profile pilot   timing pilot on non-experimental seeds (plan section 2, E-3)
    --profile smoke   plumbing check on non-experimental seeds and shrunken worlds
    --list            print the unit plan and exit
    --tiers A         restrict to tier A (default: A,B,C)
    --approval TEXT   written approval for a soft environment mismatch (recorded)

The environment guard must pass for the ``full`` and ``pilot`` profiles.
Results are written atomically per unit; a re-run resumes, skipping units whose
done-record and result digest verify. Interrupted units restart from scratch,
except kinds whose per-fold checkpoints are resume-safe.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import platform
import shutil
import socket
import sys
import tarfile
import time
import traceback

from . import __version__
from .config import PROFILES
from .units import EXECUTORS, RESUMABLE_KINDS, enumerate_units, unit_dir


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit_is_done(out_dir, unit):
    d = unit_dir(out_dir, unit["uid"])
    done, res = os.path.join(d, "done.json"), os.path.join(d, "result.json")
    if not (os.path.exists(done) and os.path.exists(res)):
        return False
    with open(done) as f:
        rec = json.load(f)
    return rec.get("status") == "ok" and rec.get("result_sha256") == _sha256(res)


def _prepare_dir(out_dir, unit):
    d = unit_dir(out_dir, unit["uid"])
    if os.path.isdir(d) and unit["kind"] not in RESUMABLE_KINDS:
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    for name in ("done.json", "result.json", "failed.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.remove(p)
    return d


def execute_unit(unit, profile_name, out_dir):
    """Worker entry point. Returns (uid, status, seconds, message)."""
    profile = PROFILES[profile_name]
    d = unit_dir(out_dir, unit["uid"])
    started, t0 = _now(), time.time()
    try:
        result = EXECUTORS[unit["kind"]](profile, unit["params"], d, out_dir)
        _atomic_write(os.path.join(d, "result.json"), result)
        dur = time.time() - t0
        _atomic_write(os.path.join(d, "done.json"), {
            "uid": unit["uid"], "kind": unit["kind"], "params": unit["params"], "tier": unit["tier"],
            "status": "ok", "started": started, "finished": _now(), "duration_s": dur,
            "result_sha256": _sha256(os.path.join(d, "result.json")), "host": socket.gethostname(),
            "pid": os.getpid()})
        return unit["uid"], "ok", dur, ""
    except Exception as e:  # recorded, never silently dropped
        _atomic_write(os.path.join(d, "failed.json"), {
            "uid": unit["uid"], "status": "failed", "started": started, "finished": _now(),
            "error": repr(e), "traceback": traceback.format_exc()})
        return unit["uid"], "failed", time.time() - t0, repr(e)


def _log(out_dir, msg):
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    with open(os.path.join(out_dir, "logs", "orchestrator.log"), "a") as f:
        f.write(line + "\n")


def run_stage_parallel(units, profile_name, out_dir, workers, fn=None):
    """Runs every unit not yet done; a failed unit is retried once (plan section 6)."""
    fn = fn or execute_unit
    ctx = multiprocessing.get_context("fork" if platform.system() == "Linux" else "spawn")
    todo = [u for u in units if not unit_is_done(out_dir, u)]
    skipped = len(units) - len(todo)
    for attempt in (1, 2):
        if not todo:
            break
        for u in todo:
            _prepare_dir(out_dir, u)
        todo.sort(key=lambda u: -u["cost"])
        _log(out_dir, f"  attempt {attempt}: {len(todo)} units on {workers} workers ({skipped} already done)")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(fn, u, profile_name, out_dir): u for u in todo}
            n_done = 0
            for fut in cf.as_completed(futs):
                u = futs[fut]
                try:
                    uid_, status, dur, msg = fut.result()
                except Exception as e:  # worker crashed
                    uid_, status, dur, msg = u["uid"], "failed", 0.0, repr(e)
                n_done += 1
                _log(out_dir, f"    [{n_done}/{len(todo)}] {status:6s} {dur:8.1f}s {uid_} {msg}")
        todo = [u for u in todo if not unit_is_done(out_dir, u)]
    return {"n_units": len(units), "skipped_done": skipped, "failed": [u["uid"] for u in todo]}


def run_stage_serial(units, profile_name, out_dir, fn=None, rerun=False):
    """Serial stages (gate, decision-cost benchmark, analysis); one retry on failure.
    ``rerun`` forces re-execution of units already done (used for the analysis)."""
    fn = fn or execute_unit
    failed = []
    for u in units:
        if unit_is_done(out_dir, u) and not rerun:
            continue
        for attempt in (1, 2):
            _prepare_dir(out_dir, u)
            uid_, status, dur, msg = fn(u, profile_name, out_dir)
            _log(out_dir, f"    attempt {attempt}: {status:6s} {dur:8.1f}s {uid_} {msg}")
            if status == "ok":
                break
        if status != "ok":
            failed.append(uid_)
    return {"n_units": len(units), "failed": failed}


def write_checksums(out_dir):
    entries = []
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
            if rel in ("SHA256SUMS",) or name.endswith(".tmp"):
                continue
            entries.append(f"{_sha256(p)}  {rel}")
    entries.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", newline="\n") as f:
        f.write("\n".join(entries) + "\n")
    return len(entries)


def _inside(path, root):
    try:
        return os.path.normcase(os.path.commonpath([path, root])) == os.path.normcase(root)
    except ValueError:  # different drives
        return False


def make_archive(out_dir):
    base = os.path.abspath(out_dir).rstrip(os.sep)
    tar_path = base + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(base, arcname=os.path.basename(base))
    digest = _sha256(tar_path)
    with open(tar_path + ".sha256", "w", newline="\n") as f:
        f.write(f"{digest}  {os.path.basename(tar_path)}\n")
    return tar_path, digest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--out", required=False)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--tiers", default="A,B,C")
    ap.add_argument("--approval", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args(argv)
    profile = PROFILES[args.profile]
    tiers = tuple(t.strip() for t in args.tiers.split(","))
    units = enumerate_units(profile, tiers)

    if args.list:
        from collections import Counter
        by = Counter((u["stage"], u["tier"], u["kind"]) for u in units)
        for (s, t, k), n in sorted(by.items()):
            print(f"stage {s}  tier {t}  {k:20s} {n:4d}")
        print(f"total units: {len(units)}")
        return 0
    if not args.out:
        ap.error("--out is required")
    out_dir = os.path.abspath(args.out)
    from .env_guard import REPO, check, format_table
    if _inside(out_dir, REPO):
        ap.error("--out must be outside the repository (the environment guard requires a clean working tree)")
    os.makedirs(out_dir, exist_ok=True)

    guard = check(approval=args.approval, workers=args.workers)
    table = format_table(guard)
    print(table, flush=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    with open(os.path.join(out_dir, "logs", "environment_gate.txt"), "a", newline="\n") as f:
        f.write(f"[{_now()}] profile={profile.name} tiers={','.join(tiers)}\n{table}\n\n")
    if profile.name in ("full", "pilot") and not guard["passed"]:
        _atomic_write(os.path.join(out_dir, "env_guard_FAILED.json"), guard)
        print("ENVIRONMENT GATE FAILED — nothing was run. See env_guard_FAILED.json.", file=sys.stderr)
        return 2

    meta_path = os.path.join(out_dir, "run_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.setdefault("runs", []).append({
        "started": _now(), "package_version": __version__, "profile": profile.name,
        "scientific": profile.scientific, "tiers": list(tiers), "workers": args.workers,
        "argv": sys.argv, "env_guard": guard})
    _atomic_write(meta_path, meta)

    if profile.name == "pilot":
        from .pilot import run_pilot
        res = run_pilot(profile, units)
        _atomic_write(os.path.join(out_dir, "pilot_timing.json"), res)
        _log(out_dir, "pilot timing written to pilot_timing.json")
        return 0

    summary = {}
    for stage in sorted({u["stage"] for u in units}):
        stage_units = [u for u in units if u["stage"] == stage]
        _log(out_dir, f"stage {stage}: {len(stage_units)} units")
        if stage == 1:
            summary[stage] = run_stage_serial(stage_units, profile.name, out_dir)
            gate = {"passed": False}
            if unit_is_done(out_dir, stage_units[0]):
                with open(os.path.join(unit_dir(out_dir, stage_units[0]["uid"]), "result.json")) as f:
                    gate = json.load(f)
            if not gate.get("passed"):
                _log(out_dir, "GATE R0 FAILED — run aborted; no revision output is valid.")
                meta["runs"][-1].update({"finished": _now(), "aborted": "gate_failed", "stages": summary})
                _atomic_write(meta_path, meta)
                return 3
            _log(out_dir, "gate R0 passed")
        elif stage == 4:
            summary[stage] = run_stage_serial(stage_units, profile.name, out_dir)
        elif stage == 5:
            summary[stage] = run_stage_serial(stage_units, profile.name, out_dir, rerun=True)
            if unit_is_done(out_dir, stage_units[0]):
                with open(os.path.join(unit_dir(out_dir, stage_units[0]["uid"]), "result.json")) as f:
                    errs = json.load(f).get("section_errors", [])
                for e in errs:
                    _log(out_dir, f"ANALYSIS SECTION ERROR {e['section']}: {e['error']}")
                summary[stage]["analysis_section_errors"] = len(errs)
                if errs:
                    summary[stage]["failed"] = summary[stage]["failed"] + [f"analysis sections: {len(errs)}"]
        else:
            summary[stage] = run_stage_parallel(stage_units, profile.name, out_dir, args.workers)

    failures = [u for s in summary.values() for u in s.get("failed", [])]
    meta["runs"][-1].update({"finished": _now(), "stages": summary})
    _atomic_write(meta_path, meta)
    _log(out_dir, f"done; failed units: {len(failures)}")
    # Nothing is written inside out_dir after the checksums.
    n = write_checksums(out_dir)
    print(f"SHA256SUMS written ({n} files)", flush=True)
    if not args.no_archive:
        tar_path, digest = make_archive(out_dir)
        print(f"archive {tar_path} sha256 {digest}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
