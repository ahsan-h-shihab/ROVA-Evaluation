"""A1 orchestrator (docs/REVISION_002_ANALYSIS_PLAN.md, section 2).

Scientific run (lab workstation only):
    python -m rova_revision.run_a1 --profile full --out /path/a1_out \
        --rev001 /path/revision_out --workers 15
Plumbing check (non-experimental seeds, shrunken worlds; needs a REV-001 smoke output):
    python -m rova_revision.run_a1 --profile smoke --out /tmp/a1_smoke --rev001 /tmp/rev001_smoke

Order: environment gate (must pass for ``full``) -> stage 1 gate (R0 replay with
exact zero difference + REV-001 input verification) -> stage 2 tuning campaigns ->
stop if any reproduction check RC1, RC2, RC4, RC5 failed -> stage 3 evaluations ->
stop if RC3 failed -> stage 4 analysis -> checksums and archive. A re-run resumes:
finished units are verified and skipped; tuning campaigns resume per outer fold.
Exit status: 0 success; 1 failed units or analysis errors; 2 environment gate
failed; 3 gate or reproduction check failed (A1 invalid).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import platform
import shutil
import socket
import sys
import time
import traceback

from .a1_config import A1_PROFILES
from .a1_units import A1_EXECUTORS, A1_RESUMABLE_KINDS, INPUTS_FILE, enumerate_a1_units
from .run_revision import _atomic_write, _inside, _now, _sha256, make_archive, unit_is_done, write_checksums
from .units import unit_dir

A1_VERSION = "rev2.0"


def _log(out_dir, msg):
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    with open(os.path.join(out_dir, "logs", "orchestrator.log"), "a", newline="\n") as f:
        f.write(line + "\n")


def _prepare_dir(out_dir, unit):
    d = unit_dir(out_dir, unit["uid"])
    if os.path.isdir(d) and unit["kind"] not in A1_RESUMABLE_KINDS:
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    for name in ("done.json", "result.json", "failed.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.remove(p)


def execute_a1_unit(unit, profile_name, out_dir):
    a1prof = A1_PROFILES[profile_name]
    d = unit_dir(out_dir, unit["uid"])
    started, t0 = _now(), time.time()
    try:
        result = A1_EXECUTORS[unit["kind"]](a1prof, unit["params"], d, out_dir)
        _atomic_write(os.path.join(d, "result.json"), result)
        dur = time.time() - t0
        _atomic_write(os.path.join(d, "done.json"), {
            "uid": unit["uid"], "kind": unit["kind"], "params": unit["params"], "status": "ok",
            "started": started, "finished": _now(), "duration_s": dur,
            "result_sha256": _sha256(os.path.join(d, "result.json")), "host": socket.gethostname(), "pid": os.getpid()})
        return unit["uid"], "ok", dur, ""
    except Exception as e:  # recorded, never silently dropped
        _atomic_write(os.path.join(d, "failed.json"), {
            "uid": unit["uid"], "status": "failed", "started": started, "finished": _now(),
            "error": repr(e), "traceback": traceback.format_exc()})
        return unit["uid"], "failed", time.time() - t0, repr(e)


def run_parallel(units, profile_name, out_dir, workers):
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
            futs = {ex.submit(execute_a1_unit, u, profile_name, out_dir): u for u in todo}
            for n, fut in enumerate(cf.as_completed(futs), 1):
                u = futs[fut]
                try:
                    uid_, status, dur, msg = fut.result()
                except Exception as e:
                    uid_, status, dur, msg = u["uid"], "failed", 0.0, repr(e)
                _log(out_dir, f"    [{n}/{len(todo)}] {status:6s} {dur:8.1f}s {uid_} {msg}")
        todo = [u for u in todo if not unit_is_done(out_dir, u)]
    return {"n_units": len(units), "skipped_done": skipped, "failed": [u["uid"] for u in todo]}


def run_serial(units, profile_name, out_dir, rerun=False):
    failed = []
    for u in units:
        if unit_is_done(out_dir, u) and not rerun:
            continue
        for attempt in (1, 2):
            _prepare_dir(out_dir, u)
            uid_, status, dur, msg = execute_a1_unit(u, profile_name, out_dir)
            _log(out_dir, f"    attempt {attempt}: {status:6s} {dur:8.1f}s {uid_} {msg}")
            if status == "ok":
                break
        if status != "ok":
            failed.append(uid_)
    return {"n_units": len(units), "failed": failed}


def _result(out_dir, u):
    with open(os.path.join(unit_dir(out_dir, u["uid"]), "result.json")) as f:
        return json.load(f)


def _repro_failures(out_dir, units):
    bad = []
    for u in units:
        if not unit_is_done(out_dir, u):
            bad.append(f"{u['uid']} (not done)")
        elif not _result(out_dir, u)["reproduction"]["ok"]:
            bad.append(u["uid"])
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(A1_PROFILES), default="smoke")
    ap.add_argument("--out", required=False)
    ap.add_argument("--rev001", required=False, help="REV-001 revision_out directory (read-only input)")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--approval", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args(argv)
    a1prof = A1_PROFILES[args.profile]
    units = enumerate_a1_units(a1prof)
    if args.list:
        from collections import Counter
        for (s, k), n in sorted(Counter((u["stage"], u["kind"]) for u in units).items()):
            print(f"stage {s}  {k:14s} {n:4d}")
        print(f"total units: {len(units)}")
        return 0
    if not args.out or not args.rev001:
        ap.error("--out and --rev001 are required")
    out_dir, rev001 = os.path.abspath(args.out), os.path.abspath(args.rev001)
    from .a1_guard import check
    from .env_guard import REPO, format_table
    if _inside(out_dir, REPO):
        ap.error("--out must be outside the repository")
    if _inside(out_dir, rev001) or _inside(rev001, out_dir):
        ap.error("--out must be separate from the REV-001 output directory")
    if not os.path.isdir(rev001):
        ap.error(f"REV-001 directory not found: {rev001}")
    os.makedirs(out_dir, exist_ok=True)

    guard = check(approval=args.approval, workers=args.workers)
    table = format_table(guard)
    print(table, flush=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    with open(os.path.join(out_dir, "logs", "environment_gate.txt"), "a", newline="\n") as f:
        f.write(f"[{_now()}] A1 profile={a1prof.name}\n{table}\n\n")
    if a1prof.scientific and not guard["passed"]:
        _atomic_write(os.path.join(out_dir, "env_guard_FAILED.json"), guard)
        print("ENVIRONMENT GATE FAILED — nothing was run. See env_guard_FAILED.json.", file=sys.stderr)
        return 2

    inputs_path = os.path.join(out_dir, INPUTS_FILE)
    if os.path.exists(inputs_path):
        with open(inputs_path) as f:
            prev = json.load(f)
        if prev["rev001_dir"] != rev001:
            ap.error(f"this output directory was started with --rev001 {prev['rev001_dir']}")
    else:
        _atomic_write(inputs_path, {"rev001_dir": rev001})

    meta_path = os.path.join(out_dir, "run_metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.setdefault("runs", []).append({
        "started": _now(), "package_version": A1_VERSION, "analysis": "A1 (docs/REVISION_002_ANALYSIS_PLAN.md)",
        "profile": a1prof.name, "scientific": a1prof.scientific, "workers": args.workers, "argv": sys.argv,
        "rev001_dir": rev001, "env_guard": guard})
    _atomic_write(meta_path, meta)

    def finish(code, summary, note=None):
        meta["runs"][-1].update({"finished": _now(), "stages": summary, **({"aborted": note} if note else {})})
        _atomic_write(meta_path, meta)
        return code

    summary = {}
    st = {s: [u for u in units if u["stage"] == s] for s in (1, 2, 3, 4)}
    _log(out_dir, f"stage 1: {len(st[1])} units")
    summary[1] = run_serial(st[1], a1prof.name, out_dir)
    if not (unit_is_done(out_dir, st[1][0]) and _result(out_dir, st[1][0]).get("passed")):
        _log(out_dir, "A1 GATE FAILED (R0 exact replay or REV-001 inputs) — run aborted; nothing else was run.")
        return finish(3, summary, "gate_failed")
    _log(out_dir, "A1 gate passed (R0 exact; REV-001 inputs verified)")
    for stage in (2, 3):
        _log(out_dir, f"stage {stage}: {len(st[stage])} units")
        summary[stage] = run_parallel(st[stage], a1prof.name, out_dir, args.workers)
        bad = _repro_failures(out_dir, st[stage])
        if bad:
            for b in bad:
                _log(out_dir, f"REPRODUCTION CHECK FAILED: {b}")
            _log(out_dir, "A1 INVALID — reproduction of REV-001 failed; later stages not run.")
            return finish(3, summary, f"reproduction_failed_stage_{stage}")
        _log(out_dir, f"stage {stage}: every reproduction check passed")
    _log(out_dir, f"stage 4: {len(st[4])} units")
    summary[4] = run_serial(st[4], a1prof.name, out_dir, rerun=True)
    if unit_is_done(out_dir, st[4][0]):
        for e in _result(out_dir, st[4][0]).get("section_errors", []):
            _log(out_dir, f"ANALYSIS SECTION ERROR {e['section']}: {e['error']}")
            summary[4]["failed"] = summary[4]["failed"] + [f"analysis section {e['section']}"]
    failures = [u for s in summary.values() for u in s.get("failed", [])]
    finish(0, summary)
    _log(out_dir, f"done; failed units: {len(failures)}")
    n = write_checksums(out_dir)
    print(f"SHA256SUMS written ({n} files)", flush=True)
    if not args.no_archive:
        tar_path, digest = make_archive(out_dir)
        print(f"archive {tar_path} sha256 {digest}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
