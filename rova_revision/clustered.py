"""A2: fold-clustered statistical re-analysis (docs/REVISION_002_ANALYSIS_PLAN.md, section 3).

A post-run, deterministic re-analysis of stored per-world regrets. It runs no
simulation and imports no simulator code: the only project import is the frozen
Cliff's delta of ``dtcoa_val.analysis.stats`` (numpy and scipy only). A test runs
this module with every simulator package blocked from import.

Resampling (fixed by the plan, section 3.4). For each comparison a fresh
``numpy.random.default_rng(0)`` draws, in this order, the fold choices
``integers(0, K, (B, K))`` and then the within-fold positions
``integers(0, m, (B, K, m))`` (K outer folds of m worlds each); the same indices
are applied to both methods, so pairing is preserved. The fold-only bootstrap uses
its own ``default_rng(0)`` and draws ``integers(0, K, (B, K))``.

Scientific run (laptop permitted; nothing is simulated):
    python -m rova_revision.clustered --rev001-archive revision_out.tar.gz \
        --work WORK_DIR --out OUT_DIR
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile

import numpy as np

from dtcoa_val.analysis.stats import cliffs_delta

from .rev002_constants import (A2_N_BOOT, A2_SEED, FROZEN_TAG_REV002, REV001_ARCHIVE_SHA256, REV001_SUMS_ENTRIES,
                        REV001_SUMS_SHA256)

NEGLIGIBLE = 0.147
CEW_UNIT_PREFIXES = ("S3__eval_cew__", "S3__eval_cew_b__", "S3__eval_cew_c__")
GRID = (0.5, 1.5, 3.5, 8.0, 20.0)
ENVS = ("favorable_hmm_coupled", "unfavorable_oscillation_uncoupled", "collapse_moderate_coupled")

# Plan section 3.3: (REV-001 family, A, B).
A2_PAIRS = (
    [("F1", "rova", "static"), ("F1", "rova", "ewma")]
    + [("F2", m, "A0") for m in ("rova", "static", "ewma")]
    + [("F3", "rova", "learned"), ("F3", "rova", "qlearn")]
    + [("F4", "rova", "static_ext"), ("F4", "rova", "ewma_ext")]
    + [("F5", "rova", "rova_single"), ("F5", "rova", "clairvoyant")]
    + [("F6", "rova_tpe", "static_tpe"), ("F6", "rova_tpe", "ewma_tpe")]
    + [("F7", "rova_tpe", "rova"), ("F7", "static_tpe", "static"), ("F7", "ewma_tpe", "ewma")]
    + [("F8", f"abl_no_{x}", "abl_full") for x in ("memory", "decay", "propagation", "stability")]
)


# --------------------------------------------------------------------------- #
# Statistics (pure functions)
# --------------------------------------------------------------------------- #
def fold_structure(folds):
    """Index matrix (K, m) of worlds per outer fold; folds must be equal-sized."""
    folds = np.asarray(folds)
    levels = np.unique(folds)
    rows = [np.flatnonzero(folds == lv) for lv in levels]
    sizes = {len(r) for r in rows}
    if len(sizes) != 1:
        raise ValueError(f"unequal fold sizes: {sorted(len(r) for r in rows)}")
    return np.vstack(rows), levels


def two_stage_indices(F, n_boot, seed=A2_SEED):
    K, m = F.shape
    rng = np.random.default_rng(seed)
    fsel = rng.integers(0, K, (n_boot, K))
    pos = rng.integers(0, m, (n_boot, K, m))
    return F[fsel[:, :, None], pos].reshape(n_boot, K * m)


def fold_only_indices(F, n_boot, seed=A2_SEED):
    K, m = F.shape
    rng = np.random.default_rng(seed)
    fsel = rng.integers(0, K, (n_boot, K))
    return F[fsel].reshape(n_boot, K * m)


def _pct(x):
    lo, hi = np.percentile(np.asarray(x, float), [2.5, 97.5])
    return [float(lo), float(hi)]


def _delta_resampled(xa, xb, idx):
    """Cliff's delta(A, B) of the resampled samples A[idx], B[idx], per resample."""
    S = np.sign(xa[:, None] - xb[None, :])
    n = len(xa)
    W = np.zeros((idx.shape[0], n))
    for r in range(idx.shape[0]):
        W[r] = np.bincount(idx[r], minlength=n)
    tot = idx.shape[1] ** 2
    return np.einsum("ri,ij,rj->r", W, S, W) / tot


def sign_flip_p(fold_means):
    m = np.asarray(fold_means, float)
    K = len(m)
    obs = abs(m.mean())
    signs = np.array([[1.0 if (s >> k) & 1 else -1.0 for k in range(K)] for s in range(2 ** K)])
    stats = np.abs((signs * m).mean(axis=1))
    return float(np.mean(stats >= obs * (1.0 - 1e-12))), int(2 ** K)


def icc1(x, groups):
    """One-way random-effects ICC(1) with the ANOVA estimator, truncated at 0."""
    x, g = np.asarray(x, float), np.asarray(groups)
    levels = np.unique(g)
    k, n = len(levels), len(x)
    sizes = np.array([np.sum(g == lv) for lv in levels])
    means = np.array([x[g == lv].mean() for lv in levels])
    ssb = float(np.sum(sizes * (means - x.mean()) ** 2))
    ssw = float(sum(np.sum((x[g == lv] - mu) ** 2) for lv, mu in zip(levels, means)))
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(sizes ** 2) / n) / (k - 1)
    var_b = max(0.0, (msb - msw) / n0)
    tot = var_b + msw
    return float(var_b / tot) if tot > 0 else 0.0


def analyze_pair(xa, xb, folds, n_boot=A2_N_BOOT):
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    d = xa - xb
    F, levels = fold_structure(folds)
    K, m = F.shape
    two = two_stage_indices(F, n_boot)
    one = fold_only_indices(F, n_boot)
    mean_two = d[two].mean(axis=1)
    mean_one = d[one].mean(axis=1)
    delta_two = _delta_resampled(xa, xb, two)
    fold_means = [float(d[F[k]].mean()) for k in range(K)]
    p, n_vec = sign_flip_p(fold_means)
    icc = icc1(d, folds)
    de = 1.0 + (m - 1) * icc
    return {"n": int(len(d)), "folds": int(K), "worlds_per_fold": int(m),
            "mean_diff": float(d.mean()), "cliffs_delta": float(cliffs_delta(xa, xb)),
            "two_stage_mean_diff_ci": _pct(mean_two), "two_stage_delta_ci": _pct(delta_two),
            "fold_only_mean_diff_ci": _pct(mean_one), "fold_means": fold_means,
            "sign_flip_p": p, "sign_flip_vectors": n_vec, "sign_flip_min_p": 2.0 / n_vec,
            "icc1_of_difference": icc, "design_effect": de, "effective_n": float(len(d) / de),
            "n_boot": int(n_boot)}


def generalization_flag(label_code, ci):
    """Plan section 3.5 rule 2."""
    if label_code == "a_lower":
        return "generalizes across tuning runs" if ci[1] < 0 else "not established across tuning runs"
    if label_code == "b_lower":
        return "generalizes across tuning runs" if ci[0] > 0 else "not established across tuning runs"
    return "no per-world label (no detected difference)"


def g_clustered(data, folds, n_boot=A2_N_BOOT, names=("A0", "rova", "static", "ewma")):
    """G = (A0 - ROVA) / (best baseline - ROVA); the best baseline is fixed from the
    full-sample means, as in REV-001; same resample indices for every method."""
    a0n, rvn, sn, en = names
    if any(k not in data for k in names):
        return {"status": "missing methods"}
    a0, rv = np.asarray(data[a0n], float), np.asarray(data[rvn], float)
    best = sn if np.mean(data[sn]) <= np.mean(data[en]) else en
    bb = np.asarray(data[best], float)
    num, den = float(a0.mean() - rv.mean()), float(bb.mean() - rv.mean())
    out = {"best_baseline": best, "numerator": num, "denominator": den}
    if den <= 0:
        out["status"] = "not defined: best tuned baseline mean regret <= ROVA mean regret"
        return out
    F, _ = fold_structure(folds)
    idx = two_stage_indices(F, n_boot)
    nums = a0[idx].mean(axis=1) - rv[idx].mean(axis=1)
    dens = bb[idx].mean(axis=1) - rv[idx].mean(axis=1)
    nonpos = int(np.sum(dens <= 0))
    out.update({"status": "defined", "G": num / den, "resamples_with_nonpositive_denominator": nonpos,
                "two_stage_denominator_ci": _pct(dens)})
    out["two_stage_G_ci"] = _pct(nums / dens) if nonpos == 0 else None
    if nonpos:
        out["G_ci_note"] = "not reported: the denominator is non-positive in some resamples"
    return out


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def aligned_records(recs_by_method):
    """Vectors in (outer_fold, seed) order; every method must have identical
    (outer_fold, seed, tag) keys (matched worlds)."""
    ref, data, folds = None, {}, None
    for name, recs in sorted(recs_by_method.items()):
        rs = sorted(recs, key=lambda r: (int(r["outer_fold"]), int(r["seed"])))
        keys = [(int(r["outer_fold"]), int(r["seed"]), r.get("tag")) for r in rs]
        if ref is None:
            ref, folds = keys, np.array([k[0] for k in keys])
        elif keys != ref:
            raise ValueError(f"method {name}: worlds differ from the other methods")
        data[name] = np.array([r["regret"] for r in rs], float)
    return data, folds


def load_rev001_cew(rev001_dir, env, c):
    recs = {}
    for pre in CEW_UNIT_PREFIXES:
        path = os.path.join(rev001_dir, "units", f"{pre}{c}__{env}", "result.json")
        with open(path) as f:
            recs.update(json.load(f)["records"])
    return recs


def rev001_rows(summary, env, c):
    cell = summary["cew"][f"{env}|{c}"]
    return {(r["a"], r["b"]): r for fam in cell["families"].values() for r in fam if "a" in r}


def run_rev001(rev001_dir, n_boot=A2_N_BOOT, cells=None):
    with open(os.path.join(rev001_dir, "analysis", "summary.json")) as f:
        summary = json.load(f)
    rows, grows = [], []
    if cells is None:
        cells = [(e, c) for e in ENVS for c in GRID if f"{e}|{c}" in summary["cew"]]
    for env, c in cells:
        data, folds = aligned_records(load_rev001_cew(rev001_dir, env, c))
        ref = rev001_rows(summary, env, c)
        for fam, a, b in A2_PAIRS:
            r = analyze_pair(data[a], data[b], folds, n_boot)
            pw = ref.get((a, b), {})
            rows.append({"env": env, "c": c, "family": fam, "a": a, "b": b,
                         "rev001_label": pw.get("label"), "rev001_code": pw.get("code"),
                         "rev001_wilcoxon_p": pw.get("wilcoxon_p"), "rev001_cliffs_delta": pw.get("cliffs_delta"),
                         "rev001_mean_diff": pw.get("mean_diff"), "rev001_mean_diff_bca_ci": pw.get("mean_diff_ci"),
                         "rev001_protocol_family_reject": pw.get("protocol_family_reject"),
                         **r, "generalization": generalization_flag(pw.get("code"), r["two_stage_mean_diff_ci"])})
        g = g_clustered(data, folds, n_boot)
        g_rev = summary["cew"][f"{env}|{c}"].get("G", {})
        grows.append({"env": env, "c": c, "rev001_status": g_rev.get("status"), "rev001_G": g_rev.get("G"),
                      "rev001_G_ci": g_rev.get("G_ci"), **g})
    return rows, grows


# --------------------------------------------------------------------------- #
# Inputs, provenance, outputs
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_rev001_dir(rev001_dir, expected_sums=REV001_SUMS_SHA256, expected_entries=REV001_SUMS_ENTRIES):
    sums = os.path.join(rev001_dir, "SHA256SUMS")
    got = sha256_file(sums)
    bad, n = [], 0
    with open(sums) as f:
        for line in f:
            if not line.strip():
                continue
            digest, rel = line.rstrip("\r\n").split("  ", 1)
            n += 1
            p = os.path.join(rev001_dir, rel)
            if not os.path.exists(p) or sha256_file(p) != digest:
                bad.append(rel)
    return {"sha256sums_sha256": got, "expected": expected_sums, "entries": n,
            "expected_entries": expected_entries, "failed_entries": bad[:50],
            "passed": got == expected_sums and n == expected_entries and not bad}


def extract_archive(archive, work, expected=REV001_ARCHIVE_SHA256):
    digest = sha256_file(archive)
    if digest != expected:
        raise SystemExit(f"archive SHA-256 {digest} != expected {expected}")
    if os.path.exists(work) and os.listdir(work):
        raise SystemExit(f"work directory {work} is not empty")
    os.makedirs(work, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(work, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(work)
    return digest, os.path.join(work, "revision_out")


def _git(*args):
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            return platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def environment_record():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "system": platform.system(), "platform": platform.platform(), "cpu": _cpu_name(),
            "logical_cpus": os.cpu_count(), "git_head": _git("rev-parse", "HEAD"),
            "git_frozen_tag_commit": _git("rev-parse", f"{FROZEN_TAG_REV002}^{{commit}}"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "simulation": "none (stored per-world regrets only)"}


def _cell(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return "" if v is None else v


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r.get(k)) for k in keys})
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(buf.getvalue())


def write_json(path, obj):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def _fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def report(rows, grows, title):
    lines = [f"# {title}", "",
             "Auto-generated; lists results without interpretation. Plan: docs/REVISION_002_ANALYSIS_PLAN.md, "
             "section 3. Post-run statistical re-analysis; no simulation.", "",
             "| env | c | pair | REV-001 label | mean d | two-stage CI (d) | fold-only CI (d) | δ | two-stage CI (δ) "
             "| sign-flip p | ICC | n_eff | generalization |", "|---" * 13 + "|"]
    for r in rows:
        lines.append(f"| {r['env']} | {r['c']} | {r['a']}–{r['b']} | {_fmt(r.get('rev001_label'))} | "
                     f"{_fmt(r['mean_diff'])} | {_fmt(r['two_stage_mean_diff_ci'])} | {_fmt(r['fold_only_mean_diff_ci'])} | "
                     f"{_fmt(r['cliffs_delta'], 3)} | {_fmt(r['two_stage_delta_ci'], 3)} | {_fmt(r['sign_flip_p'], 3)} | "
                     f"{_fmt(r['icc1_of_difference'], 3)} | {_fmt(r['effective_n'], 3)} | {r['generalization']} |")
    if grows:
        lines += ["", "## Surviving-margin ratio G", "",
                  "| env | c | status | G | two-stage CI (G) | non-positive denominators | two-stage CI (denominator) |",
                  "|---|---|---|---|---|---|---|"]
        for g in grows:
            lines.append(f"| {g['env']} | {g['c']} | {g.get('status')} | {_fmt(g.get('G'))} | {_fmt(g.get('two_stage_G_ci'))} | "
                         f"{_fmt(g.get('resamples_with_nonpositive_denominator'))} | {_fmt(g.get('two_stage_denominator_ci'))} |")
    counts = {}
    for r in rows:
        counts[r["generalization"]] = counts.get(r["generalization"], 0) + 1
    lines += ["", "## Counts", ""] + [f"- {k}: {v}" for k, v in sorted(counts.items())]
    return "\n".join(lines) + "\n"


def write_sums(out_dir):
    entries = []
    for root, _d, files in os.walk(out_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), out_dir).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                entries.append(f"{sha256_file(os.path.join(root, name))}  {rel}")
    entries.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", newline="\n") as f:
        f.write("\n".join(entries) + "\n")
    return len(entries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rev001-archive", help="REV-001 revision_out.tar.gz (verified, then extracted to --work)")
    src.add_argument("--rev001-dir", help="already-extracted REV-001 revision_out (verified in place)")
    ap.add_argument("--work", help="empty directory for the extraction (with --rev001-archive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=A2_N_BOOT)
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="code verification only: permit a commit other than the frozen tag or a dirty tree")
    args = ap.parse_args(argv)

    env = environment_record()
    frozen_ok = (env["git_head"] is not None and env["git_head"] == env["git_frozen_tag_commit"]
                 and env["git_status_porcelain"] == "")
    if not frozen_ok and not args.allow_unfrozen:
        ap.error(f"not a clean checkout of {FROZEN_TAG_REV002}")
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out):
        ap.error(f"output directory {out} is not empty")
    archive_sha = None
    if args.rev001_archive:
        if not args.work:
            ap.error("--work is required with --rev001-archive")
        archive_sha, rev_dir = extract_archive(args.rev001_archive, os.path.abspath(args.work))
    else:
        rev_dir = os.path.abspath(args.rev001_dir)
    inputs = verify_rev001_dir(rev_dir)
    if not inputs["passed"] and not args.allow_unfrozen:
        print(json.dumps(inputs, indent=1), file=sys.stderr)
        raise SystemExit("REV-001 input verification FAILED; nothing was analysed")
    print(f"REV-001 inputs verified: {inputs['entries']} entries, SHA256SUMS {inputs['sha256sums_sha256']}", flush=True)

    rows, grows = run_rev001(rev_dir, args.n_boot)
    if len(grows) != len(ENVS) * len(GRID) and not args.allow_unfrozen:
        raise SystemExit(f"expected {len(ENVS) * len(GRID)} cells, found {len(grows)}")
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "A2_clustered.csv"), rows)
    write_csv(os.path.join(out, "A2_G.csv"), grows)
    counts = {}
    for r in rows:
        counts[r["generalization"]] = counts.get(r["generalization"], 0) + 1
    write_json(os.path.join(out, "A2_summary.json"), {"rows": rows, "G": grows, "counts": counts,
                                                       "n_comparisons": len(rows), "n_boot": args.n_boot})
    with open(os.path.join(out, "A2_REPORT.md"), "w", newline="\n", encoding="utf-8") as f:
        f.write(report(rows, grows, "A2 fold-clustered re-analysis of REV-001"))
    write_json(os.path.join(out, "run_metadata.json"), {
        "analysis": "A2 (docs/REVISION_002_ANALYSIS_PLAN.md section 3)", "label": "L4: post-run statistical re-analysis",
        "environment": env, "frozen_checkout": frozen_ok, "rev001_archive_sha256": archive_sha,
        "rev001_inputs": inputs, "n_boot": args.n_boot, "seed": A2_SEED, "pairs": [list(p) for p in A2_PAIRS]})
    n = write_sums(out)
    print(f"A2 done: {len(rows)} comparisons, {len(grows)} G cells; SHA256SUMS {n} files; "
          f"digest {sha256_file(os.path.join(out, 'SHA256SUMS'))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
