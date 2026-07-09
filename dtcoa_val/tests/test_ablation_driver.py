"""ABLATION-001 driver integration/regression tests (Task C/G). Run:
    python3 -m dtcoa_val.tests.test_ablation_driver

Exercises a real single-cell pilot end-to-end, resumability, the FIX-5
race-free manifest primitives (atomic write, shard consolidation).
"""
import hashlib
import json
import os
import tempfile

from dtcoa_val.experiments import run_ablation as R


def _hash_dir(d):
    h = hashlib.sha256()
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.endswith(".json"):
                h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


def test_single_cell_end_to_end_and_resumable():
    with tempfile.TemporaryDirectory() as out:
        # num_shards=15 -> shard 0 gets exactly cell index 0.
        R.run_ablation(out, budget_name="pilot", shard_index=0, num_shards=15)
        ck = os.path.join(out, "hpo_checkpoints")
        # The single assigned cell is (first env, first cost ratio).
        from dtcoa_val.experiments.environments import ENVIRONMENTS, COST_RATIO_GRID
        env = list(ENVIRONMENTS)[0]; c = COST_RATIO_GRID[0]
        tag = f"{env}__c{c}"
        for b in R.BRANCHES:
            assert os.path.exists(os.path.join(ck, f"{tag}__{b}.json")), b
        assert os.path.exists(os.path.join(out, f"{tag}__ablation_report.json"))
        assert os.path.exists(os.path.join(out, f"{tag}__containment_report.json"))
        assert os.path.exists(os.path.join(out, "manifest.shard0.json"))
        assert os.path.exists(os.path.join(out, "manifest.json"))
        assert os.path.exists(os.path.join(out, "environment.txt"))
        assert os.path.exists(os.path.join(out, "seeds_verified.json"))

        before = _hash_dir(ck)
        man_before = json.load(open(os.path.join(out, "manifest.shard0.json")))["completed"]
        # Re-run: must resume (skip) and change nothing.
        R.run_ablation(out, budget_name="pilot", shard_index=0, num_shards=15)
        after = _hash_dir(ck)
        man_after = json.load(open(os.path.join(out, "manifest.shard0.json")))["completed"]
        assert before == after, "re-run must not alter completed checkpoints"
        assert set(man_before) == set(man_after), "re-run must not change completion set"


def test_atomic_save_leaves_no_tmp():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        R._atomic_save_json(p, {"x": 1})
        assert json.load(open(p)) == {"x": 1}
        assert [f for f in os.listdir(d)] == ["m.json"], "no leftover .tmp file"


def test_consolidate_merges_shards():
    with tempfile.TemporaryDirectory() as d:
        R._atomic_save_json(os.path.join(d, "manifest.shard0.json"),
                            {"shard": 0, "completed": {"a__full": {"summary": {}}}})
        R._atomic_save_json(os.path.join(d, "manifest.shard1.json"),
                            {"shard": 1, "completed": {"b__full": {"summary": {}}}})
        m = R.consolidate_manifest(d)
        assert set(m["completed"]) == {"a__full", "b__full"}
        assert os.path.exists(os.path.join(d, "manifest.json"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL ABLATION DRIVER TESTS PASSED")
