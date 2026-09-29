"""Tests of the execution-environment gate (synthetic observations only).

    PYTHONPATH=. python -m rova_revision.tests.test_env_gate
"""
import copy

from rova_revision.env_guard import THREAD_VARS, evaluate, format_table, parse_cpuinfo, parse_os_release, reference


def _reference_like_host(ref):
    blas = {"name": "scipy-openblas", "version": ref["blas_version"] + ".0",
            "openblas configuration": "OpenBLAS DYNAMIC_ARCH"}
    return {
        "system": "Linux", "platform": f"Linux-{ref['kernel']}-x86_64-with-glibc{ref['glibc']}",
        "kernel": ref["kernel"], "kernel_version": "#40-Ubuntu", "os_id": "ubuntu",
        "os_pretty_name": "Ubuntu", "glibc": ref["glibc"],
        "cpu_model": ref["cpu_model"] + " 16-Core Processor", "physical_cores": ref["physical_cores"],
        "sockets": 1, "logical_cpus": ref["logical_cpus"], "memory_gib": 59.0,
        "board": {"board_vendor": "Micro-Star", "board_name": "MS-7E51", "product_name": "MS-7E51"},
        "hostname": ref["hostname"], "python": ref["python"], "python_executable": ref["interpreter"],
        "numpy": ref["numpy"], "scipy": ref["scipy"], "matplotlib": ref["matplotlib"],
        "blas": {"numpy": {"blas": dict(blas)}, "scipy": {"blas": dict(blas)}},
        "thread_vars": {v: "1" for v in THREAD_VARS}, "gpu_frameworks_loaded": [], "workers": ref["workers"],
        "git_head": "abc", "git_frozen_tag_commit": "abc", "git_status_porcelain": "", "git_describe": "x",
    }


def test_reference_is_parsed_from_archived_records():
    ref = reference()
    assert ref["cpu_model"] == "AMD Ryzen 9 9950X"
    assert (ref["physical_cores"], ref["logical_cpus"], ref["sockets"]) == (16, 32, 1)
    assert ref["kernel"] == "6.17.0-40-generic" and ref["glibc"] == "2.42" and ref["os_id"] == "ubuntu"
    assert (ref["python"], ref["numpy"], ref["scipy"], ref["matplotlib"]) == ("3.13.14", "2.5.1", "1.18.0", "3.11.0")
    assert (ref["blas_family"], ref["blas_version"]) == ("scipy-openblas", "0.3.33")
    assert ref["workers"] == 15 and ref["hostname"] == "halyk-rtx5090-MS-7E51"


def test_parsers():
    blocks = []
    for cpu in range(32):
        blocks.append(f"processor\t: {cpu}\nmodel name\t: AMD Ryzen 9 9950X 16-Core Processor\n"
                      f"physical id\t: 0\ncore id\t\t: {cpu % 16}\n")
    info = parse_cpuinfo("\n".join(blocks))
    assert info == {"cpu_model": "AMD Ryzen 9 9950X 16-Core Processor", "sockets": 1, "physical_cores": 16}
    assert parse_os_release('ID=ubuntu\nPRETTY_NAME="Ubuntu 25.10"\n')["ID"] == "ubuntu"


def test_gate_passes_on_reference_host():
    ref = reference()
    res = evaluate(_reference_like_host(ref), ref)
    assert res["passed"] and not res["soft_failed"] and not res["hard_failed"], res["soft_failed"]
    assert "cpu_model" in format_table(res)


def test_host_name_and_interpreter_path_do_not_imply_the_cpu():
    ref = reference()
    info = _reference_like_host(ref)
    info["cpu_model"] = "Intel(R) Core(TM) i7-12700"
    res = evaluate(info, ref)
    assert not res["passed"] and res["soft_failed"] == ["cpu_model"]


def test_soft_mismatch_requires_written_approval():
    ref = reference()
    for key, val in (("kernel", "6.17.0-45-generic"), ("numpy", "2.5.2"), ("workers", 8),
                     ("physical_cores", 8), ("glibc", "2.43")):
        info = _reference_like_host(ref)
        info[key] = val
        res = evaluate(info, ref)
        assert not res["passed"] and len(res["soft_failed"]) == 1, (key, res["soft_failed"])
        res = evaluate(info, ref, approval="Approved by PI on <date>: kernel update only")
        assert res["passed"] and res["overridden_by_approval"]


def test_blas_checks():
    ref = reference()
    info = _reference_like_host(ref)
    info["blas"]["numpy"]["blas"]["version"] = "0.3.31.188.0"
    assert evaluate(info, ref)["passed"]
    info["blas"]["scipy"]["blas"]["version"] = "0.3.30"
    assert "blas_version (numpy, scipy)" in evaluate(info, ref)["soft_failed"]
    info = _reference_like_host(ref)
    info["blas"]["scipy"]["blas"]["name"] = "mkl"
    assert "blas_family (numpy, scipy)" in evaluate(info, ref)["soft_failed"]


def test_hard_failures_cannot_be_overridden():
    ref = reference()
    for mutate in (lambda i: i["thread_vars"].update(OMP_NUM_THREADS="4"),
                   lambda i: i.update(gpu_frameworks_loaded=["torch"]),
                   lambda i: i.update(git_status_porcelain="?? stray.txt"),
                   lambda i: i.update(git_head="def")):
        info = copy.deepcopy(_reference_like_host(ref))
        mutate(info)
        res = evaluate(info, ref, approval="any approval")
        assert not res["passed"] and len(res["hard_failed"]) == 1


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
