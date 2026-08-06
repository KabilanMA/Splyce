#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpMMH benchmark sweep.
#
# For every SuiteSparse matrix selected by
# suitesparse/download_data.py's median_square_matrix_per_group()
# sampling (one representative square matrix per group, 169 total), this:
#
#   1. Downloads it (suitesparse/download_data.py's download_and_extract).
#   2. Converts it to .tns (suitesparse/convert_to_tns.py's process_dir —
#      mirrors symmetric entries, drops explicit zeros).
#   3. Copies that single .tns into this directory as tensor_B.tns,
#      tensor_C.tns, AND tensor_D.tns — spmmh needs three operands (B, C
#      dense, D), we only have one real matrix per dataset, so all three
#      use it (i.e. this benchmarks A = D .* (B @ C) with B = C = D).
#      tensor_C is declared with a dense encoding in spmmh_dn.mlir, but
#      sparse_tensor.new zero-fills any coordinate missing from the file
#      regardless of the declared level type — confirmed against a hand-
#      computed example, so the plain (non-exhaustive) converted .tns
#      works correctly here without expanding it to a fully-enumerated
#      dense file (which would be prohibitively large for bigger N).
#   4. Before downloading anything, estimates peak memory for the dataset
#      and skips it outright if that exceeds --memory-limit-gb (default
#      250, matching this benchmark's server). Both A (the output) and C
#      (declared dense — see above) are num_rows x num_rows dense f64
#      buffers, so the estimate is 2 * num_rows**2 * 8 bytes. There is no
#      sparse-output binary for spmmh (unlike spgemm): an attempt to make
#      A a sparse CSC output was found to compile and compute correctly,
#      but the workspace-insertion pattern it requires isn't recognized by
#      Splyce's current vectorizer, so skipping oversized datasets outright
#      is the only available mitigation here rather than swapping formats.
#   5. Runs test_benchmark_spmmh_splyce_phase_001 FIRST, then
#      test_benchmark_spmmh_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 4 iterations internally per spmmh_dn.mlir's
#      @main, writing a "benchmark" file with one time per line.
#   6. Appends one summary row (dataset, group, nnz, scf_median,
#      splyce_median — median of the 3 non-cold-start iterations; scf_
#      median is "SKIPPED" when Splyce timed out, and both are
#      "SKIPPED_MEMORY" when the dataset was skipped per step 4 above) to
#      spmmh_realworld_results.csv, and every individual raw iteration
#      time to spmmh_realworld_raw_runtimes.csv as a backup.
#   7. Deletes tensor_B.tns/tensor_C.tns/tensor_D.tns and the downloaded/
#      converted suitesparse/<name>/ directory before moving to the next
#      dataset.
#
# Both CSVs are appended to, not overwritten, so the sweep can be
# interrupted and resumed (already-recorded datasets aren't re-run —
# including ones skipped for memory, since those get a CSV row too).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spmmh_scf and test_benchmark_spmmh_splyce_phase_001
# exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # full 169-dataset sweep
#   ./run_suitesparse_benchmark.py --limit 5    # only the first 5 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --memory-limit-gb 500  # override the
#                                                  # 250 GiB skip threshold

import csv
import os
import shutil
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmmh_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmmh_splyce_phase_001")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmmh_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmmh_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# spmmh has no sparse-output binary (see module docstring), so oversized
# datasets can only be skipped outright, not routed to a smaller-footprint
# variant. Default matches this benchmark's server RAM.
DEFAULT_MEMORY_LIMIT_GB = 250


def median_excl_first(times):
    rest = times[1:]
    return statistics.median(rest) if rest else None


def estimated_memory_bytes(entry):
    # A (the output) and C (declared dense — see module docstring) are
    # both num_rows x num_rows dense f64 buffers; B and D stay sparse
    # (O(nnz), negligible next to the O(n^2) terms for any n large enough
    # to matter here).
    n = entry["num_rows"]
    return 2 * n * n * 8


def run_binary(bin_path, timeout):
    if os.path.isfile(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
    try:
        subprocess.run(
            [bin_path],
            cwd=SCRIPT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"    [timeout] {os.path.basename(bin_path)} exceeded {timeout}s")
        return None, True
    if not os.path.isfile(BENCHMARK_FILE):
        print(f"    [warning] {os.path.basename(bin_path)} produced no benchmark file (likely crashed)")
        return None, False
    with open(BENCHMARK_FILE) as f:
        times = [float(line.strip()) for line in f if line.strip()]
    os.remove(BENCHMARK_FILE)
    return times, False


def already_recorded():
    if not os.path.isfile(SUMMARY_CSV):
        return set()
    with open(SUMMARY_CSV) as f:
        return {row["dataset"] for row in csv.DictReader(f)}


def main():
    args = sys.argv[1:]
    limit = None
    timeout = 300
    memory_limit_gb = DEFAULT_MEMORY_LIMIT_GB
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--memory-limit-gb" in args:
        memory_limit_gb = float(args[args.index("--memory-limit-gb") + 1])
    memory_limit_bytes = memory_limit_gb * 1024 ** 3

    if not (os.path.isfile(BASELINE_BIN) and os.path.isfile(SPLYCE_BIN)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    selected = dl.median_square_matrix_per_group(metadata)
    if limit is not None:
        selected = selected[:limit]

    done = already_recorded()
    print(f"{len(selected)} datasets selected, {len(done)} already recorded — resuming")

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow(["dataset", "group", "nnz", "scf_median_s", "splyce_median_s"])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for entry in selected:
            name = entry["name"]
            group = entry["group"]

            if name in done:
                continue

            print(f"=== {group}/{name} (nnz={entry['nnz']}) ===")

            est_bytes = estimated_memory_bytes(entry)
            if est_bytes > memory_limit_bytes:
                print(f"  [skip] {name}: estimated {est_bytes / 1024**3:.1f} GiB "
                      f"(> {memory_limit_gb:.0f} GiB memory limit) — no sparse-output "
                      f"binary for spmmh to fall back to, skipping")
                summary_writer.writerow([name, group, entry["nnz"], "SKIPPED_MEMORY", "SKIPPED_MEMORY"])
                sf.flush()
                continue

            dataset_dir = os.path.join(SUITESPARSE_DIR, name)

            try:
                dl.download_and_extract(name, group, force=False)
                if not os.path.isdir(dataset_dir):
                    print(f"  [skip] {name}: download/extract failed")
                    continue

                cvt.process_dir(dataset_dir)
                tns_path = os.path.join(dataset_dir, f"{name}.tns")
                if not os.path.isfile(tns_path):
                    print(f"  [skip] {name}: conversion failed, no .tns produced")
                    continue

                shutil.copyfile(tns_path, TENSOR_B)
                shutil.copyfile(tns_path, TENSOR_C)
                shutil.copyfile(tns_path, TENSOR_D)

                print("  [run] splyce phase_001 ...")
                splyce_times, splyce_timed_out = run_binary(SPLYCE_BIN, timeout)

                if splyce_timed_out:
                    print("  [skip] splyce timed out — skipping baseline run")
                    scf_times, scf_med = None, "SKIPPED"
                else:
                    print("  [run] baseline (scf) ...")
                    scf_times, _ = run_binary(BASELINE_BIN, timeout*5)
                    scf_med = median_excl_first(scf_times) if scf_times else "NA"

                for i, t in enumerate(splyce_times or []):
                    raw_writer.writerow([name, "splyce_phase_001", i, t])
                for i, t in enumerate(scf_times or []):
                    raw_writer.writerow([name, "scf", i, t])
                rf.flush()

                splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

                if splyce_med != "NA":
                    summary_writer.writerow([name, group, entry["nnz"], scf_med, splyce_med])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            finally:
                for f in (TENSOR_B, TENSOR_C, TENSOR_D):
                    if os.path.isfile(f):
                        os.remove(f)
                if os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
