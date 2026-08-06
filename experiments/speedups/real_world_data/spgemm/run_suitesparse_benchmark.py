#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpGEMM benchmark sweep.
#
# For every SuiteSparse matrix selected by
# suitesparse/download_data.py's median_square_matrix_per_group()
# sampling (one representative square matrix per group, 169 total), this:
#
#   1. Downloads it (suitesparse/download_data.py's download_and_extract).
#   2. Converts it to .tns (suitesparse/convert_to_tns.py's process_dir —
#      mirrors symmetric entries, drops explicit zeros).
#   3. Copies that single .tns into this directory as BOTH tensor_B.tns
#      and tensor_C.tns (spgemm needs two operands; we only have one real
#      matrix per dataset, so it's used for both — i.e. this benchmarks
#      A = B^2 for each sampled matrix B).
#   4. Picks dense (spgemm_dn.mlir) vs CSR (spgemm.mlir) output binaries
#      based on the dense output size: A = B^2 is num_rows x num_rows, so a
#      dense f64 A would need num_rows**2 * 8 bytes. If that exceeds
#      DENSE_OUTPUT_LIMIT_BYTES (128 GiB), the dense binaries would just
#      fail to allocate A and crash (confirmed: BenElechi1 — 245874 x
#      245874, ~484 GB dense — segfaults immediately, malloc returns NULL
#      and the sparsifier-generated store doesn't check it), so the CSR
#      binaries are used instead for both scf and Splyce. CSR output avoids
#      that crash, though for large enough matrices it can still be too
#      slow to finish within --timeout (the (i,k) loop nest is driven by
#      the dense levels of the *input* encodings either way, so switching
#      only the output format doesn't change that time complexity).
#   5. Runs the (dense- or CSR-selected) splyce_phase_001 binary FIRST, then
#      the scf binary — but only if Splyce didn't time out. If Splyce
#      already hit the timeout, the (unvectorized, typically no faster)
#      baseline is skipped entirely rather than wasting the same timeout on
#      a run that's essentially guaranteed to also be too slow. Each binary
#      loops 4 iterations internally per its @main, writing a "benchmark"
#      file with one time per line.
#   6. Appends one summary row (dataset, group, nnz, format, scf_median,
#      splyce_median — median of the 3 non-cold-start iterations, i.e.
#      excluding iteration 0; scf_median is "SKIPPED" when Splyce timed
#      out) to spgemm_realworld_results.csv, and every individual raw
#      iteration time to spgemm_realworld_raw_runtimes.csv as a backup.
#   7. Deletes tensor_B.tns/tensor_C.tns and the downloaded/converted
#      suitesparse/<name>/ directory before moving to the next dataset.
#
# Both CSVs are appended to, not overwritten, so the sweep can be
# interrupted and resumed (already-recorded datasets aren't re-run).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spgemm_scf, test_benchmark_spgemm_splyce_phase_001,
# test_benchmark_spgemm_csr_scf, and test_benchmark_spgemm_csr_splyce_phase_001
# all exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # full 169-dataset sweep
#   ./run_suitesparse_benchmark.py --limit 5    # only the first 5 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)

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

BASELINE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_scf")
SPLYCE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_splyce_phase_001")
BASELINE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_scf")
SPLYCE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_splyce_phase_001")

# A = B^2 is num_rows x num_rows; a dense f64 A needs num_rows**2 * 8 bytes.
# Past this, the dense-output binaries would fail to allocate A and crash —
# swap to the CSR-output binaries instead. See module docstring.
DENSE_OUTPUT_LIMIT_BYTES = 200 * 1024 ** 3  # 200 GiB

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")


def median_excl_first(times):
    rest = times[1:]
    return statistics.median(rest) if rest else None


def dense_output_bytes(entry):
    return entry["num_rows"] * entry["num_rows"] * 8


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
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])

    if not (os.path.isfile(BASELINE_BIN_DENSE) and os.path.isfile(SPLYCE_BIN_DENSE)
            and os.path.isfile(BASELINE_BIN_CSR) and os.path.isfile(SPLYCE_BIN_CSR)):
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
            summary_writer.writerow(["dataset", "group", "nnz", "format", "scf_median_s", "splyce_median_s"])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for entry in selected:
            name = entry["name"]
            group = entry["group"]

            if name in done:
                continue

            print(f"=== {group}/{name} (nnz={entry['nnz']}) ===")
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

                dense_bytes = dense_output_bytes(entry)
                if dense_bytes > DENSE_OUTPUT_LIMIT_BYTES:
                    fmt = "csr"
                    baseline_bin, splyce_bin = BASELINE_BIN_CSR, SPLYCE_BIN_CSR
                    print(f"  [format] dense output would be {dense_bytes / 1024**3:.1f} GiB "
                          f"(> {DENSE_OUTPUT_LIMIT_BYTES / 1024**3:.0f} GiB) — using CSR binaries")
                else:
                    fmt = "dense"
                    baseline_bin, splyce_bin = BASELINE_BIN_DENSE, SPLYCE_BIN_DENSE

                print(f"  [run] splyce phase_001 ({fmt}) ...")
                splyce_times, splyce_timed_out = run_binary(splyce_bin, timeout)

                if splyce_timed_out:
                    print("  [skip] splyce timed out — skipping baseline run")
                    scf_times, scf_med = None, "SKIPPED"
                else:
                    print(f"  [result] splyce phase_001 ({fmt}) runtime: {splyce_times}")
                    print(f"  [run] baseline (scf, {fmt}) ...")
                    scf_times, _ = run_binary(baseline_bin, timeout*5)
                    scf_med = median_excl_first(scf_times) if scf_times else "NA"

                for i, t in enumerate(splyce_times or []):
                    raw_writer.writerow([name, f"splyce_phase_001_{fmt}", i, t])
                for i, t in enumerate(scf_times or []):
                    raw_writer.writerow([name, f"scf_{fmt}", i, t])
                rf.flush()

                splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

                if splyce_med != "NA":
                    summary_writer.writerow([name, group, entry["nnz"], fmt, scf_med, splyce_med])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            finally:
                for f in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(f):
                        os.remove(f)
                if os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
