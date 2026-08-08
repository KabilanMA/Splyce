#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpTTSpM benchmark, single
# curated matrix.
#
# spttspm computes A(i,j,r) = Σ_k B(i,j,k) · C(k,r) (see spttspm_dn.mlir).
# Only one real SuiteSparse matrix is used here:
#
#   barth4 / Nasa group ("duplicate structural problem"): 6019 x 6019
#   square, nnz=23492 — used for C(k,r). NOTE: "barth4" also exists in the
#   Pothen group (a different, symmetric 40965-nnz matrix, also 6019x6019)
#   — suitesparse/download_data.py's load_metadata() dedupes by name alone
#   and would silently resolve to whichever entry appears last in
#   matrix_metadata.json (Pothen), so this script looks up the Nasa entry
#   directly by (name, group) instead of going through that by-name dict.
#
# B(i,j,k) is synthetically generated (there's no natural SuiteSparse
# source for a 3D tensor). Dimension choice:
#   - k must equal C's own k dimension (6019, from barth4/Nasa — barth4 is
#     square so this is also its num_cols) so B(i,j,k) actually conforms
#     against C(k,r).
#   - i and j aren't constrained by C at all, so both use a fixed 500.
#   => B(i,j,k) = 500 x 500 x 6019.
# B is generated at a fixed 0.06% nonzero density (sparsity=0.9994) via
# generate_sparse_3d_tns below (ported from experiments/gen_data.py's
# function of the same name).
#
# This script:
#   1. Downloads barth4/Nasa (suitesparse/download_data.py's
#      download_and_extract, called with the Nasa group directly — see
#      note above; matrix_metadata.json must already exist, produced by
#      suitesparse/scrape_metadata.py) and converts it to .tns
#      (suitesparse/convert_to_tns.py's process_dir — drops explicit
#      zeros; barth4/Nasa isn't symmetric so no mirroring happens),
#      copying the result into this directory as tensor_C.tns.
#   2. Generates tensor_B.tns (500 x 500 x 6019, 0.06% dense).
#   3. Runs test_benchmark_spttspm_splyce_phase_001 FIRST, then
#      test_benchmark_spttspm_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 6 iterations internally per spttspm_dn.mlir's
#      @main, writing a "benchmark" file with one time per line.
#   4. Appends one summary row (dataset, b/c shape, sparsity, scf_median,
#      splyce_median — median of the 5 non-cold-start iterations;
#      scf_median is "SKIPPED" when Splyce timed out) to
#      spttspm_realworld_results.csv, and every individual raw iteration
#      time to spttspm_realworld_raw_runtimes.csv as a backup.
#   5. Deletes tensor_B.tns/tensor_C.tns and the downloaded/converted
#      suitesparse/barth4/ directory afterward.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once
# barth4 is already recorded (unless the CSV row is removed first).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spttspm_scf and test_benchmark_spttspm_splyce_phase_001
# exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)

import csv
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spttspm_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spttspm_splyce_phase_001")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spttspm_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spttspm_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# The single curated job this script runs — see module docstring for why
# the group is pinned explicitly (barth4 is ambiguous by name alone).
MATRIX_NAME = "barth4"
MATRIX_GROUP = "Nasa"

# Free index sizes (not constrained by C) for the synthetic B tensor.
SYNTHETIC_I = 500
SYNTHETIC_J = 500

# Fixed nonzero density for the synthetic B tensor.
SYNTHETIC_DENSITY_PCT = 0.06


def load_matrix_entry(name, group):
    # suitesparse/download_data.py's load_metadata() dedupes by name alone,
    # which would silently pick the wrong entry for a name that exists in
    # more than one group (see module docstring) — so this reads the raw
    # metadata list directly and matches on (name, group).
    with open(dl.METADATA_PATH) as f:
        data = json.load(f)
    for m in data["matrices"]:
        if m["name"] == name and m["group"] == group:
            return m
    return None


def generate_sparse_3d_tns(filename, dim1, dim2, dim3, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_3d_tns: geometric
    # skip sampling (O(1) memory, visits only ~nnz elements) writes a
    # FROSTT-format sparse tensor with `1 - sparsity` nonzero density.
    total_elements = dim1 * dim2 * dim3
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, "w") as tmp:
        idx = 0
        while idx < total_elements:
            i = idx // (dim2 * dim3)
            j = (idx % (dim2 * dim3)) // dim3
            k = idx % dim3
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {k + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, "w") as f:
        f.write("# extended FROSTT format\n")
        f.write(f"3 {actual_nnz}\n")
        f.write(f"{dim1} {dim2} {dim3}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)

    print(f"Generated sparse 3D tensor: {filename} | Shape: ({dim1}, {dim2}, {dim3}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")
    return actual_nnz


def median_excl_first(times):
    rest = times[1:]
    return statistics.median(rest) if rest else None


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
    timeout = 300
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])

    if not (os.path.isfile(BASELINE_BIN) and os.path.isfile(SPLYCE_BIN)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    entry = load_matrix_entry(MATRIX_NAME, MATRIX_GROUP)
    if entry is None:
        sys.exit(f"error: {MATRIX_GROUP}/{MATRIX_NAME} not found in matrix_metadata.json")
    name, group = entry["name"], entry["group"]

    done = already_recorded()
    if name in done:
        print(f"{name} already recorded — nothing to do")
        return

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "group", "b_shape", "c_shape",
                "synthetic_density_pct", "scf_median_s", "splyce_median_s",
            ])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        print(f"=== {group}/{name} (nnz={entry['nnz']}) ===")
        dataset_dir = os.path.join(SUITESPARSE_DIR, name)

        try:
            dl.download_and_extract(name, group, force=False)
            if not os.path.isdir(dataset_dir):
                sys.exit(f"  [skip] {name}: download/extract failed")

            cvt.process_dir(dataset_dir)
            tns_path = os.path.join(dataset_dir, f"{name}.tns")
            if not os.path.isfile(tns_path):
                sys.exit(f"  [skip] {name}: conversion failed, no .tns produced")

            shutil.copyfile(tns_path, TENSOR_C)
            dim_k = entry["num_rows"]  # C(k,r); square, so num_rows == num_cols == r too

            synthetic_sparsity = 1.0 - (SYNTHETIC_DENSITY_PCT / 100.0)

            print(f"  [generate] tensor_B ({SYNTHETIC_I} x {SYNTHETIC_J} x {dim_k}) ...")
            generate_sparse_3d_tns(TENSOR_B, SYNTHETIC_I, SYNTHETIC_J, dim_k, synthetic_sparsity)

            print("  [run] splyce phase_001 ...")
            splyce_times, splyce_timed_out = run_binary(SPLYCE_BIN, timeout)

            if splyce_timed_out:
                print("  [skip] splyce timed out — skipping baseline run")
                scf_times, scf_med = None, "SKIPPED"
            else:
                print("  [run] baseline (scf) ...")
                scf_times, _ = run_binary(BASELINE_BIN, timeout * 5)
                scf_med = median_excl_first(scf_times) if scf_times else "NA"

            for i, t in enumerate(splyce_times or []):
                raw_writer.writerow([name, "splyce_phase_001", i, t])
            for i, t in enumerate(scf_times or []):
                raw_writer.writerow([name, "scf", i, t])
            rf.flush()

            splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

            if splyce_med != "NA":
                summary_writer.writerow([
                    name, group,
                    f"{SYNTHETIC_I}x{SYNTHETIC_J}x{dim_k}", f"{dim_k}x{dim_k}",
                    SYNTHETIC_DENSITY_PCT, scf_med, splyce_med,
                ])
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
