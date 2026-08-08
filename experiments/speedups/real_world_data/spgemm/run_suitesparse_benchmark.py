#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpGEMM benchmark, curated set.
#
# Runs four fixed jobs (instead of sweeping the full 169 median-per-group
# SuiteSparse sample — see suitesparse/download_data.py for that sweep):
#
#   1. internet        — square, used for BOTH tensor_B and tensor_C
#                                 (A = B^2).
#   2. exdata_1  (GHS_indef)   — square, used for BOTH tensor_B and tensor_C.
#   3. heart1    (Norris)      — square, used for BOTH tensor_B and tensor_C.
#   4. c8_mat11  (JGD_Groebner) — rectangular (4562 x 5761), so it can't be
#                                 squared against itself. Used for tensor_B
#                                 only; tensor_C is a synthetically generated
#                                 5761 x 5000 tensor (5761 rows so the shared
#                                 j dimension lines up with c8_mat11's column
#                                 count) at 9.37% nonzero density — chosen to
#                                 match c8_mat11's own density
#                                 (2462970 / (4562*5761) = 9.372%), so the two
#                                 operands are comparably sparse.
#
# For each job, this:
#   1. Downloads every real matrix it needs (suitesparse/download_data.py's
#      download_and_extract, looked up via matrix_metadata.json — run
#      suitesparse/scrape_metadata.py first if that file doesn't exist yet).
#   2. Converts each to .tns (suitesparse/convert_to_tns.py's process_dir —
#      mirrors symmetric entries, drops explicit zeros).
#   3. Populates tensor_B.tns / tensor_C.tns in this directory: for jobs 1-3
#      the same downloaded .tns is copied into both files; for job 4,
#      tensor_C.tns is instead generated synthetically (see
#      generate_sparse_2d_tns below — ported from experiments/gen_data.py's
#      function of the same name).
#   4. Picks dense (spgemm_dn.mlir) vs CSR (spgemm.mlir) output binaries
#      based on the dense output size: A is num_rows(B) x num_cols(C), so a
#      dense f64 A would need num_rows(B) * num_cols(C) * 8 bytes. If that
#      exceeds DENSE_OUTPUT_LIMIT_BYTES (200 GiB — none of the four curated
#      jobs currently reach this), the dense binaries would just fail to
#      allocate A and crash, so the CSR binaries are used instead for both
#      scf and Splyce.
#   5. Runs the (dense- or CSR-selected) splyce_phase_001 binary FIRST, then
#      the scf binary — but only if Splyce didn't time out. If Splyce
#      already hit the timeout, the (unvectorized, typically no faster)
#      baseline is skipped entirely rather than wasting the same timeout on
#      a run that's essentially guaranteed to also be too slow. Each binary
#      loops 6 iterations internally per its @main, writing a "benchmark"
#      file with one time per line.
#   6. Appends one summary row (dataset, b/c source, b/c shape+nnz, format,
#      scf_median, splyce_median — median of the 5 non-cold-start
#      iterations, i.e. excluding iteration 0; scf_median is "SKIPPED" when
#      Splyce timed out) to spgemm_realworld_results.csv, and every
#      individual raw iteration time to spgemm_realworld_raw_runtimes.csv as
#      a backup.
#   7. Deletes tensor_B.tns/tensor_C.tns and any downloaded/converted
#      suitesparse/<name>/ director(y/ies) before moving to the next job.
#
# Both CSVs are appended to, not overwritten, so the sweep can be
# interrupted and resumed (already-recorded jobs aren't re-run).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spgemm_scf, test_benchmark_spgemm_splyce_phase_001,
# test_benchmark_spgemm_csr_scf, and test_benchmark_spgemm_csr_splyce_phase_001
# all exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # all 4 curated jobs
#   ./run_suitesparse_benchmark.py --limit 2    # only the first 2 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)

import csv
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

BASELINE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_scf")
SPLYCE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_splyce_phase_001")
BASELINE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_scf")
SPLYCE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_splyce_phase_001")

# A is num_rows(B) x num_cols(C); a dense f64 A needs that many * 8 bytes.
# Past this, the dense-output binaries would fail to allocate A and crash —
# swap to the CSR-output binaries instead. See module docstring.
DENSE_OUTPUT_LIMIT_BYTES = 200 * 1024 ** 3  # 200 GiB

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# Synthetic tensor_C paired with c8_mat11 (job 4 below). 5761 rows so the
# shared j dimension matches c8_mat11's column count; 9.37% nonzero density
# matches c8_mat11's own density (2462970 / (4562*5761) = 9.372%).
SYNTHETIC_C_COLS = 5000
SYNTHETIC_C_DENSITY_PCT = 9.37

# The four curated jobs this script runs. "c": None means tensor_C is
# generated synthetically instead of downloaded (see main()).
JOBS = [
    {"name": "internet", "b": "internet", "c": "internet"},
    {"name": "exdata_1", "b": "exdata_1", "c": "exdata_1"},
    {"name": "heart1", "b": "heart1", "c": "heart1"},
    {"name": "c8_mat11", "b": "c8_mat11", "c": None},
]


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_2d_tns: geometric
    # skip sampling (O(1) memory, visits only ~nnz elements) writes a
    # FROSTT-format sparse tensor with `1 - sparsity` nonzero density. Each
    # element is included with probability `density`; the gap to the next
    # included element follows Geom(density), computed as
    # floor(log(U) / log(1 - density)) for uniform U in (0, 1).
    total_elements = rows * cols
    density = 1.0 - sparsity
    tmp_path = filename + ".tmp"

    actual_nnz = 0
    with open(tmp_path, "w") as tmp:
        idx = 0
        while idx < total_elements:
            i = idx // cols
            j = idx % cols
            val = random.uniform(0.5, 2.5)
            tmp.write(f"{i + 1} {j + 1} {val:.4f}\n")
            actual_nnz += 1
            r = random.random()
            if r == 0.0:
                break
            idx += math.floor(math.log(r) / math.log(1.0 - density)) + 1

    with open(filename, "w") as f:
        f.write("# extended FROSTT format\n")
        f.write(f"2 {actual_nnz}\n")
        f.write(f"{rows} {cols}\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                f.write(line)
    os.remove(tmp_path)

    print(f"Generated sparse 2D tensor: {filename} | Shape: ({rows}, {cols}) | NNZ: {actual_nnz} | Sparsity: {sparsity}")
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


def download_matrix(name, metadata):
    """Downloads + converts a named SuiteSparse matrix. Returns
    (tns_path, dataset_dir) or (None, dataset_dir_or_None) on failure."""
    entry = metadata.get(name)
    if entry is None:
        print(f"  [skip] {name}: not found in matrix_metadata.json")
        return None, None

    dl.download_and_extract(name, entry["group"], force=False)
    dataset_dir = os.path.join(SUITESPARSE_DIR, name)
    if not os.path.isdir(dataset_dir):
        print(f"  [skip] {name}: download/extract failed")
        return None, None

    cvt.process_dir(dataset_dir)
    tns_path = os.path.join(dataset_dir, f"{name}.tns")
    if not os.path.isfile(tns_path):
        print(f"  [skip] {name}: conversion failed, no .tns produced")
        return None, dataset_dir

    return tns_path, dataset_dir


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

    jobs = JOBS[:limit] if limit is not None else JOBS

    done = already_recorded()
    print(f"{len(jobs)} curated jobs, {len(done)} already recorded — resuming")

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "b_source", "c_source", "b_shape", "b_nnz",
                "c_shape", "c_nnz", "format", "scf_median_s", "splyce_median_s",
            ])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for job in jobs:
            name = job["name"]
            if name in done:
                continue

            print(f"=== {name} ===")
            downloaded_dirs = []

            try:
                b_name = job["b"]
                b_entry = metadata.get(b_name)
                b_tns, b_dir = download_matrix(b_name, metadata)
                if b_dir is not None:
                    downloaded_dirs.append(b_dir)
                if b_tns is None:
                    continue
                shutil.copyfile(b_tns, TENSOR_B)
                b_rows, b_cols, b_nnz = b_entry["num_rows"], b_entry["num_cols"], b_entry["nnz"]

                c_name = job["c"]
                if c_name is None:
                    c_rows, c_cols = b_cols, SYNTHETIC_C_COLS
                    synthetic_sparsity = 1.0 - (SYNTHETIC_C_DENSITY_PCT / 100.0)
                    c_nnz = generate_sparse_2d_tns(TENSOR_C, c_rows, c_cols, synthetic_sparsity)
                    c_source = f"synthetic_{SYNTHETIC_C_DENSITY_PCT}pct"
                elif c_name == b_name:
                    shutil.copyfile(b_tns, TENSOR_C)
                    c_rows, c_cols, c_nnz = b_rows, b_cols, b_nnz
                    c_source = c_name
                else:
                    c_entry = metadata.get(c_name)
                    c_tns, c_dir = download_matrix(c_name, metadata)
                    if c_dir is not None:
                        downloaded_dirs.append(c_dir)
                    if c_tns is None:
                        continue
                    shutil.copyfile(c_tns, TENSOR_C)
                    c_rows, c_cols, c_nnz = c_entry["num_rows"], c_entry["num_cols"], c_entry["nnz"]
                    c_source = c_name

                dense_bytes = b_rows * c_cols * 8
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
                    scf_times, _ = run_binary(baseline_bin, timeout * 5)
                    scf_med = median_excl_first(scf_times) if scf_times else "NA"

                for i, t in enumerate(splyce_times or []):
                    raw_writer.writerow([name, f"splyce_phase_001_{fmt}", i, t])
                for i, t in enumerate(scf_times or []):
                    raw_writer.writerow([name, f"scf_{fmt}", i, t])
                rf.flush()

                splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

                if splyce_med != "NA":
                    summary_writer.writerow([
                        name, b_name, c_source,
                        f"{b_rows}x{b_cols}", b_nnz,
                        f"{c_rows}x{c_cols}", c_nnz,
                        fmt, scf_med, splyce_med,
                    ])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            finally:
                for f in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(f):
                        os.remove(f)
                for d in downloaded_dirs:
                    if os.path.isdir(d):
                        shutil.rmtree(d)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
