#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpMTTKRP benchmark, single
# curated matrix.
#
# spmttkrp computes A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j) (see
# spmttkrp_dn.mlir). Only one real SuiteSparse matrix is used here:
#
#   heart1 (Norris group): 3557 x 3557 square, nnz=1385317 (10.95% dense)
#   — used for D(k,j).
#
# B and C are synthetically generated (there's no natural SuiteSparse
# source for a 3D tensor or for a matrix that must share D's k/j index
# ranges). Dimension choice:
#   - k and j must equal D's own dimension (3557, square) so B(i,k,l) and
#     C(l,j) actually conform against D(k,j) — i.e. those two index sizes
#     come from D.
#   - i and l aren't constrained by D at all, so both use a fixed 1000.
#   => B(i,k,l) = 1000 x 3557 x 1000, C(l,j) = 1000 x 3557.
# Both are generated at a fixed 10.97% nonzero density (sparsity=0.8903)
# via generate_sparse_3d_tns / generate_sparse_2d_tns below (ported from
# experiments/gen_data.py's functions of the same name).
#
# This script:
#   1. Downloads heart1 (suitesparse/download_data.py's
#      download_and_extract, looked up via matrix_metadata.json — run
#      suitesparse/scrape_metadata.py first if that file doesn't exist yet)
#      and converts it to .tns (suitesparse/convert_to_tns.py's
#      process_dir — mirrors symmetric entries, drops explicit zeros),
#      copying the result into this directory as tensor_D.tns.
#   2. Generates tensor_B.tns (1000 x 3557 x 1000, 10.97% dense) and
#      tensor_C.tns (1000 x 3557, 10.97% dense).
#   3. Runs test_benchmark_spmttkrp_splyce_phase_001 FIRST, then
#      test_benchmark_spmttkrp_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 6 iterations internally per spmttkrp_dn.mlir's
#      @main, writing a "benchmark" file with one time per line.
#   4. Appends one summary row (dataset, b/c/d shape, sparsity, scf_median,
#      splyce_median — median of the 5 non-cold-start iterations;
#      scf_median is "SKIPPED" when Splyce timed out) to
#      spmttkrp_realworld_results.csv, and every individual raw iteration
#      time to spmttkrp_realworld_raw_runtimes.csv as a backup.
#   5. Deletes tensor_B.tns/tensor_C.tns/tensor_D.tns and the downloaded/
#      converted suitesparse/heart1/ directory afterward.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once
# heart1 is already recorded (unless the CSV row is removed first).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spmttkrp_scf and test_benchmark_spmttkrp_splyce_phase_001
# exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --mode multicore [--cores N]
#       Runs the OpenMP dense-outer-loop parallel binaries (compile.sh's
#       *_parallel variants) with OMP_NUM_THREADS=N instead of the
#       single-threaded ones. --cores defaults to every CPU on the machine
#       (os.cpu_count()). To pin these to a specific NUMA node/CPU set,
#       wrap the whole command in `numactl ...` — nothing here sets its own
#       CPU affinity, so an outer numactl applies to every binary run.

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

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001")
BASELINE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf_parallel")
SPLYCE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001_parallel")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# The single curated job this script runs — see module docstring.
MATRIX_NAME = "heart1"

# Free index sizes (not constrained by D) for the synthetic B/C tensors.
SYNTHETIC_I = 1000
SYNTHETIC_L = 1000

# Fixed nonzero density for the synthetic B/C tensors.
SYNTHETIC_DENSITY_PCT = 10.97


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


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_2d_tns (same
    # geometric skip sampling as generate_sparse_3d_tns above).
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


def run_binary(bin_path, timeout, cores=None):
    if os.path.isfile(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
    env = os.environ.copy()
    if cores is not None:
        env["OMP_NUM_THREADS"] = str(cores)
    try:
        subprocess.run(
            [bin_path],
            cwd=SCRIPT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            env=env,
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
    # Keyed on (dataset, mode) — not dataset alone — so a job already
    # recorded in one mode doesn't shadow a run of it in the other mode.
    if not os.path.isfile(SUMMARY_CSV):
        return set()
    with open(SUMMARY_CSV) as f:
        return {(row["dataset"], row.get("mode", "single")) for row in csv.DictReader(f)}


def main():
    args = sys.argv[1:]
    timeout = 300
    mode = "single"
    cores = None
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--cores" in args:
        cores = int(args[args.index("--cores") + 1])

    if mode not in ("single", "multicore"):
        sys.exit(f"error: unsupported --mode '{mode}' (supported: single, multicore)")
    if cores is not None and mode != "multicore":
        sys.exit("error: --cores is only meaningful with --mode multicore")
    if mode == "multicore" and cores is None:
        cores = os.cpu_count()

    baseline_bin = BASELINE_BIN_PARALLEL if mode == "multicore" else BASELINE_BIN
    splyce_bin = SPLYCE_BIN_PARALLEL if mode == "multicore" else SPLYCE_BIN

    if not (os.path.isfile(baseline_bin) and os.path.isfile(splyce_bin)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    entry = metadata.get(MATRIX_NAME)
    if entry is None:
        sys.exit(f"error: {MATRIX_NAME} not found in matrix_metadata.json")
    name, group = entry["name"], entry["group"]

    done = already_recorded()
    if (name, mode) in done:
        print(f"{name} ({mode}) already recorded — nothing to do")
        return

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "group", "b_shape", "c_shape", "d_shape",
                "synthetic_density_pct", "mode", "cores",
                "scf_median_s", "splyce_median_s",
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

            shutil.copyfile(tns_path, TENSOR_D)
            dim_kj = entry["num_rows"]  # square: num_rows == num_cols; shared k/j index size

            synthetic_sparsity = 1.0 - (SYNTHETIC_DENSITY_PCT / 100.0)

            print(f"  [generate] tensor_B ({SYNTHETIC_I} x {dim_kj} x {SYNTHETIC_L}) ...")
            generate_sparse_3d_tns(TENSOR_B, SYNTHETIC_I, dim_kj, SYNTHETIC_L, synthetic_sparsity)

            print(f"  [generate] tensor_C ({SYNTHETIC_L} x {dim_kj}) ...")
            generate_sparse_2d_tns(TENSOR_C, SYNTHETIC_L, dim_kj, synthetic_sparsity)

            run_cores = cores if mode == "multicore" else None

            print("  [run] splyce phase_001 ...")
            splyce_times, splyce_timed_out = run_binary(splyce_bin, timeout, run_cores)

            if splyce_timed_out:
                print("  [skip] splyce timed out — skipping baseline run")
                scf_times, scf_med = None, "SKIPPED"
            else:
                print("  [run] baseline (scf) ...")
                scf_times, _ = run_binary(baseline_bin, timeout * 5, run_cores)
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
                    f"{SYNTHETIC_I}x{dim_kj}x{SYNTHETIC_L}", f"{SYNTHETIC_L}x{dim_kj}", f"{dim_kj}x{dim_kj}",
                    SYNTHETIC_DENSITY_PCT, mode, cores or "",
                    scf_med, splyce_med,
                ])
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
