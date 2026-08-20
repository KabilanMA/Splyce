#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Download the SuiteSparse matrices listed
# in JOBS below, pair each with tensor_B/tensor_C as the job specifies
# (generating a synthetic tensor_C when the job has none), and run the
# compiled ./spgemm binary (see compile.sh) once per job.
#
# Follows the same job structure as
# experiments/speedups/real_world_data/spgemm/run_suitesparse_benchmark.py:
# each job in JOBS names which SuiteSparse matrix (if any) supplies
# tensor_B and tensor_C — "b" is always a real matrix, but "c" can be
# None, in which case tensor_C is generated synthetically instead (needed
# for a job like c8_mat11: rectangular, 4562x5761, so it can't be squared
# against itself). Synthetic tensor_C is sized SYNTHETIC_C_COLS wide (its
# row count is always B's own column count, so the shared j dimension
# lines up) at density = max(B's own density, DENSITY_FLOOR) — same
# floor-at-the-real-matrix's-own-density rule
# real_world_data/spmttkrp/run_suitesparse_benchmark.py and its siblings
# use, generalized from this script's previous hardcoded 9.37% (tuned only
# for c8_mat11) to any job's own B.
#
# Unlike that script, this only drives the single already-vectorized
# binary compile.sh builds here (no scf baseline, no dense/CSR format
# selection, no parallel mode) — spgemm_splyce_scf.mlir's @main runs
# SpGEMM exactly once and writes its elapsed time to ./benchmark, not a
# multi-iteration loop.
#
# Reuses the shared SuiteSparse downloader/converter under
# experiments/speedups/real_world_data/suitesparse/ (download_data.py,
# convert_to_tns.py) rather than duplicating that logic — same
# matrix_metadata.json lookup, same suitesparse/<name>/<name>.mtx ->
# suitesparse/<name>/<name>.tns pipeline.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#       Downloads/converts/runs every job listed in JOBS below.
#   ./run_suitesparse_benchmark.py --force
#       Re-download even if suitesparse/<name>/ already exists.
#   ./run_suitesparse_benchmark.py --keep
#       Don't delete the downloaded suitesparse/<name>/ director(y/ies) or
#       tensor_B.tns/tensor_C.tns afterward.
#
# Appends one row per job (dataset, b_source, c_source, b_shape, b_nnz,
# c_shape, c_nnz, elapsed_s) to spgemm_clustering_results.csv — resumable,
# already-recorded jobs are skipped.

import csv
import math
import os
import random
import shutil
import subprocess
import sys

# Jobs this script runs — "c": None means tensor_C is generated
# synthetically instead of downloaded (see module docstring for
# c8_mat11). Edit this list to add/remove jobs; each real name must exist
# in matrix_metadata.json (run suitesparse/scrape_metadata.py first if it
# doesn't).
JOBS = [
    {"name": "internet", "b": "internet", "c": "internet"},  
    {"name": "heart1", "b": "heart1", "c": "heart1"},
    {"name": "c8_mat11", "b": "c8_mat11", "c": None},
    {"name": "exdata_1", "b": "exdata_1", "c": "exdata_1"}
]

# Column count for a synthetic tensor_C (row count is always the job's B
# column count, so the shared j dimension lines up).
SYNTHETIC_C_COLS = 5000

# Floor for a synthetic tensor_C's density (0.001%) — see module
# docstring. Never applies when C is a real matrix.
DENSITY_FLOOR = 0.001 / 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "speedups", "real_world_data", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BINARY = os.path.join(SCRIPT_DIR, "spgemm")
TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "spgemm_clustering_results.csv")


def already_recorded():
    if not os.path.isfile(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV) as f:
        return {row["dataset"] for row in csv.DictReader(f)}


def download_matrix(name, metadata, force):
    """Downloads + converts a named SuiteSparse matrix. Returns
    (tns_path, dataset_dir) or (None, dataset_dir_or_None) on failure."""
    entry = metadata[name]
    dl.download_and_extract(name, entry["group"], force=force)
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


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_2d_tns:
    # geometric skip sampling (O(1) memory, visits only ~nnz elements)
    # writes a FROSTT-format sparse tensor with `1 - sparsity` density.
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
    return actual_nnz


def run_binary():
    if os.path.isfile(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
    # Not check=True: this binary's exit code isn't a reliable success
    # signal (observed nonzero even on a normal run that writes a correct
    # benchmark file) — presence of the benchmark file is what matters.
    # stdout/stderr inherited (not redirected) so the binary's own
    # "Scalar/SIMD elements processed" and "Execution time" prints show up.
    subprocess.run([BINARY], cwd=SCRIPT_DIR)
    if not os.path.isfile(BENCHMARK_FILE):
        return None
    with open(BENCHMARK_FILE) as f:
        return float(f.read().strip())


def main():
    args = sys.argv[1:]
    force = "--force" in args
    keep = "--keep" in args

    if not os.path.isfile(BINARY):
        sys.exit("error: ./spgemm not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    done = already_recorded()
    write_header = not os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "b_source", "c_source", "b_shape", "b_nnz",
                              "c_shape", "c_nnz", "elapsed_s"])

        for job in JOBS:
            name = job["name"]
            if name in done:
                print(f"[skip] {name}: already recorded")
                continue

            b_name = job["b"]
            b_entry = metadata.get(b_name)
            if b_entry is None:
                print(f"[skip] {name}: '{b_name}' not found in matrix_metadata.json")
                continue

            print(f"=== {name} ===")
            downloaded_dirs = []

            b_tns, b_dir = download_matrix(b_name, metadata, force)
            if b_dir is not None:
                downloaded_dirs.append(b_dir)
            if b_tns is None:
                continue
            shutil.copyfile(b_tns, TENSOR_B)
            b_rows, b_cols, b_nnz = b_entry["num_rows"], b_entry["num_cols"], b_entry["nnz"]

            c_name = job["c"]
            if c_name is None:
                c_rows, c_cols = b_cols, SYNTHETIC_C_COLS
                b_density = b_nnz / (b_rows * b_cols)
                target_density = max(b_density, DENSITY_FLOOR)
                target_sparsity = 1.0 - target_density
                print(f"  [generate] tensor_C ({c_rows} x {c_cols}) @ target_density={target_density:.6g} "
                      f"(B's own density={b_density:.6g}) ...")
                c_nnz = generate_sparse_2d_tns(TENSOR_C, c_rows, c_cols, target_sparsity)
                c_source = f"synthetic_{target_density * 100:.4g}pct"
            elif c_name == b_name:
                shutil.copyfile(b_tns, TENSOR_C)
                c_rows, c_cols, c_nnz = b_rows, b_cols, b_nnz
                c_source = c_name
            else:
                c_entry = metadata.get(c_name)
                if c_entry is None:
                    print(f"[skip] {name}: '{c_name}' not found in matrix_metadata.json")
                    continue
                c_tns, c_dir = download_matrix(c_name, metadata, force)
                if c_dir is not None:
                    downloaded_dirs.append(c_dir)
                if c_tns is None:
                    continue
                shutil.copyfile(c_tns, TENSOR_C)
                c_rows, c_cols, c_nnz = c_entry["num_rows"], c_entry["num_cols"], c_entry["nnz"]
                c_source = c_name

            print("  [run] ./spgemm ...")
            elapsed = run_binary()
            if elapsed is None:
                print("  [warning] no benchmark file produced (likely crashed)")
            else:
                print(f"  elapsed={elapsed:.6f}s")
                writer.writerow([name, b_name, c_source, f"{b_rows}x{b_cols}", b_nnz,
                                  f"{c_rows}x{c_cols}", c_nnz, elapsed])
                f.flush()

            if not keep:
                for p in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(p):
                        os.remove(p)
                for d in downloaded_dirs:
                    if os.path.isdir(d):
                        shutil.rmtree(d)

    print(f"Done. Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
