#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Download the SuiteSparse matrices listed
# in DATASETS below, pair each with a synthetic sparse vector, and run the
# compiled ./spmspv binary (see compile.sh) once per matrix.
#
# Unlike experiments/speedups/real_world_data/spmspv/run_suitesparse_benchmark.py,
# this only drives the single already-vectorized binary compile.sh builds
# here (no scf baseline, no parallel mode) — spmspv_splyce_scf.mlir's
# @main runs SpMSpV exactly once and writes its elapsed time to
# ./benchmark, not a multi-iteration loop.
#
# Reuses the shared SuiteSparse downloader/converter under
# experiments/speedups/real_world_data/suitesparse/ (download_data.py,
# convert_to_tns.py) rather than duplicating that logic — same
# matrix_metadata.json lookup, same suitesparse/<name>/<name>.mtx ->
# suitesparse/<name>/<name>.tns pipeline.
#
# The synthetic vector x is generated the same way
# real_world_data/spmspv/run_suitesparse_benchmark.py does: length = the
# matrix's own dimension (square matrices only — B's column count doubles
# as x's length), density = max(matrix's own density,
# VECTOR_DENSITY_FLOOR) so an extremely sparse/huge matrix doesn't end up
# with a near-empty (or zero-nnz) vector.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#       Downloads/converts/runs every matrix listed in DATASETS below.
#   ./run_suitesparse_benchmark.py --force
#       Re-download even if suitesparse/<name>/ already exists.
#   ./run_suitesparse_benchmark.py --keep
#       Don't delete downloaded/generated files afterward.
#
# Appends one row per matrix (dataset, dim, matrix_nnz, matrix_sparsity,
# vector_nnz, vector_sparsity, elapsed_s) to spmspv_clustering_results.csv
# — resumable, already-recorded matrices are skipped.

import csv
import os
import random
import shutil
import subprocess
import sys

# Matrices this script runs — square only (B's column count doubles as
# x's length, so a rectangular matrix would size x wrong). Edit this list
# to add/remove datasets; each name must exist in matrix_metadata.json
# (run suitesparse/scrape_metadata.py first if it doesn't).
DATASETS = [
    "stokes",
    "mycielskian18",
    "arabic-2005",
    "hugetrace-00020"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "speedups", "real_world_data", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BINARY = os.path.join(SCRIPT_DIR, "spmspv")
TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_X = os.path.join(SCRIPT_DIR, "tensor_x.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "spmspv_clustering_results.csv")

# Floor for the synthetic vector's density (0.001%) — same rule
# real_world_data/spmspv/run_suitesparse_benchmark.py uses, so an
# extremely sparse/huge matrix still gets a usable (non-empty) vector.
VECTOR_DENSITY_FLOOR = 0.00001


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


def generate_sparse_vector_tns(path, dim, nnz):
    nnz = max(1, min(nnz, dim))
    indices = sorted(random.sample(range(1, dim + 1), nnz))
    with open(path, "w") as f:
        f.write("# extended FROSTT format\n")
        f.write(f"1 {nnz}\n")
        f.write(f"{dim}\n")
        for idx in indices:
            val = random.uniform(0.5, 2.5)
            f.write(f"{idx} {val:.4f}\n")
    return nnz


def run_binary():
    if os.path.isfile(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
    # Not check=True: this binary's exit code isn't a reliable success
    # signal (see clustering/spgemm/run_suitesparse_benchmark.py, which
    # hit the same thing) — presence of the benchmark file is what
    # matters. stdout/stderr inherited (not redirected) so the binary's
    # own "Scalar/SIMD elements processed" and "Execution time" prints
    # show up.
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
        sys.exit("error: ./spmspv not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    done = already_recorded()
    write_header = not os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "dim", "matrix_nnz", "matrix_sparsity",
                              "vector_nnz", "vector_sparsity", "elapsed_s"])

        for name in DATASETS:
            if name in done:
                print(f"[skip] {name}: already recorded")
                continue

            entry = metadata.get(name)
            if entry is None:
                print(f"[skip] {name}: not found in matrix_metadata.json")
                continue
            if entry["num_rows"] != entry["num_cols"]:
                print(f"[skip] {name}: {entry['num_rows']}x{entry['num_cols']} is not square — "
                      "B's column count doubles as x's length here")
                continue

            print(f"=== {name} ===")
            tns_path, dataset_dir = download_matrix(name, metadata, force)
            if tns_path is None:
                continue

            dim = entry["num_rows"]  # square: num_rows == num_cols
            matrix_nnz = entry["nnz"]
            matrix_density = matrix_nnz / (dim * dim)
            matrix_sparsity = 1.0 - matrix_density
            target_vector_density = max(matrix_density, VECTOR_DENSITY_FLOOR)
            target_vector_nnz = round(target_vector_density * dim)

            shutil.copyfile(tns_path, TENSOR_B)
            vector_nnz = generate_sparse_vector_tns(TENSOR_X, dim, target_vector_nnz)
            vector_sparsity = 1.0 - (vector_nnz / dim)
            print(f"  [vector] dim={dim} nnz={vector_nnz} sparsity={vector_sparsity:.6g} "
                  f"(matrix_sparsity={matrix_sparsity:.6g})")

            print("  [run] ./spmspv ...")
            elapsed = run_binary()
            if elapsed is None:
                print("  [warning] no benchmark file produced (likely crashed)")
            else:
                print(f"  elapsed={elapsed:.6f}s")
                writer.writerow([name, dim, matrix_nnz, matrix_sparsity,
                                  vector_nnz, vector_sparsity, elapsed])
                f.flush()

            if not keep:
                for p in (TENSOR_B, TENSOR_X):
                    if os.path.isfile(p):
                        os.remove(p)
                if dataset_dir and os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)

    print(f"Done. Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
