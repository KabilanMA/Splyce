#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Download the SuiteSparse matrices listed
# in JOBS below, pair each with synthetic tensor(s) as needed, and run the
# compiled ./spmttkrp binary (see compile.sh) once per job.
#
# spmttkrp computes A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j). Each job
# in JOBS defines which SuiteSparse matrix (if any) supplies tensor_C and
# tensor_D — B is always synthetic (3D, no natural SuiteSparse source).
# When c/d is None, that operand is synthetic too. Whichever of C/D is
# real must be square (per matrix_metadata.json); its own dimension
# supplies whichever axes it constrains via conformance (B's k = D's k,
# B's l = C's l, C's j = D's j), while every axis not constrained by a
# real operand uses FREE_DIM instead — same scheme
# real_world_data/spmttkrp/run_suitesparse_benchmark.py uses.
#
# Unlike that script, this only drives the single already-vectorized
# binary compile.sh builds here (no scf baseline, no parallel mode) —
# spmttkrp_splyce_scf.mlir's @main runs SpMTTKRP exactly once and writes
# its elapsed time to ./benchmark, not a multi-iteration loop.
#
# Reuses the shared SuiteSparse downloader/converter under
# experiments/speedups/real_world_data/suitesparse/ (download_data.py,
# convert_to_tns.py) rather than duplicating that logic.
#
# Every synthetic operand (B always, plus whichever of C/D is None) is
# generated at density = max(the job's real matrix density,
# DENSITY_FLOOR), same floor-at-the-real-matrix's-own-density rule
# real_world_data/spmttkrp uses.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#       Downloads/converts/runs every job listed in JOBS below.
#   ./run_suitesparse_benchmark.py --force
#       Re-download even if suitesparse/<name>/ already exists.
#   ./run_suitesparse_benchmark.py --keep
#       Don't delete downloaded/generated files afterward.
#
# Appends one row per job (dataset, b_shape, c_source, c_shape, d_source,
# d_shape, b_nnz, c_nnz, d_nnz, target_density_pct, elapsed_s) to
# spmttkrp_clustering_results.csv — resumable, already-recorded jobs are
# skipped.

import csv
import math
import os
import random
import shutil
import subprocess
import sys

# Jobs this script runs — "c"/"d": None means that operand is synthetic; a
# name means it's downloaded from SuiteSparse (must be square). There's no
# "b" key — B is always synthetic. Edit this list to add/remove jobs; each
# real name must exist in matrix_metadata.json (run
# suitesparse/scrape_metadata.py first if it doesn't).
JOBS = [
    {"name": "heart1", "c": None, "d": "heart1"},
    {"name": "CAG_mat364", "c": "CAG_mat364", "d": "CAG_mat364"},
    {"name": "struct4", "c": None, "d": "struct4"},
    {"name": "cavity26", "c": None, "d": "cavity26"},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "speedups", "real_world_data", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BINARY = os.path.join(SCRIPT_DIR, "spmttkrp")
TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_clustering_results.csv")

# Floor for every synthetic operand's density (0.001%) — see module
# docstring.
DENSITY_FLOOR = 0.001 / 100

# Dimension for any axis not constrained by a real operand (B's i always;
# B's k/D's k when D isn't real; B's l/C's l when C isn't real).
FREE_DIM = 1000


def already_recorded():
    if not os.path.isfile(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV) as f:
        return {row["dataset"] for row in csv.DictReader(f)}


def download_matrix(name, entry, force):
    """Downloads + converts a named SuiteSparse matrix. Returns
    (tns_path, dataset_dir) or (None, dataset_dir_or_None) on failure."""
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


def generate_sparse_3d_tns(filename, dim1, dim2, dim3, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_3d_tns:
    # geometric skip sampling (O(1) memory, visits only ~nnz elements)
    # writes a FROSTT-format sparse tensor with `1 - sparsity` density.
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
    return actual_nnz


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Same geometric skip sampling as generate_sparse_3d_tns above.
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
        sys.exit("error: ./spmttkrp not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    done = already_recorded()
    write_header = not os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "b_shape", "c_source", "c_shape", "d_source", "d_shape",
                              "b_nnz", "c_nnz", "d_nnz", "target_density_pct", "elapsed_s"])

        for job in JOBS:
            name = job["name"]
            if name in done:
                print(f"[skip] {name}: already recorded")
                continue

            c_name, d_name = job["c"], job["d"]
            real_names = {n for n in (c_name, d_name) if n is not None}
            if not real_names:
                print(f"[skip] {name}: job specifies no real matrix for C or D "
                      f"(need at least one to derive a dimension)")
                continue

            real_entries = {}
            bad = False
            for rn in real_names:
                entry = metadata.get(rn)
                if entry is None:
                    print(f"[skip] {name}: '{rn}' not found in matrix_metadata.json")
                    bad = True
                    break
                if entry["num_rows"] != entry["num_cols"]:
                    print(f"[skip] {name}: '{rn}' is {entry['num_rows']}x{entry['num_cols']}, not square")
                    bad = True
                    break
                real_entries[rn] = entry
            if bad:
                continue

            # j is shared between C and D — if both are real, their
            # dimensions must agree. l (B/C) and k (B/D) are each pinned
            # only by their own real operand, if any; any axis with no
            # real operand behind it uses FREE_DIM instead.
            c_dim = real_entries[c_name]["num_rows"] if c_name in real_entries else None
            d_dim = real_entries[d_name]["num_rows"] if d_name in real_entries else None
            if c_dim is not None and d_dim is not None and c_dim != d_dim:
                print(f"[skip] {name}: C ('{c_name}', dim={c_dim}) and D ('{d_name}', dim={d_dim}) "
                      f"must share the same j dimension")
                continue

            i_dim = FREE_DIM
            k_dim = d_dim if d_dim is not None else FREE_DIM
            l_dim = c_dim if c_dim is not None else FREE_DIM
            j_dim = c_dim if c_dim is not None else d_dim

            densities = {rn: e["nnz"] / (e["num_rows"] * e["num_cols"]) for rn, e in real_entries.items()}
            target_density = max(max(densities.values()), DENSITY_FLOOR)
            target_sparsity = 1.0 - target_density

            print(f"=== {name} ===")
            downloaded_dirs = []
            tns_paths = {}
            download_failed = False
            for rn in real_names:
                tns_path, ddir = download_matrix(rn, real_entries[rn], force)
                if ddir is not None:
                    downloaded_dirs.append(ddir)
                if tns_path is None:
                    download_failed = True
                    break
                tns_paths[rn] = tns_path
            if download_failed:
                continue

            if c_name is not None:
                shutil.copyfile(tns_paths[c_name], TENSOR_C)
                c_nnz = real_entries[c_name]["nnz"]
                c_source = c_name
            else:
                c_nnz = generate_sparse_2d_tns(TENSOR_C, l_dim, j_dim, target_sparsity)
                c_source = f"synthetic_{target_density * 100:.4g}pct"

            if d_name is not None:
                shutil.copyfile(tns_paths[d_name], TENSOR_D)
                d_nnz = real_entries[d_name]["nnz"]
                d_source = d_name
            else:
                d_nnz = generate_sparse_2d_tns(TENSOR_D, k_dim, j_dim, target_sparsity)
                d_source = f"synthetic_{target_density * 100:.4g}pct"

            print(f"  [generate] tensor_B ({i_dim} x {k_dim} x {l_dim}) @ target_density={target_density:.6g} ...")
            b_nnz = generate_sparse_3d_tns(TENSOR_B, i_dim, k_dim, l_dim, target_sparsity)

            print("  [run] ./spmttkrp ...")
            elapsed = run_binary()
            if elapsed is None:
                print("  [warning] no benchmark file produced (likely crashed)")
            else:
                print(f"  elapsed={elapsed:.6f}s")
                writer.writerow([name, f"{i_dim}x{k_dim}x{l_dim}", c_source, f"{l_dim}x{j_dim}",
                                  d_source, f"{k_dim}x{j_dim}", b_nnz, c_nnz, d_nnz,
                                  target_density * 100, elapsed])
                f.flush()

            if not keep:
                for p in (TENSOR_B, TENSOR_C, TENSOR_D):
                    if os.path.isfile(p):
                        os.remove(p)
                for d in downloaded_dirs:
                    if os.path.isdir(d):
                        shutil.rmtree(d)

    print(f"Done. Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
