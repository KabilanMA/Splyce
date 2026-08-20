#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Download the SuiteSparse matrix listed in
# JOBS below, generate a synthetic tensor_B to pair with it, and run the
# compiled ./spttspm binary (see compile.sh) once per job.
#
# spttspm computes A(i,j,r) = Σ_k B(i,j,k) · C(k,r). Each job in JOBS
# defines which SuiteSparse matrix supplies tensor_C — B is always
# synthetic (3D, no natural SuiteSparse source). There's no "b" key —
# only C can ever be real, and a job needs it to derive k/r and a
# reference density (unlike spmttkrp/spmmh, which each have two possible
# real-operand slots and can fall back on one when the other is missing).
#
# i and j are never constrained by anything (nothing besides B references
# them), so they always use FREE_DIM. k and r are both C's own dimension
# (C must be square, per matrix_metadata.json) — same scheme
# real_world_data/spttspm/run_suitesparse_benchmark.py uses, but with a
# much smaller FREE_DIM (100 vs 500): A is fully dense at i x j x r, and r
# is tied to C's own (possibly large) dimension, so a big FREE_DIM here
# can make a "quick smoke test" anything but quick (e.g. barth4's own
# 6019 dimension already puts A at 100 x 100 x 6019 ~= 480 MiB; at
# real_world_data's FREE_DIM=500 the same job needs ~12 GiB).
#
# Some matrix names exist in more than one SuiteSparse group (e.g. barth4
# is both Nasa and Pothen, different matrices despite the shared name) —
# suitesparse/download_data.py's load_metadata() dedupes by name alone and
# would silently resolve to whichever entry appears last in
# matrix_metadata.json, so a JOBS entry can set an explicit "group" key to
# disambiguate (load_matrix_entry looks the raw list up by (name, group)
# directly instead of going through that by-name dict).
#
# Unlike real_world_data/spttspm's script, this only drives the single
# already-vectorized binary compile.sh builds here (no scf baseline) —
# spttspm_splyce_scf.mlir's @main runs SpTTSpM exactly once and writes its
# elapsed time to ./benchmark, not a multi-iteration loop.
#
# Reuses the shared SuiteSparse downloader/converter under
# experiments/speedups/real_world_data/suitesparse/ (download_data.py,
# convert_to_tns.py) rather than duplicating that logic.
#
# Synthetic tensor_B is generated at density = max(C's own density,
# DENSITY_FLOOR) — same floor-at-the-real-matrix's-own-density rule
# real_world_data/spttspm uses.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#       Downloads/converts/runs every job listed in JOBS below.
#   ./run_suitesparse_benchmark.py --force
#       Re-download even if suitesparse/<name>/ already exists.
#   ./run_suitesparse_benchmark.py --keep
#       Don't delete downloaded/generated files afterward.
#   ./run_suitesparse_benchmark.py --memory-limit-gib 64
#       Override the auto-detected memory budget (default: 50% of this
#       machine's total RAM, see MEMORY_SAFETY_FRACTION) with an explicit
#       cap — A (dense, i x j x r) is the term that actually needs
#       watching, since r scales with C's own dimension.
#
# Appends one row per job (dataset, group, b_shape, c_shape, b_nnz, c_nnz,
# target_density_pct, elapsed_s) to spttspm_clustering_results.csv —
# resumable, already-recorded jobs are skipped.

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys

# Jobs this script runs — "c" is the real SuiteSparse matrix (must be
# square); "group" disambiguates a name that exists in more than one
# group (optional otherwise). Keep FREE_DIM/C's own dimension in mind
# before adding a much bigger matrix here (see module docstring on dense
# A's cost). Edit this list to add/remove jobs — each name must exist in
# matrix_metadata.json (run suitesparse/scrape_metadata.py first if it
# doesn't).
JOBS = [
    {"name": "barth4", "c": "barth4", "group": "Nasa"},
    {"name": "rdist1", "c": "rdist1", "group": "Zitney"},
    {"name": "psmigr_1", "c": "psmigr_1", "group": "HB"},
    {"name": "EX6", "c": "EX6", "group": "JGD_SPG"},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "speedups", "real_world_data", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BINARY = os.path.join(SCRIPT_DIR, "spttspm")
TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "spttspm_clustering_results.csv")

# Floor for synthetic tensor_B's density (0.001%).
DENSITY_FLOOR = 0.001 / 100

# Dimension for i/j — never constrained by anything (see module
# docstring; deliberately smaller than real_world_data's own FREE_DIM=500
# to keep a "quick" clustering run actually quick).
FREE_DIM = 500

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime (briefly holds a full COO intermediate alongside
# the final level-format storage before freeing the COO) — roughly 2x the
# raw coordinate size:
#   tensor_B, 3D (3 coords + 1 value, 8 bytes each) * 2 = 64 bytes/nnz
#   tensor_C, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
TENSOR_B_BYTES_PER_NNZ = 64
TENSOR_2D_BYTES_PER_NNZ = 48

# Fraction of total system RAM usable as budget — see module docstring.
MEMORY_SAFETY_FRACTION = 0.5


def detect_total_memory_bytes():
    # Linux-specific (/proc/meminfo) — falls back to a conservative 32 GiB
    # if unreadable, so a detection failure fails toward skipping too much
    # rather than too little.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # value is in KiB
    except (OSError, ValueError):
        pass
    return 32 * 1024 ** 3


def already_recorded():
    if not os.path.isfile(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV) as f:
        return {row["dataset"] for row in csv.DictReader(f)}


def load_matrix_entry(name, group):
    # suitesparse/download_data.py's load_metadata() dedupes by name alone,
    # which would silently pick the wrong entry for an ambiguous name (see
    # module docstring) — read the raw metadata list directly and match on
    # (name, group).
    with open(dl.METADATA_PATH) as f:
        data = json.load(f)
    for m in data["matrices"]:
        if m["name"] == name and m["group"] == group:
            return m
    return None


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
    memory_limit_gib = None
    if "--memory-limit-gib" in args:
        memory_limit_gib = float(args[args.index("--memory-limit-gib") + 1])

    memory_budget_bytes = (
        int(memory_limit_gib * 1024 ** 3) if memory_limit_gib is not None
        else int(detect_total_memory_bytes() * MEMORY_SAFETY_FRACTION)
    )
    print(f"Memory budget per job: {memory_budget_bytes / 1024**3:.1f} GiB"
          + (" (explicit)" if memory_limit_gib is not None else
             f" ({MEMORY_SAFETY_FRACTION:.0%} of detected total RAM)"))

    if not os.path.isfile(BINARY):
        sys.exit("error: ./spttspm not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    done = already_recorded()
    write_header = not os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "group", "b_shape", "c_shape",
                              "b_nnz", "c_nnz", "target_density_pct", "elapsed_s"])

        for job in JOBS:
            name = job["name"]
            if name in done:
                print(f"[skip] {name}: already recorded")
                continue

            c_name = job["c"]
            c_group = job.get("group")

            if c_group is not None:
                entry = load_matrix_entry(c_name, c_group)
                if entry is None:
                    print(f"[skip] {name}: {c_group}/{c_name} not found in matrix_metadata.json")
                    continue
            else:
                entry = metadata.get(c_name)
                if entry is None:
                    print(f"[skip] {name}: '{c_name}' not found in matrix_metadata.json")
                    continue
            group = entry["group"]

            if entry["num_rows"] != entry["num_cols"]:
                print(f"[skip] {name}: '{c_name}' is {entry['num_rows']}x{entry['num_cols']}, not square")
                continue

            # k and r are both C's own dimension (C is square); i and j
            # are always free.
            i_dim = FREE_DIM
            j_dim = FREE_DIM
            k_dim = entry["num_rows"]
            r_dim = entry["num_rows"]

            c_nnz = entry["nnz"]
            c_density = c_nnz / (k_dim * r_dim)
            target_density = max(c_density, DENSITY_FLOOR)
            target_sparsity = 1.0 - target_density

            # B and C, AND the dense output A, are all simultaneously
            # resident — estimate the combined peak before downloading
            # anything. A is i x j x r and fully dense, the term that
            # actually needs watching (r scales with C's own dimension).
            expected_b_nnz = target_density * (i_dim * j_dim * k_dim)
            tensor_b_bytes = expected_b_nnz * TENSOR_B_BYTES_PER_NNZ
            tensor_c_bytes = c_nnz * TENSOR_2D_BYTES_PER_NNZ
            dense_a_bytes = i_dim * j_dim * r_dim * 8
            estimated_peak_bytes = tensor_b_bytes + tensor_c_bytes + dense_a_bytes

            if estimated_peak_bytes > memory_budget_bytes:
                print(f"[skip] {name}: estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                      f"(B={tensor_b_bytes / 1024**3:.1f} C={tensor_c_bytes / 1024**3:.1f} "
                      f"A={dense_a_bytes / 1024**3:.1f} GiB) "
                      f"> budget {memory_budget_bytes / 1024**3:.1f} GiB")
                continue

            print(f"=== {group}/{c_name} (nnz={c_nnz}) ===")
            tns_path, dataset_dir = download_matrix(c_name, entry, force)
            if tns_path is None:
                continue

            shutil.copyfile(tns_path, TENSOR_C)

            print(f"  [generate] tensor_B ({i_dim} x {j_dim} x {k_dim}) @ target_density={target_density:.6g} ...")
            b_nnz = generate_sparse_3d_tns(TENSOR_B, i_dim, j_dim, k_dim, target_sparsity)

            print("  [run] ./spttspm ...")
            elapsed = run_binary()
            if elapsed is None:
                print("  [warning] no benchmark file produced (likely crashed)")
            else:
                print(f"  elapsed={elapsed:.6f}s")
                writer.writerow([name, group, f"{i_dim}x{j_dim}x{k_dim}", f"{k_dim}x{r_dim}",
                                  b_nnz, c_nnz, target_density * 100, elapsed])
                f.flush()

            if not keep:
                for p in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(p):
                        os.remove(p)
                if os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)

    print(f"Done. Results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
