#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Download the SuiteSparse matrices listed
# in JOBS below, pair each with synthetic tensor(s) as needed, and run the
# compiled ./spmmh binary (see compile.sh) once per job.
#
# spmmh computes A(i,j) = B(i,k) · C(k,j) · D(i,j). Each job in JOBS
# defines which SuiteSparse matrix (if any) supplies tensor_B and
# tensor_D — C is always synthetic, and always *fully dense* (spmmh.mlir
# declares C with an all-dense encoding; there's no sparse/CSC variant of
# C the way B/D have one). When b/d is None, that operand is synthetic
# (sparse) too.
#
# Whichever of B/D is real must be square (per matrix_metadata.json), and
# its own dimension supplies whichever axes it constrains via conformance
# — B(i,k) and D(i,j) share i (the co-iterated compressed index), while k
# (B's other axis, C's row axis) and j (D's other axis, C's column axis)
# are each pinned only by their own real operand, if any; any axis with no
# real operand behind it uses FREE_DIM instead — same scheme
# real_world_data/spmmh/run_suitesparse_benchmark.py uses. C is always
# k x j and fully dense, so keep JOBS small: dense C costs k*j*8 bytes
# regardless of how sparse B/D are (e.g. a 57735-dim matrix used for both
# B and D, like real_world_data's bayer01, would need a ~27 GiB dense C —
# fine on that benchmark's server, not something you want in a quick
# clustering smoke test).
#
# Unlike that script, this only drives the single already-vectorized
# binary compile.sh builds here (no scf baseline) — spmmh_splyce_scf.mlir's
# @main runs SpMMH exactly once and writes its elapsed time to
# ./benchmark, not a multi-iteration loop.
#
# Reuses the shared SuiteSparse downloader/converter under
# experiments/speedups/real_world_data/suitesparse/ (download_data.py,
# convert_to_tns.py) rather than duplicating that logic.
#
# Whichever of B/D is synthetic is generated at density = max(the job's
# real operand's density, DENSITY_FLOOR) — same floor-at-the-real-
# matrix's-own-density rule real_world_data/spmmh uses.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#       Downloads/converts/runs every job listed in JOBS below.
#   ./run_suitesparse_benchmark.py --force
#       Re-download even if suitesparse/<name>/ already exists.
#   ./run_suitesparse_benchmark.py --keep
#       Don't delete downloaded/generated files afterward.
#   ./run_suitesparse_benchmark.py --memory-limit-gb 180
#       Override the auto-detected dense-C skip threshold (default: 50% of
#       this machine's total RAM, see MEMORY_SAFETY_FRACTION) with an
#       explicit cap — e.g. to run the bayer01 job (needs ~25 GiB for C)
#       on a bigger machine than the one this happens to be invoked on.
#
# Appends one row per job (dataset, b_source, b_shape, d_source, d_shape,
# c_shape, b_nnz, d_nnz, target_density_pct, elapsed_s) to
# spmmh_clustering_results.csv — resumable, already-recorded jobs are
# skipped.

import csv
import math
import os
import random
import shutil
import subprocess
import sys

# Jobs this script runs — "b"/"d": None means that operand is synthetic; a
# name means it's downloaded from SuiteSparse (must be square). There's no
# "c" key — C is always synthetic and fully dense; the DENSE_C_BYTE_LIMIT
# skip-guard below (overridable via --memory-limit-gb) protects against a
# job whose dense C doesn't fit the machine this runs on. Edit this list
# to add/remove jobs — each real name must exist in matrix_metadata.json
# (run suitesparse/scrape_metadata.py first if it doesn't).
JOBS = [
    {"name": "bayer01", "b": "bayer01", "d": "bayer01"},
    {"name": "msc23052", "b": "msc23052", "d": "msc23052"},
    {"name": "smt", "b": "smt", "d": "smt"},
    {"name": "mark3jac020", "b": "mark3jac020", "d": "mark3jac020"},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "speedups", "real_world_data", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BINARY = os.path.join(SCRIPT_DIR, "spmmh")
TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "spmmh_clustering_results.csv")

# Floor for whichever of B/D is synthetic's density (0.001%) — never
# applies to C, which is always fully dense regardless.
DENSITY_FLOOR = 0.001 / 100

# Dimension for any axis not constrained by a real operand.
FREE_DIM = 5000

# C is always fully dense (k_dim * j_dim entries) — skip a job outright
# rather than risk generating a dense tensor bigger than this fraction of
# the machine's total RAM (auto-detected; override with --memory-limit-gb
# for an explicit cap instead) — same MEMORY_SAFETY_FRACTION convention
# real_world_data/*/run_suitesparse_benchmark.py and run_suitesparse_sweep.py
# use.
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


def generate_dense_2d_tns(filename, rows, cols):
    # Ported from experiments/gen_data.py's generate_dense_2d_tns: writes
    # every (row, col) coordinate explicitly.
    nnz = rows * cols
    with open(filename, "w") as f:
        f.write("# extended FROSTT format\n")
        f.write(f"2 {nnz}\n")
        f.write(f"{rows} {cols}\n")
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                f.write(f"{r} {c} {random.uniform(0.5, 2.5):.4f}\n")
    return nnz


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_2d_tns
    # (geometric skip sampling, O(1) memory, visits only ~nnz elements).
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
    memory_limit_gb = None
    if "--memory-limit-gb" in args:
        memory_limit_gb = float(args[args.index("--memory-limit-gb") + 1])

    dense_c_byte_limit = (
        int(memory_limit_gb * 1024 ** 3) if memory_limit_gb is not None
        else int(detect_total_memory_bytes() * MEMORY_SAFETY_FRACTION)
    )
    print(f"Dense-C skip threshold: {dense_c_byte_limit / 1024**3:.1f} GiB"
          + (" (explicit)" if memory_limit_gb is not None else
             f" ({MEMORY_SAFETY_FRACTION:.0%} of detected total RAM)"))

    if not os.path.isfile(BINARY):
        sys.exit("error: ./spmmh not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    done = already_recorded()
    write_header = not os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "b_source", "b_shape", "d_source", "d_shape",
                              "c_shape", "b_nnz", "d_nnz", "target_density_pct", "elapsed_s"])

        for job in JOBS:
            name = job["name"]
            if name in done:
                print(f"[skip] {name}: already recorded")
                continue

            b_name, d_name = job["b"], job["d"]
            real_names = {n for n in (b_name, d_name) if n is not None}
            if not real_names:
                print(f"[skip] {name}: job specifies no real matrix for B or D "
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

            # i is shared between B and D — if both are real, their
            # dimensions must agree. k (B only) and j (D only) are each
            # pinned only by their own real operand, if any; any axis with
            # no real operand behind it uses FREE_DIM.
            b_dim = real_entries[b_name]["num_rows"] if b_name in real_entries else None
            d_dim = real_entries[d_name]["num_rows"] if d_name in real_entries else None
            if b_dim is not None and d_dim is not None and b_dim != d_dim:
                print(f"[skip] {name}: B ('{b_name}', dim={b_dim}) and D ('{d_name}', dim={d_dim}) "
                      f"must share the same i dimension")
                continue

            k_dim = b_dim if b_dim is not None else FREE_DIM
            j_dim = d_dim if d_dim is not None else FREE_DIM

            dense_c_bytes = k_dim * j_dim * 8
            if dense_c_bytes > dense_c_byte_limit:
                print(f"[skip] {name}: dense C would be {dense_c_bytes / 1024**3:.1f} GiB "
                      f"(k={k_dim} x j={j_dim}) > {dense_c_byte_limit / 1024**3:.1f} GiB limit "
                      f"(--memory-limit-gb to raise it)")
                continue

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

            i_dim = b_dim if b_dim is not None else d_dim

            if b_name is not None:
                shutil.copyfile(tns_paths[b_name], TENSOR_B)
                b_nnz = real_entries[b_name]["nnz"]
                b_source = b_name
            else:
                b_nnz = generate_sparse_2d_tns(TENSOR_B, i_dim, k_dim, target_sparsity)
                b_source = f"synthetic_{target_density * 100:.4g}pct"

            if d_name is not None:
                shutil.copyfile(tns_paths[d_name], TENSOR_D)
                d_nnz = real_entries[d_name]["nnz"]
                d_source = d_name
            else:
                d_nnz = generate_sparse_2d_tns(TENSOR_D, i_dim, j_dim, target_sparsity)
                d_source = f"synthetic_{target_density * 100:.4g}pct"

            print(f"  [generate] dense tensor_C ({k_dim} x {j_dim}) ...")
            generate_dense_2d_tns(TENSOR_C, k_dim, j_dim)

            print("  [run] ./spmmh ...")
            elapsed = run_binary()
            if elapsed is None:
                print("  [warning] no benchmark file produced (likely crashed)")
            else:
                print(f"  elapsed={elapsed:.6f}s")
                writer.writerow([name, b_source, f"{i_dim}x{k_dim}", d_source, f"{i_dim}x{j_dim}",
                                  f"{k_dim}x{j_dim}", b_nnz, d_nnz, target_density * 100, elapsed])
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
