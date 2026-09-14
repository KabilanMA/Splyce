#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpMMH benchmark, curated set.
#
# spmmh computes A(i,j) = B(i,k) · C(k,j) · D(i,j) = D(i,j) · Σ_k B(i,k)·C(k,j)
# (see spmmh_dn.mlir). Each job in JOBS below defines which SuiteSparse
# matrix (if any) supplies tensor_B and tensor_D — C is always synthetic,
# and always *fully dense* (not just low-sparsity — spmmh_dn.mlir declares
# C with an all-dense encoding; there's no sparse/CSC variant of C the way
# B/D have one). When b/d is None, that operand is synthetic (sparse) too.
#
# Whichever of B/D is real must be square (per matrix_metadata.json), and
# its own dimension is used for whichever axes it actually constrains via
# conformance — B(i,k) and D(i,j) share i (the co-iterated compressed
# index), while k (B's other axis, and C's row axis) and j (D's other
# axis, and C's column axis) are each pinned only by their own real
# operand, if any. Any axis with no real operand behind it uses a fixed
# FREE_DIM (5000) instead:
#   - only B real:  i = k = dim(B),        j = FREE_DIM
#   - only D real:  i = j = dim(D),        k = FREE_DIM
#   - both real:    i must match between B and D (errors otherwise);
#                   k = dim(B), j = dim(D)
# C is always (k, j) and fully dense (every entry populated, no density
# concept applies to it). Whichever of B/D is synthetic is generated at
# density = max(the job's real operand's density, DENSITY_FLOOR (0.001%))
# — same floor-at-the-real-matrix's-own-density rule
# ../spmttkrp/run_suitesparse_sweep.py and
# ../spmttkrp/run_suitesparse_benchmark.py use. If both B and D are real,
# the denser of the two is used as the reference.
#
# For each job, this:
#   1. Resolves i/k/j + the target synthetic density from metadata alone
#      (matrix_metadata.json — run suitesparse/scrape_metadata.py first if
#      that file doesn't exist yet), and estimates peak memory — skipping
#      the job before downloading anything if that exceeds
#      --memory-limit-gb (default 250, matching this benchmark's server).
#      A (the output) and C (synthetic, fully dense) are the only O(n^2)
#      terms (dense i x j and k x j f64 buffers respectively); B and D
#      stay sparse (O(nnz), negligible next to those) whether real or
#      synthetic, since neither ever grows cubically the way spmttkrp's
#      3D tensor_B can — so the estimate is just their combined bytes.
#      There is no sparse-output binary for spmmh (unlike spgemm): an
#      attempt to make A a sparse CSC output was found to compile and
#      compute correctly, but the workspace-insertion pattern it requires
#      isn't recognized by Splyce's current vectorizer, so skipping
#      outright is the only available mitigation here rather than
#      swapping formats.
#   2. Downloads + converts whichever of B/D are real
#      (suitesparse/download_data.py + convert_to_tns.py — mirrors
#      symmetric entries, drops explicit zeros), deduplicating a download
#      when the same matrix name fills both roles.
#   3. Generates whichever of B/D are synthetic (generate_sparse_2d_tns
#      below), and always generates tensor_C as a fully dense k x j tensor
#      (generate_dense_2d_tns below) — both ported from
#      experiments/gen_data.py's functions of the same name.
#   4. Runs test_benchmark_spmmh_splyce_phase_001 FIRST, then
#      test_benchmark_spmmh_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 6 iterations internally per spmmh_dn.mlir's
#      @main, writing a "benchmark" file with one time per line.
#   5. Appends one summary row (dataset, b/d source, b/c/d shape, target
#      density, scf_median, splyce_median — median of the 5 non-cold-start
#      iterations; scf_median is "SKIPPED" when Splyce timed out, and both
#      are "SKIPPED_MEMORY" when the job was skipped per step 1 above) to
#      spmmh_realworld_results.csv, and every individual raw iteration
#      time to spmmh_realworld_raw_runtimes.csv as a backup.
#   6. Deletes tensor_B.tns/tensor_C.tns/tensor_D.tns and any downloaded/
#      converted suitesparse/<name>/ director(y/ies) before moving to the
#      next job.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once a
# job is already recorded (unless the CSV row is removed first).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spmmh_scf and test_benchmark_spmmh_splyce_phase_001 exist
# in this directory. There are no *_parallel variants for spmmh (compile.sh
# doesn't build any), so this has no --mode/--cores, unlike its siblings.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # every job in JOBS
#   ./run_suitesparse_benchmark.py --matrix cat_ears_2_1
#       Runs that one SuiteSparse matrix for BOTH tensor_B and tensor_D
#       instead of JOBS — must be square. Ignores JOBS/--limit entirely.
#   ./run_suitesparse_benchmark.py --limit 2    # only the first 2 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --memory-limit-gb 500  # override the
#                                                  # 250 GiB skip threshold

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

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmmh_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmmh_splyce_phase_001")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmmh_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmmh_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# spmmh has no sparse-output binary (see module docstring), so an oversized
# job can only be skipped outright, not routed to a smaller-footprint
# variant. Default matches this benchmark's server RAM.
DEFAULT_MEMORY_LIMIT_GB = 250

# Floor for whichever of B/D is synthetic's density (0.001%) — see module
# docstring. Never applies to C, which is always fully dense regardless.
DENSITY_FLOOR = 0.001 / 100

# Dimension for any axis not constrained by a real operand — see module
# docstring. Bigger than spmttkrp's equivalent (1000): spmmh has one fewer
# tensor dimension overall (2D operands, not 3D), so a larger free
# dimension here is comparable in cost to spmttkrp's smaller one.
FREE_DIM = 5000

# The curated jobs this script runs — see module docstring for why bayer01
# was chosen. "b"/"d": None means that operand is synthetic; a name means
# it's downloaded from SuiteSparse. There's no "c" key — C is always
# synthetic and fully dense.
JOBS = [
    {"name": "bayer01", "b": "bayer01", "d": "bayer01"},
]


def generate_dense_2d_tns(filename, rows, cols):
    # Ported from experiments/gen_data.py's generate_dense_2d_tns: writes
    # every (row, col) coordinate explicitly, i.e. a fully enumerated dense
    # tensor in FROSTT format.
    nnz = rows * cols

    with open(filename, "w") as f:
        f.write("# extended FROSTT format\n")
        f.write(f"2 {nnz}\n")
        f.write(f"{rows} {cols}\n")
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                val = random.uniform(0.5, 2.5)
                f.write(f"{r} {c} {val:.4f}\n")

    print(f"Generated dense tensor: {filename} | Shape: ({rows}, {cols}) | NNZ: {nnz}")
    return nnz


def generate_sparse_2d_tns(filename, rows, cols, sparsity):
    # Ported from experiments/gen_data.py's generate_sparse_2d_tns:
    # geometric skip sampling (O(1) memory, visits only ~nnz elements)
    # writes a FROSTT-format sparse tensor with `1 - sparsity` nonzero
    # density.
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


def download_matrix(name, entry):
    """Downloads + converts a named SuiteSparse matrix. Returns
    (tns_path, dataset_dir) or (None, dataset_dir_or_None) on failure."""
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
    memory_limit_gb = DEFAULT_MEMORY_LIMIT_GB
    matrix_name = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--memory-limit-gb" in args:
        memory_limit_gb = float(args[args.index("--memory-limit-gb") + 1])
    if "--matrix" in args:
        matrix_name = args[args.index("--matrix") + 1]
    memory_limit_bytes = memory_limit_gb * 1024 ** 3

    if not (os.path.isfile(BASELINE_BIN) and os.path.isfile(SPLYCE_BIN)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    metadata = dl.load_metadata()

    if matrix_name is not None:
        entry = metadata.get(matrix_name)
        if entry is None:
            sys.exit(f"error: '{matrix_name}' not found in matrix_metadata.json")
        if entry["num_rows"] != entry["num_cols"]:
            sys.exit(f"error: '{matrix_name}' is {entry['num_rows']}x{entry['num_cols']}, not square — "
                      "this script only pairs a square matrix against itself")
        jobs = [{"name": matrix_name, "b": matrix_name, "d": matrix_name}]
    else:
        jobs = JOBS[:limit] if limit is not None else JOBS

    done = already_recorded()
    print(f"{len(jobs)} job(s), {len(done)} already recorded — resuming")

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "b_source", "b_shape", "d_source", "d_shape", "c_shape",
                "b_nnz", "d_nnz", "target_density_pct",
                "scf_median_s", "splyce_median_s",
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
                d_name = job["d"]
                real_names = {n for n in (b_name, d_name) if n is not None}

                if not real_names:
                    print(f"  [skip] {name}: job specifies no real matrix for B or D "
                          f"(need at least one to derive a dimension)")
                    continue

                real_entries = {}
                bad = False
                for rn in real_names:
                    entry = metadata.get(rn)
                    if entry is None:
                        print(f"  [skip] {name}: '{rn}' not found in matrix_metadata.json")
                        bad = True
                        break
                    if entry["num_rows"] != entry["num_cols"]:
                        print(f"  [skip] {name}: '{rn}' is {entry['num_rows']}x{entry['num_cols']}, not square")
                        bad = True
                        break
                    real_entries[rn] = entry
                if bad:
                    continue

                # i is shared between B and D — if both are real, their
                # dimensions must agree. k (B only) and j (D only) are
                # each pinned only by their own real operand, if any; any
                # axis with no real operand behind it uses FREE_DIM — see
                # module docstring.
                b_dim = real_entries[b_name]["num_rows"] if b_name in real_entries else None
                d_dim = real_entries[d_name]["num_rows"] if d_name in real_entries else None
                if b_dim is not None and d_dim is not None and b_dim != d_dim:
                    print(f"  [skip] {name}: B ('{b_name}', dim={b_dim}) and D ('{d_name}', dim={d_dim}) "
                          f"must share the same i dimension")
                    continue

                i_dim = b_dim if b_dim is not None else d_dim
                k_dim = b_dim if b_dim is not None else FREE_DIM
                j_dim = d_dim if d_dim is not None else FREE_DIM

                densities = {rn: e["nnz"] / (e["num_rows"] * e["num_cols"]) for rn, e in real_entries.items()}
                reference_density = max(densities.values())
                target_density = max(reference_density, DENSITY_FLOOR)
                target_sparsity = 1.0 - target_density

                # A (the output) and C (synthetic, always fully dense) are
                # the only O(n^2) terms here — B and D stay sparse
                # (O(nnz), negligible next to those) whether real or
                # synthetic, since (unlike spmttkrp's 3D tensor_B) neither
                # ever grows cubically.
                c_bytes = k_dim * j_dim * 8
                a_bytes = i_dim * j_dim * 8
                est_bytes = c_bytes + a_bytes

                if est_bytes > memory_limit_bytes:
                    print(f"  [skip] {name}: estimated {est_bytes / 1024**3:.1f} GiB "
                          f"(C={c_bytes / 1024**3:.1f} A={a_bytes / 1024**3:.1f} GiB) "
                          f"(> {memory_limit_gb:.0f} GiB memory limit) — no sparse-output "
                          f"binary for spmmh to fall back to, skipping")
                    summary_writer.writerow([
                        name, b_name or "", f"{i_dim}x{k_dim}", d_name or "", f"{i_dim}x{j_dim}",
                        f"{k_dim}x{j_dim}", "", "", target_density * 100,
                        "SKIPPED_MEMORY", "SKIPPED_MEMORY",
                    ])
                    sf.flush()
                    continue

                tns_paths = {}
                download_failed = False
                for rn in real_names:
                    tns_path, ddir = download_matrix(rn, real_entries[rn])
                    if ddir is not None:
                        downloaded_dirs.append(ddir)
                    if tns_path is None:
                        download_failed = True
                        break
                    tns_paths[rn] = tns_path
                if download_failed:
                    continue

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
                        name, b_source, f"{i_dim}x{k_dim}", d_source, f"{i_dim}x{j_dim}",
                        f"{k_dim}x{j_dim}", b_nnz, d_nnz, target_density * 100,
                        scf_med, splyce_med,
                    ])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            finally:
                for f in (TENSOR_B, TENSOR_C, TENSOR_D):
                    if os.path.isfile(f):
                        os.remove(f)
                for d in downloaded_dirs:
                    if os.path.isdir(d):
                        shutil.rmtree(d)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
