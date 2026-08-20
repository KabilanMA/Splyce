#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpMTTKRP benchmark, curated set.
#
# spmttkrp computes A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j) (see
# spmttkrp_dn.mlir). Each job in JOBS below defines which SuiteSparse
# matrix (if any) supplies tensor_C and tensor_D — B is always synthetic,
# since there's no natural SuiteSparse source for a 3D tensor
# (run_frostt_benchmark.py handles real 3D data separately, from FROSTT
# instead). When c/d is None, that operand is synthetic too.
#
# Whichever of C/D is real must be square (per matrix_metadata.json), and
# its own dimension is used for whichever axes it actually constrains via
# conformance — B(i,k,l)/C(l,j)/D(k,j) must conform (B's k = D's k, B's l =
# C's l, C's j = D's j) — while every axis *not* constrained by a real
# operand (B's i always; B's k/D's k when D isn't real; B's l/C's l when C
# isn't real) uses a fixed FREE_DIM (1000) instead, same as the original
# single-job version of this script's SYNTHETIC_I/SYNTHETIC_L. So: D real,
# C synthetic -> B = 1000 x dim(D) x 1000, C = 1000 x dim(D); C real, D
# synthetic -> B = 1000 x 1000 x dim(C), D = 1000 x dim(C); both real (the
# same matrix, or two different same-j-dimension ones — their j dimension,
# i.e. num_rows/num_cols, must match; errors clearly otherwise) -> B = 1000
# x dim x dim.
#
# Every synthetic operand (B always, plus whichever of C/D is None) is
# generated at density = max(the job's real matrix density, DENSITY_FLOOR
# (0.001%)) — same floor-at-the-real-matrix's-own-density rule
# run_suitesparse_sweep.py and run_frostt_benchmark.py use: a real matrix
# at least as dense as the floor gets synthetic operands exactly as sparse
# as itself, while an extremely sparse/huge real matrix still gets
# floor-density synthetic operands instead of ending up with ~0 nonzeros.
# If both C and D are real, the denser of the two is used as the
# reference.
#
# For each job, this:
#   1. Resolves dim + the target synthetic density from metadata alone
#      (matrix_metadata.json — run suitesparse/scrape_metadata.py first if
#      that file doesn't exist yet), and estimates the combined peak memory
#      B + C + D + dense A would need (see MEMORY_SAFETY_FRACTION; override
#      with --memory-limit-gib) — skipping the job before downloading
#      anything if that exceeds the budget. Unlike
#      spgemm/run_suitesparse_benchmark.py, there's no CSR-output fallback
#      to fall back to here (spmttkrp's output A is always dense), so this
#      just skips.
#   2. Downloads + converts whichever of C/D are real
#      (suitesparse/download_data.py + convert_to_tns.py — mirrors
#      symmetric entries, drops explicit zeros), deduplicating a download
#      when the same matrix name fills both roles.
#   3. Generates whichever of B/C/D are synthetic (generate_sparse_3d_tns /
#      generate_sparse_2d_tns below — ported from experiments/gen_data.py's
#      functions of the same name).
#   4. Runs the splyce_phase_001 binary FIRST, then the scf binary — but
#      only if Splyce didn't time out. If Splyce already hit the timeout,
#      the (unvectorized, typically no faster) baseline is skipped entirely
#      rather than wasting the same timeout on a run that's essentially
#      guaranteed to also be too slow. Each binary loops 6 iterations
#      internally per spmttkrp_dn.mlir's @main, writing a "benchmark" file
#      with one time per line.
#   5. Appends one summary row (dataset, b/c/d shape+source, target
#      density, scf_median, splyce_median — median of the 5 non-cold-start
#      iterations; scf_median is "SKIPPED" when Splyce timed out) to
#      spmttkrp_realworld_results.csv, and every individual raw iteration
#      time to spmttkrp_realworld_raw_runtimes.csv as a backup.
#   6. Deletes tensor_B.tns/tensor_C.tns/tensor_D.tns and any downloaded/
#      converted suitesparse/<name>/ director(y/ies) before moving to the
#      next job.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once a
# job is already recorded (unless the CSV row is removed first).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spmttkrp_scf and test_benchmark_spmttkrp_splyce_phase_001
# exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # every job in JOBS
#   ./run_suitesparse_benchmark.py --matrix cat_ears_2_1
#       Runs that one SuiteSparse matrix for BOTH tensor_C and tensor_D
#       instead of JOBS — must be square. Ignores JOBS/--limit entirely.
#   ./run_suitesparse_benchmark.py --limit 2    # only the first 2 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --memory-limit-gib 64   # override the
#       auto-detected memory budget (see MEMORY_SAFETY_FRACTION)
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

# Floor for every synthetic operand's density (0.001%) — see module
# docstring.
DENSITY_FLOOR = 0.001 / 100

# Dimension for any axis not constrained by a real operand (B's i always;
# B's k/D's k when D isn't real; B's l/C's l when C isn't real) — see
# module docstring.
FREE_DIM = 1000

# The curated jobs this script runs — see module docstring. "c"/"d": None
# means that operand is synthetic; a name means it's downloaded from
# SuiteSparse. There's no "b" key — B is always synthetic (3D, no real
# SuiteSparse source).
JOBS = [
    {"name": "heart1", "c": None, "d": "heart1"},
    # {"name": "CAG_mat364", "c": "CAG_mat364", "d": "CAG_mat364"},
    # {"name": "struct4", "c": None, "d": "struct4"},
    # {"name": "cavity26", "c": None, "d": "cavity26"},
]

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime, which briefly holds a full COO intermediate
# alongside the final level-format storage before freeing the COO (see the
# sparse_tensor reader trace from the FROSTT-loading investigation) —
# roughly 2x the raw coordinate size:
#   tensor_B, 3D (3 coords + 1 value, 8 bytes each) * 2   = 64 bytes/nnz
#   tensor_C/D, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
TENSOR_B_BYTES_PER_NNZ = 64
TENSOR_2D_BYTES_PER_NNZ = 48

# Fraction of total system RAM usable as budget — only one job runs at a
# time (sequential), so this is a fraction of the *whole machine's* RAM,
# not divided across jobs; not the full total, to leave headroom for the
# OS, page cache, the Python driver itself, and the fact that the
# per-nonzero estimates above are approximate, not exact.
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
    mode = "single"
    cores = None
    memory_limit_gib = None
    matrix_name = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--cores" in args:
        cores = int(args[args.index("--cores") + 1])
    if "--memory-limit-gib" in args:
        memory_limit_gib = float(args[args.index("--memory-limit-gib") + 1])
    if "--matrix" in args:
        matrix_name = args[args.index("--matrix") + 1]

    memory_budget_bytes = (
        int(memory_limit_gib * 1024 ** 3) if memory_limit_gib is not None
        else int(detect_total_memory_bytes() * MEMORY_SAFETY_FRACTION)
    )
    print(f"Memory budget per job: {memory_budget_bytes / 1024**3:.1f} GiB"
          + (" (explicit)" if memory_limit_gib is not None else
             f" ({MEMORY_SAFETY_FRACTION:.0%} of detected total RAM)"))

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

    if matrix_name is not None:
        entry = metadata.get(matrix_name)
        if entry is None:
            sys.exit(f"error: '{matrix_name}' not found in matrix_metadata.json")
        if entry["num_rows"] != entry["num_cols"]:
            sys.exit(f"error: '{matrix_name}' is {entry['num_rows']}x{entry['num_cols']}, not square — "
                      "this script only pairs a square matrix against itself")
        jobs = [{"name": matrix_name, "c": matrix_name, "d": matrix_name}]
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
                "dataset", "b_shape", "c_source", "c_shape", "d_source", "d_shape",
                "b_nnz", "c_nnz", "d_nnz", "target_density_pct",
                "mode", "cores", "scf_median_s", "splyce_median_s",
            ])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for job in jobs:
            name = job["name"]
            if (name, mode) in done:
                continue

            print(f"=== {name} ===")
            downloaded_dirs = []

            try:
                c_name = job["c"]
                d_name = job["d"]
                real_names = {n for n in (c_name, d_name) if n is not None}

                if not real_names:
                    print(f"  [skip] {name}: job specifies no real matrix for C or D "
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

                # j is shared between C and D — if both are real, their
                # dimensions must agree. l (B/C) and k (B/D) are each
                # pinned only by their own real operand, if any; any axis
                # with no real operand behind it (i always; k or l when
                # D/C respectively isn't real) uses FREE_DIM instead — see
                # module docstring.
                c_dim = real_entries[c_name]["num_rows"] if c_name in real_entries else None
                d_dim = real_entries[d_name]["num_rows"] if d_name in real_entries else None
                if c_dim is not None and d_dim is not None and c_dim != d_dim:
                    print(f"  [skip] {name}: C ('{c_name}', dim={c_dim}) and D ('{d_name}', dim={d_dim}) "
                          f"must share the same j dimension")
                    continue

                i_dim = FREE_DIM
                k_dim = d_dim if d_dim is not None else FREE_DIM
                l_dim = c_dim if c_dim is not None else FREE_DIM
                j_dim = c_dim if c_dim is not None else d_dim

                densities = {rn: e["nnz"] / (e["num_rows"] * e["num_cols"]) for rn, e in real_entries.items()}
                reference_density = max(densities.values())
                target_density = max(reference_density, DENSITY_FLOOR)
                target_sparsity = 1.0 - target_density

                # B, C, D, AND dense output A are all simultaneously
                # resident (see module docstring) — estimate the combined
                # peak before downloading anything.
                expected_b_nnz = target_density * (i_dim * k_dim * l_dim)
                tensor_b_bytes = expected_b_nnz * TENSOR_B_BYTES_PER_NNZ
                c_nnz_for_estimate = real_entries[c_name]["nnz"] if c_name in real_entries else target_density * (l_dim * j_dim)
                d_nnz_for_estimate = real_entries[d_name]["nnz"] if d_name in real_entries else target_density * (k_dim * j_dim)
                tensor_c_bytes = c_nnz_for_estimate * TENSOR_2D_BYTES_PER_NNZ
                tensor_d_bytes = d_nnz_for_estimate * TENSOR_2D_BYTES_PER_NNZ
                dense_a_bytes = i_dim * k_dim * 8
                estimated_peak_bytes = tensor_b_bytes + tensor_c_bytes + tensor_d_bytes + dense_a_bytes

                if estimated_peak_bytes > memory_budget_bytes:
                    print(f"  [skip] estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                          f"(B={tensor_b_bytes / 1024**3:.1f} C={tensor_c_bytes / 1024**3:.1f} "
                          f"D={tensor_d_bytes / 1024**3:.1f} A={dense_a_bytes / 1024**3:.1f} GiB) "
                          f"> budget {memory_budget_bytes / 1024**3:.1f} GiB — skipping {name}")
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
                        name, f"{i_dim}x{k_dim}x{l_dim}", c_source, f"{l_dim}x{j_dim}",
                        d_source, f"{k_dim}x{j_dim}", b_nnz, c_nnz, d_nnz,
                        target_density * 100, mode, cores or "",
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
