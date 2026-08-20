#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpMSpV benchmark, single curated
# matrix.
#
# Runs one fixed job (instead of sweeping the full 169 median-per-group
# SuiteSparse sample — see suitesparse/download_data.py for that sweep):
#
#   stokes (VLSI group): 11449533 x 11449533 square, nnz=349321980
#   (~0.0000266% dense) — used for tensor_B.
#
# For this job, this script:
#   1. Downloads it (suitesparse/download_data.py's download_and_extract,
#      looked up via matrix_metadata.json — run suitesparse/scrape_metadata.py
#      first if that file doesn't exist yet).
#   2. Converts it to .tns (suitesparse/convert_to_tns.py's process_dir —
#      mirrors symmetric entries, drops explicit zeros) and copies it into
#      this directory as tensor_B.tns.
#   3. Generates a synthetic sparse vector tensor_x.tns whose length matches
#      B's column count, at density = max(matrix's own density,
#      VECTOR_DENSITY_FLOOR (0.001%)) — same floor-at-the-matrix's-own-
#      density rule run_suitesparse_sweep.py uses: a matrix at least as
#      dense as the floor gets a vector exactly as sparse as itself, while
#      an extremely sparse/huge matrix (e.g. stokes, ~0.0000266% dense)
#      still gets a floor-density vector instead of ending up with ~0
#      nonzeros. (Now that --matrix accepts any SuiteSparse matrix, not
#      just stokes, a fixed target regardless of the matrix's own density
#      no longer made sense for denser matrices.)
#   4. Runs test_benchmark_spmspv_splyce_phase_001 FIRST, then
#      test_benchmark_spmspv_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 6 iterations internally per spmspv.mlir's @main,
#      writing a "benchmark" file with one time per line.
#   5. Appends one summary row (dataset, group, matrix nnz/dim, vector
#      nnz/density, scf_median, splyce_median — median of the 5
#      non-cold-start iterations; scf_median is "SKIPPED" when Splyce
#      timed out) to spmspv_realworld_results.csv, and every individual
#      raw iteration time to spmspv_realworld_raw_runtimes.csv as a
#      backup.
#   6. Deletes tensor_B.tns/tensor_x.tns and the downloaded/converted
#      suitesparse/stokes/ directory afterward.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once
# stokes is already recorded (unless the CSV row is removed first).
#
# Before downloading anything, this also estimates the combined peak memory
# B, x, AND the dense output y would need simultaneously resident (see
# spmspv.mlir's main()) — computed from matrix_metadata.json alone, same as
# ../spmspv/run_suitesparse_sweep.py's equivalent guard — and exits without
# downloading if it exceeds a budget (see MEMORY_SAFETY_FRACTION; override
# with --memory-limit-gib). Unlike spgemm/run_suitesparse_benchmark.py,
# there's no dense/CSR output choice to fall back between here, so this
# just skips rather than picking an alternate format.
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spmspv_scf and test_benchmark_spmspv_splyce_phase_001
# exist in this directory.
#
# Usage:
#   ./run_suitesparse_benchmark.py
#   ./run_suitesparse_benchmark.py --matrix cat_ears_2_1
#       Runs that one SuiteSparse matrix instead of stokes. Must be square
#       (per matrix_metadata.json), same as run_suitesparse_sweep.py's
#       --matrix — B's column count doubles as x's length here (a single
#       "dim" derived from num_rows), so a rectangular matrix would size x
#       wrong.
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --memory-limit-gib 64   # override the
#       auto-detected memory budget with a fixed cap
#   ./run_suitesparse_benchmark.py --mode multicore [--cores N]
#       Runs the OpenMP dense-outer-loop parallel binaries (compile.sh's
#       *_parallel variants) with OMP_NUM_THREADS=N instead of the
#       single-threaded ones. --cores defaults to every CPU on the machine
#       (os.cpu_count()). To pin these to a specific NUMA node/CPU set,
#       wrap the whole command in `numactl ...` — nothing here sets its own
#       CPU affinity, so an outer numactl applies to every binary run.

import csv
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

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmspv_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmspv_splyce_phase_001")
BASELINE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmspv_scf_parallel")
SPLYCE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmspv_splyce_phase_001_parallel")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmspv_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmspv_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_X = os.path.join(SCRIPT_DIR, "tensor_x.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# Floor for the synthetic vector's density (0.001%) — see module docstring.
VECTOR_DENSITY_FLOOR = 0.00001

# The single curated job this script runs.
MATRIX_NAME = "stokes"

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime, which briefly holds a full COO intermediate
# alongside the final level-format storage before freeing the COO (see the
# sparse_tensor reader trace from the FROSTT-loading investigation) —
# roughly 2x the raw coordinate size:
#   B, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
#   x, 1D (1 coord + 1 value, 8 bytes each) * 2  = 32 bytes/nnz
TENSOR_2D_BYTES_PER_NNZ = 48
TENSOR_1D_BYTES_PER_NNZ = 32

# Fraction of total system RAM usable as budget — see module docstring.
MEMORY_SAFETY_FRACTION = 0.5


def detect_total_memory_bytes():
    # Linux-specific (/proc/meminfo) — falls back to a conservative 32 GiB
    # if unreadable, so a detection failure fails toward skipping rather
    # than risking an OOM.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # value is in KiB
    except (OSError, ValueError):
        pass
    return 32 * 1024 ** 3


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
    memory_limit_gib = None
    matrix_name = MATRIX_NAME
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
    print(f"Memory budget: {memory_budget_bytes / 1024**3:.1f} GiB"
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
    entry = metadata.get(matrix_name)
    if entry is None:
        sys.exit(f"error: '{matrix_name}' not found in matrix_metadata.json")
    if entry["num_rows"] != entry["num_cols"]:
        sys.exit(f"error: '{matrix_name}' is {entry['num_rows']}x{entry['num_cols']}, not square — "
                  "this script only pairs a square matrix with a same-length vector")
    name, group = entry["name"], entry["group"]

    done = already_recorded()
    if (name, mode) in done:
        print(f"{name} ({mode}) already recorded — nothing to do")
        return

    # B, x, AND the dense output y are all simultaneously resident (see
    # module docstring) — estimate the combined peak from metadata alone,
    # before downloading anything.
    dim = entry["num_rows"]  # square: num_rows == num_cols
    matrix_nnz = entry["nnz"]
    matrix_density = matrix_nnz / (dim * dim)
    target_vector_density = max(matrix_density, VECTOR_DENSITY_FLOOR)
    target_vector_nnz = round(target_vector_density * dim)
    tensor_b_bytes = matrix_nnz * TENSOR_2D_BYTES_PER_NNZ
    tensor_x_bytes = target_vector_nnz * TENSOR_1D_BYTES_PER_NNZ
    dense_y_bytes = dim * 8
    estimated_peak_bytes = tensor_b_bytes + tensor_x_bytes + dense_y_bytes

    if estimated_peak_bytes > memory_budget_bytes:
        sys.exit(f"error: estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                  f"(B={tensor_b_bytes / 1024**3:.1f} x={tensor_x_bytes / 1024**3:.1f} "
                  f"y={dense_y_bytes / 1024**3:.1f} GiB) > budget "
                  f"{memory_budget_bytes / 1024**3:.1f} GiB — skipping {name} "
                  f"without downloading")

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "group", "matrix_dim", "matrix_nnz", "matrix_sparsity",
                "vector_nnz", "vector_sparsity", "mode", "cores",
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

            # sparsity = 1 - density (fraction of *zero* entries), matching
            # run_suitesparse_sweep.py's convention and the rest of the
            # codebase's generators (e.g. gen_data.py) — not nnz/total
            # directly, which is density. matrix_density itself was already
            # computed above (needed for the vector-density floor before
            # the memory check).
            matrix_sparsity = 1.0 - matrix_density

            shutil.copyfile(tns_path, TENSOR_B)
            vector_nnz = generate_sparse_vector_tns(TENSOR_X, dim, target_vector_nnz)
            vector_density = vector_nnz / dim
            vector_sparsity = 1.0 - vector_density
            print(f"  [vector] dim={dim} nnz={vector_nnz} sparsity={vector_sparsity:.6g} "
                  f"(matrix_sparsity={matrix_sparsity:.6g})")

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
                    name, group, dim, matrix_nnz, matrix_sparsity,
                    vector_nnz, vector_sparsity, mode, cores or "",
                    scf_med, splyce_med,
                ])
            sf.flush()
            print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

        finally:
            for f in (TENSOR_B, TENSOR_X):
                if os.path.isfile(f):
                    os.remove(f)
            if os.path.isdir(dataset_dir):
                shutil.rmtree(dataset_dir)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
