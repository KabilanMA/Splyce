#!/usr/bin/env python3
# run_frostt_benchmark.py — Real-world SpMTTKRP benchmark using a real 3D
# tensor from the FROSTT collection (frostt.io) for tensor_B, instead of
# the entirely-synthetic B/C run_suitesparse_benchmark.py uses (which only
# pulls a real 2D matrix for D).
#
# spmttkrp computes A(i,k) = Σ_l Σ_j B(i,k,l) · C(l,j) · D(k,j) (see
# spmttkrp_dn.mlir). B(i,k,l) is each named FROSTT tensor in turn — see
# ../frostt/download_data.py's FROSTT_TENSORS for the full hardcoded name ->
# download-URL list; FROSTT publishes no machine-readable index to look
# names up in the way SuiteSparse's ssstats.csv does, so unlike the
# suitesparse/-based scripts here, each tensor has to be added there by
# hand. tensor_C and tensor_D are synthetic, shaped from B's own dims so
# everything conforms:
#   - C(l,j): l = B's 3rd dim, j = B's 1st dim.
#   - D(k,j): k = B's 2nd dim, j = B's 1st dim (same j as C, per spmttkrp's
#     shared j index).
# (i.e. each one's first dim comes from the matching B dim it must conform
# against; both share their second dim, taken from B's own first dim.)
#
# C/D's density is derived from B's own global density (b_nnz / (dim1 *
# dim2 * dim3)), floored at MIN_DENSITY (0.001%) — B here can have one
# enormous dimension (FROSTT tensors are often extremely skewed, e.g.
# darpa's 3rd dim is ~23.7M, a fine-grained timestamp), so without a floor
# a real B density applied to C/D's own shape could produce an essentially
# empty synthetic operand.
#
# For the job, this downloads + converts B (../frostt/download_data.py,
# ../frostt/convert_to_tns.py), generates tensor_C.tns/tensor_D.tns (via
# ../../../gen_data.sh's gen_2d_tensor), runs splyce_phase_001 then scf
# (skipping scf if splyce already timed out), appends one row to
# spmttkrp_frostt_results.csv (and every raw iteration to
# spmttkrp_frostt_raw_runtimes.csv), and deletes tensor_B/C/D.tns plus the
# downloaded frostt/<name>/ directory afterward. Both CSVs are appended to,
# not overwritten, so a re-run of an already-recorded (tensor, mode) is a
# no-op.
#
# Prerequisite: ./compile.sh has already been run.
#
# Which tensors get benchmarked isn't passed on the command line — this
# runs every tensor listed in ../frostt/download_data.py's FROSTT_TENSORS
# map (skipping ones already recorded, per already_recorded()), so adding a
# new tensor there is enough to have the next run pick it up.
#
# spmttkrp_dn.mlir's main() keeps B, C, D, AND the dense output A all
# simultaneously resident for the whole benchmark loop (none of B/C/D are
# freed until after it finishes), and FROSTT tensors can be huge — so
# before generating C/D or running anything, this estimates the combined
# peak (B + C + D + A) and skips the tensor if it exceeds a memory budget
# (see MEMORY_SAFETY_FRACTION; override with --memory-limit-gib). Unlike
# run_suitesparse_sweep.py's equivalent guard (same directory), this can't
# check *before* downloading — FROSTT publishes no
# advance size metadata (see download_data.py's module docstring), so
# tensor_B's real shape/nnz are only known after it's already been
# downloaded and converted. The check still guards the actually
# memory-heavy steps (C/D generation, and running the binaries), just not
# the download bandwidth for an oversized tensor.
#
# Usage:
#   ./run_frostt_benchmark.py
#   ./run_frostt_benchmark.py --timeout 600        # per-binary-run timeout
#                                                   # in seconds (default 300)
#   ./run_frostt_benchmark.py --memory-limit-gib 64   # override the
#       auto-detected memory budget with a fixed cap
#   ./run_frostt_benchmark.py --mode multicore [--cores N]
#       Runs the OpenMP dense-outer-loop parallel binaries (compile.sh's
#       *_parallel variants) with OMP_NUM_THREADS=N instead of the
#       single-threaded ones. --cores defaults to every CPU on the machine
#       (os.cpu_count()). To pin these to a specific NUMA node/CPU set,
#       wrap the whole command in `numactl ...` — nothing here sets its own
#       CPU affinity, so an outer numactl applies to every binary run.

import csv
import os
import shutil
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FROSTT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "frostt"))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
GEN_DATA_SH = os.path.join(EXPERIMENTS_DIR, "gen_data.sh")
sys.path.insert(0, FROSTT_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001")
BASELINE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf_parallel")
SPLYCE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001_parallel")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_frostt_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_frostt_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

MIN_DENSITY = 0.1 / 100  # 0.1% floor, for when B's own density is ~0

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime, which briefly holds a full COO intermediate
# alongside the final level-format storage before freeing the COO (see the
# sparse_tensor reader trace from the earlier FROSTT-loading investigation)
# — roughly 2x the raw coordinate size:
#   tensor_B, 3D (3 coords + 1 value, 8 bytes each) * 2   = 64 bytes/nnz
#   tensor_C/D, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
TENSOR_B_BYTES_PER_NNZ = 64
TENSOR_2D_BYTES_PER_NNZ = 48

# Fraction of *currently available* (not total) memory usable as budget —
# available already accounts for what's in use by other processes and for
# reclaimable page cache, so an aggressive fraction of it is safe to spend
# without the same risk an aggressive fraction of total RAM would carry on
# a machine already under memory pressure. FROSTT tensors are often huge,
# so this deliberately runs hotter than the SuiteSparse sweep's 50%-of-total
# — the goal here is using as much of what's actually free as safely
# possible, not leaving half the machine idle by default.
MEMORY_SAFETY_FRACTION = 0.85


def detect_available_memory_bytes():
    # Linux-specific (/proc/meminfo). Falls back to half of MemTotal if
    # MemAvailable specifically is missing (pre-3.14 kernels), then to a
    # conservative 16 GiB if /proc/meminfo itself is unreadable — a
    # detection failure fails toward skipping too much, not too little.
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemTotal"):
                    info[key] = int(rest.split()[0]) * 1024  # value is in KiB
        if "MemAvailable" in info:
            return info["MemAvailable"]
        if "MemTotal" in info:
            return info["MemTotal"] // 2
    except (OSError, ValueError):
        pass
    return 16 * 1024 ** 3


def generate_2d_tensor(path, rows, cols, density):
    density = min(0.99, density)
    sparsity = 1.0 - density
    subprocess.run(
        [GEN_DATA_SH, "gen_2d_tensor", path, str(rows), str(cols), str(sparsity)],
        cwd=EXPERIMENTS_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with open(path) as f:
        f.readline()
        header_line = f.readline()  # header line 2: "2 <nnz>"
    return int(header_line.split()[1]), density


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
    # Keyed on (dataset, mode) — not dataset alone — so a tensor already
    # recorded in one mode doesn't shadow a run of it in the other mode.
    if not os.path.isfile(SUMMARY_CSV):
        return set()
    with open(SUMMARY_CSV) as f:
        return {(row["dataset"], row.get("mode", "single")) for row in csv.DictReader(f)}


def process_tensor(tensor_name, timeout, mode, cores, memory_budget_bytes, summary_writer, raw_writer, sf, rf):
    print(f"=== {tensor_name} ===")
    downloaded_dir = os.path.join(FROSTT_DIR, tensor_name)

    try:
        raw_path = dl.download_and_extract(tensor_name, force=False)
        (dim1, dim2, dim3), b_nnz = cvt.convert_raw_to_tns(raw_path, os.path.join(downloaded_dir, f"{tensor_name}.tns"))
        shutil.copyfile(os.path.join(downloaded_dir, f"{tensor_name}.tns"), TENSOR_B)

        b_density = b_nnz / (dim1 * dim2 * dim3)
        target_density = max(b_density, MIN_DENSITY)
        print(f"  [tensor_B] {tensor_name}: shape=({dim1},{dim2},{dim3}) nnz={b_nnz} density={b_density:.6g}")

        c_rows, c_cols = dim3, dim1
        d_rows, d_cols = dim2, dim1

        # B, C, D, and dense output A are all simultaneously resident (see
        # module docstring) — estimate the combined peak before generating
        # C/D or running anything. b_nnz is exact (just measured from the
        # real download); expected C/D nnz are estimated from their target
        # density, same as generate_2d_tensor will actually produce.
        tensor_b_bytes = b_nnz * TENSOR_B_BYTES_PER_NNZ
        expected_c_nnz = target_density * (c_rows * c_cols)
        expected_d_nnz = target_density * (d_rows * d_cols)
        tensor_c_bytes = expected_c_nnz * TENSOR_2D_BYTES_PER_NNZ
        tensor_d_bytes = expected_d_nnz * TENSOR_2D_BYTES_PER_NNZ
        dense_a_bytes = dim1 * dim2 * 8
        estimated_peak_bytes = tensor_b_bytes + tensor_c_bytes + tensor_d_bytes + dense_a_bytes

        if estimated_peak_bytes > memory_budget_bytes:
            print(f"  [skip] estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                  f"(B={tensor_b_bytes / 1024**3:.1f} C={tensor_c_bytes / 1024**3:.1f} "
                  f"D={tensor_d_bytes / 1024**3:.1f} A={dense_a_bytes / 1024**3:.1f} GiB) "
                  f"> budget {memory_budget_bytes / 1024**3:.1f} GiB — skipping {tensor_name} "
                  f"(download already completed — FROSTT has no advance size metadata to "
                  f"check before downloading)")
            # Record this as an NA row instead of just skipping silently —
            # already_recorded() only sees a tensor as done once it has a
            # row here, so without this, every future run would re-download
            # and re-convert this same (potentially many-GB) tensor only to
            # hit the same memory limit again. plot_results.py-style scripts
            # already treat a non-numeric scf/splyce time as "no comparison
            # available" (excluded from the chart, reported separately) —
            # same handling the splyce-timeout path below relies on via its
            # own "SKIPPED" value, so NA fits the same convention.
            summary_writer.writerow([
                tensor_name, f"{dim1}x{dim2}x{dim3}", b_nnz,
                f"{c_rows}x{c_cols}", "NA", "NA",
                f"{d_rows}x{d_cols}", "NA", "NA",
                mode, cores or "",
                "NA", "NA",
            ])
            sf.flush()
            return

        c_nnz, c_density = generate_2d_tensor(TENSOR_C, c_rows, c_cols, target_density)
        d_nnz, d_density = generate_2d_tensor(TENSOR_D, d_rows, d_cols, target_density)
        print(f"  [tensor_C] shape=({c_rows},{c_cols}) nnz={c_nnz} density={c_density:.6g}")
        print(f"  [tensor_D] shape=({d_rows},{d_cols}) nnz={d_nnz} density={d_density:.6g}")

        baseline_bin = BASELINE_BIN_PARALLEL if mode == "multicore" else BASELINE_BIN
        splyce_bin = SPLYCE_BIN_PARALLEL if mode == "multicore" else SPLYCE_BIN
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
            raw_writer.writerow([tensor_name, "splyce_phase_001", i, t])
        for i, t in enumerate(scf_times or []):
            raw_writer.writerow([tensor_name, "scf", i, t])
        rf.flush()

        splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

        if splyce_med != "NA":
            summary_writer.writerow([
                tensor_name, f"{dim1}x{dim2}x{dim3}", b_nnz,
                f"{c_rows}x{c_cols}", c_nnz, c_density,
                f"{d_rows}x{d_cols}", d_nnz, d_density,
                mode, cores or "",
                scf_med, splyce_med,
            ])
        sf.flush()
        print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

    except Exception as e:
        print(f"  [error] {tensor_name}: {e} — aborting")
        raise
    finally:
        for f in (TENSOR_B, TENSOR_C, TENSOR_D):
            if os.path.isfile(f):
                os.remove(f)
        if os.path.isdir(downloaded_dir):
            shutil.rmtree(downloaded_dir)


def main():
    args = sys.argv[1:]
    timeout = 300
    mode = "single"
    cores = None
    memory_limit_gib = None
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if "--cores" in args:
        cores = int(args[args.index("--cores") + 1])
    if "--memory-limit-gib" in args:
        memory_limit_gib = float(args[args.index("--memory-limit-gib") + 1])

    if mode not in ("single", "multicore"):
        sys.exit(f"error: unsupported --mode '{mode}' (supported: single, multicore)")
    if cores is not None and mode != "multicore":
        sys.exit("error: --cores is only meaningful with --mode multicore")
    if mode == "multicore" and cores is None:
        cores = os.cpu_count()

    memory_budget_bytes = (
        int(memory_limit_gib * 1024 ** 3) if memory_limit_gib is not None
        else int(detect_available_memory_bytes() * MEMORY_SAFETY_FRACTION)
    )
    print(f"Memory budget per tensor: {memory_budget_bytes / 1024**3:.1f} GiB"
          + (" (explicit)" if memory_limit_gib is not None else
             f" ({MEMORY_SAFETY_FRACTION:.0%} of currently available RAM)"))

    baseline_bin = BASELINE_BIN_PARALLEL if mode == "multicore" else BASELINE_BIN
    splyce_bin = SPLYCE_BIN_PARALLEL if mode == "multicore" else SPLYCE_BIN

    if not (os.path.isfile(baseline_bin) and os.path.isfile(splyce_bin)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    recorded = already_recorded()
    pending = [name for name in dl.FROSTT_TENSORS if (name, mode) not in recorded]
    if not pending:
        print(f"all known tensors already recorded for mode={mode} — nothing to do")
        return

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "b_shape", "b_nnz", "c_shape", "c_nnz", "c_density",
                "d_shape", "d_nnz", "d_density", "mode", "cores",
                "scf_median_s", "splyce_median_s",
            ])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for tensor_name in pending:
            process_tensor(tensor_name, timeout, mode, cores, memory_budget_bytes,
                            summary_writer, raw_writer, sf, rf)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
