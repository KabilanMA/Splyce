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
# Usage:
#   ./run_frostt_benchmark.py
#   ./run_frostt_benchmark.py --timeout 600        # per-binary-run timeout
#                                                   # in seconds (default 300)
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

MIN_DENSITY = 0.001 / 100  # 0.001% floor, for when B's own density is ~0


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


def process_tensor(tensor_name, timeout, mode, cores, summary_writer, raw_writer, sf, rf):
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
            process_tensor(tensor_name, timeout, mode, cores, summary_writer, raw_writer, sf, rf)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
