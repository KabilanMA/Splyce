#!/usr/bin/env python3
# run_suitesparse_sweep.py — Real-world SpGEMM benchmark, full sweep.
#
# Unlike run_suitesparse_benchmark.py's four hand-picked jobs (one of which
# pairs a real matrix with a synthetic one), every job here uses two real
# SuiteSparse matrices: A = M^2 for some square matrix M (tensor_B and
# tensor_C are both M). The M's come from
# suitesparse/download_data.py's median_square_matrix_per_group(): one
# matrix per SuiteSparse group — the median-nnz one among that group's
# square matrices — giving ~169 jobs, one representative per group.
#
# For each job, this downloads M (suitesparse/download_data.py, looked up
# in matrix_metadata.json — run suitesparse/scrape_metadata.py first if
# that doesn't exist yet), converts it to .tns
# (suitesparse/convert_to_tns.py), copies it into both tensor_B.tns and
# tensor_C.tns, picks dense (spgemm_dn.mlir) vs CSR (spgemm.mlir) output
# binaries the same way run_suitesparse_benchmark.py does (by dense output
# size), runs splyce_phase_001 then scf (skipping scf if splyce already
# timed out), appends one row to spgemm_realworld_sweep_results.csv (and
# every raw iteration to spgemm_realworld_sweep_raw_runtimes.csv), and
# deletes tensor_B.tns/tensor_C.tns plus the downloaded suitesparse/<name>/
# before moving to the next job. Both CSVs are appended to, not
# overwritten, so an interrupted sweep resumes where it left off.
#
# Some selected matrices are enormous (multi-billion nnz) — use --matrix,
# --limit, or a tight --timeout to avoid downloading/running all of them.
#
# Prerequisite: ./compile.sh has already been run.
#
# Usage:
#   ./run_suitesparse_sweep.py                  # every group's matrix
#   ./run_suitesparse_sweep.py --matrix cat_ears_2_1   # just that one
#   ./run_suitesparse_sweep.py --limit 10        # first 10 (testing)
#   ./run_suitesparse_sweep.py --timeout 600     # per-binary-run timeout
#                                                 # in seconds (default 300)
#   ./run_suitesparse_sweep.py --mode multicore [--cores N]
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
SUITESPARSE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "suitesparse"))
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BASELINE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_scf")
SPLYCE_BIN_DENSE = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_splyce_phase_001")
BASELINE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_scf")
SPLYCE_BIN_CSR = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_splyce_phase_001")

BASELINE_BIN_DENSE_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_scf_parallel")
SPLYCE_BIN_DENSE_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_splyce_phase_001_parallel")
BASELINE_BIN_CSR_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_scf_parallel")
SPLYCE_BIN_CSR_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spgemm_csr_splyce_phase_001_parallel")

# A is num_rows(M) x num_cols(M); a dense f64 A needs that many * 8 bytes.
# Past this, the dense-output binaries would fail to allocate A and crash —
# swap to the CSR-output binaries instead.
DENSE_OUTPUT_LIMIT_BYTES = 200 * 1024 ** 3  # 200 GiB

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_sweep_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spgemm_realworld_sweep_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")


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
    # Keyed on (dataset, mode) — not dataset alone — so a matrix already
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


def select_jobs(metadata, matrix_name):
    if matrix_name is not None:
        entry = metadata.get(matrix_name)
        if entry is None:
            sys.exit(f"error: '{matrix_name}' not found in matrix_metadata.json")
        if entry["num_rows"] != entry["num_cols"]:
            sys.exit(f"error: '{matrix_name}' is {entry['num_rows']}x{entry['num_cols']}, not square — "
                      "this script only squares a matrix against itself")
        return [entry]
    return dl.median_square_matrix_per_group(metadata)


def main():
    args = sys.argv[1:]
    limit = None
    timeout = 300
    matrix_name = None
    mode = "single"
    cores = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
    if "--matrix" in args:
        matrix_name = args[args.index("--matrix") + 1]
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

    if mode == "multicore":
        baseline_bin_dense, splyce_bin_dense = BASELINE_BIN_DENSE_PARALLEL, SPLYCE_BIN_DENSE_PARALLEL
        baseline_bin_csr, splyce_bin_csr = BASELINE_BIN_CSR_PARALLEL, SPLYCE_BIN_CSR_PARALLEL
        print(f"Number of cores used: {cores}")
    else:
        baseline_bin_dense, splyce_bin_dense = BASELINE_BIN_DENSE, SPLYCE_BIN_DENSE
        baseline_bin_csr, splyce_bin_csr = BASELINE_BIN_CSR, SPLYCE_BIN_CSR

    if not (os.path.isfile(baseline_bin_dense) and os.path.isfile(splyce_bin_dense)
            and os.path.isfile(baseline_bin_csr) and os.path.isfile(splyce_bin_csr)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    metadata = dl.load_metadata()
    jobs = select_jobs(metadata, matrix_name)
    if limit is not None:
        jobs = jobs[:limit]

    done = already_recorded()
    print(f"{len(jobs)} job(s), {len(done)} already recorded — resuming")

    write_summary_header = not os.path.isfile(SUMMARY_CSV)
    write_raw_header = not os.path.isfile(RAW_BACKUP_CSV)

    with open(SUMMARY_CSV, "a", newline="") as sf, open(RAW_BACKUP_CSV, "a", newline="") as rf:
        summary_writer = csv.writer(sf)
        raw_writer = csv.writer(rf)
        if write_summary_header:
            summary_writer.writerow([
                "dataset", "group", "shape", "nnz", "format", "mode", "cores",
                "scf_median_s", "splyce_median_s",
            ])
        if write_raw_header:
            raw_writer.writerow(["dataset", "config", "iteration", "time_s"])

        for entry in jobs:
            name = entry["name"]
            if (name, mode) in done:
                continue

            print(f"=== {entry['group']}/{name} (nnz={entry['nnz']}) ===")
            # Set to the expected path up front (matching download_matrix's
            # own computation), not just whatever it returns — if it raises
            # partway through (e.g. cvt.process_dir fails on an unsupported
            # matrix), the assignment below never happens, and this is what
            # the except/finally cleanup needs to find + remove the partial
            # download instead of leaving it on disk.
            downloaded_dir = os.path.join(SUITESPARSE_DIR, name)

            try:
                rows, cols, nnz = entry["num_rows"], entry["num_cols"], entry["nnz"]

                dense_bytes = rows * cols * 8
                if dense_bytes > DENSE_OUTPUT_LIMIT_BYTES:
                    print(f"  [skip] dense output would be {dense_bytes / 1024**3:.1f} GiB "
                          f"(> {DENSE_OUTPUT_LIMIT_BYTES / 1024**3:.0f} GiB) — skipping {name}")
                    continue
                m_tns, downloaded_dir = download_matrix(name, entry)
                if m_tns is None:
                    continue
                shutil.copyfile(m_tns, TENSOR_B)
                shutil.copyfile(m_tns, TENSOR_C)
                fmt = "dense"
                baseline_bin, splyce_bin = baseline_bin_dense, splyce_bin_dense

                run_cores = cores if mode == "multicore" else None

                print(f"  [run] splyce phase_001 ({fmt}) ...")
                splyce_times, splyce_timed_out = run_binary(splyce_bin, timeout, run_cores)

                if splyce_timed_out:
                    print("  [skip] splyce timed out — skipping baseline run")
                    scf_times, scf_med = None, "SKIPPED"
                else:
                    print(f"  [result] splyce phase_001 ({fmt}) runtime: {splyce_times}")
                    print(f"  [run] baseline (scf, {fmt}) ...")
                    scf_times, _ = run_binary(baseline_bin, timeout * 5, run_cores)
                    scf_med = median_excl_first(scf_times) if scf_times else "NA"

                for i, t in enumerate(splyce_times or []):
                    raw_writer.writerow([name, f"splyce_phase_001_{fmt}", i, t])
                for i, t in enumerate(scf_times or []):
                    raw_writer.writerow([name, f"scf_{fmt}", i, t])
                rf.flush()

                splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

                if splyce_med != "NA":
                    summary_writer.writerow([
                        name, entry["group"], f"{rows}x{cols}", nnz,
                        fmt, mode, cores or "",
                        scf_med, splyce_med,
                    ])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            except Exception as e:
                print(f"  [error] {name}: {e} — skipping")
                continue
            finally:
                for f in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(f):
                        os.remove(f)
                if downloaded_dir is not None and os.path.isdir(downloaded_dir):
                    shutil.rmtree(downloaded_dir)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
