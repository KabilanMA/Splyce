#!/usr/bin/env python3
# run_suitesparse_sweep.py — Real-world SpMTTKRP benchmark, full sweep.
#
# spmttkrp computes A(i,k) = Σ_l Σ_j B(i,k,l)·C(l,j)·D(k,j) (see
# spmttkrp_dn.mlir). Unlike run_suitesparse_benchmark.py's single curated
# job (heart1, used only for D, with B/C entirely synthetic), every job
# here uses one real SuiteSparse matrix M for *both* tensor_C and
# tensor_D — since C(l,j) and D(k,j) share index j, using the same square
# M for both forces k = l = j = dim(M) — and a synthetic 3D tensor_B(i,k,l)
# sized dim(M) x dim(M) x dim(M) (i also set to dim(M), so B's shape is
# entirely derived from M) at a density of max(M's own density,
# TENSOR_B_DENSITY_FLOOR) — the same floor-at-the-matrix's-own-density rule
# speedups/real_world_data/spmspv/run_suitesparse_sweep.py uses for its
# synthetic vector: a matrix at least as dense as the floor gets a
# tensor_B exactly as sparse as itself, while an extremely sparse/huge
# matrix still gets a floor-density tensor_B instead of ending up with ~0
# nonzeros. The M's come from suitesparse/download_data.py's
# median_square_matrix_per_group(): one matrix per SuiteSparse group — the
# median-nnz one among that group's square matrices — giving ~169 jobs, one
# representative per group (same selection the spgemm/spmspv sweep
# scripts use). tensor_B is generated via gen_data.sh's gen_3d_tensor
# command (../../../gen_data.sh) rather than duplicating that generator
# here.
#
# For each job, this downloads M (suitesparse/download_data.py, looked up
# in matrix_metadata.json — run suitesparse/scrape_metadata.py first if
# that doesn't exist yet), converts it to .tns
# (suitesparse/convert_to_tns.py), copies it into both tensor_C.tns and
# tensor_D.tns, generates tensor_B.tns, runs splyce_phase_001 then scf
# (skipping scf if splyce already timed out), appends one row to
# spmttkrp_realworld_sweep_results.csv (and every raw iteration to
# spmttkrp_realworld_sweep_raw_runtimes.csv), and deletes
# tensor_B.tns/tensor_C.tns/tensor_D.tns plus the downloaded
# suitesparse/<name>/ before moving to the next job. Both CSVs are appended
# to, not overwritten, so an interrupted sweep resumes where it left off.
#
# Some selected matrices are enormous — use --matrix, --limit, or a tight
# --timeout to avoid downloading/running all of them. spmttkrp_dn.mlir's
# main() keeps B, C, D, AND the dense output A all simultaneously resident
# for the whole benchmark loop (none of B/C/D are freed until after it
# finishes) — so the memory guard below estimates the *combined* peak
# (B + C + D + A), not each one independently, and skips the job up front
# if that exceeds a budget derived from the machine's actual RAM (see
# MEMORY_SAFETY_FRACTION). Checking each tensor against its own fixed cap
# independently is not enough: a job can pass every individual check and
# still OOM the machine once all four are loaded together.
#
# Prerequisite: ./compile.sh has already been run.
#
# Usage:
#   ./run_suitesparse_sweep.py                  # every group's matrix
#   ./run_suitesparse_sweep.py --matrix cat_ears_2_1   # just that one
#   ./run_suitesparse_sweep.py --limit 10        # first 10 (testing)
#   ./run_suitesparse_sweep.py --timeout 600     # per-binary-run timeout
#                                                 # in seconds (default 300)
#   ./run_suitesparse_sweep.py --memory-limit-gib 64   # override the
#       auto-detected memory budget (see MEMORY_SAFETY_FRACTION) with a
#       fixed cap, e.g. on a shared machine where using half of total RAM
#       still isn't safe.
#   ./run_suitesparse_sweep.py --mode multicore [--cores N]
#       Runs the OpenMP dense-outer-loop parallel binaries (compile.sh's
#       *_parallel variants) with OMP_NUM_THREADS=N instead of the
#       single-threaded ones. --cores defaults to every CPU on the machine
#       (os.cpu_count()). To pin these to a specific NUMA node/CPU set,
#       wrap the whole command in `numactl ...` — nothing here sets its own
#       CPU affinity, so an outer numactl applies to every binary run.

import csv
import os
import statistics
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUITESPARSE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "suitesparse"))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
GEN_DATA_SH = os.path.join(EXPERIMENTS_DIR, "gen_data.sh")
sys.path.insert(0, SUITESPARSE_DIR)

import download_data as dl  # noqa: E402
import convert_to_tns as cvt  # noqa: E402

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001")
BASELINE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_scf_parallel")
SPLYCE_BIN_PARALLEL = os.path.join(SCRIPT_DIR, "test_benchmark_spmttkrp_splyce_phase_001_parallel")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_realworld_sweep_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spmttkrp_realworld_sweep_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
TENSOR_D = os.path.join(SCRIPT_DIR, "tensor_D.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# Floor for tensor_B's density (0.001%) — see module docstring.
TENSOR_B_DENSITY_FLOOR = 0.00001

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime, which briefly holds a full COO intermediate
# alongside the final level-format storage before freeing the COO (see the
# sparse_tensor reader trace from the FROSTT investigation) — roughly 2x
# the raw coordinate size:
#   tensor_B, 3D (3 coords + 1 value, 8 bytes each) * 2   = 64 bytes/nnz
#   tensor_C/D, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
TENSOR_B_BYTES_PER_NNZ = 64
TENSOR_2D_BYTES_PER_NNZ = 48

# Only one job runs at a time (this is a sequential sweep), so the budget
# is a fraction of the *whole machine's* RAM — not the full total, to leave
# headroom for the OS, page cache, the Python driver itself, and the fact
# that the per-nonzero estimates above are approximate, not exact.
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


def generate_tensor_b(path, dim, sparsity):
    subprocess.run(
        [GEN_DATA_SH, "gen_3d_tensor", path, str(dim), str(dim), str(dim), str(sparsity)],
        cwd=EXPERIMENTS_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with open(path) as f:
        f.readline()
        header_line = f.readline()  # header line 2: "3 <nnz>"
    return int(header_line.split()[1])


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
                      "this script only pairs a square matrix (as both C and D) with a same-dimension tensor_B")
        return [entry]
    return dl.median_square_matrix_per_group(metadata)


def main():
    args = sys.argv[1:]
    limit = None
    timeout = 300
    matrix_name = None
    mode = "single"
    cores = None
    memory_limit_gib = None
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
    if "--memory-limit-gib" in args:
        memory_limit_gib = float(args[args.index("--memory-limit-gib") + 1])

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
                "dataset", "group", "matrix_dim", "matrix_nnz", "matrix_sparsity",
                "tensor_b_nnz", "tensor_b_sparsity", "mode", "cores",
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
                dim = entry["num_rows"]
                matrix_nnz = entry["nnz"]
                matrix_density = matrix_nnz / (dim * dim)
                matrix_sparsity = 1.0 - matrix_density

                # B, C, D, and dense output A are all simultaneously
                # resident (see module docstring) — estimate the combined
                # peak, computed from metadata alone, before downloading
                # anything.
                tensor_b_density = max(matrix_density, TENSOR_B_DENSITY_FLOOR)
                expected_tensor_b_nnz = tensor_b_density * dim ** 3
                tensor_b_bytes = expected_tensor_b_nnz * TENSOR_B_BYTES_PER_NNZ
                tensor_c_bytes = matrix_nnz * TENSOR_2D_BYTES_PER_NNZ
                tensor_d_bytes = matrix_nnz * TENSOR_2D_BYTES_PER_NNZ
                dense_a_bytes = dim * dim * 8
                estimated_peak_bytes = tensor_b_bytes + tensor_c_bytes + tensor_d_bytes + dense_a_bytes

                if estimated_peak_bytes > memory_budget_bytes:
                    print(f"  [skip] estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                          f"(B={tensor_b_bytes / 1024**3:.1f} C={tensor_c_bytes / 1024**3:.1f} "
                          f"D={tensor_d_bytes / 1024**3:.1f} A={dense_a_bytes / 1024**3:.1f} GiB) "
                          f"> budget {memory_budget_bytes / 1024**3:.1f} GiB — skipping {name}")
                    continue

                m_tns, downloaded_dir = download_matrix(name, entry)
                if m_tns is None:
                    continue
                shutil.copyfile(m_tns, TENSOR_C)
                shutil.copyfile(m_tns, TENSOR_D)

                tensor_b_sparsity = 1.0 - tensor_b_density
                tensor_b_nnz = generate_tensor_b(TENSOR_B, dim, tensor_b_sparsity)
                print(f"  [tensor_B] shape=({dim},{dim},{dim}) nnz={tensor_b_nnz} "
                      f"sparsity={tensor_b_sparsity:.6g} (matrix_sparsity={matrix_sparsity:.6g})")

                run_cores = cores if mode == "multicore" else None

                print("  [run] splyce phase_001 ...")
                splyce_times, splyce_timed_out = run_binary(splyce_bin, timeout, run_cores)
                print(f"{splyce_times}")

                if splyce_timed_out:
                    print("  [skip] splyce timed out — skipping baseline run")
                    scf_times, scf_med = None, "SKIPPED"
                else:
                    print("  [run] baseline (scf) ...")
                    scf_times, _ = run_binary(baseline_bin, timeout * 5, run_cores)
                    print(f"{scf_times}")
                    scf_med = median_excl_first(scf_times) if scf_times else "NA"

                for i, t in enumerate(splyce_times or []):
                    raw_writer.writerow([name, "splyce_phase_001", i, t])
                for i, t in enumerate(scf_times or []):
                    raw_writer.writerow([name, "scf", i, t])
                rf.flush()

                splyce_med = median_excl_first(splyce_times) if splyce_times else "NA"

                if splyce_med != "NA":
                    summary_writer.writerow([
                        name, entry["group"], dim, matrix_nnz, matrix_sparsity,
                        tensor_b_nnz, tensor_b_sparsity, mode, cores or "",
                        scf_med, splyce_med,
                    ])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            except Exception as e:
                print(f"  [error] {name}: {e} — skipping")
                continue
            finally:
                for f in (TENSOR_B, TENSOR_C, TENSOR_D):
                    if os.path.isfile(f):
                        os.remove(f)
                if downloaded_dir is not None and os.path.isdir(downloaded_dir):
                    shutil.rmtree(downloaded_dir)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
