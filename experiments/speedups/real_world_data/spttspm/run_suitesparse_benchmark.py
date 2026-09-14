#!/usr/bin/env python3
# run_suitesparse_benchmark.py — Real-world SpTTSpM benchmark, curated set.
#
# spttspm computes A(i,j,r) = Σ_k B(i,j,k) · C(k,r) (see spttspm_dn.mlir).
# Each job in JOBS below defines which SuiteSparse matrix supplies
# tensor_C — B is always synthetic, since there's no natural SuiteSparse
# source for a 3D tensor. There's no "b" key — only C can ever be real.
#
# i and j are never constrained by anything (nothing besides B references
# them), so they always use a fixed FREE_DIM (500). k and r are both C's
# own dimension when C is real (r appears only in C and the output; k is
# shared between B and C) — whichever real matrix supplies C must be
# square (per matrix_metadata.json), so its one dimension covers both.
# There's no case where C is absent: unlike spmttkrp/spmmh (which each
# have two possible real-operand slots and can fall back on one when the
# other is missing), spttspm has only one — a job with no real C has
# nothing to derive k/r or a reference density from, so it's skipped.
#
# Synthetic tensor_B is generated at density = max(C's own density,
# DENSITY_FLOOR (0.001%)) — same floor-at-the-real-matrix's-own-density
# rule ../spmttkrp/run_suitesparse_sweep.py and
# ../spmttkrp/run_suitesparse_benchmark.py use.
#
# barth4 (the default job's matrix) also exists in the Pothen group (a
# different, symmetric 40965-nnz matrix, also 6019x6019, distinct from the
# Nasa "duplicate structural problem" one this uses) —
# suitesparse/download_data.py's load_metadata() dedupes by name alone and
# would silently resolve to whichever entry appears last in
# matrix_metadata.json, so JOBS entries needing a specific group set an
# explicit "group" key (load_matrix_entry looks the raw list up by
# (name, group) directly instead of going through that by-name dict).
# --matrix doesn't support this disambiguation — add a JOBS entry instead
# if you need a specific group for an ambiguous name.
#
# For each job, this:
#   1. Resolves k/r + the target synthetic density from metadata alone
#      (matrix_metadata.json — run suitesparse/scrape_metadata.py first if
#      that file doesn't exist yet), and estimates the combined peak memory
#      B + C + dense A would need (see MEMORY_SAFETY_FRACTION; override
#      with --memory-limit-gib) — skipping the job before downloading
#      anything if that exceeds the budget. Unlike spgemm, there's no
#      sparse-output binary for spttspm to fall back to (A is inherently a
#      dense 3D tensor here — i x j x r, which can get very large if r,
#      tied to a real matrix's own dimension, is big), so this just skips.
#   2. Downloads + converts C (suitesparse/download_data.py +
#      convert_to_tns.py — drops explicit zeros, mirrors symmetric
#      entries).
#   3. Generates synthetic tensor_B (generate_sparse_3d_tns below — ported
#      from experiments/gen_data.py's function of the same name).
#   4. Runs test_benchmark_spttspm_splyce_phase_001 FIRST, then
#      test_benchmark_spttspm_scf — but only if Splyce didn't time out. If
#      Splyce already hit the timeout, the (unvectorized, typically no
#      faster) baseline is skipped entirely rather than wasting the same
#      timeout on a run that's essentially guaranteed to also be too slow.
#      Each binary loops 6 iterations internally per spttspm_dn.mlir's
#      @main, writing a "benchmark" file with one time per line.
#   5. Appends one summary row (dataset, group, b/c shape, b_nnz, c_nnz,
#      target density, scf_median, splyce_median — median of the 5
#      non-cold-start iterations; scf_median is "SKIPPED" when Splyce
#      timed out) to spttspm_realworld_results.csv, and every individual
#      raw iteration time to spttspm_realworld_raw_runtimes.csv as a
#      backup.
#   6. Deletes tensor_B.tns/tensor_C.tns and the downloaded/converted
#      suitesparse/<name>/ directory afterward.
#
# Both CSVs are appended to, not overwritten, so a re-run is a no-op once a
# job is already recorded (unless the CSV row is removed first).
#
# Prerequisite: ./compile.sh has already been run, so
# test_benchmark_spttspm_scf and test_benchmark_spttspm_splyce_phase_001
# exist in this directory. There are no *_parallel variants for spttspm
# (compile.sh doesn't build any), so this has no --mode/--cores, unlike
# some of its siblings.
#
# Usage:
#   ./run_suitesparse_benchmark.py             # every job in JOBS
#   ./run_suitesparse_benchmark.py --matrix cat_ears_2_1
#       Runs that one SuiteSparse matrix for tensor_C instead of JOBS —
#       must be square. Ignores JOBS/--limit/group-disambiguation
#       entirely.
#   ./run_suitesparse_benchmark.py --limit 2    # only the first 2 (testing)
#   ./run_suitesparse_benchmark.py --timeout 600  # per-binary-run timeout
#                                                  # in seconds (default 300)
#   ./run_suitesparse_benchmark.py --memory-limit-gib 64   # override the
#       auto-detected memory budget (see MEMORY_SAFETY_FRACTION)

import csv
import json
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

BASELINE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spttspm_scf")
SPLYCE_BIN = os.path.join(SCRIPT_DIR, "test_benchmark_spttspm_splyce_phase_001")

SUMMARY_CSV = os.path.join(SCRIPT_DIR, "spttspm_realworld_results.csv")
RAW_BACKUP_CSV = os.path.join(SCRIPT_DIR, "spttspm_realworld_raw_runtimes.csv")

TENSOR_B = os.path.join(SCRIPT_DIR, "tensor_B.tns")
TENSOR_C = os.path.join(SCRIPT_DIR, "tensor_C.tns")
BENCHMARK_FILE = os.path.join(SCRIPT_DIR, "benchmark")

# Floor for synthetic tensor_B's density (0.001%) — see module docstring.
DENSITY_FLOOR = 0.001 / 100

# Dimension for i/j — never constrained by anything (see module
# docstring).
FREE_DIM = 500

# The curated jobs this script runs — see module docstring for why barth4
# was chosen, and why its group is pinned explicitly. "group" is optional
# (only needed to disambiguate an otherwise-ambiguous name).
JOBS = [
    {"name": "barth4", "c": "barth4", "group": "Nasa"},
]

# Estimated peak bytes/nonzero once a sparse tensor is loaded by the MLIR
# sparse tensor runtime, which briefly holds a full COO intermediate
# alongside the final level-format storage before freeing the COO (see the
# sparse_tensor reader trace from the FROSTT-loading investigation) —
# roughly 2x the raw coordinate size:
#   tensor_B, 3D (3 coords + 1 value, 8 bytes each) * 2 = 64 bytes/nnz
#   tensor_C, 2D (2 coords + 1 value, 8 bytes each) * 2 = 48 bytes/nnz
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


def load_matrix_entry(name, group):
    # suitesparse/download_data.py's load_metadata() dedupes by name alone,
    # which would silently pick the wrong entry for a name that exists in
    # more than one group (see module docstring) — so this reads the raw
    # metadata list directly and matches on (name, group).
    with open(dl.METADATA_PATH) as f:
        data = json.load(f)
    for m in data["matrices"]:
        if m["name"] == name and m["group"] == group:
            return m
    return None


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
    memory_limit_gib = None
    matrix_name = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--timeout" in args:
        timeout = int(args[args.index("--timeout") + 1])
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

    if not (os.path.isfile(BASELINE_BIN) and os.path.isfile(SPLYCE_BIN)):
        sys.exit("error: binaries not found — run ./compile.sh first")

    metadata = dl.load_metadata()

    if matrix_name is not None:
        entry = metadata.get(matrix_name)
        if entry is None:
            sys.exit(f"error: '{matrix_name}' not found in matrix_metadata.json")
        if entry["num_rows"] != entry["num_cols"]:
            sys.exit(f"error: '{matrix_name}' is {entry['num_rows']}x{entry['num_cols']}, not square")
        jobs = [{"name": matrix_name, "c": matrix_name, "group": None}]
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
                "dataset", "group", "b_shape", "c_shape",
                "b_nnz", "c_nnz", "target_density_pct",
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
                c_name = job["c"]
                c_group = job.get("group")

                if c_name is None:
                    print(f"  [skip] {name}: job specifies no real matrix for C "
                          f"(need one to derive k/r and a reference density)")
                    continue

                if c_group is not None:
                    entry = load_matrix_entry(c_name, c_group)
                    if entry is None:
                        print(f"  [skip] {name}: {c_group}/{c_name} not found in matrix_metadata.json")
                        continue
                else:
                    entry = metadata.get(c_name)
                    if entry is None:
                        print(f"  [skip] {name}: '{c_name}' not found in matrix_metadata.json")
                        continue
                group = entry["group"]

                if entry["num_rows"] != entry["num_cols"]:
                    print(f"  [skip] {name}: '{c_name}' is {entry['num_rows']}x{entry['num_cols']}, not square")
                    continue

                # k and r are both C's own dimension (C is square); i and j
                # are always free (see module docstring).
                i_dim = FREE_DIM
                j_dim = FREE_DIM
                k_dim = entry["num_rows"]
                r_dim = entry["num_rows"]

                c_nnz = entry["nnz"]
                c_density = c_nnz / (k_dim * r_dim)
                target_density = max(c_density, DENSITY_FLOOR)
                target_sparsity = 1.0 - target_density

                # B and C, AND the dense output A, are all simultaneously
                # resident (see module docstring) — estimate the combined
                # peak before downloading anything. A is i x j x r and
                # fully dense — the term that actually needs watching here,
                # since r scales with C's own (possibly huge) dimension.
                expected_b_nnz = target_density * (i_dim * j_dim * k_dim)
                tensor_b_bytes = expected_b_nnz * TENSOR_B_BYTES_PER_NNZ
                tensor_c_bytes = c_nnz * TENSOR_2D_BYTES_PER_NNZ
                dense_a_bytes = i_dim * j_dim * r_dim * 8
                estimated_peak_bytes = tensor_b_bytes + tensor_c_bytes + dense_a_bytes

                if estimated_peak_bytes > memory_budget_bytes:
                    print(f"  [skip] estimated peak memory {estimated_peak_bytes / 1024**3:.1f} GiB "
                          f"(B={tensor_b_bytes / 1024**3:.1f} C={tensor_c_bytes / 1024**3:.1f} "
                          f"A={dense_a_bytes / 1024**3:.1f} GiB) "
                          f"> budget {memory_budget_bytes / 1024**3:.1f} GiB — skipping {name}")
                    continue

                print(f"=== {group}/{c_name} (nnz={c_nnz}) ===")

                tns_path, ddir = download_matrix(c_name, entry)
                if ddir is not None:
                    downloaded_dirs.append(ddir)
                if tns_path is None:
                    continue

                shutil.copyfile(tns_path, TENSOR_C)

                print(f"  [generate] tensor_B ({i_dim} x {j_dim} x {k_dim}) @ target_density={target_density:.6g} ...")
                b_nnz = generate_sparse_3d_tns(TENSOR_B, i_dim, j_dim, k_dim, target_sparsity)

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
                        name, group, f"{i_dim}x{j_dim}x{k_dim}", f"{k_dim}x{r_dim}",
                        b_nnz, c_nnz, target_density * 100,
                        scf_med, splyce_med,
                    ])
                sf.flush()
                print(f"  scf_median={scf_med}  splyce_median={splyce_med}")

            finally:
                for f in (TENSOR_B, TENSOR_C):
                    if os.path.isfile(f):
                        os.remove(f)
                for d in downloaded_dirs:
                    if os.path.isdir(d):
                        shutil.rmtree(d)

    print(f"Done. Summary: {SUMMARY_CSV}")
    print(f"Raw backup: {RAW_BACKUP_CSV}")


if __name__ == "__main__":
    main()
