#!/usr/bin/env python3
# convert_to_tns.py — Convert every downloaded SuiteSparse Matrix Market
# (.mtx) dataset under this directory into the FROSTT-style .tns format
# that gen_data.py (and every kernel's compile.sh/run.sh) expects, e.g.:
#
#   # extended FROSTT format
#   2 <nnz>
#   <rows> <cols>
#   <row> <col> <val>
#   ...
#
# Each subdirectory of suitesparse/ is expected to hold one downloaded
# matrix (e.g. suitesparse/olm2000/olm2000.mtx, straight from the
# SuiteSparse Matrix Collection). Some directories come with extra files
# (other .mtx variants, READMEs, extracted junk) alongside the one that
# actually matters — the file named "<directory>.mtx" is always the one
# whose data we want. For each subdirectory, this script:
#
#   1. Locates "<directory>.mtx" (falling back to a lone *.mtx if no file
#      matches the directory name exactly).
#   2. Removes everything else in that subdirectory — any other file or
#      folder that isn't the chosen .mtx or the .tns this script produces
#      is deleted, regardless of what it is.
#   3. Parses the Matrix Market banner/header, strips all "%" comment
#      lines, expands symmetric/skew-symmetric matrices into their
#      explicit general form (mirroring off-diagonal entries), fills in
#      an implicit value of 1.0 for pattern matrices, and re-emits the
#      data with the header reordered to match gen_data.py's tns pattern.
#   4. Writes "<name>.tns" into the subdirectory.
#   5. Deletes the source .mtx — once the .tns exists, the data has been
#      carried over and the original download is no longer needed.
#
# Usage:
#   ./convert_to_tns.py            # convert every subdirectory
#   ./convert_to_tns.py olm2000 ... # convert only the named subdirectories

import os
import sys
import glob
import shutil


def drop_explicit_zeros(entries):
    # Some matrices (e.g. FEM assembly matrices) store explicit zeros to
    # preserve a fixed sparsity pattern — these are stored entries per the
    # .mtx header's declared count, but aren't real nonzeros, so
    # SuiteSparse's own nnz metadata excludes them. Comment out the call to
    # this function in convert_mtx_to_tns to keep explicit zeros instead of
    # dropping them.
    kept = [(r, c, val) for r, c, val in entries if val != 0.0]
    dropped = len(entries) - len(kept)
    return kept, dropped


def _read_mtx_header(f, mtx_path):
    first_line = f.readline()
    if not first_line.startswith("%%MatrixMarket"):
        raise ValueError(f"{mtx_path}: missing %%MatrixMarket banner")

    banner = first_line.lower()
    if "coordinate" not in banner:
        raise ValueError(
            f"{mtx_path}: only 'coordinate' (sparse) matrices are supported, "
            f"found: {first_line.strip()}"
        )
    if "complex" in banner or "hermitian" in banner:
        raise ValueError(f"{mtx_path}: complex/hermitian matrices are not supported")

    is_pattern = "pattern" in banner
    is_skew = "skew-symmetric" in banner
    is_symmetric = (not is_skew) and "symmetric" in banner

    line = f.readline()
    while line.startswith("%"):
        line = f.readline()
    rows, cols, nnz_declared = (int(t) for t in line.split())

    return rows, cols, nnz_declared, is_pattern, is_skew, is_symmetric


def convert_mtx_to_tns(mtx_path, tns_path):
    # Large matrices (e.g. GAP-web) can have hundreds of millions of
    # entries — buffering the whole file (or an expanded entries list) in
    # memory gets OOM-killed. Stream instead: one pass to count the final
    # (post-symmetric-expansion) entry total, then a second pass to stream
    # data lines straight to the .tns file.
    with open(mtx_path) as f:
        rows, cols, nnz_declared, is_pattern, is_skew, is_symmetric = _read_mtx_header(
            f, mtx_path
        )

        raw_count = 0
        total_count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            r, c = int(parts[0]), int(parts[1])
            raw_count += 1
            total_count += 1
            if (is_symmetric or is_skew) and r != c:
                total_count += 1

    if raw_count != nnz_declared:
        raise ValueError(
            f"{mtx_path}: expected {nnz_declared} stored entries, found {raw_count}"
        )

    dropped_zeros = 0
    # explicit-zero dropping (drop_explicit_zeros) isn't supported by this
    # streaming path — it's already disabled by default (see comment above
    # drop_explicit_zeros' definition).

    with open(mtx_path) as f, open(tns_path, "w") as out:
        _read_mtx_header(f, mtx_path)
        out.write("# extended FROSTT format\n")
        out.write(f"2 {total_count}\n")
        out.write(f"{rows} {cols}\n")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            r, c = int(parts[0]), int(parts[1])
            val = 1.0 if is_pattern else float(parts[2])
            out.write(f"{r} {c} {val:.4f}\n")
            if (is_symmetric or is_skew) and r != c:
                mirrored_val = -val if is_skew else val
                out.write(f"{c} {r} {mirrored_val:.4f}\n")

    # Guard against the header/body ever drifting apart (e.g. a future
    # edit to the write loop above): stream back through the file and
    # confirm the declared nnz on line 2 matches the number of data lines
    # that follow.
    with open(tns_path) as f:
        f.readline()
        header_line = f.readline()
        f.readline()
        written_nnz = int(header_line.split()[1])
        written_data_lines = sum(1 for _ in f)
    if written_nnz != written_data_lines or written_nnz != total_count:
        raise ValueError(
            f"{tns_path}: header declares nnz={written_nnz} but wrote "
            f"{written_data_lines} data lines ({total_count} entries computed)"
        )

    return rows, cols, total_count, dropped_zeros


def process_dir(d):
    name = os.path.basename(d.rstrip("/"))
    tns_path = os.path.join(d, f"{name}.tns")

    named_match = os.path.join(d, f"{name}.mtx")
    if os.path.isfile(named_match):
        mtx_path = named_match
    else:
        mtx_candidates = glob.glob(os.path.join(d, "*.mtx"))
        if not mtx_candidates:
            if os.path.isfile(tns_path):
                print(f"[skip] {name}: already converted, no .mtx left")
            else:
                print(f"[skip] {name}: no .mtx file found")
            return
        if len(mtx_candidates) > 1:
            print(
                f"[skip] {name}: multiple .mtx files found and none named "
                f"'{name}.mtx', expected exactly one: {mtx_candidates}"
            )
            return
        mtx_path = mtx_candidates[0]

    keep = {os.path.basename(mtx_path), os.path.basename(tns_path)}
    for entry in sorted(os.listdir(d)):
        if entry in keep:
            continue
        full = os.path.join(d, entry)
        if os.path.isdir(full):
            print(f"[clean] {name}: removing unwanted directory {entry}/")
            shutil.rmtree(full)
        else:
            print(f"[clean] {name}: removing unwanted file {entry}")
            os.remove(full)

    print(f"[convert] {name}: {os.path.basename(mtx_path)} -> {os.path.basename(tns_path)}")
    rows, cols, nnz, dropped_zeros = convert_mtx_to_tns(mtx_path, tns_path)
    print(f"          shape=({rows}, {cols}) nnz={nnz}")
    if dropped_zeros:
        print(f"          dropped {dropped_zeros} explicit-zero entries")

    os.remove(mtx_path)
    print(f"[clean] {name}: removing source {os.path.basename(mtx_path)} (data now in {os.path.basename(tns_path)})")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names = sys.argv[1:]

    if names:
        dirs = [os.path.join(script_dir, n) for n in names]
    else:
        dirs = sorted(
            os.path.join(script_dir, e)
            for e in os.listdir(script_dir)
            if os.path.isdir(os.path.join(script_dir, e))
        )

    for d in dirs:
        if not os.path.isdir(d):
            print(f"[skip] {d}: not a directory")
            continue
        process_dir(d)


if __name__ == "__main__":
    main()
