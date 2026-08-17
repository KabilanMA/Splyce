#!/usr/bin/env python3
# convert_to_tns.py — Add this repo's "extended FROSTT format" header
# (# extended FROSTT format / <ndim> <nnz> / <dim1> ... <dimN>) to a raw
# FROSTT tensor download, which has no header at all — see
# download_data.py's module docstring.
#
# ndim/nnz/dims aren't declared anywhere in the raw file, so this makes one
# streaming pass over it (nnz = line count; each dimension's size = the max
# index seen in that column — FROSTT tensors are 1-indexed, and a
# dimension's declared size is its max index, not (max - min + 1), same
# convention gen_data.c/.py's own generators use). The data lines
# themselves are copied through unchanged (whitespace-delimited either
# way, so the raw file's actual delimiter doesn't need to match this
# repo's own space-delimited generators').
#
# Usage:
#   ./convert_to_tns.py <name>   # frostt/<name>/<name>_raw.tns -> <name>.tns

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_raw_to_tns(raw_path, tns_path):
    dims = None
    nnz = 0
    tmp_path = tns_path + ".tmp"

    with open(raw_path) as src, open(tmp_path, "w") as tmp:
        for line in src:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if dims is None:
                dims = [0] * (len(fields) - 1)
            for i in range(len(dims)):
                idx = int(fields[i])
                if idx > dims[i]:
                    dims[i] = idx
            tmp.write(" ".join(fields) + "\n")
            nnz += 1

    if dims is None:
        os.remove(tmp_path)
        raise ValueError(f"{raw_path}: no data lines found")

    with open(tns_path, "w") as out:
        out.write("# extended FROSTT format\n")
        out.write(f"{len(dims)} {nnz}\n")
        out.write(" ".join(str(d) for d in dims) + "\n")
        with open(tmp_path) as tmp:
            for line in tmp:
                out.write(line)
    os.remove(tmp_path)

    print(f"Converted {raw_path} -> {tns_path} | Shape: {tuple(dims)} | NNZ: {nnz}")
    return tuple(dims), nnz


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: ./convert_to_tns.py <name>")
    name = sys.argv[1]
    raw_path = os.path.join(SCRIPT_DIR, name, f"{name}_raw.tns")
    tns_path = os.path.join(SCRIPT_DIR, name, f"{name}.tns")
    if not os.path.isfile(raw_path):
        sys.exit(f"error: {raw_path} not found — run download_data.py first")
    convert_raw_to_tns(raw_path, tns_path)
    os.remove(raw_path)


if __name__ == "__main__":
    main()
