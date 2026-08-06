#!/usr/bin/env python3
# scrape_metadata.py — Fetch metadata for every matrix in the SuiteSparse
# Matrix Collection and write it to a single JSON file.
#
# The collection website (suitesparse-collection-website.herokuapp.com)
# has no per-matrix JSON API, but its /files/ route redirects to a single
# machine-readable stats file (ssstats.csv) on the underlying file host
# that covers every matrix in the database in one shot — far simpler and
# more reliable than scraping ~2900 individual matrix pages. This script
# downloads that file and reshapes it into a single JSON document.
#
# Usage:
#   ./scrape_metadata.py [output.json]   # defaults to matrix_metadata.json

import csv
import json
import sys
import urllib.request

STATS_URL = "https://suitesparse-collection-website.herokuapp.com/files/ssstats.csv"

# Column order as published in ssstats.csv.
FIELDS = [
    "group",
    "name",
    "num_rows",
    "num_cols",
    "nnz",
    "is_real",
    "is_binary",
    "is_2d_or_3d",
    "positive_definite",
    "pattern_symmetry",
    "numeric_symmetry",
    "kind",
    "nnz_pattern_symmetrized",
]

INT_FIELDS = ("num_rows", "num_cols", "nnz", "nnz_pattern_symmetrized")
BOOL_FIELDS = ("is_real", "is_binary", "is_2d_or_3d", "positive_definite")
FLOAT_FIELDS = ("pattern_symmetry", "numeric_symmetry")


def fetch_lines(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8").splitlines()


def parse(lines):
    declared_count = int(lines[0].strip())
    generated = lines[1].strip()

    matrices = []
    for row in csv.reader(lines[2:]):
        if not row:
            continue
        entry = dict(zip(FIELDS, row))
        for key in INT_FIELDS:
            entry[key] = int(entry[key])
        for key in BOOL_FIELDS:
            entry[key] = bool(int(entry[key]))
        for key in FLOAT_FIELDS:
            entry[key] = float(entry[key])
        matrices.append(entry)

    if len(matrices) != declared_count:
        print(
            f"warning: header declared {declared_count} matrices, parsed {len(matrices)}",
            file=sys.stderr,
        )

    return {
        "source": STATS_URL,
        "generated": generated,
        "count": len(matrices),
        "matrices": matrices,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "matrix_metadata.json"

    print(f"Fetching {STATS_URL} ...")
    lines = fetch_lines(STATS_URL)

    data = parse(lines)
    print(f"Parsed {data['count']} matrices (dataset generated {data['generated']})")

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
