#!/usr/bin/env python3
# download_data.py — Download and extract SuiteSparse matrices by name.
#
# Looks each name up in matrix_metadata.json (produced by
# scrape_metadata.py) to find its group, downloads
#   https://suitesparse-collection-website.herokuapp.com/MM/<group>/<name>.tar.gz
# and extracts it right here, so the result is the same
# suitesparse/<name>/<name>.mtx (+ companion files) layout that
# convert_to_tns.py expects as input.
#
# Usage:
#   ./download_data.py <name> [<name> ...]
#   ./download_data.py --force <name> ...   # re-download even if
#                                            # suitesparse/<name>/ already
#                                            # exists
#
#   ./download_data.py --sample-per-group [--force]
#       Downloading all ~2900 matrices isn't practical, so this instead
#       picks one square matrix per group (169 groups have at least one
#       square matrix) — specifically the median-nnz square matrix within
#       each group — giving one representative, runnable sample per group.
#   ./download_data.py --sample-per-group --list
#       Print the selected names/groups/nnz without downloading anything.

import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(SCRIPT_DIR, "matrix_metadata.json")
DOWNLOAD_URL = "https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"


def load_metadata():
    if not os.path.isfile(METADATA_PATH):
        sys.exit(f"error: {METADATA_PATH} not found — run scrape_metadata.py first")
    with open(METADATA_PATH) as f:
        data = json.load(f)
    return {m["name"]: m for m in data["matrices"]}


def download_and_extract(name, group, force):
    dest_dir = os.path.join(SCRIPT_DIR, name)
    if os.path.isdir(dest_dir) and not force:
        print(f"[skip] {name}: {dest_dir} already exists (use --force to re-download)")
        return

    url = DOWNLOAD_URL.format(group=group, name=name)
    print(f"[download] {name}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            archive_bytes = resp.read()
    except urllib.error.HTTPError as e:
        print(f"[error] {name}: download failed ({e.code} {e.reason})")
        return
    except urllib.error.URLError as e:
        print(f"[error] {name}: download failed ({e.reason})")
        return

    print(f"[extract] {name}: -> {dest_dir}")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tf:
        try:
            tf.extractall(SCRIPT_DIR, filter="data")
        except TypeError:
            # Python < 3.12 without the PEP 706 'filter' backport.
            tf.extractall(SCRIPT_DIR)

    if not os.path.isdir(dest_dir):
        print(f"[warning] {name}: extraction finished but {dest_dir} not found — archive layout may differ")


def median_square_matrix_per_group(metadata):
    """One matrix per group: among that group's square matrices, the
    median one by nnz (lower median if the group has an even count)."""
    by_group = {}
    for entry in metadata.values():
        if entry["num_rows"] == entry["num_cols"]:
            by_group.setdefault(entry["group"], []).append(entry)

    selected = []
    for group, entries in sorted(by_group.items()):
        entries.sort(key=lambda e: (e["nnz"], e["name"]))
        selected.append(entries[(len(entries) - 1) // 2])
    return selected


def main():
    args = sys.argv[1:]
    force = "--force" in args
    sample_per_group = "--sample-per-group" in args
    list_only = "--list" in args
    names = [a for a in args if a not in ("--force", "--sample-per-group", "--list")]

    metadata = load_metadata()

    if sample_per_group:
        selected = median_square_matrix_per_group(metadata)
        print(f"Selected {len(selected)} matrices (median-nnz square matrix per group)")
        if list_only:
            for entry in selected:
                print(f"  {entry['group']}/{entry['name']}  nnz={entry['nnz']}")
            return
        names = [entry["name"] for entry in selected]

    if not names:
        sys.exit(
            "Usage: ./download_data.py [--force] <name> [<name> ...]\n"
            "       ./download_data.py [--force] --sample-per-group [--list]"
        )

    for name in names:
        entry = metadata.get(name)
        if entry is None:
            print(f"[error] {name}: not found in {os.path.basename(METADATA_PATH)}")
            continue
        download_and_extract(name, entry["group"], force)


if __name__ == "__main__":
    main()
