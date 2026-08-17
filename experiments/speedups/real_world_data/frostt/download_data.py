#!/usr/bin/env python3
# download_data.py — Download and extract named tensors from the FROSTT
# (Formidable Repository of Open Sparse Tensors and Tools, frostt.io)
# collection.
#
# Unlike suitesparse/download_data.py, there's no bulk metadata file to
# look names up in — FROSTT doesn't publish one machine-readable index the
# way the SuiteSparse Matrix Collection's ssstats.csv does, so each tensor
# this repo knows about is hardcoded below by name.
#
# Each FROSTT .tns.gz is actually a gzipped *tar* archive (not a plain
# gzipped text file) that also carries a macOS "AppleDouble" resource-fork
# entry (a second, junk "._<name>.tns" member alongside the real one) —
# both are true of 1998DARPA.tns.gz at least, so this assumes it's true of
# any future entry added here too, and extracts only the real member.
#
# The raw member has no header at all (unlike this repo's own .tns
# convention) — just whitespace-separated "<idx1> ... <idxN> <val>" rows,
# one per nonzero, with no declared ndim/nnz/dims. convert_to_tns.py adds
# that header; this module only downloads + extracts the raw file.
#
# Usage:
#   ./download_data.py <name> [<name> ...]
#   ./download_data.py --force <name> ...   # re-download even if
#                                            # frostt/<name>/ already exists
#   ./download_data.py --list               # print known tensor names

import os
import sys
import tarfile
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# name -> (download URL, real member name inside the gzipped tar).
FROSTT_TENSORS = {
    "darpa": (
        "https://frostt-tensors.s3.us-east-2.amazonaws.com/1998DARPA/1998darpa.tns.gz",
        "1998DARPA.tns",
    ),
}


def raw_tns_path(name):
    return os.path.join(SCRIPT_DIR, name, f"{name}_raw.tns")


def download_and_extract(name, force=False):
    entry = FROSTT_TENSORS.get(name)
    if entry is None:
        sys.exit(f"error: '{name}' not found — known tensors: {', '.join(FROSTT_TENSORS)}")
    url, member = entry

    dest_dir = os.path.join(SCRIPT_DIR, name)
    dest_path = raw_tns_path(name)
    if os.path.isfile(dest_path) and not force:
        print(f"[skip] {name}: {dest_path} already exists (use --force to re-download)")
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)

    # FROSTT archives can be multiple GB (even compressed) — buffering the
    # whole response body in memory (e.g. via resp.read()) risks an OOM
    # kill, so this streams straight to a temp file on disk instead, and
    # tarfile then reads/decompresses that file incrementally rather than
    # from an in-memory buffer.
    archive_path = os.path.join(dest_dir, f"{name}_archive.tar.gz")
    print(f"[download] {name}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(archive_path, "wb") as archive:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                archive.write(chunk)
    except urllib.error.HTTPError as e:
        sys.exit(f"[error] {name}: download failed ({e.code} {e.reason})")
    except urllib.error.URLError as e:
        sys.exit(f"[error] {name}: download failed ({e.reason})")

    print(f"[extract] {name}: {member} -> {dest_path}")
    try:
        with tarfile.open(archive_path, mode="r:gz") as tf:
            try:
                src = tf.extractfile(member)
            except KeyError:
                sys.exit(f"[error] {name}: expected member '{member}' not found in archive "
                          f"(archive contains: {tf.getnames()})")
            if src is None:
                sys.exit(f"[error] {name}: '{member}' is not a regular file in the archive")
            with open(dest_path, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    finally:
        os.remove(archive_path)

    return dest_path


def main():
    args = sys.argv[1:]
    force = "--force" in args
    list_only = "--list" in args
    names = [a for a in args if a not in ("--force", "--list")]

    if list_only:
        for name in FROSTT_TENSORS:
            print(name)
        return

    if not names:
        sys.exit("Usage: ./download_data.py [--force] <name> [<name> ...]\n"
                  "       ./download_data.py --list")

    for name in names:
        download_and_extract(name, force)


if __name__ == "__main__":
    main()
