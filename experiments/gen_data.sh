#!/usr/bin/env bash
# gen_data.sh — wrapper used by every experiment's data-generation step in
# place of calling gen_data.py or the gen_data C binary directly.
#
# Builds (and caches) gen_data.c into ./gen_data, rebuilding whenever the
# source is newer than the cached binary, then execs it. If no C compiler
# is available or the build fails, falls back to gen_data.py so the
# pipeline still works on a machine without a toolchain.
#
# Usage: ./gen_data.sh <experiment_name> [generator_args...]
#        (same CLI as gen_data.py / gen_data.c — must be run with cwd set
#        to experiments/, since output paths are relative to it)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/gen_data.c"
BIN="$SCRIPT_DIR/gen_data"

fallback() {
  echo "gen_data.sh: falling back to gen_data.py" >&2
  exec python3 "$SCRIPT_DIR/gen_data.py" "$@"
}

if ! command -v cc >/dev/null 2>&1; then
  fallback "$@"
fi

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  if ! cc -O2 -o "$BIN" "$SRC" -lm; then
    echo "gen_data.sh: build of gen_data.c failed" >&2
    fallback "$@"
  fi
fi

exec "$BIN" "$@"
