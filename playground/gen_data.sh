#!/usr/bin/env bash
# gen_data.sh — generate this directory's tensor_B.tns/tensor_C.tns
# (square sparse 2D matrices), via gen_data.py.
#
# Usage: ./gen_data.sh [dimension] [sparsity]
#   dimension   Square matrix dimension for both tensors (default: 5000)
#   sparsity    Fraction of zero entries, in [0, 1) (default: 0.95)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/gen_data.py" "$@"
