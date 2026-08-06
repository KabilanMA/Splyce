import csv
import sys

PHASES = ["000", "001", "010", "011", "100", "101", "110", "111"]
ROW_ORDER = ["spgemm_scf"] + PHASES + [f"{p}_fastmath" for p in PHASES]

def print_table(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"{path} is empty")
        return

    for row in rows[1:]:
        if row:
            row[0] = row[0].replace("spgemm_splyce_phase_", "")

    rows[1:] = sorted(
        rows[1:],
        key=lambda row: ROW_ORDER.index(row[0]) if row and row[0] in ROW_ORDER else len(ROW_ORDER),
    )

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def print_row(row):
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    print_row(rows[0])
    print("  ".join("-" * w for w in widths))
    for row in rows[1:]:
        print_row(row)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./tma_results.csv"
    print_table(path)

if __name__ == "__main__":
    main()
