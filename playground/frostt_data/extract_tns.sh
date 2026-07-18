#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 <path/to/tensor.tns.gz>"
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

input_file="$1"

if [[ ! -f "$input_file" ]]; then
    echo "Error: input file does not exist: $input_file" >&2
    exit 1
fi

if [[ "$input_file" != *.tns.gz ]]; then
    echo "Error: input file must end with .tns.gz" >&2
    exit 1
fi

for command_name in gzip tar awk find mktemp sort head cut wc; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command is unavailable: $command_name" >&2
        exit 1
    fi
done

input_basename="$(basename "$input_file")"
tensor_name="${input_basename%.gz}"
tensor_stem="${tensor_name%.tns}"

output_file="./${tensor_stem}-extended.tns"

temp_dir="$(mktemp -d "./.tns_conversion.XXXXXX")"
decompressed_payload="$temp_dir/decompressed_payload"
extract_dir="$temp_dir/extracted"
clean_coordinate_file="$temp_dir/coordinates.tns"
metadata_file="$temp_dir/metadata.txt"

cleanup() {
    rm -rf "$temp_dir"
}

trap cleanup EXIT INT TERM

echo "Input:  $input_file"
echo "Output: $output_file"
echo

if [[ -e "$output_file" ]]; then
    echo "Error: output file already exists: $output_file" >&2
    exit 1
fi

echo "Checking gzip file..."

if ! gzip -t "$input_file"; then
    echo "Error: invalid or corrupted gzip file: $input_file" >&2
    exit 1
fi

echo "Decompressing..."
gzip -dc "$input_file" > "$decompressed_payload"

if tar -tf "$decompressed_payload" >/dev/null 2>&1; then
    echo "Detected a tar archive inside the gzip file."
    echo "Extracting archive..."

    mkdir -p "$extract_dir"

    # The warning messages about LIBARCHIVE.xattr are harmless.
    tar -xf "$decompressed_payload" -C "$extract_dir"

    mapfile -d '' tensor_files < <(
        find "$extract_dir" \
            -type f \
            -iname "*.tns" \
            ! -name "._*" \
            ! -path "*/__MACOSX/*" \
            -print0
    )

    if [[ ${#tensor_files[@]} -eq 0 ]]; then
        echo "Error: no valid .tns file found inside archive." >&2
        exit 1
    fi

    if [[ ${#tensor_files[@]} -gt 1 ]]; then
        echo "Multiple .tns files found. Selecting the largest one."

        source_tns="$(
            for file in "${tensor_files[@]}"; do
                printf '%s\t%s\n' "$(wc -c < "$file")" "$file"
            done |
                sort -nr |
                head -n 1 |
                cut -f2-
        )"
    else
        source_tns="${tensor_files[0]}"
    fi

    echo "Selected: ${source_tns#"$extract_dir"/}"
else
    echo "The decompressed payload is already a tensor text file."
    source_tns="$decompressed_payload"
fi

first_nonempty="$(
    awk 'NF > 0 { print; exit }' "$source_tns"
)"

if [[ "$first_nonempty" == "# extended FROSTT format"* ]]; then
    echo "The file already has an extended FROSTT header."
    cp "$source_tns" "$output_file"

    echo
    echo "Created: $output_file"
    head -n 8 "$output_file"
    exit 0
fi

echo "Preparing coordinate data..."

awk '
NF == 0 {
    next
}

$1 ~ /^#/ {
    next
}

{
    print
}
' "$source_tns" > "$clean_coordinate_file"

if [[ ! -s "$clean_coordinate_file" ]]; then
    echo "Error: no coordinate rows found in the tensor file." >&2
    exit 1
fi

echo "Calculating tensor metadata..."

awk '
BEGIN {
    order = 0
    nnz = 0
    failed = 0
}

NF > 0 {
    current_order = NF - 1

    if (order == 0) {
        order = current_order

        if (order < 1) {
            print "Error: invalid first tensor row." > "/dev/stderr"
            failed = 1
            exit 2
        }

        for (mode = 1; mode <= order; mode++) {
            dimensions[mode] = 0
        }
    }

    if (current_order != order) {
        printf "Error: inconsistent fields at row %d. Expected %d fields, found %d.\n", nnz + 1, order + 1, NF > "/dev/stderr"
        failed = 1
        exit 3
    }

    nnz++

    for (mode = 1; mode <= order; mode++) {
        coordinate = $mode + 0

        if (coordinate < 1 || coordinate != int(coordinate)) {
            printf "Error: invalid coordinate at row %d, mode %d: %s\n", nnz, mode, $mode > "/dev/stderr"
            failed = 1
            exit 4
        }

        if (coordinate > dimensions[mode]) {
            dimensions[mode] = coordinate
        }
    }
}

END {
    if (failed) {
        exit
    }

    if (order == 0 || nnz == 0) {
        print "Error: no valid tensor entries found." > "/dev/stderr"
        exit 5
    }

    print "# extended FROSTT format"
    print order, nnz

    for (mode = 1; mode <= order; mode++) {
        if (mode > 1) {
            printf " "
        }

        printf "%d", dimensions[mode]
    }

    printf "\n"
}
' "$clean_coordinate_file" > "$metadata_file"

cat "$metadata_file" "$clean_coordinate_file" > "$output_file"

echo
echo "Conversion complete."
echo
echo "Metadata:"
cat "$metadata_file"

echo
echo "First rows:"
head -n 8 "$output_file"

echo
echo "Created:"
echo "  $output_file"