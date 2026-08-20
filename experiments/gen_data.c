// gen_data.c — C port of gen_data.py: generates synthetic sparse/dense
// tensors in extended FROSTT format for the experiments/ pipeline.
//
// Ported 1:1 from gen_data.py's geometric-skip-sampling algorithm (see the
// comment on generate_sparse_2d_tns for the derivation) — this exists
// purely for speed. Python's per-element interpreter overhead (f-string
// formatting, a file write, and two random() calls per sampled nonzero)
// makes the largest cases here (e.g. spmspv_speedup's ~500 million
// expected nonzeros) take minutes; this does the same work in a small
// fraction of that.
//
// NOT a bit-for-bit reproduction: this uses a different PRNG than Python's
// `random` module (xoshiro256** here vs. Mersenne Twister there), so the
// exact values and even the exact nnz count for a given run will differ
// from gen_data.py's output. That's fine — this is synthetic filler data;
// only shape, nonzero density, and the FROSTT format matter to every
// downstream consumer.
//
// Usage: gen_data <experiment_name> [generator_args...]
//        (mirrors gen_data.py's CLI exactly — same experiment names, same
//        optional sparsity_pct argument for sparsity_scaling/vector_width)
//
// Build: cc -O2 -o gen_data gen_data.c -lm
//        (see gen_data.sh, which does this automatically and falls back
//        to gen_data.py if no C compiler is available or this fails to
//        build)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// PRNG — xoshiro256** (Blackman & Vigna). Fast, high-quality, non-
// cryptographic; exactly what synthetic filler data needs. Seeded from
// time + pid so repeated runs vary, matching Python's unseeded `random`.
// ---------------------------------------------------------------------------

static uint64_t rng_state[4];

static uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(void) {
    uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

// splitmix64, used only to expand one seed word into xoshiro's 4-word state.
static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

// Uniform double in [0, 1) — top 53 bits, the standard construction.
static double rand01(void) {
    return (double)(xoshiro_next() >> 11) * (1.0 / 9007199254740992.0);
}

// Uniform double in [lo, hi) — matches Python's random.uniform(lo, hi).
static double rand_uniform(double lo, double hi) {
    return lo + rand01() * (hi - lo);
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

static FILE *open_or_die(const char *path, const char *mode) {
    FILE *f = fopen(path, mode);
    if (!f) {
        fprintf(stderr, "gen_data: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    return f;
}

// Width reserved for the nnz field in a placeholder header so it can be
// patched in place once the real count is known (see write_header_placeholder
// / patch_nnz below) — wide enough for any nnz this program could produce.
#define NNZ_FIELD_WIDTH 20

// Writes "# extended FROSTT format\n<ndim> " then a blank, space-padded
// NNZ_FIELD_WIDTH-byte field, then "\n<dims...>\n". Returns the byte offset
// of the start of that field, to be passed to patch_nnz once the actual
// nnz is known. Avoids the write-to-temp-file-then-copy dance gen_data.py
// uses to work around not knowing nnz in advance.
static long write_header_placeholder(FILE *f, int ndim, const long long *dims) {
    fprintf(f, "# extended FROSTT format\n%d ", ndim);
    long nnz_offset = ftell(f);
    fprintf(f, "%-*s\n", NNZ_FIELD_WIDTH, "");
    for (int i = 0; i < ndim; i++)
        fprintf(f, "%lld%c", dims[i], (i + 1 < ndim) ? ' ' : '\n');
    return nnz_offset;
}

static void patch_nnz(FILE *f, long nnz_offset, long long nnz) {
    fflush(f);
    fseek(f, nnz_offset, SEEK_SET);
    fprintf(f, "%-*lld", NNZ_FIELD_WIDTH, nnz);
    fseek(f, 0, SEEK_END);
}

// ---------------------------------------------------------------------------
// Generators — one per gen_data.py function of the same name.
// ---------------------------------------------------------------------------

// Geometric skip-sampling: visits only ~nnz of the total_elements cells,
// each included independently with probability `density` = 1 - sparsity.
// The gap to the next included cell follows Geom(density), sampled as
// floor(log(U) / log(1 - density)) for uniform U in (0, 1) — the standard
// inverse-CDF trick for a geometric distribution. O(1) memory, O(nnz) time,
// regardless of how sparse total_elements is (see spmspv_speedup: a
// 10-billion-cell space walked in ~500 million steps, not 10 billion).
static long long next_gap(double density) {
    double r = rand01();
    if (r == 0.0) return -1; // signals "stop"
    return (long long)floor(log(r) / log(1.0 - density));
}

static void generate_sparse_3d_tns(const char *filename, long long dim1,
                                    long long dim2, long long dim3, double sparsity) {
    long long total_elements = dim1 * dim2 * dim3;
    double density = 1.0 - sparsity;

    FILE *f = open_or_die(filename, "w");
    long long dims[3] = { dim1, dim2, dim3 };
    long nnz_offset = write_header_placeholder(f, 3, dims);

    long long actual_nnz = 0;
    long long idx = 0;
    while (idx < total_elements) {
        long long i = idx / (dim2 * dim3);
        long long j = (idx % (dim2 * dim3)) / dim3;
        long long k = idx % dim3;
        double val = rand_uniform(0.5, 2.5);
        fprintf(f, "%lld %lld %lld %.4f\n", i + 1, j + 1, k + 1, val);
        actual_nnz++;

        long long gap = next_gap(density);
        if (gap < 0) break;
        idx += gap + 1;
    }

    patch_nnz(f, nnz_offset, actual_nnz);
    fclose(f);

    printf("Generated sparse 3D tensor: %s | Shape: (%lld, %lld, %lld) | NNZ: %lld | Sparsity: %g\n",
           filename, dim1, dim2, dim3, actual_nnz, sparsity);
}

static void generate_sparse_1d_tns(const char *filename, long long dim1, double sparsity) {
    long long total_elements = dim1;
    double density = 1.0 - sparsity;

    FILE *f = open_or_die(filename, "w");
    long long dims[1] = { dim1 };
    long nnz_offset = write_header_placeholder(f, 1, dims);

    long long actual_nnz = 0;
    long long idx = 0;
    while (idx < total_elements) {
        double val = rand_uniform(0.5, 2.5);
        fprintf(f, "%lld %.4f\n", idx + 1, val);
        actual_nnz++;

        long long gap = next_gap(density);
        if (gap < 0) break;
        idx += gap + 1;
    }

    patch_nnz(f, nnz_offset, actual_nnz);
    fclose(f);

    printf("Generated sparse 1D tensor: %s | Shape: (%lld,) | NNZ: %lld | Sparsity: %g\n",
           filename, dim1, actual_nnz, sparsity);
}

static void generate_dense_2d_tns(const char *filename, long long rows, long long cols) {
    long long nnz = rows * cols;

    FILE *f = open_or_die(filename, "w");
    fprintf(f, "# extended FROSTT format\n");
    fprintf(f, "2 %lld\n", nnz);
    fprintf(f, "%lld %lld\n", rows, cols);

    for (long long r = 1; r <= rows; r++)
        for (long long c = 1; c <= cols; c++)
            fprintf(f, "%lld %lld %.4f\n", r, c, rand_uniform(0.5, 2.5));

    fclose(f);

    printf("Generated dense tensor: %s | Shape: (%lld, %lld) | NNZ: %lld\n",
           filename, rows, cols, nnz);
}

static void generate_sparse_2d_tns(const char *filename, long long rows,
                                    long long cols, double sparsity) {
    long long total_elements = rows * cols;
    double density = 1.0 - sparsity;

    FILE *f = open_or_die(filename, "w");
    long long dims[2] = { rows, cols };
    long nnz_offset = write_header_placeholder(f, 2, dims);

    long long actual_nnz = 0;
    long long idx = 0;
    while (idx < total_elements) {
        long long i = idx / cols;
        long long j = idx % cols;
        double val = rand_uniform(0.5, 2.5);
        fprintf(f, "%lld %lld %.4f\n", i + 1, j + 1, val);
        actual_nnz++;

        long long gap = next_gap(density);
        if (gap < 0) break;
        idx += gap + 1;
    }

    patch_nnz(f, nnz_offset, actual_nnz);
    fclose(f);

    printf("Generated sparse 2D tensor: %s | Shape: (%lld, %lld) | NNZ: %lld | Sparsity: %g\n",
           filename, rows, cols, actual_nnz, sparsity);
}

// Same geometric skip sampling as generate_sparse_2d_tns, but idx walks
// column-major (j outer, i inner) instead of row-major, so sparsity is
// applied down each column instead of across each row.
static void generate_sparse_2d_tns_col_sparsity(const char *filename, long long rows,
                                                 long long cols, double sparsity) {
    long long total_elements = rows * cols;
    double density = 1.0 - sparsity;

    FILE *f = open_or_die(filename, "w");
    long long dims[2] = { rows, cols };
    long nnz_offset = write_header_placeholder(f, 2, dims);

    long long actual_nnz = 0;
    long long idx = 0;
    while (idx < total_elements) {
        long long j = idx / rows;
        long long i = idx % rows;
        double val = rand_uniform(0.5, 2.5);
        fprintf(f, "%lld %lld %.4f\n", i + 1, j + 1, val);
        actual_nnz++;

        long long gap = next_gap(density);
        if (gap < 0) break;
        idx += gap + 1;
    }

    patch_nnz(f, nnz_offset, actual_nnz);
    fclose(f);

    printf("Generated sparse 2D tensor (column sparsity): %s | Shape: (%lld, %lld) | NNZ: %lld | Sparsity: %g\n",
           filename, rows, cols, actual_nnz, sparsity);
}

// ---------------------------------------------------------------------------
// Per-experiment dispatch — mirrors gen_data.py's generate_*_data functions.
// ---------------------------------------------------------------------------

static void generate_phase_ablation_data(void) {
    generate_sparse_2d_tns("./phase_ablation/tensor_B.tns", 5000, 5000, 0.95);
    generate_sparse_2d_tns("./phase_ablation/tensor_C.tns", 5000, 5000, 0.95);
}

// kernel_name: one of "spgemm" / "spmmh" / "spmspv" / "spmttkrp" / "spttspm".
// Unlike gen_data.py (which has five separate zero-arg wrapper functions
// feeding this, one per EXPERIMENTS dict entry, since Python's dict needs
// zero-arg callables), main()'s dispatch below just passes the kernel name
// straight through — same behavior, one less layer of indirection.
static void generate_speedup_data(const char *kernel_name) {
    if (strcmp(kernel_name, "spgemm") == 0) {
        generate_sparse_2d_tns("./speedups/synthetic_data/spgemm/tensor_B.tns", 5000, 5000, 0.95);
        generate_sparse_2d_tns("./speedups/synthetic_data/spgemm/tensor_C.tns", 5000, 5000, 0.95);
    }
    if (strcmp(kernel_name, "spmmh") == 0) {
        generate_sparse_2d_tns("./speedups/synthetic_data/spmmh/tensor_B.tns", 5000, 5000, 0.95);
        generate_dense_2d_tns("./speedups/synthetic_data/spmmh/tensor_C.tns", 5000, 5000);
        generate_sparse_2d_tns("./speedups/synthetic_data/spmmh/tensor_D.tns", 5000, 5000, 0.95);
    }
    if (strcmp(kernel_name, "spmspv") == 0) {
        generate_sparse_2d_tns("./speedups/synthetic_data/spmspv/tensor_B.tns", 100, 100000000, 0.95);
        generate_sparse_1d_tns("./speedups/synthetic_data/spmspv/tensor_x.tns", 100000000, 0.95);
    }
    if (strcmp(kernel_name, "spmttkrp") == 0) {
        generate_sparse_3d_tns("./speedups/synthetic_data/spmttkrp/tensor_B.tns", 1000, 1000, 1000, 0.95);
        generate_sparse_2d_tns("./speedups/synthetic_data/spmttkrp/tensor_C.tns", 1000, 1000, 0.95);
        generate_sparse_2d_tns("./speedups/synthetic_data/spmttkrp/tensor_D.tns", 1000, 1000, 0.95);
    }
    if (strcmp(kernel_name, "spttspm") == 0) {
        generate_sparse_3d_tns("./speedups/synthetic_data/spttspm/tensor_B.tns", 500, 500, 500, 0.95);
        generate_sparse_2d_tns("./speedups/synthetic_data/spttspm/tensor_C.tns", 500, 500, 0.95);
    }
}

// Same shape/sparsity as speedups/synthetic_data/spmttkrp — this experiment
// reuses that kernel's compiled-for-parallel-execution binaries (see
// multicore/compile.sh) for a core-count scaling study, not a different
// dataset.
static void generate_multicore_data(void) {
    generate_sparse_3d_tns("./multicore/tensor_B.tns", 1000, 1000, 1000, 0.95);
    generate_sparse_2d_tns("./multicore/tensor_C.tns", 1000, 1000, 0.95);
    generate_sparse_2d_tns("./multicore/tensor_D.tns", 1000, 1000, 0.95);
}

// Same 5000x5000 shape as speedups/synthetic_data/spgemm — sweeps sparsity
// instead of holding it fixed at 0.95. sparsity_pct is the *nonzero
// density*, in percent (e.g. "0.01", "1", "10" — see sparsity_scaling/
// run.sh, which regenerates data at each sweep point via this function
// before running both of sparsity_scaling/compile.sh's binaries against it).
static void generate_sparsity_scaling_data(const char *sparsity_pct) {
    double sparsity = 1.0 - (atof(sparsity_pct) / 100.0);
    generate_sparse_2d_tns("./sparsity_scaling/tensor_B.tns", 5000, 5000, sparsity);
    generate_sparse_2d_tns("./sparsity_scaling/tensor_C.tns", 5000, 5000, sparsity);
}

// Same 5000x5000 shape and sparsity_pct convention as
// generate_sparsity_scaling_data — see vector_width/run.sh, which
// regenerates data at each sweep point (1%, 5%, 10%) via this function
// before running all 5 of vector_width/compile.sh's binaries against it.
static void generate_vector_width_data(const char *sparsity_pct) {
    double sparsity = 1.0 - (atof(sparsity_pct) / 100.0);
    generate_sparse_2d_tns("./vector_width/tensor_B.tns", 5000, 5000, sparsity);
    generate_sparse_2d_tns("./vector_width/tensor_C.tns", 5000, 5000, sparsity);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static const char *EXPERIMENTS[] = {
    "phase_ablation", "multicore", "sparsity_scaling", "vector_width",
    "spgemm_speedup", "spmmh_speedup", "spmspv_speedup",
    "spmttkrp_speedup", "spttspm_speedup",
    "gen_1d_vector", "gen_2d_tensor", "gen_3d_tensor",
};
#define NUM_EXPERIMENTS (sizeof(EXPERIMENTS) / sizeof(EXPERIMENTS[0]))

static void print_available(void) {
    fprintf(stderr, "Available experiments: ");
    for (size_t i = 0; i < NUM_EXPERIMENTS; i++)
        fprintf(stderr, "%s%s", EXPERIMENTS[i], (i + 1 < NUM_EXPERIMENTS) ? ", " : "\n");
}

int main(int argc, char **argv) {
    rng_seed((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32));

    if (argc < 2) {
        fprintf(stderr, "Usage: gen_data <experiment_name> [generator_args...]\n");
        print_available();
        return 1;
    }

    const char *experiment = argv[1];
    const char *arg2 = (argc > 2) ? argv[2] : NULL;
    const char *arg3 = (argc > 3) ? argv[3] : NULL;
    const char *arg4 = (argc > 4) ? argv[4] : NULL;
    const char *arg5 = (argc > 5) ? argv[5] : NULL;
    const char *arg6 = (argc > 6) ? argv[6] : NULL;

    if (strcmp(experiment, "gen_1d_vector") == 0) {
        if (!arg2 || !arg3 || !arg4) {
            fprintf(stderr, "Usage: gen_data gen_1d_vector <filename> <dim> <sparsity>\n");
            return 1;
        }
        generate_sparse_1d_tns(arg2, atoll(arg3), atof(arg4));
    } else if (strcmp(experiment, "gen_2d_tensor") == 0) {
        if (!arg2 || !arg3 || !arg4 || !arg5) {
            fprintf(stderr, "Usage: gen_data gen_2d_tensor <filename> <rows> <cols> <sparsity>\n");
            return 1;
        }
        generate_sparse_2d_tns(arg2, atoll(arg3), atoll(arg4), atof(arg5));
    } else if (strcmp(experiment, "gen_3d_tensor") == 0) {
        if (!arg2 || !arg3 || !arg4 || !arg5 || !arg6) {
            fprintf(stderr, "Usage: gen_data gen_3d_tensor <filename> <dim1> <dim2> <dim3> <sparsity>\n");
            return 1;
        }
        generate_sparse_3d_tns(arg2, atoll(arg3), atoll(arg4), atoll(arg5), atof(arg6));
    } else if (strcmp(experiment, "phase_ablation") == 0) {
        generate_phase_ablation_data();
    } else if (strcmp(experiment, "multicore") == 0) {
        generate_multicore_data();
    } else if (strcmp(experiment, "sparsity_scaling") == 0) {
        generate_sparsity_scaling_data(arg2 ? arg2 : "1");
    } else if (strcmp(experiment, "vector_width") == 0) {
        generate_vector_width_data(arg2 ? arg2 : "1");
    } else if (strcmp(experiment, "spgemm_speedup") == 0) {
        generate_speedup_data("spgemm");
    } else if (strcmp(experiment, "spmmh_speedup") == 0) {
        generate_speedup_data("spmmh");
    } else if (strcmp(experiment, "spmspv_speedup") == 0) {
        generate_speedup_data("spmspv");
    } else if (strcmp(experiment, "spmttkrp_speedup") == 0) {
        generate_speedup_data("spmttkrp");
    } else if (strcmp(experiment, "spttspm_speedup") == 0) {
        generate_speedup_data("spttspm");
    } else {
        fprintf(stderr, "Unknown experiment: %s\n", experiment);
        print_available();
        return 1;
    }

    return 0;
}
