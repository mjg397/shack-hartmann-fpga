#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Model for quick reciprocal of a 16-bit integer value.
 *   input : unsigned 16-bit integer (not fixed-point)
 *   output: unsigned Q0.16 reciprocal
 *
 * Pipeline model:
 *   1) 256-entry LUT seed for reciprocal of normalized input in [1, 2)
 *   2) One Newton iteration x1 = x0 * (2 - a*x0)
 *
 * Notes:
 *   - Q0.16 cannot exactly represent 1.0, so 1.0 saturates to 0xFFFF.
 *   - v = 0 saturates output to 0xFFFF and sets divide-by-zero flag.
 */

#define LUT_SIZE 256u
#define Q16_ONE 65536u
#define Q16_MAX 65535u

typedef struct {
    uint16_t q16;
    bool divide_by_zero;
    bool saturated;
} reciprocal_result_t;

static const uint16_t reciprocal_seed_lut[LUT_SIZE] = {
    0xFF80, 0xFE82, 0xFD86, 0xFC8C, 0xFB94, 0xFA9E, 0xF9A9, 0xF8B7,
    0xF7C6, 0xF6D7, 0xF5EA, 0xF4FF, 0xF415, 0xF32D, 0xF247, 0xF163,
    0xF080, 0xEF9F, 0xEEBF, 0xEDE1, 0xED05, 0xEC2A, 0xEB51, 0xEA7A,
    0xE9A4, 0xE8CF, 0xE7FC, 0xE72B, 0xE65B, 0xE58C, 0xE4BF, 0xE3F4,
    0xE329, 0xE260, 0xE199, 0xE0D3, 0xE00E, 0xDF4B, 0xDE88, 0xDDC8,
    0xDD08, 0xDC4A, 0xDB8D, 0xDAD1, 0xDA17, 0xD95E, 0xD8A6, 0xD7EF,
    0xD73A, 0xD685, 0xD5D2, 0xD520, 0xD46F, 0xD3BF, 0xD311, 0xD263,
    0xD1B7, 0xD10C, 0xD062, 0xCFB9, 0xCF11, 0xCE6A, 0xCDC4, 0xCD1F,
    0xCC7B, 0xCBD8, 0xCB36, 0xCA96, 0xC9F6, 0xC957, 0xC8B9, 0xC81C,
    0xC780, 0xC6E5, 0xC64B, 0xC5B2, 0xC51A, 0xC482, 0xC3EC, 0xC357,
    0xC2C2, 0xC22E, 0xC19B, 0xC109, 0xC078, 0xBFE8, 0xBF59, 0xBECA,
    0xBE3C, 0xBDAF, 0xBD23, 0xBC98, 0xBC0D, 0xBB83, 0xBAFB, 0xBA72,
    0xB9EB, 0xB964, 0xB8DE, 0xB859, 0xB7D5, 0xB751, 0xB6CE, 0xB64C,
    0xB5CB, 0xB54A, 0xB4CA, 0xB44B, 0xB3CC, 0xB34E, 0xB2D1, 0xB254,
    0xB1D8, 0xB15D, 0xB0E3, 0xB069, 0xAFF0, 0xAF77, 0xAEFF, 0xAE88,
    0xAE11, 0xAD9B, 0xAD26, 0xACB1, 0xAC3D, 0xABC9, 0xAB56, 0xAAE4,
    0xAA72, 0xAA01, 0xA990, 0xA920, 0xA8B1, 0xA842, 0xA7D3, 0xA766,
    0xA6F8, 0xA68C, 0xA620, 0xA5B4, 0xA549, 0xA4DF, 0xA475, 0xA40C,
    0xA3A3, 0xA33A, 0xA2D3, 0xA26B, 0xA204, 0xA19E, 0xA138, 0xA0D3,
    0xA06E, 0xA00A, 0x9FA6, 0x9F43, 0x9EE0, 0x9E7E, 0x9E1C, 0x9DBA,
    0x9D59, 0x9CF9, 0x9C99, 0x9C39, 0x9BDA, 0x9B7C, 0x9B1D, 0x9AC0,
    0x9A62, 0x9A05, 0x99A9, 0x994D, 0x98F1, 0x9896, 0x983B, 0x97E1,
    0x9787, 0x972E, 0x96D5, 0x967C, 0x9624, 0x95CC, 0x9574, 0x951D,
    0x94C7, 0x9470, 0x941B, 0x93C5, 0x9370, 0x931B, 0x92C7, 0x9273,
    0x921F, 0x91CC, 0x9179, 0x9127, 0x90D5, 0x9083, 0x9032, 0x8FE1,
    0x8F90, 0x8F40, 0x8EF0, 0x8EA0, 0x8E51, 0x8E02, 0x8DB3, 0x8D65,
    0x8D17, 0x8CC9, 0x8C7C, 0x8C2F, 0x8BE2, 0x8B96, 0x8B4A, 0x8AFF,
    0x8AB3, 0x8A68, 0x8A1E, 0x89D3, 0x8989, 0x8940, 0x88F6, 0x88AD,
    0x8864, 0x881C, 0x87D3, 0x878C, 0x8744, 0x86FD, 0x86B6, 0x866F,
    0x8628, 0x85E2, 0x859C, 0x8557, 0x8511, 0x84CC, 0x8488, 0x8443,
    0x83FF, 0x83BB, 0x8377, 0x8334, 0x82F1, 0x82AE, 0x826B, 0x8229,
    0x81E7, 0x81A5, 0x8164, 0x8123, 0x80E2, 0x80A1, 0x8060, 0x8020
};

static uint32_t clz16(uint16_t x) {
    uint32_t n = 0;

    if (x == 0) {
        return 16u;
    }

    while ((x & 0x8000u) == 0u) {
        x <<= 1;
        n++;
    }
    return n;
}

static uint32_t msb_index_u16(uint16_t x) {
    return 15u - clz16(x);
}

static uint16_t q16_from_double(double x) {
    double scaled;
    long v;

    if (x <= 0.0) {
        return 0u;
    }
    if (x >= 1.0) {
        return Q16_MAX;
    }

    scaled = x * (double)Q16_ONE;
    v = (long)(scaled + 0.5);

    if (v < 0) {
        return 0u;
    }
    if ((uint32_t)v > Q16_MAX) {
        return Q16_MAX;
    }
    return (uint16_t)v;
}

static double q16_to_double(uint16_t x) {
    return (double)x / (double)Q16_ONE;
}

static uint16_t reciprocal_newton_step_q16(uint16_t a_q1_15, uint16_t x_q0_16) {
    uint32_t ax_q1_15;
    uint32_t two_minus_ax_q1_15;
    uint32_t prod_q1_31;
    uint32_t x1_q0_16;

    /*
     * Newton update in real-valued form:
     *   x1 = x0 * (2 - a*x0)
     *
     * Fixed-point mapping used here:
     *   a  in Q1.15  (normalized input in [1, 2))
     *   x0 in Q0.16  (seed reciprocal)
     *   x1 in Q0.16  (refined reciprocal)
     */

    /* ax = a*x0, with scale conversion: Q1.15 * Q0.16 -> Q1.15 via >> 16. */
    ax_q1_15 = ((uint32_t)a_q1_15 * (uint32_t)x_q0_16) >> 16;

    /* two_minus_ax = (2 - ax) in Q1.15, where 2.0 is 0x10000 in Q1.15. */
    if (ax_q1_15 > 0x10000u) {
        ax_q1_15 = 0x10000u;
    }
    two_minus_ax_q1_15 = 0x10000u - ax_q1_15;

    /* x1 = x0 * (2 - ax), with scale conversion: Q0.16 * Q1.15 -> Q0.16 via >> 15. */
    /* Add 2^14 before shift for round-to-nearest in the >> 15 step. */
    prod_q1_31 = (uint32_t)x_q0_16 * two_minus_ax_q1_15;
    x1_q0_16 = (prod_q1_31 + (1u << 14)) >> 15;

    if (x1_q0_16 > Q16_MAX) {
        x1_q0_16 = Q16_MAX;
    }
    return (uint16_t)x1_q0_16;
}

static reciprocal_result_t reciprocal_u16_to_q16(uint16_t v) {
    reciprocal_result_t result;
    uint32_t shift_left;
    uint32_t msb;
    uint16_t a_q1_15;
    uint16_t x0_q0_16;
    uint16_t x1_q0_16;
    uint32_t lut_index;
    uint32_t out_q0_16;

    result.q16 = 0u;
    result.divide_by_zero = false;
    result.saturated = false;

    if (v == 0u) {
        result.q16 = Q16_MAX;
        result.divide_by_zero = true;
        result.saturated = true;
        return result;
    }

    /* Normalize integer input to Q1.15 in [1, 2). */
    shift_left = clz16(v);
    a_q1_15 = (uint16_t)(v << shift_left);

    /* LUT index from the 8 bits under the leading 1. */
    lut_index = (((uint32_t)a_q1_15 >> 7) - 256u) & 0xFFu;
    x0_q0_16 = reciprocal_seed_lut[lut_index];

    /* One Newton refinement. */
    x1_q0_16 = reciprocal_newton_step_q16(a_q1_15, x0_q0_16);

    /* De-normalize by original power-of-two scale: shift right by msb(v). */
    msb = msb_index_u16(v);
    if (msb == 0u) {
        out_q0_16 = x1_q0_16;
    } else {
        out_q0_16 = ((uint32_t)x1_q0_16 + (1u << (msb - 1u))) >> msb;
    }

    if (out_q0_16 > Q16_MAX) {
        out_q0_16 = Q16_MAX;
        result.saturated = true;
    }

    result.q16 = (uint16_t)out_q0_16;
    return result;
}

static void print_trace_for_input(uint16_t v) {
    uint32_t shift_left;
    uint16_t a_q1_15;
    uint32_t lut_index;
    uint16_t x0_q0_16;
    uint16_t x1_q0_16;
    reciprocal_result_t out;

    if (v == 0u) {
        out = reciprocal_u16_to_q16(v);
        printf("v=%5u | div0=1 out=0x%04X\n", v, out.q16);
        return;
    }

    shift_left = clz16(v);
    a_q1_15 = (uint16_t)(v << shift_left);
    lut_index = (((uint32_t)a_q1_15 >> 7) - 256u) & 0xFFu;
    x0_q0_16 = reciprocal_seed_lut[lut_index];
    x1_q0_16 = reciprocal_newton_step_q16(a_q1_15, x0_q0_16);
    out = reciprocal_u16_to_q16(v);

    printf(
        "v=%5u | norm_shift=%2u a_q1_15=0x%04X idx=%3u x0=0x%04X x1=0x%04X out=0x%04X\n",
        v,
        shift_left,
        a_q1_15,
        lut_index,
        x0_q0_16,
        x1_q0_16,
        out.q16
    );
}

static void run_validation(void) {
    uint32_t v;
    uint32_t max_abs_err_lsb = 0u;
    uint16_t max_abs_v = 0u;
    double max_rel_err = 0.0;
    uint16_t max_rel_v = 0u;
    double sum_rel_err = 0.0;
    uint32_t count = 0u;
    uint32_t mismatch_count = 0u;
    const uint16_t traces[] = {0u, 1u, 2u, 3u, 7u, 255u, 256u, 257u, 1023u, 65535u};
    size_t i;

    for (v = 1u; v <= 65535u; v++) {
        reciprocal_result_t approx = reciprocal_u16_to_q16((uint16_t)v);
        double exact_real = 1.0 / (double)v;
        uint16_t exact_q16 = q16_from_double(exact_real);
        uint32_t abs_err_lsb;
        double approx_real = q16_to_double(approx.q16);
        double rel_err = fabs(approx_real - exact_real) / exact_real;

        if (approx.q16 >= exact_q16) {
            abs_err_lsb = (uint32_t)approx.q16 - (uint32_t)exact_q16;
        } else {
            abs_err_lsb = (uint32_t)exact_q16 - (uint32_t)approx.q16;
        }

        if (abs_err_lsb > max_abs_err_lsb) {
            max_abs_err_lsb = abs_err_lsb;
            max_abs_v = (uint16_t)v;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
            max_rel_v = (uint16_t)v;
        }

        if (approx.q16 != exact_q16) {
            mismatch_count++;
        }

        sum_rel_err += rel_err;
        count++;
    }

    printf("Validation over v=1..65535\n");
    printf("  max absolute error : %u LSB (at v=%u)\n", max_abs_err_lsb, max_abs_v);
    printf("  max relative error : %.6f%% (at v=%u)\n", max_rel_err * 100.0, max_rel_v);
    printf("  mean relative error: %.6f%%\n", (sum_rel_err / (double)count) * 100.0);
    printf("  exact-q16 mismatches: %u / %u\n", mismatch_count, count);

    printf("\nTrace vectors:\n");
    for (i = 0; i < sizeof(traces) / sizeof(traces[0]); i++) {
        print_trace_for_input(traces[i]);
    }
}

static void emit_lut_as_hdl_vector(FILE *out) {
    uint32_t i;

    fprintf(out, "// Indexing convention for [16*idx +: 16]: idx=0 is least-significant chunk.\n");
    fprintf(out, "localparam [4095:0] RECIP_SEED_LUT_Q0_16 = {\n");

    /* Emit idx=255 down to idx=0 so idx=0 lands at vector LSBs. */
    for (i = LUT_SIZE; i > 0u; i--) {
        uint32_t idx = i - 1u;
        fprintf(out, "    16'h%04X", reciprocal_seed_lut[idx]);
        if (idx != 0u) {
            fprintf(out, ",");
        }
        if ((idx % 8u) == 0u) {
            fprintf(out, "\n");
        } else {
            fprintf(out, " ");
        }
    }

    fprintf(out, "};\n");
}

static int write_lut_memh(const char *path) {
    FILE *f = fopen(path, "w");
    uint32_t i;

    if (f == NULL) {
        fprintf(stderr, "Failed to open LUT memh output: %s\n", path);
        return 1;
    }

    for (i = 0; i < LUT_SIZE; i++) {
        fprintf(f, "%04X\n", reciprocal_seed_lut[i]);
    }

    fclose(f);
    return 0;
}

static int write_golden_vectors(const char *csv_path, const char *memh_path) {
    FILE *csv = fopen(csv_path, "w");
    FILE *memh = fopen(memh_path, "w");
    uint32_t v;

    if (csv == NULL) {
        fprintf(stderr, "Failed to open golden CSV output: %s\n", csv_path);
        if (memh != NULL) {
            fclose(memh);
        }
        return 1;
    }
    if (memh == NULL) {
        fprintf(stderr, "Failed to open golden MEMH output: %s\n", memh_path);
        fclose(csv);
        return 1;
    }

    fprintf(csv, "input_u16,output_q16_hex,output_q16,div0,sat,norm_shift,lut_idx,a_q1_15,x0_q0_16,x1_q0_16\n");

    for (v = 0u; v <= 65535u; v++) {
        reciprocal_result_t out = reciprocal_u16_to_q16((uint16_t)v);
        uint32_t shift_left = 0u;
        uint16_t a_q1_15 = 0u;
        uint32_t lut_idx = 0u;
        uint16_t x0_q0_16 = 0u;
        uint16_t x1_q0_16 = 0u;

        if (v != 0u) {
            shift_left = clz16((uint16_t)v);
            a_q1_15 = (uint16_t)((uint16_t)v << shift_left);
            lut_idx = (((uint32_t)a_q1_15 >> 7) - 256u) & 0xFFu;
            x0_q0_16 = reciprocal_seed_lut[lut_idx];
            x1_q0_16 = reciprocal_newton_step_q16(a_q1_15, x0_q0_16);
        }

        fprintf(
            csv,
            "%u,0x%04X,%u,%u,%u,%u,%u,0x%04X,0x%04X,0x%04X\n",
            v,
            out.q16,
            out.q16,
            out.divide_by_zero ? 1u : 0u,
            out.saturated ? 1u : 0u,
            shift_left,
            lut_idx,
            a_q1_15,
            x0_q0_16,
            x1_q0_16
        );

        /* Pack stimulus+expected for $readmemh: {input_u16, expected_q16}. */
        fprintf(memh, "%04X%04X\n", (uint16_t)v, out.q16);

        if (v == 65535u) {
            break;
        }
    }

    fclose(csv);
    fclose(memh);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "--emit-lut-vector") == 0) {
            emit_lut_as_hdl_vector(stdout);
            return 0;
        }

        if (strcmp(argv[1], "--write-lut-memh") == 0) {
            const char *path = (argc > 2) ? argv[2] : "verilog/testbench/vectors/reciprocal_seed_lut_q16.memh";
            int rc = write_lut_memh(path);

            if (rc == 0) {
                printf("Wrote LUT memh: %s\n", path);
            }
            return rc;
        }

        if (strcmp(argv[1], "--write-golden") == 0) {
            const char *csv_path = (argc > 2) ? argv[2] : "verilog/testbench/vectors/reciprocal_golden_vectors.csv";
            const char *memh_path = (argc > 3) ? argv[3] : "verilog/testbench/vectors/reciprocal_golden_io32.memh";
            int rc = write_golden_vectors(csv_path, memh_path);

            if (rc == 0) {
                printf("Wrote golden CSV : %s\n", csv_path);
                printf("Wrote golden MEMH: %s\n", memh_path);
            }
            return rc;
        }
    }

    if (argc > 1) {
        long input = strtol(argv[1], NULL, 10);

        if (input < 0 || input > 65535) {
            fprintf(stderr, "Input out of range. Use 0..65535.\n");
            return 1;
        }

        reciprocal_result_t out = reciprocal_u16_to_q16((uint16_t)input);
        printf(
            "v=%ld reciprocal_q16=0x%04X (%u) reciprocal=%.10f div0=%d sat=%d\n",
            input,
            out.q16,
            out.q16,
            q16_to_double(out.q16),
            out.divide_by_zero ? 1 : 0,
            out.saturated ? 1 : 0
        );
        print_trace_for_input((uint16_t)input);
        return 0;
    }

    run_validation();
    return 0;
}