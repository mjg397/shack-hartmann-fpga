#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static uint16_t reciprocal_seed_lut[LUT_SIZE];

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

static void init_seed_lut(void) {
    uint32_t i;

    for (i = 0; i < LUT_SIZE; i++) {
        /* Midpoint seed for bucket a in [1 + i/256, 1 + (i+1)/256). */
        double a_mid = 1.0 + ((double)i + 0.5) / 256.0;
        reciprocal_seed_lut[i] = q16_from_double(1.0 / a_mid);
    }
}

static uint16_t reciprocal_newton_step_q16(uint16_t a_q1_15, uint16_t x_q0_16) {
    uint32_t ax_q1_15;
    uint32_t two_minus_ax_q1_15;
    uint32_t prod_q1_31;
    uint32_t x1_q0_16;

    /* Q1.15 = (Q1.15 * Q0.16) >> 16 */
    ax_q1_15 = ((uint32_t)a_q1_15 * (uint32_t)x_q0_16) >> 16;

    /* 2.0 in Q1.15 is 0x10000 (17-bit constant). */
    if (ax_q1_15 > 0x10000u) {
        ax_q1_15 = 0x10000u;
    }
    two_minus_ax_q1_15 = 0x10000u - ax_q1_15;

    /* Q0.16 = (Q0.16 * Q1.15) >> 15 */
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

int main(int argc, char **argv) {
    init_seed_lut();

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