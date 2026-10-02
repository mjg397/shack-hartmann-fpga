`timescale 1ns/1ps
// =============================================================================
// shwfs_assertions.sv -- PROTOCOL AND RANGE ASSERTIONS
//
// Checks invariants that hold regardless of the data, so they catch classes of
// bug the scoreboard cannot: a scoreboard only knows "the number is wrong",
// these say "the protocol was violated".
//
// Written as immediate checks inside clocked blocks rather than SVA
// `assert property`, because Icarus Verilog does not support concurrent
// assertions. The same source therefore runs under both Icarus and Verilator.
// Under Verilator, `--assert` additionally enables its own checks.
//
// Every failure increments `fail_count`; the testbench fails the run if it is
// non-zero at the end, so a violation cannot be missed in scrollback.
// =============================================================================

`define CHECK(cond, msg) \
    if (!(cond)) begin \
        fail_count = fail_count + 1; \
        $display("  [ASSERT] %0t %s", $time, msg); \
    end

module shwfs_assertions #(
    parameter integer N_VALID  = 168,
    parameter integer N_TILES  = 256,
    parameter integer SI_MAX   = 65280,
    parameter integer MOM_MAX  = 489600
)(
    input wire        clk,
    input wire        reset,

    input wire        s_valid,
    input wire [7:0]  s_data,

    input wire [7:0]  subaps_done_accumulator,
    input wire [15:0] intensity_accumulator,
    input wire [19:0] xI_accumulator,
    input wire [19:0] yI_accumulator,

    input wire [18:0] rI_reciprocal,
    input wire [7:0]  subaps_done_reciprocal,

    input wire [23:0] x_centroid,
    input wire [23:0] y_centroid,
    input wire [24:0] x_slopes,
    input wire [24:0] y_slopes,
    input wire        subap_valid,

    input wire [249:0] zernike_out,
    input wire         done,

    output integer fail_count,
    output integer subap_count,
    output integer done_count
);

    reg [7:0] prev_accum, prev_recip;
    reg       seen_reset;

    initial begin
        fail_count  = 0;
        subap_count = 0;
        done_count  = 0;
        prev_accum  = 8'd0;
        prev_recip  = 8'd0;
        seen_reset  = 1'b0;
    end

    always @(posedge clk) begin
        if (reset) begin
            prev_accum <= 8'd0;
            prev_recip <= 8'd0;
            seen_reset <= 1'b1;
        end else if (seen_reset) begin

            // ---- input protocol ----
            if (s_valid) begin
                `CHECK(!$isunknown(s_data), "s_data is X while s_valid asserted")
            end

            // ---- accumulator ----
            if (subaps_done_accumulator !== prev_accum) begin
                `CHECK(subaps_done_accumulator == ((prev_accum + 8'd1) & 8'hFF),
                       "subapetures_completed did not increment by exactly 1")
                `CHECK(!$isunknown(intensity_accumulator),
                       "intensity is X at tile completion")
                `CHECK(intensity_accumulator <= SI_MAX,
                       "sI exceeded its documented bound (65280)")
                `CHECK(xI_accumulator <= MOM_MAX,
                       "xI exceeded its documented bound (489600)")
                `CHECK(yI_accumulator <= MOM_MAX,
                       "yI exceeded its documented bound (489600)")
                prev_accum <= subaps_done_accumulator;
            end

            // ---- reciprocal stage ----
            if (subaps_done_reciprocal !== prev_recip) begin
                `CHECK(subaps_done_reciprocal == ((prev_recip + 8'd1) & 8'hFF),
                       "reciprocal completion count did not increment by 1")
                `CHECK(!$isunknown(rI_reciprocal),
                       "reciprocal output is X -- ROM likely failed to load")
                `CHECK(rI_reciprocal != 19'd0,
                       "reciprocal is zero, which is never a valid 1/sI")
                prev_recip <= subaps_done_reciprocal;
            end

            // ---- slope stage ----
            if (subap_valid) begin
                subap_count <= subap_count + 1;
                `CHECK(subap_count < N_VALID,
                       "more subap_valid pulses than valid subapertures")
                `CHECK(!$isunknown(x_centroid) && !$isunknown(y_centroid),
                       "centroid is X while subap_valid asserted")
                `CHECK(!$isunknown(x_slopes) && !$isunknown(y_slopes),
                       "slope is X while subap_valid asserted")
                // A centroid must lie inside its own 16-pixel subaperture:
                // Q18 means 16 px == 16*2^18 == 4194304.
                `CHECK(x_centroid <= 24'd4194304,
                       "x_centroid outside the 16-pixel subaperture")
                `CHECK(y_centroid <= 24'd4194304,
                       "y_centroid outside the 16-pixel subaperture")
            end

            // ---- output ----
            if (done) begin
                done_count <= done_count + 1;
                `CHECK(!$isunknown(zernike_out),
                       "zernike_out contains X when done asserted")
                `CHECK(subap_count == N_VALID,
                       "done asserted without all valid subapertures processed")
            end
        end
    end

    task automatic reset_counts();
        begin
            subap_count = 0;
            done_count  = 0;
            prev_accum  = 8'd0;
            prev_recip  = 8'd0;
        end
    endtask

endmodule
