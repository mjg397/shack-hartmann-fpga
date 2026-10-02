`timescale 1ns/1ps
// =============================================================================
// shwfs_scoreboard.sv -- DUT vs FUNCTIONAL REFERENCE MODEL
//
// Samples DUT outputs as they are produced and compares them against
// shwfs_ref_model, which computes the same quantities from the specification by
// a different method. Comparison is EXACT: both implement the same fixed-point
// arithmetic, so any difference is a real discrepancy, never rounding.
//
// Checks, in the order the DUT produces them:
//   * every subap_valid pulse -> centroid x/y and slope x/y against the model
//   * done                    -> all 10 Zernike coefficients against the model
//
// Mismatches are reported with the subaperture index and both values, then
// tallied. The first few are printed in full; the rest are counted, so a
// systematic failure does not bury the log.
// =============================================================================

module shwfs_scoreboard #(
    parameter integer N_TILES     = 256,
    parameter integer N_VALID     = 168,
    parameter integer N_MODES     = 10,
    parameter integer MAX_REPORTS = 5
)(
    input wire clk,
    input wire reset,

    input wire         subap_valid,
    input wire [23:0]  x_centroid,
    input wire [23:0]  y_centroid,
    input wire [24:0]  x_slopes,
    input wire [24:0]  y_slopes,
    input wire         done,
    input wire [249:0] zernike_out,

    output integer checks_run,
    output integer mismatches,
    output integer centroid_mismatches,
    output integer slope_mismatches,
    output integer zernike_mismatches
);

    // Expected values, loaded from the reference model before each test.
    integer exp_cx [0:N_TILES-1];
    integer exp_cy [0:N_TILES-1];
    integer exp_sx [0:N_TILES-1];
    integer exp_sy [0:N_TILES-1];
    integer exp_idx [0:N_VALID-1];
    integer exp_z  [0:N_MODES-1];

    integer sub_ptr;
    integer reports;
    reg     enabled;

    initial begin
        checks_run = 0; mismatches = 0;
        centroid_mismatches = 0; slope_mismatches = 0; zernike_mismatches = 0;
        sub_ptr = 0; reports = 0; enabled = 1'b0;
    end

    // The testbench copies expected values into exp_* directly (hierarchical
    // assignment) and then calls start_frame(). Array task arguments are not
    // portable across Icarus and Verilator, so they are avoided.
    task automatic start_frame();
        begin
            sub_ptr = 0;
            reports = 0;
            enabled = 1'b1;
        end
    endtask

    task automatic report_mismatch(input string what, input integer where,
                                   input integer got, input integer expv);
        begin
            if (reports < MAX_REPORTS) begin
                $display("  [SCOREBOARD] %s mismatch @%0d: dut=%0d ref=%0d",
                         what, where, got, expv);
                reports = reports + 1;
                if (reports == MAX_REPORTS)
                    $display("  [SCOREBOARD] further mismatches counted, not printed");
            end
        end
    endtask

    integer tile;
    integer dut_v, ref_v;

    always @(posedge clk) begin
        if (!reset && enabled) begin

            if (subap_valid) begin
                if (sub_ptr < N_VALID) begin
                    tile = exp_idx[sub_ptr];

                    // centroids (unsigned, 24-bit)
                    checks_run = checks_run + 2;
                    if (x_centroid !== exp_cx[tile][23:0]) begin
                        centroid_mismatches = centroid_mismatches + 1;
                        mismatches = mismatches + 1;
                        report_mismatch("x_centroid", tile, x_centroid, exp_cx[tile]);
                    end
                    if (y_centroid !== exp_cy[tile][23:0]) begin
                        centroid_mismatches = centroid_mismatches + 1;
                        mismatches = mismatches + 1;
                        report_mismatch("y_centroid", tile, y_centroid, exp_cy[tile]);
                    end

                    // slopes (signed, 25-bit)
                    checks_run = checks_run + 2;
                    dut_v = $signed(x_slopes);
                    ref_v = exp_sx[tile];
                    if (dut_v !== ref_v) begin
                        slope_mismatches = slope_mismatches + 1;
                        mismatches = mismatches + 1;
                        report_mismatch("x_slope", tile, dut_v, ref_v);
                    end
                    dut_v = $signed(y_slopes);
                    ref_v = exp_sy[tile];
                    if (dut_v !== ref_v) begin
                        slope_mismatches = slope_mismatches + 1;
                        mismatches = mismatches + 1;
                        report_mismatch("y_slope", tile, dut_v, ref_v);
                    end

                    sub_ptr = sub_ptr + 1;
                end
            end

            if (done) begin
                for (tile = 0; tile < N_MODES; tile = tile + 1) begin
                    checks_run = checks_run + 1;
                    dut_v = $signed(zernike_out[tile*25 +: 25]);
                    ref_v = exp_z[tile];
                    if (dut_v !== ref_v) begin
                        zernike_mismatches = zernike_mismatches + 1;
                        mismatches = mismatches + 1;
                        report_mismatch("zernike", tile, dut_v, ref_v);
                    end
                end
                enabled = 1'b0;   // one result per frame
            end
        end
    end

endmodule
