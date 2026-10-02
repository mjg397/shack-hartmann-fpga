`timescale 1ns/1ps

// =============================================================================
// tb_latency_trace.v
//
// Independent, race-free confirmation of the final-pixel-to-result latency.
//
// tb_full_pipeline.v increments an integer `cycle_count` with a BLOCKING
// assignment in its own always block, while other always blocks read it.
// Verilog does not order those blocks, so any cycle number derived that way can
// be off by one. This testbench instead timestamps every event with $time, which
// is identical in every block at a given edge, and converts to cycles only at
// the end. Differences are therefore exact.
//
// Clock period is 10 ns (always #5), so cycles = (t2 - t1) / 10.
// =============================================================================

module tb_latency_trace;

    localparam integer PERIOD_NS = 10;
    parameter integer FINAL_TILE_PIXEL = 61375;

    reg clk = 1'b0;
    reg reset = 1'b1;
    always #5 clk = ~clk;

    wire [7:0]   streamer_data;
    wire         streamer_fv, streamer_lv, streamer_valid, frame_complete;
    wire         full_frame_complete_accumulator;
    wire [7:0]   subaps_done_accumulator;
    wire [15:0]  intensity_accumulator;
    wire [19:0]  xI_accumulator, yI_accumulator;
    wire [19:0]  xI_reciprocal, yI_reciprocal;
    wire [18:0]  rI_reciprocal;
    wire [7:0]   subaps_done_reciprocal;
    wire [23:0]  x_centroid, y_centroid;
    wire [24:0]  x_slopes, y_slopes;
    wire         new_subapeture, subap_valid;
    wire [249:0] zernike_out;
    wire         done;

    tb_zynq_top #(.IMAGE_FILE("image_packed.mem")) dut (
        .clk (clk), .reset (reset),
        .streamer_data (streamer_data), .streamer_fv (streamer_fv),
        .streamer_lv (streamer_lv), .streamer_valid (streamer_valid),
        .frame_complete (frame_complete),
        .full_frame_complete_accumulator (full_frame_complete_accumulator),
        .subaps_done_accumulator (subaps_done_accumulator),
        .intensity_accumulator (intensity_accumulator),
        .xI_accumulator (xI_accumulator), .yI_accumulator (yI_accumulator),
        .xI_reciprocal (xI_reciprocal), .yI_reciprocal (yI_reciprocal),
        .rI_reciprocal (rI_reciprocal),
        .subaps_done_reciprocal (subaps_done_reciprocal),
        .x_centroid (x_centroid), .y_centroid (y_centroid),
        .x_slopes (x_slopes), .y_slopes (y_slopes),
        .new_subapeture (new_subapeture), .subap_valid (subap_valid),
        .zernike_out (zernike_out), .done (done)
    );

    // ---- event timestamps, in simulation time ----
    time t_pixel  = 0;   // FINAL_TILE_PIXEL present on the input bus
    time t_s1     = 0;   // that pixel captured by the accumulator's s1 register
    time t_accum  = 0;   // subaps_done_accumulator increments
    time t_recip  = 0;   // subaps_done_reciprocal increments
    time t_slope  = 0;   // subap_valid asserts
    time t_done   = 0;   // done asserts

    integer pixel_index = 0;
    reg [7:0] prev_accum = 8'd0;
    reg [7:0] prev_recip = 8'd0;
    reg captured = 1'b0;

    initial begin
        reset = 1'b1;
        repeat (10) @(posedge clk);
        reset = 1'b0;
    end

    // Input bus: timestamp the cycle on which the final contributing pixel is
    // presented to the accumulator.
    always @(posedge clk) begin
        if (!reset && streamer_valid) begin
            if (pixel_index == FINAL_TILE_PIXEL) t_pixel = $time;
            pixel_index <= pixel_index + 1;
        end
    end

    // One register deeper: the accumulator's s1 input stage. s1_subap_done is
    // high on the cycle s1 holds a subaperture's final pixel, so the last such
    // pulse before `done` is the s1-stage event for the final in-pupil tile.
    // Hierarchical probe -- observation only, the RTL is untouched.
    always @(posedge clk) begin
        if (!reset && dut.pipeline.accumulator.s1_valid
                   && dut.pipeline.accumulator.s1_subap_done) begin
            t_s1 = $time;
        end
    end

    always @(posedge clk) begin
        if (!reset) begin
            if (subaps_done_accumulator != prev_accum) begin
                t_accum = $time;
                prev_accum <= subaps_done_accumulator;
            end
            if (subaps_done_reciprocal != prev_recip) begin
                t_recip = $time;
                prev_recip <= subaps_done_reciprocal;
            end
            if (subap_valid) t_slope = $time;
        end
    end

    always @(posedge clk) begin
        if (!reset && done && !captured) begin
            captured = 1'b1;
            t_done = $time;

            $display("");
            $display("=== RACE-FREE LATENCY TRACE (timestamps, not cycle counters) ===");
            $display("  final tile pixel index          : %0d", FINAL_TILE_PIXEL);
            $display("");
            $display("  event                              time(ns)   cycle");
            $display("  final pixel on input bus          %8t  %6d", t_pixel, t_pixel / PERIOD_NS);
            $display("  captured by accumulator s1 reg    %8t  %6d", t_s1, t_s1 / PERIOD_NS);
            $display("  accumulator completion            %8t  %6d", t_accum, t_accum / PERIOD_NS);
            $display("  reciprocal completion             %8t  %6d", t_recip, t_recip / PERIOD_NS);
            $display("  slope valid (subap_valid)         %8t  %6d", t_slope, t_slope / PERIOD_NS);
            $display("  zernike done                      %8t  %6d", t_done, t_done / PERIOD_NS);
            $display("");
            $display("  STAGE DELTAS (cycles)");
            $display("    pixel      -> accumulator       : %0d",
                     (t_accum - t_pixel) / PERIOD_NS);
            $display("    accumulator-> reciprocal        : %0d",
                     (t_recip - t_accum) / PERIOD_NS);
            $display("    reciprocal -> slope             : %0d",
                     (t_slope - t_recip) / PERIOD_NS);
            $display("    slope      -> done              : %0d",
                     (t_done - t_slope) / PERIOD_NS);
            $display("");
            $display("");
            $display("  ANCHOR COMPARISON (same run, different start point)");
            $display("    from emulator output bus        : %0d cycles  (%0d.%0d ns)",
                     (t_done - t_pixel) / PERIOD_NS,
                     (((t_done - t_pixel) / PERIOD_NS) * 1000) / 150,
                     (((t_done - t_pixel) / PERIOD_NS) * 1000 / 15) % 10);
            $display("    from accumulator s1 register    : %0d cycles  (%0d.%0d ns)",
                     (t_done - t_s1) / PERIOD_NS,
                     (((t_done - t_s1) / PERIOD_NS) * 1000) / 150,
                     (((t_done - t_s1) / PERIOD_NS) * 1000 / 15) % 10);
            $display("    from accumulator completion     : %0d cycles  (%0d.%0d ns)",
                     (t_done - t_accum) / PERIOD_NS,
                     (((t_done - t_accum) / PERIOD_NS) * 1000) / 150,
                     (((t_done - t_accum) / PERIOD_NS) * 1000 / 15) % 10);
            $display("");
            $display("  TOTAL final pixel -> done         : %0d cycles",
                     (t_done - t_pixel) / PERIOD_NS);
            $display("  at 150 MHz                        : %0d.%0d ns",
                     (((t_done - t_pixel) / PERIOD_NS) * 1000) / 150,
                     (((t_done - t_pixel) / PERIOD_NS) * 1000 / 15) % 10);
            $display("");
            repeat (4) @(posedge clk);
            $finish;
        end
    end

    initial begin
        #4000000;
        $display("  [TIMEOUT] no done");
        $finish;
    end

endmodule
