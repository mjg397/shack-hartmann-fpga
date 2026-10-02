`timescale 1ns/1ps

// =============================================================================
// tb_full_pipeline.v
//
// Streams one full 256x256 frame through the Zynq SHWFS pipeline and captures
// the 10 Zernike coefficients.
//
// Frame timing (from streaming_emulator defaults):
//     256 rows x (256 active + 4 HBLANK) + 152 VBLANK ~= 66712 cycles
// =============================================================================

module tb_full_pipeline;

    // Frame geometry -- must match streaming_emulator's parameters.
    localparam integer HSIZE      = 256;
    localparam integer VSIZE      = 256;
    localparam integer NUM_PIXELS = HSIZE * VSIZE;
    localparam integer NUM_MODES  = 10;
    localparam integer MODE_BITS  = 25;

    localparam integer TIMEOUT_CYCLES = 200000;

    // Clock period in ns. Event cycles are derived from $time rather than from
    // an integer counter: `cycle_count` below is incremented with a blocking
    // assignment in its own always block, and Verilog does not order that block
    // against the blocks that read it, so counter-derived timestamps can be off
    // by one. $time is identical in every block at a given edge, so differences
    // taken from it are exact. Verified against tb_latency_trace.v.
    localparam integer PERIOD_NS = 10;

    // Raster index of the pixel that completes the LAST in-pupil subaperture.
    // Overridden from the harness (-Ptb_full_pipeline.FINAL_TILE_PIXEL=...) so
    // the true final-pixel-to-result latency can be measured against the pixel
    // ARRIVING, not against the accumulator's completion event -- the two differ
    // by the accumulator's internal pipeline depth.
    parameter integer FINAL_TILE_PIXEL = 61375;

    reg clk   = 1'b0;
    reg reset = 1'b1;

    // 100 MHz -- period is arbitrary for a functional sim.
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

    tb_zynq_top #(
        .IMAGE_FILE ("image_packed.mem")
    ) dut (
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

    // -------------------------------------------------------------------------
    // Reference image, for verifying the streamed pixel order
    // -------------------------------------------------------------------------
    reg [7:0] expected_img [0:NUM_PIXELS-1];

    integer pixel_count      = 0;
    integer pixel_mismatches = 0;
    integer frame_done_count = 0;
    integer subap_count      = 0;
    integer recip_msb_trips  = 0;
    integer unknown_valid_slopes = 0;
    integer unknown_valid_coefficients = 0;
    integer cycle_count      = 0;
    integer first_mismatch   = -1;

    // Latency instrumentation: the project's headline figure is measured from
    // the arrival of the final contributing pixel to the Zernike output.
    integer last_pixel_cycle = 0;
    integer last_pixel_index = 0;
    integer last_subap_cycle = 0;
    integer done_cycle       = 0;
    reg [7:0] prev_subaps_done = 8'd0;

    // Throughput instrumentation. s_ready is tied high in shwfs_processing, so
    // the pipeline never back-pressures; these counters prove it empirically
    // rather than by inspection.
    integer first_pixel_cycle   = -1;
    integer last_stream_cycle   = 0;
    integer run_len             = 0;
    integer max_run_len         = 0;
    integer stall_events        = 0;
    integer first_subap_cycle   = -1;

    // Per-stage valid-event tracing, for the pipeline latency breakdown.
    // Each records the cycle of the LAST event at that stage boundary, so the
    // differences give the per-stage cycle cost for the final subaperture.
    integer final_tile_pixel_cycle = -1;  // arrival of FINAL_TILE_PIXEL
    integer stage_s1_cycle      = 0;   // accumulator s1 input register
    integer stage_accum_cycle   = 0;   // subaps_done_accumulator increments
    integer stage_recip_cycle   = 0;   // subaps_done_reciprocal increments
    reg [7:0] prev_subaps_recip = 8'd0;

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("waves/zynq_pipeline.vcd");
        $dumpvars(0, tb_full_pipeline);

        $readmemh("hcipy_image_rotating.mem", expected_img);

        reset = 1'b1;
        repeat (10) @(posedge clk);
        reset = 1'b0;
    end

    // -------------------------------------------------------------------------
    // Pixel integrity: the streamed bytes must match the source image, in order.
    // A byte-order error in pack_image.py is otherwise silent -- the pipeline
    // would happily produce plausible-looking but wrong coefficients.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!reset && streamer_valid) begin
            if (pixel_count < NUM_PIXELS) begin
                if (streamer_data !== expected_img[pixel_count]) begin
                    pixel_mismatches = pixel_mismatches + 1;
                    if (first_mismatch == -1) begin
                        first_mismatch = pixel_count;
                        $display("  [FAIL] pixel %0d: got %02h, expected %02h",
                                 pixel_count, streamer_data,
                                 expected_img[pixel_count]);
                    end
                end
            end
            if (first_pixel_cycle == -1) first_pixel_cycle = $time / PERIOD_NS;
            last_stream_cycle = $time / PERIOD_NS;
            if (pixel_count == FINAL_TILE_PIXEL) final_tile_pixel_cycle = $time / PERIOD_NS;
            pixel_count = pixel_count + 1;
        end
    end

    // Back-to-back acceptance: track the longest unbroken run of accepted
    // pixels, and count interruptions during the active region.
    always @(posedge clk) begin
        if (!reset) begin
            if (streamer_valid) begin
                run_len = run_len + 1;
                if (run_len > max_run_len) max_run_len = run_len;
            end else begin
                if (run_len > 0 && streamer_fv) stall_events = stall_events + 1;
                run_len = 0;
            end
        end
    end

    // Slope dump -- lets a host-side model isolate front-end (centroid/slope)
    // errors from back-end (E-matrix reconstruction) errors.
    integer slope_fh;
    initial begin
        slope_fh = $fopen("slopes_raw.txt", "w");
        $fwrite(slope_fh, "idx x_slope y_slope x_centroid y_centroid\n");
    end

    always @(posedge clk) begin
        if (!reset && frame_complete) frame_done_count = frame_done_count + 1;

        // The final CONTRIBUTING pixel is the one that completes the last valid
        // subaperture -- not the most recent pixel streamed, since the emulator
        // keeps streaming the rest of the frame after the pupil is finished.
        // subaps_done_accumulator increments the cycle a tile completes, so its
        // last increment before `done` marks that pixel.
        if (!reset && subaps_done_accumulator != prev_subaps_done) begin
            last_pixel_cycle = $time / PERIOD_NS;
            last_pixel_index = pixel_count;
            stage_accum_cycle = $time / PERIOD_NS;
            prev_subaps_done = subaps_done_accumulator;
        end

        // Accumulator s1 input register -- one stage inside the processing
        // chain. Observation only; the RTL is untouched.
        if (!reset && dut.pipeline.accumulator.s1_valid
                   && dut.pipeline.accumulator.s1_subap_done) begin
            stage_s1_cycle = $time / PERIOD_NS;
        end

        if (!reset && subaps_done_reciprocal != prev_subaps_recip) begin
            stage_recip_cycle = $time / PERIOD_NS;
            prev_subaps_recip = subaps_done_reciprocal;
        end

        if (!reset && subap_valid) begin
            if ((^x_slopes === 1'bx) || (^y_slopes === 1'bx) ||
                (^x_centroid === 1'bx) || (^y_centroid === 1'bx)) begin
                unknown_valid_slopes = unknown_valid_slopes + 1;
            end
            if (first_subap_cycle == -1) first_subap_cycle = $time / PERIOD_NS;
            last_subap_cycle = $time / PERIOD_NS;
            $fwrite(slope_fh, "%0d %0d %0d %0d %0d\n",
                    subap_count,
                    $signed(x_slopes), $signed(y_slopes),
                    $signed({1'b0, x_centroid}), $signed({1'b0, y_centroid}));
            subap_count = subap_count + 1;
        end

        // rI_reciprocal is 19 bits but slope_calculation's rec_intensity port is
        // 18 -- bit 18 is pruned. Only reachable when sI <= 1 (dark subaperture).
        if (!reset && subap_valid && rI_reciprocal[18]) begin
            recip_msb_trips = recip_msb_trips + 1;
        end
    end

    // -------------------------------------------------------------------------
    // Capture + report
    // -------------------------------------------------------------------------
    integer fh;
    integer m;
    reg signed [MODE_BITS-1:0] coeff;
    reg captured = 1'b0;

    always @(posedge clk) begin
        if (!reset && done && !captured) begin
            captured = 1'b1;

            $display("");
            $display("=====================================================");
            $display(" Zynq SHWFS pipeline -- frame complete");
            $display("=====================================================");
            $display("  cycles elapsed        : %0d", cycle_count);
            $display("  pixels streamed       : %0d (expected %0d)",
                     pixel_count, NUM_PIXELS);
            $display("  pixel mismatches      : %0d", pixel_mismatches);
            $display("  frame_complete pulses : %0d", frame_done_count);
            $display("  subapertures valid    : %0d", subap_count);
            $display("  reciprocal MSB trips  : %0d", recip_msb_trips);
            $display("  unknown valid slope values : %0d", unknown_valid_slopes);
            $display("");
            done_cycle = $time / PERIOD_NS;
            $display("  LATENCY (cycles, and ns at 150 MHz)");
            $display("    final contributing pixel was #%0d", last_pixel_index);
            $display("    that pixel -> done              : %0d cy  (%0d.%0d ns)",
                     done_cycle - last_pixel_cycle,
                     ((done_cycle - last_pixel_cycle) * 1000) / 150,
                     (((done_cycle - last_pixel_cycle) * 1000) / 15) % 10);
            $display("    last subap_valid        -> done : %0d cy  (%0d.%0d ns)",
                     done_cycle - last_subap_cycle,
                     ((done_cycle - last_subap_cycle) * 1000) / 150,
                     (((done_cycle - last_subap_cycle) * 1000) / 15) % 10);
            $display("");
            $display("  THROUGHPUT");
            $display("    first input pixel cycle         : %0d", first_pixel_cycle);
            $display("    last input pixel cycle          : %0d", last_stream_cycle);
            $display("    first subap_valid cycle         : %0d", first_subap_cycle);
            $display("    done cycle                      : %0d", done_cycle);
            $display("    longest back-to-back pixel run  : %0d", max_run_len);
            $display("    active-region stall events      : %0d", stall_events);
            $display("");
            $display("  TRUE final-pixel-to-result (pixel ARRIVAL -> done)");
            $display("    final tile pixel #%0d arrived at cycle : %0d",
                     FINAL_TILE_PIXEL, final_tile_pixel_cycle);
            $display("    arrival -> done                 : %0d cy  (%0d.%0d ns)",
                     done_cycle - final_tile_pixel_cycle,
                     ((done_cycle - final_tile_pixel_cycle) * 1000) / 150,
                     (((done_cycle - final_tile_pixel_cycle) * 1000) / 15) % 10);
            $display("    accumulator internal depth      : %0d cy",
                     stage_accum_cycle - final_tile_pixel_cycle);
            $display("    anchor s1 -> done               : %0d cy",
                     done_cycle - stage_s1_cycle);
            $display("    anchor accum -> done            : %0d cy",
                     done_cycle - stage_accum_cycle);
            $display("");
            $display("  STAGE BOUNDARY CYCLES (last subaperture)");
            $display("    accumulator completion cycle    : %0d", stage_accum_cycle);
            $display("    reciprocal completion cycle     : %0d", stage_recip_cycle);
            $display("    slope (subap_valid) cycle       : %0d", last_subap_cycle);
            $display("    zernike (done) cycle            : %0d", done_cycle);
            $display("");

            fh = $fopen("zernikes_raw.txt", "w");
            $fwrite(fh, "raw_fpga_coeff\n");

            $display("  mode  raw (signed Q22)");
            for (m = 0; m < NUM_MODES; m = m + 1) begin
                coeff = zernike_out[m*MODE_BITS +: MODE_BITS];
                if (^coeff === 1'bx) begin
                    unknown_valid_coefficients = unknown_valid_coefficients + 1;
                end
                $display("   %2d   %0d", m + 1, coeff);
                $fwrite(fh, "%0d\n", coeff);

                // 25-bit field: |value| near 2^24 means the accumulator is
                // close to saturating, where the old 27-bit field had headroom.
                if (coeff > 16000000 || coeff < -16000000) begin
                    $display("  [WARN] mode %0d near 25-bit saturation", m + 1);
                end
            end
            $fclose(fh);
            $fclose(slope_fh);
            $display("  unknown valid coefficients : %0d", unknown_valid_coefficients);

            $display("");
            if (pixel_mismatches != 0) begin
                $display("  RESULT: FAIL -- %0d pixel mismatches", pixel_mismatches);
            end else begin
                $display("  RESULT: pixel stream verified, coefficients written");
            end
            $display("=====================================================");
            $display("");

            repeat (10) @(posedge clk);
            $finish;
        end
    end

    // -------------------------------------------------------------------------
    // Timeout guard
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        cycle_count = cycle_count + 1;
        if (cycle_count > TIMEOUT_CYCLES) begin
            $display("");
            $display("  [TIMEOUT] no 'done' after %0d cycles", TIMEOUT_CYCLES);
            $display("    pixels streamed       : %0d", pixel_count);
            $display("    frame_complete pulses : %0d", frame_done_count);
            $display("    subapertures valid    : %0d", subap_count);
            $display("");
            $finish;
        end
    end

endmodule
