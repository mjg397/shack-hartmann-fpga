`timescale 1ns/1ps
// =============================================================================
// tb_shwfs.sv -- SHWFS verification environment
//
// A layered, UVM-shaped environment built from plain SystemVerilog modules so
// it runs under Icarus and Verilator (neither supports UVM, and cocotb is not
// installed).
//
//   DRIVER      streams pixels straight into shwfs_processing, replicating the
//               camera framing (active line, HBLANK, frame_complete). It does
//               NOT use bram_model/streaming_emulator, so stimulus is fully
//               under test control and randomised frames can be built at run
//               time rather than pre-baked into a ROM.
//   REF MODEL   shwfs_ref_model -- an independent behavioural implementation of
//               the fixed-point spec.
//   SCOREBOARD  shwfs_scoreboard -- exact comparison of every centroid, slope
//               and Zernike coefficient against the reference model.
//   ASSERTIONS  shwfs_assertions -- protocol and range invariants that hold for
//               any data, catching bugs a value comparison cannot.
//
// Tests: directed frames supplied as .mem files, plus randomised frames
// generated in SystemVerilog from a seed. Every randomised frame is written to
// disk before use, so any failure is exactly reproducible.
//
//   +TESTS=<file>   manifest, one frame path per line   (directed)
//   +RANDOM=<n>     number of randomised frames          (default 8)
//   +SEED=<n>       base seed                            (default 1)
// =============================================================================

module tb_shwfs;

    localparam integer FRAME_W  = 256;
    localparam integer FRAME_H  = 256;
    localparam integer HBLANK   = 4;
    localparam integer N_TILES  = 256;
    localparam integer N_VALID  = 168;
    localparam integer N_MODES  = 10;
    localparam integer SUBAP_PX = 16;

    reg clk = 1'b0;
    reg reset = 1'b1;
    always #5 clk = ~clk;

    // ---- DUT interface ----
    reg  [7:0] s_data  = 8'd0;
    reg        s_valid = 1'b0;
    reg        frame_complete = 1'b0;
    wire       s_ready;

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

    shwfs_processing dut (
        .clk (clk), .reset (reset), .frame_complete (frame_complete),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready), .m_ready (1'b1),
        .m_data (zernike_out), .m_valid (done),
        .full_frame_complete_accumulator (full_frame_complete_accumulator),
        .subaps_done_accumulator (subaps_done_accumulator),
        .intensity_accumulator (intensity_accumulator),
        .xI_accumulator (xI_accumulator), .yI_accumulator (yI_accumulator),
        .xI_reciprocal (xI_reciprocal), .yI_reciprocal (yI_reciprocal),
        .rI_reciprocal (rI_reciprocal),
        .subaps_done_reciprocal (subaps_done_reciprocal),
        .x_centroid (x_centroid), .y_centroid (y_centroid),
        .x_slopes (x_slopes), .y_slopes (y_slopes),
        .new_subapeture (new_subapeture), .subap_valid (subap_valid)
    );

    shwfs_ref_model refm ();

    shwfs_scoreboard sb (
        .clk (clk), .reset (reset),
        .subap_valid (subap_valid),
        .x_centroid (x_centroid), .y_centroid (y_centroid),
        .x_slopes (x_slopes), .y_slopes (y_slopes),
        .done (done), .zernike_out (zernike_out),
        .checks_run (), .mismatches (),
        .centroid_mismatches (), .slope_mismatches (), .zernike_mismatches ()
    );

    shwfs_assertions asr (
        .clk (clk), .reset (reset),
        .s_valid (s_valid), .s_data (s_data),
        .subaps_done_accumulator (subaps_done_accumulator),
        .intensity_accumulator (intensity_accumulator),
        .xI_accumulator (xI_accumulator), .yI_accumulator (yI_accumulator),
        .rI_reciprocal (rI_reciprocal),
        .subaps_done_reciprocal (subaps_done_reciprocal),
        .x_centroid (x_centroid), .y_centroid (y_centroid),
        .x_slopes (x_slopes), .y_slopes (y_slopes),
        .subap_valid (subap_valid),
        .zernike_out (zernike_out), .done (done),
        .fail_count (), .subap_count (), .done_count ()
    );

    // ---- test state ----
    reg [7:0] frame [0:FRAME_W*FRAME_H-1];
    integer   tests_run, tests_passed, tests_failed;
    integer   total_checks, total_mismatches, total_assert_fails;
    integer   n_random, base_seed, fh, code;
    string    tests_file, line;
    integer   k, m;

    // -------------------------------------------------------------------------
    // DRIVER: stream one frame, replicating the camera's line framing.
    // -------------------------------------------------------------------------
    task automatic drive_frame();
        integer row, col, b;
        begin
            @(negedge clk);
            for (row = 0; row < FRAME_H; row = row + 1) begin
                for (col = 0; col < FRAME_W; col = col + 1) begin
                    s_data  = frame[row*FRAME_W + col];
                    s_valid = 1'b1;
                    @(negedge clk);
                end
                s_valid = 1'b0;
                s_data  = 8'd0;
                if (row == FRAME_H-1) begin
                    frame_complete = 1'b1;
                    @(negedge clk);
                    frame_complete = 1'b0;
                end else begin
                    for (b = 0; b < HBLANK; b = b + 1) @(negedge clk);
                end
            end
            // let the tail drain
            repeat (40) @(negedge clk);
        end
    endtask

    task automatic do_reset();
        begin
            reset = 1'b1;
            s_valid = 1'b0; s_data = 8'd0; frame_complete = 1'b0;
            repeat (6) @(negedge clk);
            reset = 1'b0;
            asr.reset_counts();
            @(negedge clk);
        end
    endtask

    // Copy the reference model's expectations into the scoreboard.
    task automatic arm_scoreboard();
        integer i;
        begin
            for (i = 0; i < N_TILES; i = i + 1) begin
                sb.exp_cx[i] = refm.exp_centroid_x[i];
                sb.exp_cy[i] = refm.exp_centroid_y[i];
                sb.exp_sx[i] = refm.exp_slope_x[i];
                sb.exp_sy[i] = refm.exp_slope_y[i];
            end
            for (i = 0; i < N_VALID; i = i + 1) sb.exp_idx[i] = refm.exp_valid_idx[i];
            for (i = 0; i < N_MODES; i = i + 1) sb.exp_z[i]   = refm.exp_zernike[i];
            sb.start_frame();
        end
    endtask

    // -------------------------------------------------------------------------
    // One test: load/emit a frame, compute expectations, run, judge.
    // -------------------------------------------------------------------------
    task automatic run_test(input string name, input string path);
        integer before_m, before_a, before_c;
        begin
            before_m = sb.mismatches;
            before_a = asr.fail_count;
            before_c = sb.checks_run;

            $readmemh(path, frame);
            refm.compute(path);

            do_reset();
            arm_scoreboard();
            drive_frame();

            tests_run = tests_run + 1;
            if (asr.done_count != 1) begin
                $display("  FAIL  %-28s no result produced (done_count=%0d)",
                         name, asr.done_count);
                tests_failed = tests_failed + 1;
            end else if (sb.mismatches != before_m || asr.fail_count != before_a) begin
                $display("  FAIL  %-28s %0d mismatches, %0d assertion failures",
                         name, sb.mismatches - before_m, asr.fail_count - before_a);
                tests_failed = tests_failed + 1;
            end else begin
                $display("  pass  %-28s %0d checks", name, sb.checks_run - before_c);
                tests_passed = tests_passed + 1;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Randomised frame generation, in SystemVerilog.
    //
    // Places a spot of random amplitude and random sub-pixel position in each
    // in-pupil subaperture. This is NOT the physical optical model -- that lives
    // in Python -- it is deliberately harsher, exercising intensity and position
    // corners the optical model never reaches, which is what the arithmetic
    // needs stressing against.
    // -------------------------------------------------------------------------
    task automatic make_random_frame(input integer seed, input string path);
        integer tile, row, col, py, px, cx, cy, amp, val, d2, i;
        integer sd;
        begin
            sd = seed;
            for (i = 0; i < FRAME_W*FRAME_H; i = i + 1) frame[i] = 8'd0;

            for (tile = 0; tile < N_TILES; tile = tile + 1) begin
                if (refm.pupil_bitmap[0][tile]) begin
                    row = tile / 16;  col = tile % 16;
                    cx  = 4 + ($urandom(sd) % 8);   sd = sd + 1;
                    cy  = 4 + ($urandom(sd) % 8);   sd = sd + 1;
                    amp = 20 + ($urandom(sd) % 236); sd = sd + 1;
                    for (py = 0; py < SUBAP_PX; py = py + 1)
                        for (px = 0; px < SUBAP_PX; px = px + 1) begin
                            d2  = (px-cx)*(px-cx) + (py-cy)*(py-cy);
                            val = (d2 > 6) ? 0 : (amp >> d2);
                            frame[(row*SUBAP_PX+py)*FRAME_W + (col*SUBAP_PX+px)] = val[7:0];
                        end
                end
            end
            // Persist it so any failure is reproducible.
            $writememh(path, frame);
        end
    endtask

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------
    initial begin
        tests_run = 0; tests_passed = 0; tests_failed = 0;

        if (!$value$plusargs("RANDOM=%d", n_random)) n_random = 8;
        if (!$value$plusargs("SEED=%d",   base_seed)) base_seed = 1;

        $display("");
        $display("=====================================================");
        $display(" SHWFS verification environment");
        $display("   driver + reference model + scoreboard + assertions");
        $display("=====================================================");
        $display("");

        // ---- directed tests from a manifest ----
        if ($value$plusargs("TESTS=%s", tests_file)) begin
            $display("DIRECTED TESTS (%s)", tests_file);
            fh = $fopen(tests_file, "r");
            if (fh == 0) begin
                $display("  cannot open manifest %s", tests_file);
            end else begin
                while (!$feof(fh)) begin
                    code = $fscanf(fh, "%s\n", line);
                    if (code == 1 && line.len() > 0)
                        run_test(line, line);
                end
                $fclose(fh);
            end
            $display("");
        end

        // ---- randomised tests generated in SystemVerilog ----
        if (n_random > 0) begin
            $display("RANDOMISED TESTS (%0d frames, base seed %0d)", n_random, base_seed);
            for (k = 0; k < n_random; k = k + 1) begin
                make_random_frame(base_seed + k*977, $sformatf("rand_%0d.mem", k));
                run_test($sformatf("random[seed=%0d]", base_seed + k*977),
                         $sformatf("rand_%0d.mem", k));
            end
            $display("");
        end

        // ---- verdict ----
        $display("=====================================================");
        $display("  tests run            : %0d", tests_run);
        $display("  passed               : %0d", tests_passed);
        $display("  failed               : %0d", tests_failed);
        $display("  scoreboard checks    : %0d", sb.checks_run);
        $display("    centroid mismatches: %0d", sb.centroid_mismatches);
        $display("    slope mismatches   : %0d", sb.slope_mismatches);
        $display("    zernike mismatches : %0d", sb.zernike_mismatches);
        $display("  assertion failures   : %0d", asr.fail_count);
        $display("");
        if (tests_failed == 0 && sb.mismatches == 0 && asr.fail_count == 0
            && tests_run > 0)
            $display("  RESULT: PASS");
        else
            $display("  RESULT: FAIL");
        $display("=====================================================");
        $display("");
        $finish;
    end

    // global timeout
    initial begin
        #40000000;
        $display("  RESULT: FAIL (timeout)");
        $finish;
    end

endmodule
