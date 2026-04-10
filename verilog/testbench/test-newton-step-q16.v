`timescale 1ns/1ns

module test_newton_step_q16;
  reg  [15:0] a_q1_15;
  reg  [15:0] x0_q0_16;
  wire [15:0] x1_q0_16;
  wire        saturated;

  integer fail_count;
  integer pass_count;
  integer test_count;
  integer i;
  integer j;
  integer seed;
  integer random_test_count;
  integer dense_window_start;
  integer dense_window_end;

  reg [15:0] exp_x1;
  reg        exp_sat;

  reg [15:0] x0_boundary [0:9];

  newton_step_q16 dut (
    .a_q1_15(a_q1_15),
    .x0_q0_16(x0_q0_16),
    .x1_q0_16(x1_q0_16),
    .saturated(saturated)
  );

  task compute_expected;
    input  [15:0] in_a;
    input  [15:0] in_x0;
    output [15:0] out_x1;
    output        out_sat;

    reg [31:0] ax_mul_local;
    reg [16:0] ax_q1_15_local;
    reg [16:0] ax_clamped_local;
    reg [16:0] two_minus_ax_local;
    reg [32:0] prod_mul_local;
    reg [17:0] x1_round_ext_local;
    begin
      ax_mul_local = in_a * in_x0;
      ax_q1_15_local = ax_mul_local[31:16];

      if (ax_q1_15_local > 17'h10000) begin
        ax_clamped_local = 17'h10000;
      end else begin
        ax_clamped_local = ax_q1_15_local;
      end

      two_minus_ax_local = 17'h10000 - ax_clamped_local;
      prod_mul_local = in_x0 * two_minus_ax_local;
      x1_round_ext_local = (prod_mul_local + 33'h000004000) >> 15;

      out_sat = (x1_round_ext_local > 18'h0FFFF);
      out_x1 = out_sat ? 16'hFFFF : x1_round_ext_local[15:0];
    end
  endtask

  task check_vector_exact;
    input [15:0] in_a;
    input [15:0] in_x0;
    input [15:0] exp_x1;
    input        exp_sat;
    begin
      a_q1_15 = in_a;
      x0_q0_16 = in_x0;
      #1;

      test_count = test_count + 1;

      if (x1_q0_16 !== exp_x1 || saturated !== exp_sat) begin
        $display(
          "FAIL a=0x%04h x0=0x%04h -> x1=0x%04h sat=%0d (expected 0x%04h sat=%0d)",
          in_a,
          in_x0,
          x1_q0_16,
          saturated,
          exp_x1,
          exp_sat
        );
        fail_count = fail_count + 1;
      end else begin
        pass_count = pass_count + 1;
      end
    end
  endtask

  task check_vector_ref;
    input [15:0] in_a;
    input [15:0] in_x0;
    begin
      compute_expected(in_a, in_x0, exp_x1, exp_sat);
      check_vector_exact(in_a, in_x0, exp_x1, exp_sat);
    end
  endtask

  initial begin
    fail_count = 0;
    pass_count = 0;
    test_count = 0;
    seed = 32'h1BADB002;
    random_test_count = 100000;
    dense_window_start = 16'h8000;
    dense_window_end = 16'h80FF;

    x0_boundary[0] = 16'h0000;
    x0_boundary[1] = 16'h0001;
    x0_boundary[2] = 16'h0002;
    x0_boundary[3] = 16'h7FFF;
    x0_boundary[4] = 16'h8000;
    x0_boundary[5] = 16'h8001;
    x0_boundary[6] = 16'hBFE8;
    x0_boundary[7] = 16'hFF80;
    x0_boundary[8] = 16'hFFFE;
    x0_boundary[9] = 16'hFFFF;

    /* Directed vectors from reciprocal-model.c trace output. */
    check_vector_exact(16'h8000, 16'hFF80, 16'hFFFF, 1'b1);
    check_vector_exact(16'hC000, 16'hAA72, 16'hAAAB, 1'b0);
    check_vector_exact(16'hE000, 16'h921F, 16'h9249, 1'b0);
    check_vector_exact(16'hAAAC, 16'hBFE8, 16'hC000, 1'b0);
    check_vector_exact(16'hFFFF, 16'h8020, 16'h8001, 1'b0);

    /* Boundary sweep on normalized a-range using interesting x0 boundaries. */
    for (i = 16'h8000; i <= 16'hFFFF; i = i + 16'h0111) begin
      for (j = 0; j < 10; j = j + 1) begin
        check_vector_ref(i[15:0], x0_boundary[j]);
      end
    end

    /* Dense local sweep around a lower edge of normalized range. */
    for (i = dense_window_start; i <= dense_window_end; i = i + 1) begin
      for (j = 0; j < 10; j = j + 1) begin
        check_vector_ref(i[15:0], x0_boundary[j]);
      end
    end

    /* Pseudo-random stress: normalized a-range [0x8000, 0xFFFF], full x0 range. */
    for (i = 0; i < random_test_count; i = i + 1) begin
      a_q1_15 = 16'h8000 + ($random(seed) & 16'h7FFF);
      x0_q0_16 = $random(seed);
      check_vector_ref(a_q1_15, x0_q0_16);
    end

    /* Additional structured sweep: full normalized-a with x0 in coarse ramps. */
    for (i = 16'h8000; i <= 16'hFFFF; i = i + 16'h0041) begin
      check_vector_ref(i[15:0], i[15:0]);
      check_vector_ref(i[15:0], ~i[15:0]);
      check_vector_ref(i[15:0], {i[7:0], i[15:8]});
    end

    /* A few out-of-range a values to verify clamping behavior path. */
    check_vector_ref(16'h0000, 16'h0000);
    check_vector_ref(16'h0001, 16'hFFFF);
    check_vector_ref(16'h7FFF, 16'hFFFF);

    if (fail_count == 0) begin
      $display(
        "PASS: newton_step_q16 comprehensive regression (%0d tests, %0d passed)",
        test_count,
        pass_count
      );
    end else begin
      $display(
        "FAIL: newton_step_q16 (%0d tests, %0d passed, %0d failed)",
        test_count,
        pass_count,
        fail_count
      );
    end

    $finish;
  end
endmodule
