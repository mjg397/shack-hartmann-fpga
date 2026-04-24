`timescale 1ns/1ns

module test_reciprocal_u16_q27;
  localparam integer GOLDEN_COUNT = 65536;

  reg  [15:0] v_u16;
  wire [26:0] reciprocal_q27;
  wire        divide_by_zero;
  wire        saturated;

  // Each entry: bits[47:32] = v_u16 input, bits[31:27] = 5'b0, bits[26:0] = expected reciprocal_q27.
  reg [47:0] golden_io [0:GOLDEN_COUNT-1];

  integer fail_count;
  integer pass_count;
  integer test_count;
  integer i;

  reg [15:0] in_v_from_file;
  reg [26:0] exp_q27_from_file;
  reg        exp_div0_from_file;
  reg        exp_sat_from_file;

  reciprocal_u16_q27 dut (
    .v_u16(v_u16),
    .reciprocal_q27(reciprocal_q27),
    .divide_by_zero(divide_by_zero),
    .saturated(saturated)
  );

  task check_vector;
    input [15:0] in_v;
    input [26:0] exp_q27;
    input        exp_div0;
    input        exp_sat;
    begin
      v_u16 = in_v;
      #1;

      test_count = test_count + 1;

      if (reciprocal_q27 !== exp_q27 || divide_by_zero !== exp_div0 || saturated !== exp_sat) begin
        $display(
          "FAIL v=%0d -> q27=0x%07h div0=%0d sat=%0d (expected 0x%07h div0=%0d sat=%0d)",
          in_v,
          reciprocal_q27,
          divide_by_zero,
          saturated,
          exp_q27,
          exp_div0,
          exp_sat
        );
        fail_count = fail_count + 1;
      end else begin
        pass_count = pass_count + 1;
      end
    end
  endtask

  initial begin
    fail_count = 0;
    pass_count = 0;
    test_count = 0;

    /* Directed vectors — expected values from gen_reciprocal_q27_vectors.py. */
    check_vector(16'd0,     27'h7FFFFFF, 1'b1, 1'b1);
    check_vector(16'd1,     27'h7FFFFFF, 1'b0, 1'b0);
    check_vector(16'd2,     27'h4000000, 1'b0, 1'b0);
    check_vector(16'd3,     27'h2AAAAAB, 1'b0, 1'b0);
    check_vector(16'd7,     27'h1249249, 1'b0, 1'b0);
    check_vector(16'd255,   27'h0080808, 1'b0, 1'b0);
    check_vector(16'd256,   27'h0080000, 1'b0, 1'b0);
    check_vector(16'd257,   27'h007F808, 1'b0, 1'b0);
    check_vector(16'd1023,  27'h0020080, 1'b0, 1'b0);
    check_vector(16'd43692, 27'h0000C00, 1'b0, 1'b0);
    check_vector(16'd65535, 27'h0000800, 1'b0, 1'b0);

    /* Full golden sweep — packed as {v_u16[15:0], 5'b0, expected_q27[26:0]}. */
    $readmemh("verilog/testbench/vectors/reciprocal_golden_q27.memh", golden_io);

    for (i = 0; i < GOLDEN_COUNT; i = i + 1) begin
      in_v_from_file    = golden_io[i][47:32];
      exp_q27_from_file = golden_io[i][26:0];

      if (in_v_from_file !== i[15:0]) begin
        $display(
          "FAIL golden index mismatch at i=%0d: file_input=%0d",
          i,
          in_v_from_file
        );
        fail_count = fail_count + 1;
      end

      exp_div0_from_file = (in_v_from_file == 16'h0000);
      exp_sat_from_file  = (in_v_from_file == 16'h0000);

      check_vector(in_v_from_file, exp_q27_from_file, exp_div0_from_file, exp_sat_from_file);
    end

    if (fail_count == 0) begin
      $display(
        "PASS: reciprocal_u16_q27 directed + golden sweep (%0d tests, %0d passed)",
        test_count,
        pass_count
      );
    end else begin
      $display(
        "FAIL: reciprocal_u16_q27 (%0d tests, %0d passed, %0d failed)",
        test_count,
        pass_count,
        fail_count
      );
    end

    $finish;
  end
endmodule
