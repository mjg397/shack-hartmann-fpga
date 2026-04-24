`timescale 1ns/1ns

module test_reciprocal_u16_q32;
  localparam integer GOLDEN_COUNT = 65536;

  reg  [15:0] v_u16;
  wire [31:0] reciprocal_q32;
  wire        divide_by_zero;
  wire        saturated;

  // Each entry: bits[47:32] = v_u16 input, bits[31:0] = expected reciprocal_q32.
  // $readmemh with 48-bit words requires a reg wide enough; use [47:0].
  reg [47:0] golden_io [0:GOLDEN_COUNT-1];

  integer fail_count;
  integer pass_count;
  integer test_count;
  integer i;

  reg [15:0] in_v_from_file;
  reg [31:0] exp_q32_from_file;
  reg        exp_div0_from_file;
  reg        exp_sat_from_file;

  reciprocal_u16_q32 dut (
    .v_u16(v_u16),
    .reciprocal_q32(reciprocal_q32),
    .divide_by_zero(divide_by_zero),
    .saturated(saturated)
  );

  task check_vector;
    input [15:0] in_v;
    input [31:0] exp_q32;
    input        exp_div0;
    input        exp_sat;
    begin
      v_u16 = in_v;
      #1;

      test_count = test_count + 1;

      if (reciprocal_q32 !== exp_q32 || divide_by_zero !== exp_div0 || saturated !== exp_sat) begin
        $display(
          "FAIL v=%0d -> q32=0x%08h div0=%0d sat=%0d (expected 0x%08h div0=%0d sat=%0d)",
          in_v,
          reciprocal_q32,
          divide_by_zero,
          saturated,
          exp_q32,
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

    /* Directed vectors — expected values from gen_reciprocal_q32_vectors.py. */
    check_vector(16'd0,     32'hFFFFFFFF, 1'b1, 1'b1);
    check_vector(16'd1,     32'hFFFFFFFF, 1'b0, 1'b0);
    check_vector(16'd2,     32'h80000000, 1'b0, 1'b0);
    check_vector(16'd3,     32'h55555556, 1'b0, 1'b0);
    check_vector(16'd7,     32'h24924925, 1'b0, 1'b0);
    check_vector(16'd255,   32'h01010101, 1'b0, 1'b0);
    check_vector(16'd256,   32'h01000000, 1'b0, 1'b0);
    check_vector(16'd257,   32'h00FF00FF, 1'b0, 1'b0);
    check_vector(16'd1023,  32'h00401004, 1'b0, 1'b0);
    check_vector(16'd43692, 32'h00017FFD, 1'b0, 1'b0);
    check_vector(16'd65535, 32'h00010001, 1'b0, 1'b0);

    /* Full golden sweep — packed as {v_u16[15:0], expected_q32[31:0]}. */
    $readmemh("verilog/testbench/vectors/reciprocal_golden_q32.memh", golden_io);

    for (i = 0; i < GOLDEN_COUNT; i = i + 1) begin
      in_v_from_file   = golden_io[i][47:32];
      exp_q32_from_file = golden_io[i][31:0];

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

      check_vector(in_v_from_file, exp_q32_from_file, exp_div0_from_file, exp_sat_from_file);
    end

    if (fail_count == 0) begin
      $display(
        "PASS: reciprocal_u16_q32 directed + golden sweep (%0d tests, %0d passed)",
        test_count,
        pass_count
      );
    end else begin
      $display(
        "FAIL: reciprocal_u16_q32 (%0d tests, %0d passed, %0d failed)",
        test_count,
        pass_count,
        fail_count
      );
    end

    $finish;
  end
endmodule
