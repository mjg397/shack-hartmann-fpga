`timescale 1ns/1ns

module test_reciprocal_u16_q16;
  localparam integer GOLDEN_COUNT = 65536;

  reg  [15:0] v_u16;
  wire [15:0] reciprocal_q16;
  wire        divide_by_zero;
  wire        saturated;

  reg [31:0] golden_io [0:GOLDEN_COUNT-1];

  integer fail_count;
  integer pass_count;
  integer test_count;
  integer i;

  reg [15:0] in_v_from_file;
  reg [15:0] exp_q16_from_file;
  reg        exp_div0_from_file;
  reg        exp_sat_from_file;

  reciprocal_u16_q16_comb dut (
    .v_u16(v_u16),
    .reciprocal_q16(reciprocal_q16),
    .divide_by_zero(divide_by_zero),
    .saturated(saturated)
  );

  task check_vector;
    input [15:0] in_v;
    input [15:0] exp_q16;
    input        exp_div0;
    input        exp_sat;
    begin
      v_u16 = in_v;
      #1;

      test_count = test_count + 1;

      if (reciprocal_q16 !== exp_q16 || divide_by_zero !== exp_div0 || saturated !== exp_sat) begin
        $display(
          "FAIL v=%0d -> q16=0x%04h div0=%0d sat=%0d (expected 0x%04h div0=%0d sat=%0d)",
          in_v,
          reciprocal_q16,
          divide_by_zero,
          saturated,
          exp_q16,
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

    /* Directed vectors from reciprocal-model.c traces and single-input runs. */
    check_vector(16'd0,     16'hFFFF, 1'b1, 1'b1);
    check_vector(16'd1,     16'hFFFF, 1'b0, 1'b0);
    check_vector(16'd2,     16'h8000, 1'b0, 1'b0);
    check_vector(16'd3,     16'h5556, 1'b0, 1'b0);
    check_vector(16'd7,     16'h2492, 1'b0, 1'b0);
    check_vector(16'd255,   16'h0101, 1'b0, 1'b0);
    check_vector(16'd256,   16'h0100, 1'b0, 1'b0);
    check_vector(16'd257,   16'h00FF, 1'b0, 1'b0);
    check_vector(16'd1023,  16'h0040, 1'b0, 1'b0);
    check_vector(16'd43692, 16'h0002, 1'b0, 1'b0);
    check_vector(16'd65535, 16'h0001, 1'b0, 1'b0);

    /* Full golden sweep from C model: packed as {input_u16, expected_q16}. */
    $readmemh("verilog/testbench/vectors/reciprocal_golden_io32.memh", golden_io);

    for (i = 0; i < GOLDEN_COUNT; i = i + 1) begin
      in_v_from_file = golden_io[i][31:16];
      exp_q16_from_file = golden_io[i][15:0];

      if (in_v_from_file !== i[15:0]) begin
        $display(
          "FAIL golden index mismatch at i=%0d: file_input=%0d",
          i,
          in_v_from_file
        );
        fail_count = fail_count + 1;
      end

      exp_div0_from_file = (in_v_from_file == 16'h0000);
      exp_sat_from_file = (in_v_from_file == 16'h0000);

      check_vector(in_v_from_file, exp_q16_from_file, exp_div0_from_file, exp_sat_from_file);
    end

    if (fail_count == 0) begin
      $display(
        "PASS: reciprocal_u16_q16_comb directed + golden sweep (%0d tests, %0d passed)",
        test_count,
        pass_count
      );
    end else begin
      $display(
        "FAIL: reciprocal_u16_q16_comb (%0d tests, %0d passed, %0d failed)",
        test_count,
        pass_count,
        fail_count
      );
    end

    $finish;
  end
endmodule
