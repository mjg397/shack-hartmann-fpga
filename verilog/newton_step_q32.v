`timescale 1ns/1ps

module newton_step_q32 (
  input  wire [31:0] a_q1_31,
  input  wire [31:0] x0_u32,
  output wire [31:0] x1_q0_32,
  output wire        saturated
);
  wire [63:0] ax_mul;
  wire [32:0] ax_q1_31;
  wire [32:0] ax_q1_31_clamped;
  wire [32:0] two_minus_ax_q1_31;
  wire [64:0] prod_mul;
  wire [33:0] x1_rounded_ext;

  /*
   * One Newton refinement step in fixed-point:
   *   x1 = x0 * (2 - a*x0)
   *
   * Format mapping:
   *   a  : Q1.31
   *   x0 : u32 (Q0.32)
   *   x1 : Q0.32
   */

  /* ax = (a * x0) >> 32 : Q1.31 * Q0.32 -> Q1.31 */
  assign ax_mul = a_q1_31 * x0_u32;
  assign ax_q1_31 = {1'b0, ax_mul[63:32]};

  /* Clamp ax to 2.0 (0x100000000 in Q1.31) before subtraction. */
  assign ax_q1_31_clamped = (ax_q1_31 > 33'h100000000) ? 33'h100000000 : ax_q1_31;
  assign two_minus_ax_q1_31 = 33'h100000000 - ax_q1_31_clamped;

  /* x1 = (x0 * (2 - ax) + 2^30) >> 31 : round-to-nearest */
  assign prod_mul = x0_u32 * two_minus_ax_q1_31;
  assign x1_rounded_ext = {(prod_mul + 65'h0000000040000000) >> 31}[33:0];

  assign saturated = (x1_rounded_ext > 34'h0FFFFFFFF);
  assign x1_q0_32 = saturated ? 32'hFFFFFFFF : x1_rounded_ext[31:0];

endmodule
