`timescale 1ns/1ps

module newton_step_q16 (
  input  wire [15:0] a_q1_15,
  input  wire [15:0] x0_q0_16,
  output wire [15:0] x1_q0_16,
  output wire        saturated
);
  wire [31:0] ax_mul;
  wire [16:0] ax_q1_15;
  wire [16:0] ax_q1_15_clamped;
  wire [16:0] two_minus_ax_q1_15;
  wire [32:0] prod_mul;
  wire [17:0] x1_rounded_ext;

  /*
   * One Newton refinement step in fixed-point:
   *   x1 = x0 * (2 - a*x0)
   *
   * Format mapping:
   *   a  : Q1.15
   *   x0 : Q0.16
   *   x1 : Q0.16
   */

  /* ax = (a * x0) >> 16 : Q1.15 * Q0.16 -> Q1.15 */
  assign ax_mul = a_q1_15 * x0_q0_16;
  assign ax_q1_15 = {1'b0, ax_mul[31:16]};

  /* Clamp ax to 2.0 (0x10000 in Q1.15) before subtraction. */
  assign ax_q1_15_clamped = (ax_q1_15 > 17'h10000) ? 17'h10000 : ax_q1_15;
  assign two_minus_ax_q1_15 = 17'h10000 - ax_q1_15_clamped;

  /* x1 = (x0 * (2 - ax) + 2^14) >> 15 : round-to-nearest */
  assign prod_mul = x0_q0_16 * two_minus_ax_q1_15;
  assign x1_rounded_ext = {(prod_mul + 33'h000004000) >> 15}[17:0];

  assign saturated = (x1_rounded_ext > 18'h0FFFF);
  assign x1_q0_16 = saturated ? 16'hFFFF : x1_rounded_ext[15:0];

endmodule
