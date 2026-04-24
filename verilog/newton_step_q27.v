`timescale 1ns/1ps

module newton_step_q27 (
  input  wire [26:0] a_q1_26,
  input  wire [26:0] x0_u27,
  output wire [26:0] x1_q0_27,
  output wire        saturated
);
  wire [53:0] ax_mul;
  wire [27:0] ax_q1_26;
  wire [27:0] ax_q1_26_clamped;
  wire [27:0] two_minus_ax_q1_26;
  wire [54:0] prod_mul;
  wire [28:0] x1_rounded_ext;

  /*
   * One Newton refinement step in fixed-point:
   *   x1 = x0 * (2 - a*x0)
   *
   * Format mapping:
   *   a  : Q1.26
   *   x0 : u27 (Q0.27)
   *   x1 : Q0.27
   */

  /* ax = (a * x0) >> 27 : Q1.26 * Q0.27 -> Q1.26 */
  assign ax_mul = a_q1_26 * x0_u27;
  assign ax_q1_26 = {1'b0, ax_mul[53:27]};

  /* Clamp ax to 2.0 (0x8000000 in Q1.26) before subtraction. */
  assign ax_q1_26_clamped = (ax_q1_26 > 28'h8000000) ? 28'h8000000 : ax_q1_26;
  assign two_minus_ax_q1_26 = 28'h8000000 - ax_q1_26_clamped;

  /* x1 = (x0 * (2 - ax) + 2^25) >> 26 : round-to-nearest */
  assign prod_mul = x0_u27 * two_minus_ax_q1_26;
  assign x1_rounded_ext = (prod_mul + 55'h0000000002000000) >> 26;

  assign saturated = (x1_rounded_ext > 29'h07FFFFFF);
  assign x1_q0_27 = saturated ? 27'h7FFFFFF : x1_rounded_ext[26:0];

endmodule
