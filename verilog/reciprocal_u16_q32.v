// `include "newton_step_q32.v"
`timescale 1ns/1ps

module reciprocal_u16_q32(
  input  wire [15:0] v_u16,
  output wire [31:0] reciprocal_q32,
  output wire        divide_by_zero,
  output wire        saturated
);
  // 8 bit LUT in ROM (logic elements)
  localparam [4095:0] RECIP_SEED_LUT_Q0_16 = {
    16'h8020, 16'h8060, 16'h80A1, 16'h80E2, 16'h8123, 16'h8164, 16'h81A5, 16'h81E7,
    16'h8229, 16'h826B, 16'h82AE, 16'h82F1, 16'h8334, 16'h8377, 16'h83BB, 16'h83FF,
    16'h8443, 16'h8488, 16'h84CC, 16'h8511, 16'h8557, 16'h859C, 16'h85E2, 16'h8628,
    16'h866F, 16'h86B6, 16'h86FD, 16'h8744, 16'h878C, 16'h87D3, 16'h881C, 16'h8864,
    16'h88AD, 16'h88F6, 16'h8940, 16'h8989, 16'h89D3, 16'h8A1E, 16'h8A68, 16'h8AB3,
    16'h8AFF, 16'h8B4A, 16'h8B96, 16'h8BE2, 16'h8C2F, 16'h8C7C, 16'h8CC9, 16'h8D17,
    16'h8D65, 16'h8DB3, 16'h8E02, 16'h8E51, 16'h8EA0, 16'h8EF0, 16'h8F40, 16'h8F90,
    16'h8FE1, 16'h9032, 16'h9083, 16'h90D5, 16'h9127, 16'h9179, 16'h91CC, 16'h921F,
    16'h9273, 16'h92C7, 16'h931B, 16'h9370, 16'h93C5, 16'h941B, 16'h9470, 16'h94C7,
    16'h951D, 16'h9574, 16'h95CC, 16'h9624, 16'h967C, 16'h96D5, 16'h972E, 16'h9787,
    16'h97E1, 16'h983B, 16'h9896, 16'h98F1, 16'h994D, 16'h99A9, 16'h9A05, 16'h9A62,
    16'h9AC0, 16'h9B1D, 16'h9B7C, 16'h9BDA, 16'h9C39, 16'h9C99, 16'h9CF9, 16'h9D59,
    16'h9DBA, 16'h9E1C, 16'h9E7E, 16'h9EE0, 16'h9F43, 16'h9FA6, 16'hA00A, 16'hA06E,
    16'hA0D3, 16'hA138, 16'hA19E, 16'hA204, 16'hA26B, 16'hA2D3, 16'hA33A, 16'hA3A3,
    16'hA40C, 16'hA475, 16'hA4DF, 16'hA549, 16'hA5B4, 16'hA620, 16'hA68C, 16'hA6F8,
    16'hA766, 16'hA7D3, 16'hA842, 16'hA8B1, 16'hA920, 16'hA990, 16'hAA01, 16'hAA72,
    16'hAAE4, 16'hAB56, 16'hABC9, 16'hAC3D, 16'hACB1, 16'hAD26, 16'hAD9B, 16'hAE11,
    16'hAE88, 16'hAEFF, 16'hAF77, 16'hAFF0, 16'hB069, 16'hB0E3, 16'hB15D, 16'hB1D8,
    16'hB254, 16'hB2D1, 16'hB34E, 16'hB3CC, 16'hB44B, 16'hB4CA, 16'hB54A, 16'hB5CB,
    16'hB64C, 16'hB6CE, 16'hB751, 16'hB7D5, 16'hB859, 16'hB8DE, 16'hB964, 16'hB9EB,
    16'hBA72, 16'hBAFB, 16'hBB83, 16'hBC0D, 16'hBC98, 16'hBD23, 16'hBDAF, 16'hBE3C,
    16'hBECA, 16'hBF59, 16'hBFE8, 16'hC078, 16'hC109, 16'hC19B, 16'hC22E, 16'hC2C2,
    16'hC357, 16'hC3EC, 16'hC482, 16'hC51A, 16'hC5B2, 16'hC64B, 16'hC6E5, 16'hC780,
    16'hC81C, 16'hC8B9, 16'hC957, 16'hC9F6, 16'hCA96, 16'hCB36, 16'hCBD8, 16'hCC7B,
    16'hCD1F, 16'hCDC4, 16'hCE6A, 16'hCF11, 16'hCFB9, 16'hD062, 16'hD10C, 16'hD1B7,
    16'hD263, 16'hD311, 16'hD3BF, 16'hD46F, 16'hD520, 16'hD5D2, 16'hD685, 16'hD73A,
    16'hD7EF, 16'hD8A6, 16'hD95E, 16'hDA17, 16'hDAD1, 16'hDB8D, 16'hDC4A, 16'hDD08,
    16'hDDC8, 16'hDE88, 16'hDF4B, 16'hE00E, 16'hE0D3, 16'hE199, 16'hE260, 16'hE329,
    16'hE3F4, 16'hE4BF, 16'hE58C, 16'hE65B, 16'hE72B, 16'hE7FC, 16'hE8CF, 16'hE9A4,
    16'hEA7A, 16'hEB51, 16'hEC2A, 16'hED05, 16'hEDE1, 16'hEEBF, 16'hEF9F, 16'hF080,
    16'hF163, 16'hF247, 16'hF32D, 16'hF415, 16'hF4FF, 16'hF5EA, 16'hF6D7, 16'hF7C6,
    16'hF8B7, 16'hF9A9, 16'hFA9E, 16'hFB94, 16'hFC8C, 16'hFD86, 16'hFE82, 16'hFF80
  };

  function [4:0] clz16;
    input [15:0] x;
    begin
      casez (x)
        16'b1???????????????: clz16 = 5'd0;
        16'b01??????????????: clz16 = 5'd1;
        16'b001?????????????: clz16 = 5'd2;
        16'b0001????????????: clz16 = 5'd3;
        16'b00001???????????: clz16 = 5'd4;
        16'b000001??????????: clz16 = 5'd5;
        16'b0000001?????????: clz16 = 5'd6;
        16'b00000001????????: clz16 = 5'd7;
        16'b000000001???????: clz16 = 5'd8;
        16'b0000000001??????: clz16 = 5'd9;
        16'b00000000001?????: clz16 = 5'd10;
        16'b000000000001????: clz16 = 5'd11;
        16'b0000000000001???: clz16 = 5'd12;
        16'b00000000000001??: clz16 = 5'd13;
        16'b000000000000001?: clz16 = 5'd14;
        16'b0000000000000001: clz16 = 5'd15;
        default:             clz16 = 5'd16;
      endcase
    end
  endfunction

  wire        v_is_zero;
  wire [15:0] v_safe;
  wire [4:0]  shift_left;
  wire [15:0] a_q1_15;
  wire [31:0] a_q1_31;
  wire [7:0]  lut_idx;
  wire [15:0] seed_q0_16;
  wire [31:0] x0_u32;
  wire [31:0] x1_q0_32;
  wire        x1_saturated;
  wire [31:0] x2_q0_32;
  wire        x2_saturated;
  wire [4:0]  msb_index;
  wire [31:0] denorm_round_bias;
  wire [32:0] denorm_numer;
  wire [32:0] out_q0_32_ext;
  wire        out_sat;

  assign v_is_zero = (v_u16 == 16'h0000);
  assign v_safe = v_is_zero ? 16'h0001 : v_u16;

  /* Normalize integer input to Q1.15 in [1, 2). */
  assign shift_left = clz16(v_safe);
  assign a_q1_15 = v_safe << shift_left;

  /* Zero-extend to Q1.31 for the 32-bit Newton step. */
  assign a_q1_31 = {a_q1_15, 16'h0000};

  /* Use the 8 bits under the leading one as the LUT index. */
  assign lut_idx = a_q1_15[14:7];
  assign seed_q0_16 = RECIP_SEED_LUT_Q0_16[16*lut_idx +: 16];

  /* Zero-extend seed to Q0.32. */
  assign x0_u32 = {seed_q0_16, 16'h0000};

  /* First Newton refinement step. */
  newton_step_q32 u_newton_step_0 (
    .a_q1_31(a_q1_31),
    .x0_u32(x0_u32),
    .x1_q0_32(x1_q0_32),
    .saturated(x1_saturated)
  );

  /* Second Newton refinement step. */
  newton_step_q32 u_newton_step_1 (
    .a_q1_31(a_q1_31),
    .x0_u32(x1_q0_32),
    .x1_q0_32(x2_q0_32),
    .saturated(x2_saturated)
  );

  /* De-normalize: result = x2 >> msb_index (with rounding). */
  assign msb_index = 5'd15 - shift_left;
  assign denorm_round_bias = (msb_index == 5'd0) ? 32'h00000000 : (32'h00000001 << (msb_index - 1'b1));
  assign denorm_numer = {1'b0, x2_q0_32} + {1'b0, denorm_round_bias};
  assign out_q0_32_ext = (msb_index == 5'd0) ? {1'b0, x2_q0_32} : (denorm_numer >> msb_index);

  assign out_sat = (out_q0_32_ext > 33'h0FFFFFFFF);
  assign divide_by_zero = v_is_zero;
  assign reciprocal_q32 = v_is_zero ? 32'hFFFFFFFF : (out_sat ? 32'hFFFFFFFF : out_q0_32_ext[31:0]);
  assign saturated = v_is_zero | out_sat;

endmodule
