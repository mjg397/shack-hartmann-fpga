`timescale 1ns/1ps

module reciprocal_u16_q18 #(
  // Set USE_BRAM to 1 for block RAM style, or 0 for distributed LUT ROM.
  parameter integer USE_BRAM = 1
) (
  input  wire        clk,
  input  wire        reset,
  input  wire [15:0] v_u16,
  output reg  [18:0] reciprocal_q18,
  output reg         divide_by_zero,
  output reg         saturated
);
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

  wire [15:0] v_safe     = (v_u16 == 16'h0000) ? 16'h0001 : v_u16;
  wire        v_is_zero  = (v_u16 == 16'h0000);
  wire [4:0]  shift_left = clz16(v_safe);
  wire [15:0] a_q1_15    = v_safe << shift_left;
  wire [7:0]  lut_idx    = a_q1_15[14:7];

  generate
    if (USE_BRAM) begin : gen_bram
      (* ram_style = "block" *) reg [18:0] recip_rom [0:65535];
      reg [18:0] rom_data;

      initial begin
        $readmemh("vivado-rtl/reciprocal_u16_q18.mem", recip_rom);
      end

      always @(posedge clk) begin
        if (reset) begin
          rom_data       <= 19'd0;
          reciprocal_q18 <= 19'd0;
          divide_by_zero <= 1'b0;
          saturated      <= 1'b0;
        end else begin
          rom_data       <= recip_rom[v_u16];
          reciprocal_q18 <= v_is_zero ? 19'h7FFFF : rom_data;
          divide_by_zero <= v_is_zero;
          saturated      <= v_is_zero;
        end
      end
    end else begin : gen_lut
      (* ram_style = "distributed" *) reg [18:0] recip_rom [0:65535];
      reg [18:0] rom_data;

      initial begin
        $readmemh("vivado-rtl/reciprocal_u16_q18.mem", recip_rom);
      end

      always @(posedge clk) begin
        if (reset) begin
          rom_data       <= 19'd0;
          reciprocal_q18 <= 19'd0;
          divide_by_zero <= 1'b0;
          saturated      <= 1'b0;
        end else begin
          rom_data       <= recip_rom[v_u16];
          reciprocal_q18 <= v_is_zero ? 19'h7FFFF : rom_data;
          divide_by_zero <= v_is_zero;
          saturated      <= v_is_zero;
        end
      end
    end
  endgenerate

endmodule
