`timescale 1ns/1ps

// Experimental reciprocal representation. The frozen baseline remains in
// rtl/reciprocal_u16_q18.v. For d=a*2^k, 1<=a<2, this ROM returns round(2^17/a)
// and k. The 18-bit mantissa still fits one DSP48E1 B port.
module reciprocal_u16_normalized_q17_rne (
    input  wire        clk,
    input  wire        reset,
    input  wire [15:0] divisor_u16,
    output reg  [17:0] mantissa_q17,
    output reg  [3:0]  exponent,
    output reg         divide_by_zero
);
    reg [17:0] reciprocal_rom [0:65535];
    reg [17:0] rom_data;
    reg [3:0] exponent_data;
    reg zero_data;

    initial begin
        $readmemh("reciprocal_normalized_q17_rne.mem", reciprocal_rom);
    end

    function [3:0] floor_log2_u16;
        input [15:0] value;
        begin
            casex (value)
                16'b1xxxxxxxxxxxxxxx: floor_log2_u16 = 4'd15;
                16'b01xxxxxxxxxxxxxx: floor_log2_u16 = 4'd14;
                16'b001xxxxxxxxxxxxx: floor_log2_u16 = 4'd13;
                16'b0001xxxxxxxxxxxx: floor_log2_u16 = 4'd12;
                16'b00001xxxxxxxxxxx: floor_log2_u16 = 4'd11;
                16'b000001xxxxxxxxxx: floor_log2_u16 = 4'd10;
                16'b0000001xxxxxxxxx: floor_log2_u16 = 4'd9;
                16'b00000001xxxxxxxx: floor_log2_u16 = 4'd8;
                16'b000000001xxxxxxx: floor_log2_u16 = 4'd7;
                16'b0000000001xxxxxx: floor_log2_u16 = 4'd6;
                16'b00000000001xxxxx: floor_log2_u16 = 4'd5;
                16'b000000000001xxxx: floor_log2_u16 = 4'd4;
                16'b0000000000001xxx: floor_log2_u16 = 4'd3;
                16'b00000000000001xx: floor_log2_u16 = 4'd2;
                16'b000000000000001x: floor_log2_u16 = 4'd1;
                default:              floor_log2_u16 = 4'd0;
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            rom_data       <= 18'd0;
            exponent_data  <= 4'd0;
            zero_data      <= 1'b0;
            mantissa_q17   <= 18'd0;
            exponent       <= 4'd0;
            divide_by_zero <= 1'b0;
        end else begin
            rom_data       <= reciprocal_rom[divisor_u16];
            exponent_data  <= floor_log2_u16(divisor_u16);
            zero_data      <= (divisor_u16 == 16'd0);
            mantissa_q17   <= rom_data;
            exponent       <= exponent_data;
            divide_by_zero <= zero_data;
        end
    end
endmodule
