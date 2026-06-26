// `include "reciprocal_u16_q16.v"
`timescale 1ns/1ps

module intensity_reciprocal (
    input               clk,
    input               reset,
    input [19:0]        xI_in,
    input [19:0]        yI_in,
    input [15:0]        sI,
    input [7:0]         centroids_done_in,

    output reg [19:0]   xI_out,
    output reg [19:0]   yI_out,
    output reg [26:0]   rI_out,
    output reg [7:0]    centroids_done_out
);

wire [26:0] reciprocal_wire;
wire divide_by_zero_wire;
wire saturated_wire;

always @(posedge clk) begin
    if (reset) begin
        xI_out <= 0;
        yI_out <= 0;
        rI_out <= 0;
        centroids_done_out <= 0;
    end
    else begin
        xI_out <= xI_in;
        yI_out <= yI_in;
        rI_out <= reciprocal_wire;
        centroids_done_out <= centroids_done_in;
    end
end

reciprocal_u16_q27 recip (
    .v_u16          (sI),
    .reciprocal_q27 (reciprocal_wire),
    .divide_by_zero (divide_by_zero_wire),
    .saturated      (saturated_wire)
);

endmodule
