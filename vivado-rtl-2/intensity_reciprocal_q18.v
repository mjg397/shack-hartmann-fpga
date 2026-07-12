`timescale 1ns/1ps

module intensity_reciprocal_q18 (
    input               clk,
    input               reset,
    input  wire [19:0]  xI_in,
    input  wire [19:0]  yI_in,
    input  wire [15:0]  sI,
    input  wire [7:0]   centroids_done_in,

    output reg [19:0]   xI_out,
    output reg [19:0]   yI_out,
    output reg [18:0]   rI_out,
    output reg [7:0]    centroids_done_out
);

wire [18:0] reciprocal_wire;
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

reciprocal_u16_q18 recip (
    .clk            (clk),
    .reset          (reset),
    .v_u16          (sI),
    .reciprocal_q18 (reciprocal_wire),
    .divide_by_zero (divide_by_zero_wire),
    .saturated      (saturated_wire)
);

endmodule
