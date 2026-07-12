`timescale 1ns/1ps

module intensity_reciprocal_pipeline (
    input               clk,
    input               reset,
    input [19:0]        xI_in,
    input [19:0]        yI_in,
    input [15:0]        sI,
    input [7:0]         centroids_done_in,

    output wire [19:0]   xI_out,
    output wire [19:0]   yI_out,
    output reg  [26:0]   rI_out,
    output wire [7:0]    centroids_done_out
);

    wire [26:0] reciprocal_wire;
    wire divide_by_zero_wire;
    wire saturated_wire;

    reg [19:0] xI_internal[3:0];
    reg [19:0] yI_internal[3:0];
    reg [7:0]  centroids_done_internal[3:0];

    assign xI_out = xI_internal[3];
    assign yI_out = yI_internal[3];
    assign centroids_done_out = centroids_done_internal[3];

    always @(posedge clk) begin
        if (reset) begin
            rI_out <= 0;

            xI_internal[3] <= 0;
            xI_internal[2] <= 0;
            xI_internal[1] <= 0;
            xI_internal[0] <= 0;

            yI_internal[3] <= 0;
            yI_internal[2] <= 0;
            yI_internal[1] <= 0;
            yI_internal[0] <= 0;

            centroids_done_internal[3] <= 0;
            centroids_done_internal[2] <= 0;
            centroids_done_internal[1] <= 0;
            centroids_done_internal[0] <= 0;
        end
        else begin
            xI_internal[3] <= xI_internal[2];
            xI_internal[2] <= xI_internal[1];
            xI_internal[1] <= xI_internal[0];
            xI_internal[0] <= xI_in;

            yI_internal[3] <= yI_internal[2];
            yI_internal[2] <= yI_internal[1];
            yI_internal[1] <= yI_internal[0];
            yI_internal[0] <= yI_in;

            centroids_done_internal[3] <= centroids_done_internal[2];
            centroids_done_internal[2] <= centroids_done_internal[1];
            centroids_done_internal[1] <= centroids_done_internal[0];
            centroids_done_internal[0] <= centroids_done_in;

            rI_out <= reciprocal_wire;
        end
    end

    reciprocal_u16_q27_3stage recip (
        .clk            (clk),
        .reset          (reset),
        .v_u16          (sI),

        .reciprocal_q27 (reciprocal_wire),
        .divide_by_zero (divide_by_zero_wire),
        .saturated      (saturated_wire)
    );

endmodule 
