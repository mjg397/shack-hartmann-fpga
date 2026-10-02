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

// reciprocal_u16_q18 has two internal pipeline stages, so reciprocal_wire is
// valid 2 cycles after sI is presented. Delay the side-band signals by the
// same 2 cycles so each centroid pairs with its own subaperture's reciprocal.
localparam integer RECIP_LATENCY = 2;

reg [19:0] xI_pipe [0:RECIP_LATENCY-1];
reg [19:0] yI_pipe [0:RECIP_LATENCY-1];
reg [7:0]  cd_pipe [0:RECIP_LATENCY-1];

integer i;

always @(posedge clk) begin
    if (reset) begin
        xI_out <= 0;
        yI_out <= 0;
        rI_out <= 0;
        centroids_done_out <= 0;
        for (i = 0; i < RECIP_LATENCY; i = i + 1) begin
            xI_pipe[i] <= 20'd0;
            yI_pipe[i] <= 20'd0;
            cd_pipe[i] <= 8'd0;
        end
    end
    else begin
        xI_pipe[0] <= xI_in;
        yI_pipe[0] <= yI_in;
        cd_pipe[0] <= centroids_done_in;

        for (i = 1; i < RECIP_LATENCY; i = i + 1) begin
            xI_pipe[i] <= xI_pipe[i-1];
            yI_pipe[i] <= yI_pipe[i-1];
            cd_pipe[i] <= cd_pipe[i-1];
        end

        xI_out             <= xI_pipe[RECIP_LATENCY-1];
        yI_out             <= yI_pipe[RECIP_LATENCY-1];
        rI_out             <= reciprocal_wire;
        centroids_done_out <= cd_pipe[RECIP_LATENCY-1];
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
