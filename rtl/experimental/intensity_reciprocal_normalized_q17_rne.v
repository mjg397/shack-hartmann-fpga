`timescale 1ns/1ps

// Side-band alignment wrapper for the separate normalized reciprocal experiment.
module intensity_reciprocal_normalized_q17_rne (
    input  wire        clk,
    input  wire        reset,
    input  wire [19:0] xI_in,
    input  wire [19:0] yI_in,
    input  wire [15:0] sI,
    input  wire [7:0]  centroids_done_in,
    output reg  [19:0] xI_out,
    output reg  [19:0] yI_out,
    output reg  [17:0] reciprocal_mantissa_q17,
    output reg  [3:0]  reciprocal_exponent,
    output reg  [18:0] reciprocal_debug,
    output reg  [7:0]  centroids_done_out
);
    localparam integer RECIP_LATENCY = 2;
    wire [17:0] mantissa_wire;
    wire [3:0] exponent_wire;
    wire divide_by_zero_wire;
    reg [19:0] xI_pipe [0:RECIP_LATENCY-1];
    reg [19:0] yI_pipe [0:RECIP_LATENCY-1];
    reg [7:0] cd_pipe [0:RECIP_LATENCY-1];
    integer i;

    always @(posedge clk) begin
        if (reset) begin
            xI_out <= 0;
            yI_out <= 0;
            reciprocal_mantissa_q17 <= 0;
            reciprocal_exponent <= 0;
            reciprocal_debug <= 0;
            centroids_done_out <= 0;
            for (i = 0; i < RECIP_LATENCY; i = i + 1) begin
                xI_pipe[i] <= 0;
                yI_pipe[i] <= 0;
                cd_pipe[i] <= 0;
            end
        end else begin
            xI_pipe[0] <= xI_in;
            yI_pipe[0] <= yI_in;
            cd_pipe[0] <= centroids_done_in;
            for (i = 1; i < RECIP_LATENCY; i = i + 1) begin
                xI_pipe[i] <= xI_pipe[i-1];
                yI_pipe[i] <= yI_pipe[i-1];
                cd_pipe[i] <= cd_pipe[i-1];
            end
            xI_out <= xI_pipe[RECIP_LATENCY-1];
            yI_out <= yI_pipe[RECIP_LATENCY-1];
            reciprocal_mantissa_q17 <= mantissa_wire;
            reciprocal_exponent <= exponent_wire;
            reciprocal_debug <= {1'b0, mantissa_wire};
            centroids_done_out <= cd_pipe[RECIP_LATENCY-1];
        end
    end

    reciprocal_u16_normalized_q17_rne reciprocal (
        .clk(clk), .reset(reset), .divisor_u16(sI),
        .mantissa_q17(mantissa_wire), .exponent(exponent_wire),
        .divide_by_zero(divide_by_zero_wire)
    );
endmodule
