// `include "TCoG_intensity_accumulator.v"
// `include "streaming_emulator.v"
// `include "intensity_reciprocal.v"
`timescale 1ns/1ps

module shwfs_top (
    input wire clk,
    input wire reset,

    output reg [19:0] xI_reciprocal,
    output reg [19:0] yI_reciprocal,
    output reg [15:0] rI_reciprocal,
    output reg [7:0] subaps_done_reciprocal
);

    reg [7:0] streamer_data;
    reg streamer_fv;
    reg streamer_lv;
    reg frame_complete;
    reg streamer_valid;

    streaming_emulator streamer 
    (
        .clk            (clk),
        .reset          (reset),
        .data           (streamer_data),
        .fv             (streamer_fv),
        .lv             (streamer_lv),
        .frame_complete (frame_complete),
        .valid          (streamer_valid)
    );

    reg full_frame_complete_accumulator;
    reg [7:0] subaps_done_accumulator;
    reg [15:0] intensity_accumulator;
    reg [19:0] xI_accumulator;
    reg [19:0] yI_accumulator;

    TCoG_intensity_accumulator accumulator
    (
        .clk            (clk),
        .reset          (reset),

        // Inputs from streamer
        .valid          (streamer_valid),
        .data_in        (streamer_data),

        // Outputs 
        .full_frame_complete    (full_frame_complete_accumulator),
        .subapetures_completed  (subaps_done_accumulator),
        .intensity              (intensity_accumulator),
        .x_intensity            (xI_accumulator),
        .y_intensity            (yI_accumulator)
    );

    // reg [19:0] xI_reciprocal;
    // reg [19:0] yI_reciprocal;
    // reg [15:0] rI_reciprocal;
    // reg [7:0] subaps_done_reciprocal;

    intensity_reciprocal reciprocal
    (
        .clk                (clk),
        .reset              (reset),
        .xI_in              (xI_accumulator),
        .yI_in              (yI_accumulator),
        .sI                 (intensity_accumulator),
        .centroids_done_in  (subaps_done_accumulator),

        .xI_out             (xI_reciprocal),
        .yI_out             (yI_reciprocal),
        .rI_out             (rI_reciprocal),
        .centroids_done_out (subaps_done_reciprocal)
    );

endmodule
