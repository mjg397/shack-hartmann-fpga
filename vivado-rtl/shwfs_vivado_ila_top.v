`timescale 1ns/1ps

module shwfs_vivado_ila_top (
    input wire clk,
    input wire reset
);

    shwfs_vivado_top pipeline (
        .clk                              (clk),
        .reset                            (reset),
        .streamer_data                    (),
        .streamer_fv                      (),
        .streamer_lv                      (),
        .streamer_valid                   (),
        .frame_complete                   (),
        .full_frame_complete_accumulator  (),
        .subaps_done_accumulator          (),
        .intensity_accumulator            (),
        .xI_accumulator                   (),
        .yI_accumulator                   (),
        .xI_reciprocal                    (),
        .yI_reciprocal                    (),
        .rI_reciprocal                    (),
        .subaps_done_reciprocal           (),
        .x_centroid                       (),
        .y_centroid                       (),
        .x_slopes                         (),
        .y_slopes                         (),
        .new_subapeture                   (),
        .subap_valid                      (),
        .zernike_out                      (),
        .done                             ()
    );

endmodule
