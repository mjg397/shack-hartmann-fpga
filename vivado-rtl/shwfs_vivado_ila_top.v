`timescale 1ns/1ps

module shwfs_vivado_ila_top (
    input wire clk,
    input wire reset,
    input wire [3:0] zernike_mode_select,
    output wire [26:0] zernike_mode_data,
    output wire zernike_done
);

    wire [269:0] zernike_out_internal;
    wire         done_internal;

    assign zernike_mode_data = (zernike_mode_select < 4'd10)
                              ? zernike_out_internal[zernike_mode_select * 27 +: 27]
                              : 27'd0;
    assign zernike_done = done_internal;

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
        .zernike_out                      (zernike_out_internal),
        .done                             (done_internal)
    );

endmodule
