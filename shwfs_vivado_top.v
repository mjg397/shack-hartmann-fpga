`timescale 1ns/1ps

module shwfs_vivado_top (
    input wire clk,
    input wire reset,

    output wire [7:0] streamer_data,
    output wire streamer_fv,
    output wire streamer_lv,
    output wire streamer_valid,
    output wire frame_complete,

    output wire full_frame_complete_accumulator,
    output wire [7:0] subaps_done_accumulator,
    output wire [15:0] intensity_accumulator,
    output wire [19:0] xI_accumulator,
    output wire [19:0] yI_accumulator,

    output wire [19:0] xI_reciprocal,
    output wire [19:0] yI_reciprocal,
    output wire [26:0] rI_reciprocal,
    output wire [7:0] subaps_done_reciprocal,

    output wire [26:0] x_centroid,
    output wire [26:0] y_centroid,
    output wire [26:0] x_slopes,
    output wire [26:0] y_slopes,
    output wire new_subapeture,
    output wire subap_valid,

    output wire [269:0] zernike_out,
    output wire done
);

    (* mark_debug = "true" *) wire [7:0] streamer_data_dbg;
    (* mark_debug = "true" *) wire streamer_fv_dbg;
    (* mark_debug = "true" *) wire streamer_lv_dbg;
    (* mark_debug = "true" *) wire streamer_valid_dbg;
    (* mark_debug = "true" *) wire frame_complete_dbg;
    (* mark_debug = "true" *) wire full_frame_complete_accumulator_dbg;
    (* mark_debug = "true" *) wire [7:0] subaps_done_accumulator_dbg;
    (* mark_debug = "true" *) wire [15:0] intensity_accumulator_dbg;
    (* mark_debug = "true" *) wire [19:0] xI_accumulator_dbg;
    (* mark_debug = "true" *) wire [19:0] yI_accumulator_dbg;
    (* mark_debug = "true" *) wire [19:0] xI_reciprocal_dbg;
    (* mark_debug = "true" *) wire [19:0] yI_reciprocal_dbg;
    (* mark_debug = "true" *) wire [26:0] rI_reciprocal_dbg;
    (* mark_debug = "true" *) wire [7:0] subaps_done_reciprocal_dbg;
    (* mark_debug = "true" *) wire [26:0] x_centroid_dbg;
    (* mark_debug = "true" *) wire [26:0] y_centroid_dbg;
    (* mark_debug = "true" *) wire [26:0] x_slopes_dbg;
    (* mark_debug = "true" *) wire [26:0] y_slopes_dbg;
    (* mark_debug = "true" *) wire new_subapeture_dbg;
    (* mark_debug = "true" *) wire subap_valid_dbg;
    (* mark_debug = "true" *) wire [269:0] zernike_out_dbg;
    (* mark_debug = "true" *) wire done_dbg;
    (* mark_debug = "true" *) wire [639:0] ila_probe_bus;

    assign streamer_data = streamer_data_dbg;
    assign streamer_fv = streamer_fv_dbg;
    assign streamer_lv = streamer_lv_dbg;
    assign streamer_valid = streamer_valid_dbg;
    assign frame_complete = frame_complete_dbg;
    assign full_frame_complete_accumulator = full_frame_complete_accumulator_dbg;
    assign subaps_done_accumulator = subaps_done_accumulator_dbg;
    assign intensity_accumulator = intensity_accumulator_dbg;
    assign xI_accumulator = xI_accumulator_dbg;
    assign yI_accumulator = yI_accumulator_dbg;
    assign xI_reciprocal = xI_reciprocal_dbg;
    assign yI_reciprocal = yI_reciprocal_dbg;
    assign rI_reciprocal = rI_reciprocal_dbg;
    assign subaps_done_reciprocal = subaps_done_reciprocal_dbg;
    assign x_centroid = x_centroid_dbg;
    assign y_centroid = y_centroid_dbg;
    assign x_slopes = x_slopes_dbg;
    assign y_slopes = y_slopes_dbg;
    assign new_subapeture = new_subapeture_dbg;
    assign subap_valid = subap_valid_dbg;
    assign zernike_out = zernike_out_dbg;
    assign done = done_dbg;

    streaming_emulator streamer (
        .clk            (clk),
        .reset          (reset),
        .data           (streamer_data_dbg),
        .fv             (streamer_fv_dbg),
        .lv             (streamer_lv_dbg),
        .frame_complete (frame_complete_dbg),
        .valid          (streamer_valid_dbg)
    );

    TCoG_intensity_accumulator accumulator (
        .clk                   (clk),
        .reset                 (reset),
        .valid                 (streamer_valid_dbg),
        .data_in               (streamer_data_dbg),
        .full_frame_complete   (full_frame_complete_accumulator_dbg),
        .subapetures_completed (subaps_done_accumulator_dbg),
        .intensity             (intensity_accumulator_dbg),
        .x_intensity           (xI_accumulator_dbg),
        .y_intensity           (yI_accumulator_dbg)
    );

    intensity_reciprocal reciprocal (
        .clk                (clk),
        .reset              (reset),
        .xI_in              (xI_accumulator_dbg),
        .yI_in              (yI_accumulator_dbg),
        .sI                 (intensity_accumulator_dbg),
        .centroids_done_in  (subaps_done_accumulator_dbg),
        .xI_out             (xI_reciprocal_dbg),
        .yI_out             (yI_reciprocal_dbg),
        .rI_out             (rI_reciprocal_dbg),
        .centroids_done_out (subaps_done_reciprocal_dbg)
    );

    slope_calculation slopes (
        .clk                   (clk),
        .rst                   (reset),
        .subapetures_completed (subaps_done_reciprocal_dbg),
        .frame_complete        (frame_complete_dbg),
        .rec_intensity         (rI_reciprocal_dbg),
        .x_intensity           (xI_reciprocal_dbg),
        .y_intensity           (yI_reciprocal_dbg),
        .x_centroid            (x_centroid_dbg),
        .y_centroid            (y_centroid_dbg),
        .x_slope               (x_slopes_dbg),
        .y_slope               (y_slopes_dbg),
        .new_subapeture        (new_subapeture_dbg),
        .subap_valid           (subap_valid_dbg)
    );

    ematrix_accumulator em (
        .clk         (clk),
        .rst         (reset),
        .sub_valid   (subap_valid_dbg),
        .x_slope     (x_slopes_dbg),
        .y_slope     (y_slopes_dbg),
        .zernike_out (zernike_out_dbg),
        .done        (done_dbg)
    );

    assign ila_probe_bus = {
        107'd0,
        zernike_out_dbg,
        done_dbg,
        subap_valid_dbg,
        new_subapeture_dbg,
        y_slopes_dbg,
        x_slopes_dbg,
        y_centroid_dbg,
        x_centroid_dbg,
        subaps_done_reciprocal_dbg,
        rI_reciprocal_dbg,
        yI_reciprocal_dbg,
        xI_reciprocal_dbg,
        yI_accumulator_dbg,
        xI_accumulator_dbg,
        intensity_accumulator_dbg,
        subaps_done_accumulator_dbg,
        full_frame_complete_accumulator_dbg,
        frame_complete_dbg,
        streamer_valid_dbg,
        streamer_lv_dbg,
        streamer_fv_dbg,
        streamer_data_dbg
    };

`ifdef USE_ILA
    ila_0 shwfs_ila (
        .clk    (clk),
        .probe0 (ila_probe_bus)
    );
`endif

endmodule
