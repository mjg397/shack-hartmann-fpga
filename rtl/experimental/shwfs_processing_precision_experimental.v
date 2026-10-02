`timescale 1ns/1ps

// Explicit opt-in experimental processing path. The default build continues to
// instantiate rtl/shwfs_processing.v and is unaffected by this file.
module shwfs_processing_precision_experimental (
    input wire clk, input wire reset, input wire frame_complete,
    input wire [7:0] s_data, input wire s_valid,
    output wire s_ready, input wire m_ready,
    output wire [249:0] m_data, output wire m_valid,
    output wire full_frame_complete_accumulator,
    output wire [7:0] subaps_done_accumulator,
    output wire [15:0] intensity_accumulator,
    output wire [19:0] xI_accumulator, output wire [19:0] yI_accumulator,
    output wire [19:0] xI_reciprocal, output wire [19:0] yI_reciprocal,
    output wire [18:0] rI_reciprocal,
    output wire [7:0] subaps_done_reciprocal,
    output wire [23:0] x_centroid, output wire [23:0] y_centroid,
    output wire [24:0] x_slopes, output wire [24:0] y_slopes,
    output wire new_subapeture, output wire subap_valid
);
    assign s_ready = 1'b1;
    wire done_int;
    wire [17:0] reciprocal_mantissa_q17;
    wire [3:0] reciprocal_exponent;
    wire [18:0] xI_accumulator_internal;
    wire [18:0] yI_accumulator_internal;
    assign xI_accumulator = {1'b0, xI_accumulator_internal};
    assign yI_accumulator = {1'b0, yI_accumulator_internal};

    Thresholding_CoG_intensity_accumulator accumulator (
        .clk(clk), .reset(reset), .valid(s_valid), .data_in(s_data),
        .full_frame_complete(full_frame_complete_accumulator),
        .subapetures_completed(subaps_done_accumulator),
        .intensity(intensity_accumulator), .x_intensity(xI_accumulator_internal),
        .y_intensity(yI_accumulator_internal)
    );

    intensity_reciprocal_normalized_q17_rne reciprocal (
        .clk(clk), .reset(reset), .xI_in(xI_accumulator), .yI_in(yI_accumulator),
        .sI(intensity_accumulator), .centroids_done_in(subaps_done_accumulator),
        .xI_out(xI_reciprocal), .yI_out(yI_reciprocal),
        .reciprocal_mantissa_q17(reciprocal_mantissa_q17),
        .reciprocal_exponent(reciprocal_exponent),
        .reciprocal_debug(rI_reciprocal),
        .centroids_done_out(subaps_done_reciprocal)
    );

    slope_calculation_normalized_q17_rne slopes (
        .clk(clk), .rst(reset),
        .subapetures_completed(subaps_done_reciprocal),
        .frame_complete(frame_complete),
        .reciprocal_mantissa_q17(reciprocal_mantissa_q17),
        .reciprocal_exponent(reciprocal_exponent),
        .x_intensity(xI_reciprocal[18:0]), .y_intensity(yI_reciprocal[18:0]),
        .x_centroid(x_centroid), .y_centroid(y_centroid),
        .x_slope(x_slopes), .y_slope(y_slopes),
        .new_subapeture(new_subapeture), .subap_valid(subap_valid)
    );

    ematrix_accumulator em (
        .clk(clk), .rst(reset), .sub_valid(subap_valid),
        .x_slope(x_slopes), .y_slope(y_slopes),
        .zernike_out(m_data), .done(done_int)
    );
    assign m_valid = done_int;
endmodule
