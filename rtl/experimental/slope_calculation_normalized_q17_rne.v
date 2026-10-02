`timescale 1ns/1ps

// Separate experimental centroid/slope path. It retains the baseline Q18
// centroid, Q18 slope, reference ROM, output widths, and two-stage slope timing.
module slope_calculation_normalized_q17_rne (
    input  wire               clk,
    input  wire               rst,
    input  wire [7:0]         subapetures_completed,
    input  wire               frame_complete,
    input  wire [17:0]        reciprocal_mantissa_q17,
    input  wire [3:0]         reciprocal_exponent,
    input  wire [18:0]        x_intensity,
    input  wire [18:0]        y_intensity,
    output reg  [23:0]        x_centroid,
    output reg  [23:0]        y_centroid,
    output reg signed [24:0]  x_slope,
    output reg signed [24:0]  y_slope,
    output reg                new_subapeture,
    output reg                subap_valid
);
    reg [255:0] subap_bitmap_mem [0:0];
    reg signed [24:0] slopes_ref_x_mem [0:255];
    reg signed [24:0] slopes_ref_y_mem [0:255];
    wire [255:0] subap_bitmap = subap_bitmap_mem[0];

    initial begin
        $readmemh("subaperture_bitmap.mem", subap_bitmap_mem);
        $readmemh("slopes_ref_x.mem", slopes_ref_x_mem);
        $readmemh("slopes_ref_y.mem", slopes_ref_y_mem);
    end

    reg [7:0] current_subap;
    wire new_subap_event = (subapetures_completed > current_subap);
    wire [7:0] completed_subap_addr = subapetures_completed - 8'd1;

    (* use_dsp = "yes" *) wire [36:0] x_product =
        x_intensity * reciprocal_mantissa_q17;
    (* use_dsp = "yes" *) wire [36:0] y_product =
        y_intensity * reciprocal_mantissa_q17;

    // Convert xI * normalized_Q17 into centroid_Q18. For exponent zero the
    // product shifts left once. Otherwise it shifts right exponent-1 places.
    // The discarded portion is rounded nearest, ties upward.
    function [23:0] normalized_product_to_q18;
        input [36:0] product;
        input [3:0] exponent_value;
        reg [37:0] extended_product;
        reg [37:0] shifted_product;
        reg [4:0] right_shift;
        begin
            extended_product = {1'b0, product};
            if (exponent_value == 4'd0) begin
                shifted_product = extended_product << 1;
            end else begin
                right_shift = exponent_value - 1'b1;
                if (right_shift == 0) begin
                    shifted_product = extended_product;
                end else begin
                    shifted_product =
                        (extended_product + (38'd1 << (right_shift - 1'b1)))
                        >> right_shift;
                end
            end
            normalized_product_to_q18 = shifted_product[23:0];
        end
    endfunction

    wire [23:0] x_centroid_scaled =
        normalized_product_to_q18(x_product, reciprocal_exponent);
    wire [23:0] y_centroid_scaled =
        normalized_product_to_q18(y_product, reciprocal_exponent);

    reg stage1_valid;
    reg stage1_in_pupil;
    reg [23:0] stage1_x_centroid;
    reg [23:0] stage1_y_centroid;
    reg signed [24:0] stage1_x_ref;
    reg signed [24:0] stage1_y_ref;

    always @(posedge clk) begin
        if (rst) begin
            current_subap <= 0;
            stage1_valid <= 0;
            stage1_in_pupil <= 0;
            stage1_x_centroid <= 0;
            stage1_y_centroid <= 0;
            stage1_x_ref <= 0;
            stage1_y_ref <= 0;
        end else begin
            stage1_valid <= 0;
            if (new_subap_event) begin
                current_subap <= subapetures_completed;
                stage1_x_centroid <= x_centroid_scaled;
                stage1_y_centroid <= y_centroid_scaled;
                stage1_x_ref <= slopes_ref_x_mem[completed_subap_addr];
                stage1_y_ref <= slopes_ref_y_mem[completed_subap_addr];
                stage1_in_pupil <= subap_bitmap[completed_subap_addr];
                stage1_valid <= 1'b1;
            end
            if (frame_complete) current_subap <= 0;
        end
    end

    wire signed [24:0] stage1_x_centroid_s =
        $signed({1'b0, stage1_x_centroid});
    wire signed [24:0] stage1_y_centroid_s =
        $signed({1'b0, stage1_y_centroid});
    wire signed [25:0] raw_x_slope =
        $signed({stage1_x_centroid_s[24], stage1_x_centroid_s}) -
        $signed({stage1_x_ref[24], stage1_x_ref});
    wire signed [25:0] raw_y_slope =
        $signed({stage1_y_centroid_s[24], stage1_y_centroid_s}) -
        $signed({stage1_y_ref[24], stage1_y_ref});

    always @(posedge clk) begin
        if (rst) begin
            x_centroid <= 0;
            y_centroid <= 0;
            x_slope <= 0;
            y_slope <= 0;
            new_subapeture <= 0;
            subap_valid <= 0;
        end else begin
            new_subapeture <= 0;
            subap_valid <= 0;
            if (stage1_valid) begin
                x_centroid <= stage1_x_centroid;
                y_centroid <= stage1_y_centroid;
                x_slope <= raw_x_slope[24:0];
                y_slope <= raw_y_slope[24:0];
                new_subapeture <= 1'b1;
                subap_valid <= stage1_in_pupil;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst && stage1_valid) begin
            if (raw_x_slope[25] != raw_x_slope[24])
                $error("experimental x_slope overflow: raw=%0d", raw_x_slope);
            if (raw_y_slope[25] != raw_y_slope[24])
                $error("experimental y_slope overflow: raw=%0d", raw_y_slope);
        end
    end
endmodule
