`timescale 1ns/1ps

// =============================================================================
// Thresholding Center of Gravity Intensity Accumulator
//
// ONE added pipeline stage:
//
//   Input stage:
//       data_in
//         -> threshold
//         -> pipeline registers
//
//   Accumulator stage:
//       registered pixel * registered coordinate
//         -> add to selected accumulator
//         -> accumulator register
//
// Throughput:
//   1 pixel / clock
//
// Added latency:
//   1 clock cycle
// =============================================================================

module Thresholding_CoG_intensity_accumulator #(
    parameter NUM_SUBAPETURES_SQRT        = 16,
    parameter NUM_PIXELS_SUBAPETURE_SQRT = 16,
    parameter PIXEL_THRESHOLD             = 8'd5
)(
    input  wire clk,
    input  wire reset,
    input  wire valid,
    input  wire [7:0] data_in,

    output reg full_frame_complete,

    output reg [
        $clog2(
            NUM_PIXELS_SUBAPETURE_SQRT *
            NUM_PIXELS_SUBAPETURE_SQRT
        )-1:0
    ] subapetures_completed,

    output reg [
        $clog2(
            255 *
            NUM_PIXELS_SUBAPETURE_SQRT *
            NUM_PIXELS_SUBAPETURE_SQRT
        )-1:0
    ] intensity,

    output reg [18:0] x_intensity,
    output reg [18:0] y_intensity
);

    // =========================================================================
    // Widths
    // =========================================================================

    localparam integer SUBAP_W =
        $clog2(NUM_SUBAPETURES_SQRT);

    localparam integer PIXEL_POS_W =
        $clog2(NUM_PIXELS_SUBAPETURE_SQRT);

    localparam integer INTENSITY_W =
        $clog2(
            255 *
            NUM_PIXELS_SUBAPETURE_SQRT *
            NUM_PIXELS_SUBAPETURE_SQRT
        );

    localparam integer MOMENT_W = 20;


    // =========================================================================
    // Traversal state
    // =========================================================================

    reg [SUBAP_W-1:0] subap_col;
    reg [SUBAP_W-1:0] subap_row;

    reg [PIXEL_POS_W-1:0] count_pixel_h;
    reg [PIXEL_POS_W-1:0] count_pixel_v;


    // =========================================================================
    // Current input-pixel position flags
    // =========================================================================

    wire last_pixel_h;
    wire last_pixel_v;
    wire last_subap_col;
    wire last_subap_row;

    assign last_pixel_h =
        (count_pixel_h == NUM_PIXELS_SUBAPETURE_SQRT - 1);

    assign last_pixel_v =
        (count_pixel_v == NUM_PIXELS_SUBAPETURE_SQRT - 1);

    assign last_subap_col =
        (subap_col == NUM_SUBAPETURES_SQRT - 1);

    assign last_subap_row =
        (subap_row == NUM_SUBAPETURES_SQRT - 1);


    // =========================================================================
    // Current accumulator bank
    // =========================================================================

    wire [SUBAP_W-1:0] subap_idx;

    assign subap_idx = subap_col;


    // =========================================================================
    // Accumulator banks
    // =========================================================================

    reg [INTENSITY_W-1:0]
        i [0:NUM_SUBAPETURES_SQRT-1];

    reg [MOMENT_W-1:0]
        x_i [0:NUM_SUBAPETURES_SQRT-1];

    reg [MOMENT_W-1:0]
        y_i [0:NUM_SUBAPETURES_SQRT-1];


    // =========================================================================
    // Combinational threshold
    //
    // This is BEFORE the one added pipeline register.
    //
    // New first path:
    //   upstream data register
    //      -> threshold compare/mux
    //      -> s1_data register
    //
    // The multiplier is NOT in this path.
    // =========================================================================

    wire [7:0] thresholded_data;

    assign thresholded_data =
        (data_in < PIXEL_THRESHOLD)
            ? 8'd0
            : data_in;


    // =========================================================================
    // SINGLE ADDED PIPELINE STAGE
    //
    // Register:
    //   - thresholded pixel
    //   - x coordinate
    //   - y coordinate
    //   - destination accumulator bank
    //   - subaperture completion flag
    //   - frame completion flag
    //
    // Everything stays aligned to the same pixel.
    // =========================================================================

    reg s1_valid;

    reg [7:0] s1_data;

    reg [PIXEL_POS_W-1:0] s1_count_pixel_h;
    reg [PIXEL_POS_W-1:0] s1_count_pixel_v;

    reg [SUBAP_W-1:0] s1_subap_idx;

    reg s1_subap_done;
    reg s1_frame_done;


    always @(posedge clk) begin
        if (reset) begin
            s1_valid         <= 1'b0;
            s1_data          <= 8'd0;
            s1_count_pixel_h <= {PIXEL_POS_W{1'b0}};
            s1_count_pixel_v <= {PIXEL_POS_W{1'b0}};
            s1_subap_idx     <= {SUBAP_W{1'b0}};
            s1_subap_done    <= 1'b0;
            s1_frame_done    <= 1'b0;

        end else begin
            // Pipeline valid every cycle.
            s1_valid <= valid;

            if (valid) begin
                // Threshold result is registered BEFORE multiplier.
                s1_data <= thresholded_data;

                // Register multiplier operands.
                s1_count_pixel_h <= count_pixel_h;
                s1_count_pixel_v <= count_pixel_v;

                // Register destination bank.
                s1_subap_idx <= subap_idx;

                // Register metadata for this exact pixel.
                s1_subap_done <=
                    last_pixel_h &&
                    last_pixel_v;

                s1_frame_done <=
                    last_pixel_h   &&
                    last_pixel_v   &&
                    last_subap_col &&
                    last_subap_row;

            end else begin
                // Bubble through pipeline.
                s1_subap_done <= 1'b0;
                s1_frame_done <= 1'b0;
            end
        end
    end


    // =========================================================================
    // Input traversal control
    //
    // This still advances from the incoming stream.
    //
    // At a clock edge:
    //   - Stage 1 captures the OLD counter values for the current pixel
    //   - This block advances counters for the next pixel
    //
    // Nonblocking assignments preserve correct alignment.
    // =========================================================================

    always @(posedge clk) begin
        if (reset) begin
            subap_col     <= {SUBAP_W{1'b0}};
            subap_row     <= {SUBAP_W{1'b0}};
            count_pixel_h <= {PIXEL_POS_W{1'b0}};
            count_pixel_v <= {PIXEL_POS_W{1'b0}};

        end else if (valid) begin

            // -----------------------------------------------------------------
            // Horizontal pixel traversal
            // -----------------------------------------------------------------

            if (last_pixel_h) begin
                count_pixel_h <= {PIXEL_POS_W{1'b0}};

                if (last_subap_col) begin
                    subap_col <= {SUBAP_W{1'b0}};
                end else begin
                    subap_col <= subap_col + 1'b1;
                end

            end else begin
                count_pixel_h <= count_pixel_h + 1'b1;
            end


            // -----------------------------------------------------------------
            // Vertical pixel traversal
            // -----------------------------------------------------------------

            if (last_pixel_h && last_subap_col) begin

                if (last_pixel_v) begin
                    count_pixel_v <= {PIXEL_POS_W{1'b0}};

                    if (last_subap_row) begin
                        subap_row <= {SUBAP_W{1'b0}};
                    end else begin
                        subap_row <= subap_row + 1'b1;
                    end

                end else begin
                    count_pixel_v <= count_pixel_v + 1'b1;
                end
            end
        end
    end


    // =========================================================================
    // Accumulator combinational logic
    //
    // IMPORTANT:
    //
    // The multiplier and add remain TOGETHER.
    //
    // Path is now:
    //
    //   s1_data register
    //       \
    //        multiply
    //       /
    //   s1_coord register
    //       |
    //       + accumulator feedback
    //       |
    //       v
    //   accumulator register
    //
    // This is the one-cycle MAC stage.
    // =========================================================================

    wire [INTENSITY_W-1:0] i_next;
    wire [MOMENT_W-1:0]    x_i_next;
    wire [MOMENT_W-1:0]    y_i_next;

    assign i_next =
        i[s1_subap_idx] +
        s1_data;

    assign x_i_next =
        x_i[s1_subap_idx] +
        s1_data * s1_count_pixel_h;

    assign y_i_next =
        y_i[s1_subap_idx] +
        s1_data * s1_count_pixel_v;


    // =========================================================================
    // Accumulator update
    //
    // Still sustains one pixel every clock.
    //
    // Consecutive pixels targeting the same bank are safe:
    //
    //   cycle N:
    //       x_i <= x_i + pixel_N
    //
    //   cycle N+1:
    //       next combinational calculation sees updated x_i
    //       x_i <= x_i + pixel_N+1
    //
    // No extra feedback pipeline was inserted.
    // =========================================================================

    integer j;

    always @(posedge clk) begin
        if (reset) begin
            intensity             <= {INTENSITY_W{1'b0}};
            x_intensity           <= 19'd0;
            y_intensity           <= 19'd0;
            subapetures_completed <= 0;
            full_frame_complete   <= 1'b0;

            for (
                j = 0;
                j < NUM_SUBAPETURES_SQRT;
                j = j + 1
            ) begin
                i[j]   <= {INTENSITY_W{1'b0}};
                x_i[j] <= {MOMENT_W{1'b0}};
                y_i[j] <= {MOMENT_W{1'b0}};
            end

        end else begin
            // Default: completion is a one-cycle pulse.
            full_frame_complete <= 1'b0;

            if (s1_valid) begin

                // -------------------------------------------------------------
                // Final pixel of a subaperture
                // -------------------------------------------------------------

                if (s1_subap_done) begin

                    // *_next includes the final pixel.
                    intensity   <= i_next;
                    x_intensity <= x_i_next[18:0];
                    y_intensity <= y_i_next[18:0];

                    subapetures_completed <=
                        subapetures_completed + 1'b1;

                    // Clear completed bank for later reuse.
                    i[s1_subap_idx] <=
                        {INTENSITY_W{1'b0}};

                    x_i[s1_subap_idx] <=
                        {MOMENT_W{1'b0}};

                    y_i[s1_subap_idx] <=
                        {MOMENT_W{1'b0}};

                end else begin

                    // ---------------------------------------------------------
                    // Normal accumulation
                    // ---------------------------------------------------------

                    i[s1_subap_idx] <=
                        i_next;

                    x_i[s1_subap_idx] <=
                        x_i_next;

                    y_i[s1_subap_idx] <=
                        y_i_next;
                end


                // -------------------------------------------------------------
                // Full frame completion
                //
                // Aligned with the cycle in which the final pixel is actually
                // accumulated and emitted.
                // -------------------------------------------------------------

                if (s1_frame_done) begin
                    full_frame_complete <= 1'b1;
                end
            end
        end
    end


    // =========================================================================
    // Assertions
    // =========================================================================

    always @(posedge clk) begin
        if (!reset) begin

            // Incoming-stream coordinate checks.
            if (valid) begin

                if (
                    count_pixel_h >
                    NUM_PIXELS_SUBAPETURE_SQRT - 1
                ) begin
                    $error(
                        "local_x out of range: %0d",
                        count_pixel_h
                    );
                end

                if (
                    count_pixel_v >
                    NUM_PIXELS_SUBAPETURE_SQRT - 1
                ) begin
                    $error(
                        "local_y out of range: %0d",
                        count_pixel_v
                    );
                end
            end


            // Accumulator checks are aligned to pipelined valid.
            if (s1_valid) begin

                if (i_next > 16'd65280) begin
                    $error(
                        "si_sum overflow: %0d",
                        i_next
                    );
                end

                if (x_i_next > 20'd489600) begin
                    $error(
                        "mx_sum overflow: %0d",
                        x_i_next
                    );
                end

                if (y_i_next > 20'd489600) begin
                    $error(
                        "my_sum overflow: %0d",
                        y_i_next
                    );
                end
            end
        end
    end

endmodule