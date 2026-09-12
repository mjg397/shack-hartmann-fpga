`timescale 1ns/1ps

// =============================================================================
// slope_calculation.v
//
// Low-latency pipelined slope calculation.
//
// Pipeline:
//
//   Stage 0 combinational:
//       x_intensity * rec_intensity
//       y_intensity * rec_intensity
//
//   Stage 1 registers:                 <-- ONE ADDED PIPELINE STAGE
//       centroid products
//       matching X/Y reference values
//       pupil-valid metadata
//
//   Stage 2 registers:
//       centroid - reference
//       x_slope / y_slope
//       subap_valid / new_subapeture
//
// Throughput:
//       1 completed subaperture / clock once pipeline is full
//
// Added latency versus original:
//       exactly 1 clock cycle
//
// Fixed-point intent:
//       centroid product output : 24 bits
//       reference slope         : signed 25 bits
//       subtraction intermediate: signed 26 bits
//       slope output            : signed 25 bits
// =============================================================================

module slope_calculation (
    input  wire               clk,
    input  wire               rst,

    input  wire        [7:0]  subapetures_completed,
    input  wire               frame_complete,

    input  wire        [17:0] rec_intensity,
    input  wire        [18:0] x_intensity,
    input  wire        [18:0] y_intensity,

    output reg         [23:0] x_centroid,
    output reg         [23:0] y_centroid,

    output reg  signed [24:0] x_slope,
    output reg  signed [24:0] y_slope,

    output reg                new_subapeture,
    output reg                subap_valid
);


    // =========================================================================
    // Pupil bitmap
    // =========================================================================

    reg [255:0] subap_bitmap_mem [0:0];
    wire [255:0] subap_bitmap;

    initial begin
        $readmemh(
            "subaperture_bitmap.mem",
            subap_bitmap_mem
        );
    end

    assign subap_bitmap = subap_bitmap_mem[0];


    // =========================================================================
    // Reference slope ROMs
    //
    // IMPORTANT:
    // Keep these values signed and preserve bit 24.
    //
    // Original code did:
    //
    //   slopes_ref_x_mem[idx][23:0]
    //
    // which discarded the sign bit.
    // =========================================================================

    (* rom_style = "block" *)
    reg signed [24:0] slopes_ref_x_mem [0:255];

    (* rom_style = "block" *)
    reg signed [24:0] slopes_ref_y_mem [0:255];

    initial begin
        $readmemh(
            "slopes_ref_x.mem",
            slopes_ref_x_mem
        );

        $readmemh(
            "slopes_ref_y.mem",
            slopes_ref_y_mem
        );
    end


    // =========================================================================
    // Detect arrival of a newly completed subaperture
    //
    // current_subap represents how many completion events have already been
    // accepted into this slope pipeline.
    // =========================================================================

    reg [7:0] current_subap;

    wire new_subap_event;

    assign new_subap_event =
        (subapetures_completed > current_subap);


    // =========================================================================
    // Address corresponding to the most recently completed subaperture
    //
    // Only consumed when new_subap_event is true, so completed count is
    // expected to be nonzero.
    // =========================================================================

    wire [7:0] completed_subap_addr;

    assign completed_subap_addr =
        subapetures_completed - 8'd1;


    // =========================================================================
    // Stage 0: combinational centroid multiplication
    //
    // Current timing cone:
    //
    //   intensity
    //      *
    //   reciprocal
    //      |
    //      v
    //   DSP output
    //
    // The new Stage-1 registers terminate that path BEFORE subtraction.
    // =========================================================================

    wire [23:0] x_centroid_mult;
    wire [23:0] y_centroid_mult;

    unsigned_mult x_mult (
        .out (x_centroid_mult),
        .a   (x_intensity),
        .b   (rec_intensity)
    );

    unsigned_mult y_mult (
        .out (y_centroid_mult),
        .a   (y_intensity),
        .b   (rec_intensity)
    );


    // =========================================================================
    // Stage 1 pipeline registers
    //
    // This is the ONE newly added pipeline stage.
    //
    // Register together:
    //   - centroid product
    //   - corresponding signed reference
    //   - pupil-valid bit
    //
    // This guarantees all metadata remains associated with the same
    // subaperture.
    // =========================================================================

    reg        stage1_valid;
    reg        stage1_in_pupil;

    reg [23:0] stage1_x_centroid;
    reg [23:0] stage1_y_centroid;

    reg signed [24:0] stage1_x_ref;
    reg signed [24:0] stage1_y_ref;


    always @(posedge clk) begin
        if (rst) begin
            current_subap      <= 8'd0;

            stage1_valid       <= 1'b0;
            stage1_in_pupil    <= 1'b0;

            stage1_x_centroid  <= 24'd0;
            stage1_y_centroid  <= 24'd0;

            stage1_x_ref       <= 25'sd0;
            stage1_y_ref       <= 25'sd0;

        end else begin
            // Default: bubble unless a new completed subaperture arrives.
            stage1_valid <= 1'b0;

            if (new_subap_event) begin
                // Track the completion count that has entered the pipeline.
                //
                // Using the observed completed count is safer than merely +1
                // if there is ever a gap or delayed observation.
                current_subap <= subapetures_completed;

                // ---------------------------------------------------------
                // Register DSP multiplication results.
                // ---------------------------------------------------------
                stage1_x_centroid <= x_centroid_mult;
                stage1_y_centroid <= y_centroid_mult;

                // ---------------------------------------------------------
                // Register matching signed references.
                //
                // This coding style also exposes a synchronous registered
                // ROM-read boundary suitable for block ROM inference.
                // ---------------------------------------------------------
                stage1_x_ref <=
                    slopes_ref_x_mem[completed_subap_addr];

                stage1_y_ref <=
                    slopes_ref_y_mem[completed_subap_addr];

                // ---------------------------------------------------------
                // Register matching pupil metadata.
                // ---------------------------------------------------------
                stage1_in_pupil <=
                    subap_bitmap[completed_subap_addr];

                stage1_valid <= 1'b1;
            end

            // Optional frame resynchronization.
            //
            // The original module did not use frame_complete. Resetting the
            // event tracker here allows the completion count to restart at
            // zero for a new frame.
            if (frame_complete) begin
                current_subap <= 8'd0;
            end
        end
    end


    // =========================================================================
    // Stage 2 combinational subtraction
    //
    // Centroid is unsigned/nonnegative:
    //
    //     {1'b0, stage1_x_centroid}
    //
    // Reference is signed:
    //
    //     stage1_x_ref
    //
    // Use a 26-bit intermediate to preserve subtraction range.
    // =========================================================================

    wire signed [24:0] stage1_x_centroid_s;
    wire signed [24:0] stage1_y_centroid_s;

    assign stage1_x_centroid_s =
        $signed({1'b0, stage1_x_centroid});

    assign stage1_y_centroid_s =
        $signed({1'b0, stage1_y_centroid});


    wire signed [25:0] raw_x_slope;
    wire signed [25:0] raw_y_slope;

    assign raw_x_slope =
        $signed({
            stage1_x_centroid_s[24],
            stage1_x_centroid_s
        })
        -
        $signed({
            stage1_x_ref[24],
            stage1_x_ref
        });

    assign raw_y_slope =
        $signed({
            stage1_y_centroid_s[24],
            stage1_y_centroid_s
        })
        -
        $signed({
            stage1_y_ref[24],
            stage1_y_ref
        });


    // =========================================================================
    // Stage 2 output registers
    //
    // Existing architectural slope register boundary.
    //
    // subap_valid and new_subapeture are registered from the SAME stage1_valid
    // token, so they are aligned with x_slope/y_slope.
    // =========================================================================

    always @(posedge clk) begin
        if (rst) begin
            x_centroid      <= 24'd0;
            y_centroid      <= 24'd0;

            x_slope         <= 25'sd0;
            y_slope         <= 25'sd0;

            new_subapeture  <= 1'b0;
            subap_valid     <= 1'b0;

        end else begin
            // Default one-cycle pulses.
            new_subapeture <= 1'b0;
            subap_valid    <= 1'b0;

            if (stage1_valid) begin
                // Preserve centroid outputs for observation/debugging.
                x_centroid <= stage1_x_centroid;
                y_centroid <= stage1_y_centroid;

                // Explicit 25-bit output selection.
                //
                // This matches your existing 25-bit slope interface.
                x_slope <= raw_x_slope[24:0];
                y_slope <= raw_y_slope[24:0];

                // These pulses correspond to the slope values loaded above.
                new_subapeture <= 1'b1;
                subap_valid    <= stage1_in_pupil;
            end
        end
    end


    // =========================================================================
    // Runtime checks
    // =========================================================================

    always @(posedge clk) begin
        if (!rst) begin

            // A valid output slope must belong to a pupil subaperture.
            if (subap_valid && !new_subapeture) begin
                $error(
                    "subap_valid asserted without new_subapeture"
                );
            end

            // 26-bit subtraction must fit the intended 25-bit output.
            //
            // Valid 25-bit signed extension requires bits [25] and [24]
            // to match.
            if (stage1_valid) begin
                if (raw_x_slope[25] != raw_x_slope[24]) begin
                    $error(
                        "x_slope overflow: raw=%0d",
                        raw_x_slope
                    );
                end

                if (raw_y_slope[25] != raw_y_slope[24]) begin
                    $error(
                        "y_slope overflow: raw=%0d",
                        raw_y_slope
                    );
                end
            end
        end
    end

endmodule


// =============================================================================
// Unsigned centroid multiplier
//
// 19-bit intensity moment × 18-bit reciprocal = 37-bit exact product.
//
// Output selects bits [35:12], preserving your original fixed-point scaling.
// =============================================================================

module unsigned_mult (
    output wire [23:0] out,
    input  wire [18:0] a,
    input  wire [17:0] b
);

    wire [36:0] mult_out;

    assign mult_out = a * b;

    assign out = mult_out[35:12];

endmodule