`timescale 1ns/1ps

// =============================================================================
// tb_zynq_top.v
//
// Simulation-only top for the Zynq SHWFS pipeline.
//
// This is what rtl/shwfs_vivado_top.v should have been. That file is a
// previous generation: it wires a 27-bit Q27 reciprocal into slope_calculation's
// 18-bit Q18 port (9 high bits pruned), declares zernike_out as 270 bits against
// a 250-bit accumulator, and instantiates streaming_emulator WITHOUT the BRAM
// ports the current emulator requires -- leaving bram_rddata floating, so every
// streamed pixel would be X. It is left untouched in rtl/ and is not
// compiled here.
//
// Chain (all current-generation modules):
//
//   bram_model -> streaming_emulator -> shwfs_processing
//                                         |- Thresholding_CoG_intensity_accumulator
//                                         |- intensity_reciprocal_q18
//                                         |- slope_calculation
//                                         `- ematrix_accumulator
//
// Every tap from the old ILA probe bus is exposed so the testbench can observe
// each pipeline stage.
// =============================================================================

module tb_zynq_top #(
    parameter IMAGE_FILE = "image_packed.mem"
)(
    input  wire        clk,
    input  wire        reset,

    // Streaming emulator taps
    output wire [7:0]  streamer_data,
    output wire        streamer_fv,
    output wire        streamer_lv,
    output wire        streamer_valid,
    output wire        frame_complete,

    // Intensity accumulator taps
    output wire        full_frame_complete_accumulator,
    output wire [7:0]  subaps_done_accumulator,
    output wire [15:0] intensity_accumulator,
    output wire [19:0] xI_accumulator,
    output wire [19:0] yI_accumulator,

    // Reciprocal taps (Q18: rI is 19 bits here, not 27)
    output wire [19:0] xI_reciprocal,
    output wire [19:0] yI_reciprocal,
    output wire [18:0] rI_reciprocal,
    output wire [7:0]  subaps_done_reciprocal,

    // Slope taps
    output wire [23:0] x_centroid,
    output wire [23:0] y_centroid,
    output wire [24:0] x_slopes,
    output wire [24:0] y_slopes,
    output wire        new_subapeture,
    output wire        subap_valid,

    // Zernike output (250 bits = 10 modes x 25)
    output wire [249:0] zernike_out,
    output wire         done
);

    // -------------------------------------------------------------------------
    // Image BRAM
    // -------------------------------------------------------------------------
    wire [13:0] bram_addr;
    wire        bram_en;
    wire [31:0] bram_rddata;

    bram_model #(
        .INIT_FILE (IMAGE_FILE)
    ) image_bram (
        .clk    (clk),
        .addr   (bram_addr),
        .en     (bram_en),
        .rddata (bram_rddata)
    );

    // -------------------------------------------------------------------------
    // Pixel stream source
    // -------------------------------------------------------------------------
    streaming_emulator streamer (
        .clk            (clk),
        .reset          (reset),

        .bram_addr      (bram_addr),
        .bram_en        (bram_en),
        .bram_rddata    (bram_rddata),

        .data           (streamer_data),
        .fv             (streamer_fv),
        .lv             (streamer_lv),
        .frame_complete (frame_complete),
        .valid          (streamer_valid)
    );

    // -------------------------------------------------------------------------
    // Processing pipeline
    // -------------------------------------------------------------------------
    wire s_ready_unused;

    shwfs_processing pipeline (
        .clk            (clk),
        .reset          (reset),
        .frame_complete (frame_complete),

        .s_data         (streamer_data),
        .s_valid        (streamer_valid),
        .s_ready        (s_ready_unused),
        .m_ready        (1'b1),

        .m_data         (zernike_out),
        .m_valid        (done),

        .full_frame_complete_accumulator (full_frame_complete_accumulator),
        .subaps_done_accumulator         (subaps_done_accumulator),
        .intensity_accumulator           (intensity_accumulator),
        .xI_accumulator                  (xI_accumulator),
        .yI_accumulator                  (yI_accumulator),

        .xI_reciprocal          (xI_reciprocal),
        .yI_reciprocal          (yI_reciprocal),
        .rI_reciprocal          (rI_reciprocal),
        .subaps_done_reciprocal (subaps_done_reciprocal),

        .x_centroid     (x_centroid),
        .y_centroid     (y_centroid),
        .x_slopes       (x_slopes),
        .y_slopes       (y_slopes),
        .new_subapeture (new_subapeture),
        .subap_valid    (subap_valid)
    );

endmodule
