`timescale 1ns/1ps

// =============================================================================
// bram_model.v
//
// Simulation stand-in for the Zynq block RAM that streaming_emulator reads.
//
// The read is COMBINATIONAL, which is what streaming_emulator actually requires.
//
// In STATE_ACTIVE_FRAME the emulator registers the address for pixel N+1 during
// cycle N:
//
//     bram_addr <= pixel_index_next[15:2];
//
// so during cycle N+1 bram_addr already points at that pixel's word and the data
// must be available in the same cycle. A registered (1-cycle latency) read makes
// bram_rddata lag by one word, corrupting every pixel at a word boundary while
// leaving bytes 1-3 of each word correct -- a subtle, mostly-invisible error.
//
// NOTE FOR HARDWARE: real Zynq block RAM is synchronous. The emulator's
// STATE_LINE_PRIME comment ("wait one cycle for BRAM data") shows synchronous
// BRAM was intended, but the ACTIVE_FRAME address timing above contradicts it --
// it primes only the first word of each line. On real BRAM the same one-word lag
// would appear. This model matches what the RTL asks for; the RTL itself needs
// either an extra address pipeline stage or a registered-output BRAM to be
// correct on silicon.
//
// Holds 16384 x 32-bit words = 65536 packed 8-bit pixels (see pack_image.py).
// =============================================================================

module bram_model #(
    parameter ADDR_WIDTH = 14,
    parameter DEPTH      = 1 << ADDR_WIDTH,
    parameter INIT_FILE  = "image_packed.mem"
)(
    input  wire                  clk,
    input  wire [ADDR_WIDTH-1:0] addr,
    input  wire                  en,
    output wire [31:0]           rddata
);

    (* ram_style = "block" *) reg [31:0] mem [0:DEPTH-1];

    integer i;
    initial begin
        // Default to zero so an undersized init file cannot leave X's in the
        // stream, which would silently poison the accumulator.
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = 32'd0;
        end
        $readmemh(INIT_FILE, mem);
    end

    // Combinational read -- see the header note on why this is not registered.
    // `en` is ignored for data: the emulator drops it during blanking, and
    // gating the output would hand the pipeline X's on the first active cycle.
    assign rddata = mem[addr];

endmodule
