`timescale 1ns/1ps
// =============================================================================
// ematrix_accumulator.v
// Fully parallel E-matrix accumulator for Shack-Hartmann wavefront reconstruction.
// All 10 Zernike modes are computed in a single cycle per subaperture.
//
// Fixed-point formats
//   E matrix   : Q1.16  18-bit signed
//   Slopes     : Q4.23  27-bit signed
//   Products   : Q5.39  45-bit signed  (18 + 27 bits, no overflow)
//   x+y sum    : Q5.39  46-bit signed  (one guard bit)
//   Accumulator: sign-extended to 55 bits
//   Output     : Q4.23  27-bit signed  (bits [43:17] of accumulator)
//
// Key fix: intermediate e_x[]/e_y[] wire arrays removed. ModelSim loses the
// signed attribute on wire arrays indexed through a genvar, so the multiplier
// silently switches to unsigned. Reading directly from the ROM inside the
// generate block and casting with $signed() on both operands guarantees
// correct two's-complement multiplication in all Verilog simulators.
// =============================================================================

module ematrix_accumulator #(
    parameter NUM_MODES  = 10,   // Zernike modes  (rows of E)
    parameter NUM_SUBS   = 168,  // valid subapertures
    parameter NUM_SLOPES = 392   // 2 * NUM_SUBS (kept for port compatibility)
)(
    input  wire        clk,
    input  wire        rst,

    // sub_valid pulses high for one cycle when a subaperture slope is ready
    input  wire        sub_valid,
    input  wire signed [26:0] x_slope,   // Q4.23
    input  wire signed [26:0] y_slope,   // Q4.23

    // flattened output bus: mode m occupies bits [m*27 +: 27]
    output reg  [269:0] zernike_out,     // 10 × 27 bits, Q4.23 per mode
    output reg          done
);

// ---------------------------------------------------------------------------
// Local parameters
// ---------------------------------------------------------------------------
localparam integer SUB_BITS  = 8;    // ceil(log2(196)) = 8
localparam integer ACC_WIDTH = 55;   // Q5.39 + sign-extension headroom

localparam integer STATE_IDLE = 0;
localparam integer STATE_DONE = 1;

// Explicit-width constant avoids width-mismatch warnings on comparison
localparam [SUB_BITS-1:0] LAST_SUB = NUM_SUBS - 1;  // 8'hC3 = 195

// ---------------------------------------------------------------------------
// E-matrix ROM — split x / y halves for fully parallel reads
//   e_rom_x[m * NUM_SUBS + s] = E[m, 2s]    (x column, subaperture s)
//   e_rom_y[m * NUM_SUBS + s] = E[m, 2s+1]  (y column, subaperture s)
// Declared unsigned [17:0]; $signed() cast is applied at point of use so
// $readmemh loads raw bits without sign-extension surprises.
// ---------------------------------------------------------------------------
reg [17:0] e_rom_x [0:(NUM_MODES * NUM_SUBS)-1];
reg [17:0] e_rom_y [0:(NUM_MODES * NUM_SUBS)-1];

initial begin
    $readmemh("C:/Users/sjbar/OneDrive/Desktop/ECE5760/shack-hartmann-fpga/src/e_matrix_x.hex", e_rom_x);
    $readmemh("C:/Users/sjbar/OneDrive/Desktop/ECE5760/shack-hartmann-fpga/src/e_matrix_y.hex", e_rom_y);
end

// ---------------------------------------------------------------------------
// State and subaperture counter
// ---------------------------------------------------------------------------
reg                state;
reg [SUB_BITS-1:0] sub_counter;

// ---------------------------------------------------------------------------
// Parallel MAC — all NUM_MODES modes in one combinatorial stage
//
// Wires are declared inside the named generate scope (not as arrays) to avoid
// the ModelSim bug where signed attributes are dropped on genvar-indexed arrays.
// $signed() casts on both operands of each multiply ensure the tool treats the
// operation as signed regardless of synthesis / simulation tool version.
// The product MSB is bit [44]; it is used (not the operand MSBs) to
// sign-extend to ACC_WIDTH.
// ---------------------------------------------------------------------------
wire signed [ACC_WIDTH-1:0] mac_sum [0:NUM_MODES-1];

genvar m;
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : mac_gen
        // Force signed on the ROM read
        wire signed [17:0] ex = $signed(e_rom_x[m * NUM_SUBS + sub_counter]);
        wire signed [17:0] ey = $signed(e_rom_y[m * NUM_SUBS + sub_counter]);

        // 18-bit signed × 27-bit signed = 45-bit signed (exact, no overflow)
        wire signed [44:0] prod_x = ex * $signed(x_slope);
        wire signed [44:0] prod_y = ey * $signed(y_slope);

        // Sign-extend each 45-bit product to ACC_WIDTH using the product MSB [44]
        assign mac_sum[m] = {{10{prod_x[44]}}, prod_x}
                          + {{10{prod_y[44]}}, prod_y};
    end
endgenerate

// ---------------------------------------------------------------------------
// Accumulators — one per mode, Q5.39, ACC_WIDTH bits
// ---------------------------------------------------------------------------
reg signed [ACC_WIDTH-1:0] acc [0:NUM_MODES-1];

// Combinatorial next-accumulator value (acc + mac_sum this cycle)
wire signed [ACC_WIDTH-1:0] acc_next [0:NUM_MODES-1];

generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : acc_next_gen
        assign acc_next[m] = acc[m] + mac_sum[m];
    end
endgenerate

integer i;

// ---------------------------------------------------------------------------
// Sequential logic
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (rst) begin
        state       <= STATE_IDLE;
        sub_counter <= 0;
        done        <= 0;
        zernike_out <= 0;
        for (i = 0; i < NUM_MODES; i = i + 1)
            acc[i] <= 0;
    end
    else begin
        done <= 0;  // default: de-assert every cycle

        case (state)

            // ----------------------------------------------------------------
            // STATE_IDLE — wait for subaperture data
            // ----------------------------------------------------------------
            STATE_IDLE: begin
                if (sub_valid) begin

                    if (sub_counter == LAST_SUB) begin
                        // Last subaperture: latch  (not stale acc) and
                        // truncate Q5.39 → Q4.23 by selecting bits [43:17]:
                        //   [43:40] = 4 integer bits
                        //   [39:17] = 23 fractional bits
                        for (i = 0; i < NUM_MODES; i = i + 1) begin
                            acc[i]                  <= acc_next[i];
                            zernike_out[i*27 +: 27] <= acc_next[i][43:17];
                        end
                        state <= STATE_DONE;
                    end
                    else begin
                        // Mid-frame: accumulate and advance counter
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= acc_next[i];
                        sub_counter <= sub_counter + 1;
                    end

                end
            end

            // ----------------------------------------------------------------
            // STATE_DONE — pulse done for exactly one cycle, then reset
            // ----------------------------------------------------------------
            STATE_DONE: begin
                done        <= 1;
                sub_counter <= 0;
                for (i = 0; i < NUM_MODES; i = i + 1)
                    acc[i] <= 0;
                state <= STATE_IDLE;
            end

            default: state <= STATE_IDLE;

        endcase
    end
end

endmodule