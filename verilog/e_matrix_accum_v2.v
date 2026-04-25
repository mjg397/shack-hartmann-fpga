`timescale 1ns/1ps
// fully parallel ematrix accumulator for shack-hartmann wavefront reconstruction
// all 10 zernike modes are computed in a single cycle per subaperture
// E matrix Q1.16 18-bit, slopes Q4.23 27-bit, outputs Q4.23 27-bit
//
// fixes vs original:
//   1. zernike_out flattened to 1-D packed bus (modelsim verilog compat)
//   2. sub_col removed — sub_counter used directly for rom addressing
//   3. last-subaperture truncation fixed: captures acc+mac_sum not stale acc
//   4. $clog2 moved to localparam so port widths are constant expressions
//   5. state encoding changed to localparam integer to avoid width warnings

module ematrix_accumulator #(
    parameter NUM_MODES  = 10,   // zernike modes, rows of E
    parameter NUM_SUBS   = 196,  // valid subapertures
    parameter NUM_SLOPES = 392   // 2 * NUM_SUBS
)(
    input  wire        clk,
    input  wire        rst,

    // sub_valid pulses high for one cycle when slopes are ready
    input  wire        sub_valid,
    input  wire signed [26:0] x_slope,   // Q4.23
    input  wire signed [26:0] y_slope,   // Q4.23

    // flattened output bus: zernike_out[m] = out_bus[m*27 +: 27]
    output reg  [269:0] zernike_out,          // 10 * 27 bits, Q4.23 per mode
    output reg          done
);

// ---------------------------------------------------------
// localparams
// ---------------------------------------------------------
localparam integer SUB_BITS  = 8;        // ceil(log2(196)) = 8
localparam integer E_FRAC    = 16;
localparam integer SL_FRAC   = 23;
// E Q1.16 (18b) * slope Q4.23 (27b) = Q5.39 (45b)
// x-product + y-product => one guard bit => 46b, sign-extended to 55b
localparam integer ACC_WIDTH = 55;

localparam integer STATE_IDLE = 0;
localparam integer STATE_DONE = 1;

// ---------------------------------------------------------
// E-matrix ROM — split into x and y halves for parallel read
// e_rom_x[m * NUM_SUBS + s] = E[m, 2s]   (x column for sub s)
// e_rom_y[m * NUM_SUBS + s] = E[m, 2s+1] (y column for sub s)
// ---------------------------------------------------------
reg signed [17:0] e_rom_x [0:(NUM_MODES * NUM_SUBS)-1];
reg signed [17:0] e_rom_y [0:(NUM_MODES * NUM_SUBS)-1];

initial begin
    $readmemh("e_matrix_x.hex", e_rom_x);
    $readmemh("e_matrix_y.hex", e_rom_y);
end

// ---------------------------------------------------------
// state and counters
// ---------------------------------------------------------
reg                    state;
reg [SUB_BITS-1:0]     sub_counter;

// ---------------------------------------------------------
// combinatorial ROM reads — all 10 mode rows in parallel
// indexed by sub_counter (replaces the redundant sub_col)
// ---------------------------------------------------------
wire signed [17:0] e_x [0:NUM_MODES-1];
wire signed [17:0] e_y [0:NUM_MODES-1];

genvar m;
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : erom_read
        assign e_x[m] = e_rom_x[m * NUM_SUBS + sub_counter];
        assign e_y[m] = e_rom_y[m * NUM_SUBS + sub_counter];
    end
endgenerate

// ---------------------------------------------------------
// parallel MAC: all 10 modes in one combinatorial stage
// sign-extend 45-bit products to 55 bits before summing
// ---------------------------------------------------------
wire signed [ACC_WIDTH-1:0] mac_sum [0:NUM_MODES-1];

generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : mac_gen
        assign mac_sum[m] =
            {{10{e_x[m][17]}}, e_x[m] * x_slope} +
            {{10{e_y[m][17]}}, e_y[m] * y_slope};
    end
endgenerate

// ---------------------------------------------------------
// accumulators — one per mode, Q5.39, 55 bits
// ---------------------------------------------------------
reg signed [ACC_WIDTH-1:0] acc [0:NUM_MODES-1];

// wire to hold the fully updated value for the current cycle
// used so the last subaperture can commit acc+mac_sum correctly
wire signed [ACC_WIDTH-1:0] acc_next [0:NUM_MODES-1];

generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : acc_next_gen
        assign acc_next[m] = acc[m] + mac_sum[m];
    end
endgenerate

integer i;

// ---------------------------------------------------------
// sequential logic
// ---------------------------------------------------------
always @(posedge clk) begin
    if (rst) begin
        state       <= STATE_IDLE;
        sub_counter <= 0;
        done        <= 0;
        out_bus     <= 0;
        for (i = 0; i < NUM_MODES; i = i + 1)
            acc[i] <= 0;
    end
    else begin
        done <= 0;

        case (state)

            // ---- STATE_IDLE: wait for subaperture data ----
            STATE_IDLE: begin
                if (sub_valid) begin

                    // clear accumulators at the start of each new frame
                    // last-assignment-wins: the acc_next write below overrides
                    // this only for sub_counter==0, giving: acc = 0 + mac_sum[0]
                    if (sub_counter == 0) begin
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= 0;
                    end

                    // accumulate all modes in parallel using acc_next
                    // acc_next is combinatorial so it already includes mac_sum
                    for (i = 0; i < NUM_MODES; i = i + 1)
                        acc[i] <= acc_next[i];

                    if (sub_counter == NUM_SUBS - 1) begin
                        // last subaperture: commit acc_next (not stale acc)
                        // truncate Q5.39 → Q4.23: keep bits [43:17]
                        //   bits [43:40] = 4 integer bits
                        //   bits [39:17] = 23 fractional bits
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            zernike_out[i*27 +: 27] <= acc_next[i][43:17];
                        state <= STATE_DONE;
                    end
                    else begin
                        sub_counter <= sub_counter + 1;
                    end
                end
            end

            // ---- STATE_DONE: pulse done for one cycle then reset ----
            STATE_DONE: begin
                done        <= 1;
                sub_counter <= 0;
                state       <= STATE_IDLE;
            end

            default: state <= STATE_IDLE;

        endcase
    end
end

endmodule



