`timescale 1ns/1ps
// this is claude version
// fully parallel ematrix accumulator for shack-hartmann wavefront reconstruction
// all 10 zernike modes are computed in a single cycle per subaperture
// since the entire e_rom is loaded at startup we can read all rows simultaneously
// x and y slopes arrive together from reference_calculation so both are multiplied in the same cycle
// sub_valid comes from reference_calculation which needs ~256 pixel cycles per subaperture
// so sub_valid can never arrive faster than we can consume it — no fifo needed
// E matrix Q1.16 18-bit, slopes Q4.23 27-bit, outputs Q4.23 27-bit

module ematrix_accumulator #(
    parameter NUM_MODES  = 10,   // zernike modes, rows of E
    parameter NUM_SUBS   = 196,  // valid subapertures
    parameter NUM_SLOPES = 392   // 2 * NUM_SUBS, interleaved x then y per sub
)(
    input  wire        clk,
    input  wire        rst,

    // sub_valid pulses high for one cycle when reference_calculation
    // has finished computing x_slope and y_slope for a subaperture
    input  wire        sub_valid,
    input  wire signed [26:0] x_slope,   // Q4.23 from reference_calculation
    input  wire signed [26:0] y_slope,   // Q4.23 from reference_calculation

    output reg  signed [26:0] zernike_out [0:NUM_MODES-1],  // Q4.23
    output reg                done
);

localparam integer E_FRAC   = 16;   // Q1.16 e matrix
localparam integer SL_FRAC  = 23;   // Q4.23 slopes
localparam integer ACC_FRAC = 39;   // Q5.39 accumulator (16+23)

// e matrix rom — 10 modes x 392 slopes = 3920 words, 18-bit wide
// split into two roms so x and y columns can be read simultaneously
// e_rom_x holds even columns (x slopes): address = mode * NUM_SUBS + sub_col
// e_rom_y holds odd columns  (y slopes): address = mode * NUM_SUBS + sub_col
reg signed [17:0] e_rom_x [0:(NUM_MODES * NUM_SUBS)-1];
reg signed [17:0] e_rom_y [0:(NUM_MODES * NUM_SUBS)-1];
initial begin
    $readmemh("e_matrix_x.hex", e_rom_x);
    $readmemh("e_matrix_y.hex", e_rom_y);
end

// read all mode rows simultaneously for the current sub_col
// these are combinatorial reads — no clock needed, output is always valid
wire signed [17:0] e_x [0:NUM_MODES-1];
wire signed [17:0] e_y [0:NUM_MODES-1];

genvar m;
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : erom_read
        assign e_x[m] = e_rom_x[m * NUM_SUBS + sub_col];
        assign e_y[m] = e_rom_y[m * NUM_SUBS + sub_col];
    end
endgenerate

// all 10 mac_sums computed in parallel — one per mode
// E Q1.16 (18-bit) * slope Q4.23 (27-bit) = Q5.39 (45-bit)
// two products summed so 46-bit result, sign extended to 55 bits for accumulator
wire signed [54:0] mac_sum [0:NUM_MODES-1];
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : mac_gen
        assign mac_sum[m] = {{10{e_x[m][17]}}, e_x[m] * x_slope}
                          + {{10{e_y[m][17]}}, e_y[m] * y_slope};
    end
endgenerate

localparam STATE_IDLE   = 2'b00;   // waiting for sub_valid
localparam STATE_DONE   = 2'b01;   // pulse done, outputs are valid

reg [1:0] state;

reg [$clog2(NUM_SUBS)-1:0] sub_counter;   // which subaperture we are on
reg [$clog2(NUM_SUBS)-1:0] sub_col;       // indexes into e_rom, advances each sub

// one accumulator per zernike mode — grows by mac_sum every sub_valid cycle
// Q5.39 + 1 guard bit for the x+y sum = 55 bits, safe for 196 accumulations
reg signed [54:0] acc [0:NUM_MODES-1];

integer i;

always @(posedge clk) begin
    if (rst == 1) begin
        state       <= STATE_IDLE;
        sub_counter <= 0;
        sub_col     <= 0;
        done        <= 0;
        for (i = 0; i < NUM_MODES; i = i + 1) begin
            acc[i]         <= 0;
            zernike_out[i] <= 0;
        end
    end
    else begin
        done <= 0;

        case (state)

            STATE_IDLE: begin
                if (sub_valid == 1) begin
                    // clear accumulators only at the start of a new frame
                    if (sub_counter == 0) begin
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= 0;
                    end

                    // all modes accumulate in this single cycle
                    // mac_sum is already on the wires combinatorially from e_rom reads
                    for (i = 0; i < NUM_MODES; i = i + 1)
                        acc[i] <= acc[i] + mac_sum[i];

                    sub_col <= sub_col + 1;

                    if (sub_counter == NUM_SUBS - 1) begin
                        // last subaperture — truncate Q5.39 to Q4.23 and commit
                        // bits [43:17]: [43:40] = 4 integer bits, [39:17] = 23 fractional bits
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            zernike_out[i] <= acc[i][43:17];
                        state <= STATE_DONE;
                    end
                    else begin
                        sub_counter <= sub_counter + 1;
                    end
                end
            end

            STATE_DONE: begin
                done        <= 1;
                sub_counter <= 0;
                sub_col     <= 0;
                state       <= STATE_IDLE;
            end

        endcase
    end
end

endmodule
