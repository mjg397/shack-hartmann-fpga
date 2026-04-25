`timescale 1ns/1ps
// instead of waiting for all slopes to be ready, we accumulate
// as soon as the first x/y pair arrives from reference_calculation
// x and y slopes are ready at the same time so we multiply both in one cycle
// since the entire e_rom is loaded at startup we can read all rows simultaneously
// one sub_valid = one cycle of work across all modes in parallel
// E matrix Q1.16 18-bit, slopes Q4.23 27-bit, outputs Q4.23 27-bit

module ematrix_accumulator #(
    parameter NUM_MODES  = 10,   // zernike modes, rows of E
    parameter NUM_SUBS   = 196,  // valid subapertures
    parameter NUM_SLOPES = 392   // 2 * NUM_SUBS, interleaved x then y per sub
)(
    input  wire        clk,
    input  wire        rst,

    input  wire        sub_valid,
    input  wire signed [26:0] x_slope,
    input  wire signed [26:0] y_slope,

    // flattened output port — zernike_out_flat[k*27 +: 27] = mode k Q4.23
    // unpack in the parent module with a generate loop
    output reg  [(NUM_MODES*27)-1:0] zernike_out_flat,
    output reg                       done
);

localparam integer E_FRAC   = 16;   // Q1.16 e matrix
localparam integer SL_FRAC  = 23;   // Q4.23 slopes
localparam integer ACC_FRAC = 39;   // Q5.39 accumulator (16+23)

// split rom — x columns in one block, y columns in another
// this lets us read both in the same cycle without dual-port complications
// address = mode * NUM_SUBS + sub_col
reg signed [17:0] e_rom_x [0:(NUM_MODES * NUM_SUBS)-1];
reg signed [17:0] e_rom_y [0:(NUM_MODES * NUM_SUBS)-1];
initial begin
    $readmemh("e_matrix_x.hex", e_rom_x);
    $readmemh("e_matrix_y.hex", e_rom_y);
end

// sub_col tracks which subaperture column pair we are on (0 to NUM_SUBS-1)
// x lives at e_rom_x[mode * NUM_SUBS + sub_col]
// y lives at e_rom_y[mode * NUM_SUBS + sub_col]
reg [$clog2(NUM_SUBS)-1:0] sub_col;

// combinatorial reads — all mode rows read simultaneously for current sub_col
// flattened to avoid unpacked wire arrays which some tools struggle with
// e_x_flat[m*18 +: 18] = e_rom_x row for mode m at current sub_col
// e_y_flat[m*18 +: 18] = e_rom_y row for mode m at current sub_col
wire [(NUM_MODES*18)-1:0] e_x_flat;
wire [(NUM_MODES*18)-1:0] e_y_flat;

genvar m;
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : erom_read
        assign e_x_flat[m*18 +: 18] = e_rom_x[m * NUM_SUBS + sub_col];
        assign e_y_flat[m*18 +: 18] = e_rom_y[m * NUM_SUBS + sub_col];
    end
endgenerate

// one mac_sum per mode — both x and y columns multiplied and summed in one cycle
// Q1.16 * Q4.23 = Q5.39, sign extended to 55 bits
// flattened — mac_sum_flat[m*55 +: 55] = mac result for mode m
wire [(NUM_MODES*55)-1:0] mac_sum_flat;
generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : mac_gen
        assign mac_sum_flat[m*55 +: 55] =
            {{10{e_x_flat[m*18+17]}}, $signed(e_x_flat[m*18 +: 18]) * x_slope} +
            {{10{e_y_flat[m*18+17]}}, $signed(e_y_flat[m*18 +: 18]) * y_slope};
    end
endgenerate

// full precision result for the last subaperture — acc + final mac_sum
// computed as a wire so we can assign to both acc and zernike_out in one cycle
// without hitting the non-blocking assignment last-write-wins problem
// flattened — acc_final_flat[m*55 +: 55] = final accumulated value for mode m
wire [(NUM_MODES*55)-1:0] acc_final_flat;

localparam STATE_IDLE = 1'b0;
localparam STATE_DONE = 1'b1;

reg state;

// one accumulator per zernike mode
// Q5.39 + 1 extra bit for the paired sum, so 55 bits to be safe
reg signed [54:0] acc [0:NUM_MODES-1];

reg [$clog2(NUM_SUBS)-1:0] sub_counter;

generate
    for (m = 0; m < NUM_MODES; m = m + 1) begin : acc_final_gen
        assign acc_final_flat[m*55 +: 55] = acc[m] + mac_sum_flat[m*55 +: 55];
    end
endgenerate

integer i;

always @(posedge clk) begin
    if (rst == 1) begin
        state       <= STATE_IDLE;
        sub_counter <= 0;
        sub_col     <= 0;
        done        <= 0;
        for (i = 0; i < NUM_MODES; i = i + 1) begin
            acc[i]                       <= 0;
            zernike_out_flat[i*27 +: 27] <= 0;
        end
    end
    else begin
        done <= 0;

        case (state)

            STATE_IDLE: begin
                if (sub_valid == 1) begin

                    if (sub_counter == 0) begin
                        // first subaperture of a new frame
                        // write mac_sum directly so no stale acc value is involved
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= $signed(mac_sum_flat[i*55 +: 55]);

                        sub_col     <= sub_col + 1;
                        sub_counter <= sub_counter + 1;
                    end

                    else if (sub_counter == NUM_SUBS - 1) begin
                        // last subaperture — acc_final_flat already holds acc + mac_sum
                        // assign to both acc and output in same cycle with no conflict
                        // truncate Q5.39 -> Q4.23: bits [43:17]
                        // [43:40] = 4 integer bits, [39:17] = 23 fractional bits
                        for (i = 0; i < NUM_MODES; i = i + 1) begin
                            acc[i]                       <= $signed(acc_final_flat[i*55 +: 55]);
                            zernike_out_flat[i*27 +: 27] <= acc_final_flat[i*55+17 +: 27];
                        end

                        sub_col <= sub_col + 1;
                        state   <= STATE_DONE;
                    end

                    else begin
                        // middle subapertures — accumulate and advance
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= $signed(acc_final_flat[i*55 +: 55]);

                        sub_col     <= sub_col + 1;
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
