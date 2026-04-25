`timescale 1ns/1ps
// instead of waiting for all slopes to be ready, we accumulate
// as soon as the first x/y pair arrives from reference_calculation
// x and y slopes are ready at the same time so we multiply both in one cycle
// E matrix is Q1.16 18-bit, slopes are Q4.23 27-bit, outputs are Q4.23 27-bit

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

    output reg  signed [26:0] zernike_out [0:NUM_MODES-1],  
    output reg                done
);

localparam integer E_FRAC   = 16;   // Q1.16 e matrix
localparam integer SL_FRAC  = 23;   // Q4.23 slopes
localparam integer ACC_FRAC = 39;   // Q5.39 accumulator (16+23)

reg signed [17:0] e_rom [0:(NUM_MODES * NUM_SLOPES)-1];
initial begin
    $readmemh("e_matrix.hex", e_rom);
end

localparam STATE_IDLE   = 2'b00;
localparam STATE_STREAM = 2'b01;
localparam STATE_DONE   = 2'b10;

reg [1:0] state;

reg [$clog2(NUM_SUBS)-1:0]   sub_counter;
reg [$clog2(NUM_MODES)-1:0]  mode_counter;

// sub_col tracks which subaperture column pair we are on (0 to NUM_SUBS-1)
// x lives at e_rom col 2*sub_col, y lives at e_rom col 2*sub_col+1
reg [$clog2(NUM_SUBS)-1:0] sub_col;

// one accumulator per zernike mode
// Q5.39 + 1 extra bit for the paired sum, so 55 bits to be safe
reg signed [54:0] acc [0:NUM_MODES-1];

// registered slope values for the current subaperture
reg signed [26:0] slope_x;
reg signed [26:0] slope_y;

// two e_rom values per cycle,  x column and y column for current mode
reg signed [17:0] e_val_x;
reg signed [17:0] e_val_y;

// Q1.16 * Q4.23 = Q5.39, sign extended to 55 bits
wire signed [54:0] mac_x;
wire signed [54:0] mac_y;
assign mac_x = {{10{e_val_x[17]}}, e_val_x * slope_x};
assign mac_y = {{10{e_val_y[17]}}, e_val_y * slope_y};

// sum of both contributions for this subaperture, added in one cycle
wire signed [54:0] mac_sum;
assign mac_sum = mac_x + mac_y;

integer i;

always @(posedge clk) begin
    if (rst == 1) begin
        state        <= STATE_IDLE;
        sub_counter  <= 0;
        mode_counter <= 0;
        sub_col      <= 0;
        slope_x      <= 0;
        slope_y      <= 0;
        e_val_x      <= 0;
        e_val_y      <= 0;
        done         <= 0;
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
                    if (sub_counter == 0) begin
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            acc[i] <= 0;
                    end
                    // latch both slopes and preload e_rom for mode 0
                    slope_x      <= x_slope;
                    slope_y      <= y_slope;
                    mode_counter <= 0;
                    e_val_x      <= e_rom[0 * NUM_SLOPES + (sub_col * 2)];
                    e_val_y      <= e_rom[0 * NUM_SLOPES + (sub_col * 2 + 1)];
                    state        <= STATE_STREAM;
                end
            end

            // one cycle per mode — both x and y columns multiplied and summed
            STATE_STREAM: begin
                acc[mode_counter] <= acc[mode_counter] + mac_sum;

                if (mode_counter < NUM_MODES - 1) begin
                    mode_counter <= mode_counter + 1;
                    e_val_x      <= e_rom[(mode_counter + 1) * NUM_SLOPES + (sub_col * 2)];
                    e_val_y      <= e_rom[(mode_counter + 1) * NUM_SLOPES + (sub_col * 2 + 1)];
                end
                else begin
                    // all modes done for this subaperture
                    mode_counter <= 0;
                    sub_col      <= sub_col + 1;

                    if (sub_counter == NUM_SUBS - 1) begin
                        for (i = 0; i < NUM_MODES; i = i + 1)
                            zernike_out[i] <= acc[i][43:17];
                        state <= STATE_DONE;
                    end
                    else begin
                        sub_counter <= sub_counter + 1;
                        if (sub_valid == 1) begin
                            slope_x  <= x_slope;
                            slope_y  <= y_slope;
                            e_val_x  <= e_rom[0 * NUM_SLOPES + ((sub_col + 1) * 2)];
                            e_val_y  <= e_rom[0 * NUM_SLOPES + ((sub_col + 1) * 2 + 1)];
                            // stay in STATE_STREAM
                        end
                        else begin
                            state <= STATE_IDLE;
                        end
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
