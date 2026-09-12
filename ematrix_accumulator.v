`timescale 1ns/1ps

// =============================================================================
// ematrix_accumulator.v
//
// Fully parallel E-matrix accumulator for Shack-Hartmann reconstruction.
//
// Architecture:
//   - 10 Zernike modes in parallel
//   - 2 multiplies per mode (x and y)
//   - 20 DSP multipliers total
//   - One subaperture accepted per cycle
//
// IMPORTANT TIMING RESTRUCTURE:
//
// OLD PATH:
//   sub_counter
//      -> distributed ROM selection
//      -> DSP multiply
//      -> product register
//
// NEW PATH:
//   sub_counter
//      -> distributed ROM selection
//      -> prefetched coefficient register
//
//   prefetched coefficient register
//      -> DSP multiply
//      -> product register
//
// While coefficient N is multiplied by slope N,
// coefficient N+1 is fetched concurrently.
//
// Therefore:
//   - no additional steady-state latency
//   - no throughput reduction
//   - sub_counter is removed from the active DSP input path
//
// Fixed-point formats:
//   E matrix   : Q1.16, 18-bit signed
//   Slopes     : 25-bit signed
//   Products   : 43-bit signed
//   Accumulator: 52-bit signed
//   Output     : 25-bit signed, acc[42:18]
//
// Assumption:
//   sub_valid samples correspond to sequential subapertures:
//
//       0, 1, 2, ... NUM_SUBS-1
//
// =============================================================================

module ematrix_accumulator #(
    parameter NUM_MODES  = 10,
    parameter NUM_SUBS   = 168,
    parameter NUM_SLOPES = 336
)(
    input  wire                     clk,
    input  wire                     rst,

    // One-cycle pulse when the next sequential slope pair is ready
    input  wire                     sub_valid,

    input  wire signed [24:0]       x_slope,
    input  wire signed [24:0]       y_slope,

    // Mode m occupies:
    //   zernike_out[m*25 +: 25]
    output reg  [249:0]             zernike_out,
    output reg                      done
);

    // =========================================================================
    // Local parameters
    // =========================================================================

    localparam integer SUB_BITS  = 8;
    localparam integer ACC_WIDTH = 52;

    localparam integer STATE_IDLE = 0;
    localparam integer STATE_DONE = 1;

    localparam [SUB_BITS-1:0] LAST_SUB = NUM_SUBS - 1;


    // =========================================================================
    // E-matrix coefficient ROMs
    //
    // Flat storage format:
    //
    //   e_rom_x[m * NUM_SUBS + s] = E[m, 2*s]
    //   e_rom_y[m * NUM_SUBS + s] = E[m, 2*s + 1]
    //
    // Keep distributed ROM for now.
    //
    // The architectural difference is that ROM output no longer directly
    // feeds the DSP multiplier for the current slope.
    // =========================================================================

    (* rom_style = "distributed" *)
    reg [17:0] e_rom_x [0:(NUM_MODES * NUM_SUBS)-1];

    (* rom_style = "distributed" *)
    reg [17:0] e_rom_y [0:(NUM_MODES * NUM_SUBS)-1];


    initial begin
        $readmemh("e_matrix_x.mem", e_rom_x);
        $readmemh("e_matrix_y.mem", e_rom_y);
    end


    // =========================================================================
    // State
    // =========================================================================

    reg                state;
    reg [SUB_BITS-1:0] sub_counter;


    // =========================================================================
    // Look-ahead address
    //
    // coeff_*_r always holds the coefficient for the CURRENT sub_counter.
    //
    // During processing of current subaperture N:
    //
    //   DSP uses:
    //       coeff_*_r = E[N]
    //
    //   concurrently ROM fetches:
    //       E[N+1]
    //
    // At the clock edge:
    //       product[N]   -> prod_*_r
    //       coefficient[N+1] -> coeff_*_r
    //
    // On the last subaperture, preload coefficient 0 for the next frame.
    // =========================================================================

    wire [SUB_BITS-1:0] next_coeff_addr;

    assign next_coeff_addr =
        (sub_counter == LAST_SUB)
            ? {SUB_BITS{1'b0}}
            : sub_counter + {{(SUB_BITS-1){1'b0}}, 1'b1};


    // =========================================================================
    // Prefetched coefficient registers
    //
    // These are the crucial new registers.
    //
    // They isolate the DSP from:
    //   - sub_counter
    //   - distributed ROM address fanout
    //   - ROM LUT selection network
    // =========================================================================

    reg signed [17:0] coeff_x_r [0:NUM_MODES-1];
    reg signed [17:0] coeff_y_r [0:NUM_MODES-1];


    // =========================================================================
    // Product registers
    // =========================================================================

    reg signed [42:0] prod_x_r [0:NUM_MODES-1];
    reg signed [42:0] prod_y_r [0:NUM_MODES-1];

    reg prod_valid_r;
    reg prod_last_sub_r;


    // =========================================================================
    // Parallel coefficient prefetch + multiply
    // =========================================================================

    genvar m;

    generate
        for (m = 0; m < NUM_MODES; m = m + 1) begin : mac_gen

            // -----------------------------------------------------------------
            // Constant-address coefficient zero
            //
            // Used to prime coeff_x_r / coeff_y_r during reset.
            //
            // Because m is a genvar, m*NUM_SUBS is a compile-time constant.
            // -----------------------------------------------------------------

            wire signed [17:0] first_coeff_x =
                $signed(e_rom_x[m * NUM_SUBS]);

            wire signed [17:0] first_coeff_y =
                $signed(e_rom_y[m * NUM_SUBS]);


            // -----------------------------------------------------------------
            // Look-ahead ROM reads
            //
            // These fetch coefficient N+1 while coefficient N is being used
            // by the DSP.
            //
            // The high-fanout counter path terminates at coeff_*_r.
            // It no longer continues through the DSP multiplier.
            // -----------------------------------------------------------------

            wire signed [17:0] next_coeff_x =
                $signed(
                    e_rom_x[
                        (m * NUM_SUBS) + next_coeff_addr
                    ]
                );

            wire signed [17:0] next_coeff_y =
                $signed(
                    e_rom_y[
                        (m * NUM_SUBS) + next_coeff_addr
                    ]
                );


            // -----------------------------------------------------------------
            // DSP multipliers
            //
            // CRITICAL DIFFERENCE:
            //
            // OLD:
            //   ROM output * slope
            //
            // NEW:
            //   registered prefetched coefficient * slope
            //
            // sub_counter is no longer in this immediate path.
            // -----------------------------------------------------------------

            (* use_dsp = "yes" *)
            wire signed [42:0] prod_x =
                $signed(coeff_x_r[m]) * $signed(x_slope);

            (* use_dsp = "yes" *)
            wire signed [42:0] prod_y =
                $signed(coeff_y_r[m]) * $signed(y_slope);


            // -----------------------------------------------------------------
            // Concurrent operation
            //
            // On a valid subaperture:
            //
            //   1. Capture product using CURRENT prefetched coefficient
            //   2. Prefetch NEXT coefficient
            //
            // Nonblocking assignments guarantee prod_x/prod_y use the OLD
            // coeff_x_r/coeff_y_r values, which correspond to the current
            // subaperture.
            // -----------------------------------------------------------------

            always @(posedge clk) begin
                if (rst) begin

                    // Prime coefficient zero so the first post-reset
                    // sub_valid can immediately process subaperture 0.
                    coeff_x_r[m] <= first_coeff_x;
                    coeff_y_r[m] <= first_coeff_y;

                    prod_x_r[m] <= 43'sd0;
                    prod_y_r[m] <= 43'sd0;

                end
                else if (sub_valid) begin

                    // CURRENT coefficient N * CURRENT slope N
                    prod_x_r[m] <= prod_x;
                    prod_y_r[m] <= prod_y;

                    // Simultaneously fetch coefficient N+1
                    coeff_x_r[m] <= next_coeff_x;
                    coeff_y_r[m] <= next_coeff_y;

                end
            end

        end
    endgenerate


    // =========================================================================
    // Valid / last pipeline
    //
    // Same effective product timing as the original implementation:
    //
    //   sub_valid at cycle N
    //       -> product registered at edge N
    //
    //   prod_valid_r identifies that product for accumulation next cycle.
    // =========================================================================

    always @(posedge clk) begin
        if (rst) begin
            prod_valid_r    <= 1'b0;
            prod_last_sub_r <= 1'b0;
        end
        else begin
            prod_valid_r <= sub_valid;

            prod_last_sub_r <=
                sub_valid &&
                (sub_counter == LAST_SUB);
        end
    end


    // =========================================================================
    // Accumulators
    // =========================================================================

    reg signed [ACC_WIDTH-1:0] acc [0:NUM_MODES-1];

    wire signed [ACC_WIDTH-1:0]
        acc_next [0:NUM_MODES-1];


    generate
        for (m = 0; m < NUM_MODES; m = m + 1) begin : acc_next_gen

            assign acc_next[m] =
                acc[m]
                + {
                    {9{prod_x_r[m][42]}},
                    prod_x_r[m]
                }
                + {
                    {9{prod_y_r[m][42]}},
                    prod_y_r[m]
                };

        end
    endgenerate


    // =========================================================================
    // Main state / counter / accumulation logic
    // =========================================================================

    integer i;

    always @(posedge clk) begin

        if (rst) begin

            state       <= STATE_IDLE;
            sub_counter <= {SUB_BITS{1'b0}};

            done        <= 1'b0;
            zernike_out <= 250'd0;

            for (i = 0; i < NUM_MODES; i = i + 1) begin
                acc[i] <= {ACC_WIDTH{1'b0}};
            end

        end
        else begin

            // Default pulse behavior
            done <= 1'b0;

            case (state)

                // =============================================================
                // IDLE / ACTIVE ACCUMULATION STATE
                // =============================================================

                STATE_IDLE: begin

                    // ---------------------------------------------------------
                    // Advance current-subaperture index.
                    //
                    // coeff_*_r is advanced concurrently in mac_gen.
                    //
                    // Before edge:
                    //   sub_counter = N
                    //   coeff_r     = E[N]
                    //
                    // After edge:
                    //   sub_counter = N+1
                    //   coeff_r     = E[N+1]
                    // ---------------------------------------------------------

                    if (
                        sub_valid &&
                        (sub_counter != LAST_SUB)
                    ) begin
                        sub_counter <=
                            sub_counter
                            + {{(SUB_BITS-1){1'b0}}, 1'b1};
                    end


                    // ---------------------------------------------------------
                    // Accumulate registered product
                    // ---------------------------------------------------------

                    if (prod_valid_r) begin

                        if (prod_last_sub_r) begin

                            // Final product of frame
                            for (
                                i = 0;
                                i < NUM_MODES;
                                i = i + 1
                            ) begin

                                acc[i] <= acc_next[i];

                                zernike_out[
                                    i*25 +: 25
                                ] <= acc_next[i][42:18];

                            end

                            done  <= 1'b1;
                            state <= STATE_DONE;

                        end
                        else begin

                            // Normal accumulation
                            for (
                                i = 0;
                                i < NUM_MODES;
                                i = i + 1
                            ) begin
                                acc[i] <= acc_next[i];
                            end

                        end
                    end
                end


                // =============================================================
                // DONE STATE
                //
                // coeff_*_r was already wrapped back to E[0] when the final
                // subaperture was accepted.
                //
                // Therefore only the logical counter and accumulators need
                // resetting here.
                // =============================================================

                STATE_DONE: begin

                    sub_counter <= {SUB_BITS{1'b0}};

                    for (
                        i = 0;
                        i < NUM_MODES;
                        i = i + 1
                    ) begin
                        acc[i] <= {ACC_WIDTH{1'b0}};
                    end

                    state <= STATE_IDLE;
                end


                default: begin
                    state <= STATE_IDLE;
                end

            endcase
        end
    end

endmodule