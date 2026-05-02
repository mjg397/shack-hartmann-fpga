`timescale 1ns/1ns

module slope_calculation_tb(

  reg clk_100;
  reg reset;
  reg [7:0] subapetures_completed;
  reg frame_complete;
  reg [26:0] rec_intensity;
  reg [19:0] x_intensity;
  reg [19:0] y_intensity;
  wire signed [26:0] x_centroid;
  wire signed [26:0] y_centroid;
  wire signed [26:0] x_slope;
  wire signed [26:0] y_slope;
  wire new_subapeture;
);

  slope_calculation DUT (
    .clk(clk_100),
    .rst(reset),
    .subapetures_completed(subapetures_completed),
    .frame_complete(frame_complete),
    .rec_intensity(rec_intensity),
    .x_intensity(x_intensity),
    .y_intensity(y_intensity),
    .x_centroid(x_centroid),
    .y_centroid(y_centroid),
    .x_slope(x_slope),
    .y_slope(y_slope),
    .new_subapeture(new_subapeture)
  );

  // ================= CLOCK =================
  initial clk_100 = 0;
  always #5 clk_100 = ~clk_100;

  // =========================================================
  // REAL ? FIXED (Q4.23)
  // =========================================================
  function [26:0] to_fixed;
    input real val;
    begin
      to_fixed = val * SCALE;
    end
  endfunction

  // =========================================================
  // FIXED ? REAL (Q4.23)
  // =========================================================
  function real to_real;
    input signed [26:0] val;
    begin
      to_real = val / SCALE;
    end
  endfunction

  // =========================================================
  // APPLY INPUTS (REAL DOMAIN 0?15)
  // =========================================================
  task apply_input;
    input integer subap;
    input real rec;
    input real xi;
    input real yi;
    begin
      subapetures_completed = subap;


      rec_intensity = rec;

      x_intensity   = xi;             // integer
      y_intensity   = yi;             // integer
    end
  endtask

  // =========================================================
  // PRINT RESULTS (REAL INTERPRETATION)
  // =========================================================
  task print_outputs;
    input integer id;

    real xc;
    real yc;
    real xs;
    real ys;

    begin
      xc = to_real(x_centroid);
      yc = to_real(y_centroid);
      xs = to_real(x_slope);
      ys = to_real(y_slope);

      $display("================================================");
      $display("TEST %0d", id);

      $display("INPUTS (REAL DOMAIN 0?15):");
      $display("  subap = %0d", subapetures_completed);
      $display("  rec   = %f", rec_intensity);
      $display("  x_in  = %f", x_intensity);
      $display("  y_in  = %f", y_intensity);

      $display("OUTPUTS (DECODED Q4.23 ? REAL):");
      $display("  x_centroid ? %f", xc);
      $display("  y_centroid ? %f", yc);
      $display("  x_slope    ? %f", xs);
      $display("  y_slope    ? %f", ys);
      $display("  new_subap  = %0d", new_subapeture);
      $display("================================================");
    end
  endtask

  // =========================================================
  // 10 TEST CASES (0?15 RANGE)   the values of x_intensity and y_intensity should be between 0 - 489600, realistically near the higher end of this
  //  the values of regular intensity is between 0 - 65280 and therefore r_intensity should be between 0.00001531862 - 1, values of regular intensity below 0 are set to 0
  // =========================================================
  task run_all_tests;
  begin
    $display("========== Q4.23 FIXED-POINT TEST START ==========");

    // RESET
    reset = 1;
    apply_input(0, 0.0, 0.0, 0.0);
    #10;
    reset = 0;
    #10;
    print_outputs(1);

    // LOW VALUES
    apply_input(1, 0.5, 200000, 300000);
    #10;
    print_outputs(2);

    // SMALL RANGE
    apply_input(2, 0.1, 4.2, 5.1);
    #10;
    print_outputs(3);

    // MID RANGE
    apply_input(3, 0.6, 7.5, 8.2);
    #10;
    print_outputs(4);

    // HIGH RANGE
    apply_input(4, 1.0, 7.4, 7.4);
    #10;
    print_outputs(5);

    // MAX RANGE
    apply_input(5, 1.0, 15.0, 15.0);
    #10;
    print_outputs(6);

    // NON-SYMMETRIC
    apply_input(6, 0.5, 1.2, 14.7);
    #10;
    print_outputs(7);

    // SMALL DIFFERENCE SENSITIVITY
    apply_input(7, 0.1, 5.01, 5.02);
    #10;
    print_outputs(8);

    // SUBAPERTURE EDGE CASE
    subapetures_completed = 0;
    apply_input(8, 0.5, 30.0, 30.0);
    #10;
    print_outputs(9);

    subapetures_completed = 1;
    #10;
    print_outputs(9);

    // DYNAMIC CHANGE TEST
    apply_input(9, 1.0, 2.0, 3.0);
    #10;

    apply_input(10, 1.0, 8.0, 9.0);
    #10;

    apply_input(11, 1.0, 13.0, 14.0);
    #10;

    print_outputs(10);

    $display("========== TEST COMPLETE ==========");
  end
  endtask

  // ================= MAIN =================
  initial begin
    clk_100 = 0;
    reset = 1;
    #20;
    reset = 0;

    run_all_tests();

    $finish;
  end

endmodule

// `timescale 1ns/1ns

// module slope_calculation_tb.v();
//     reg        clk_100,
//     reg        reset,
//     reg [7:0] subapetures_completed,
//     reg       frame_complete,
//     reg [26:0] rec_intensity,
//     reg [19:0] x_intensity,
//     reg [19:0] y_intensity,
//     wire signed [26:0] x_centroid,
//     wire signed [26:0] y_centroid,
//     wire signed [26:0] x_slope,
//     wire signed [26:0] y_slope,
//     wire         new_subapeture

//   slope_calculation DUT (
//     .clk(clk_100),
//     .rst(reset),
//     .subapetures_completed(subapetures_completed),
//     .frame_complete(frame_complete),
//     .rec_intensity(rec_intensity),
//     .x_intensity(x_intensity),
//     .y_intensity(y_intensity),
//     .x_centroid(x_centroid),
//     .y_centroid(y_centroid),
//     .x_slope(x_slope),
//     .y_slope(y_slope),
//     .new_subapeture(new_subapeture)
//   );

//   // Initialize the clk and reset
//   initial begin
//     clk_100 <= 0;
//     reset <= 1;
//     #10
//     reset <= 0;
//   end

//   //Toggle the clock
//   always begin
//     #5
//     clk_100  = !clk_100;
//   end
 
// endmodule


module slope_calculation (
    input wire        clk,
    input wire        rst,
    input wire [7:0] subapetures_completed,
    input wire       frame_complete,
    input wire [26:0] rec_intensity,
    input wire [19:0] x_intensity,
    input wire [19:0] y_intensity,
    output reg signed [26:0] x_centroid,
    output reg signed [26:0] y_centroid,
    output reg signed [26:0] x_slope,
    output reg signed [26:0] y_slope,
    output reg         new_subapeture
);

  localparam integer SCALE = 8388608; // 2^23 for 4.23 signed fixed point
  localparam x_ref = 62914560;
  localparam y_ref = 62914560;

  wire [27:0] x_centroid_mult;
  wire [27:0] y_centroid_mult;

  unsigned_mult x_mult(
    .out(x_centroid_mult),
    .a(x_intensity),
    .b(rec_intensity)
  );
  
  unsigned_mult y_mult(
    .out(y_centroid_mult),
    .a(y_intensity),
    .b(rec_intensity)
  );


  wire signed [27:0] raw_x_slope = $signed(x_centroid_mult) - x_ref;
  wire signed [27:0] raw_y_slope = $signed(y_centroid_mult) - y_ref;

  always @(posedge clk) begin
    if (rst) begin
        x_centroid <= 0;
        y_centroid <= 0;
        x_slope    <= 0;
        y_slope    <= 0;
    end else begin
        x_centroid <= x_centroid_mult[26:0];
        y_centroid <= y_centroid_mult[26:0];

        // Truncate slope to 27-bit signed (4.23)
        x_slope <= raw_x_slope[26:0];
        y_slope <= raw_y_slope[26:0];
    end
  end

  // high every time a new subap appears
    reg [7:0] current_subap;
  always @(posedge clk) begin
    if (rst) begin
      current_subap <= 0;
      new_subapeture <= 0;
    end else begin
      if (subapetures_completed > current_subap) begin
        current_subap <= current_subap + 1;
        new_subapeture <= 1;
      end else begin
        new_subapeture <= 0;
      end
    end
  end

endmodule

module unsigned_mult (
  output signed [27:0] out,
  input signed  [19:0] a,  // in this case 20.0
  input signed  [26:0] b  // in this case 0.27
  );
  // intermediate full bit length
  wire signed [46:0] mult_out;
  assign mult_out = a * b;
  // select bits for 4.23 fixed point
  assign out = mult_out[31:4];
endmodule
