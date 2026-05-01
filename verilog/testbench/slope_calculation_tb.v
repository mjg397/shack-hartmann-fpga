`timescale 1ns/1ns

module slope_calculation_tb;

  // ---------------- CLOCK + RESET ----------------
  reg clk_100;
  reg reset;

  // ---------------- INPUTS ----------------
  reg [7:0] subapetures_completed;
  reg frame_complete;
  reg [26:0] rec_intensity;
  reg [19:0] x_intensity;
  reg [19:0] y_intensity;

  // ---------------- OUTPUTS ----------------
  wire signed [26:0] x_centroid;
  wire signed [26:0] y_centroid;
  wire signed [26:0] x_slope;
  wire signed [26:0] y_slope;
  wire new_subapeture;

  // ---------------- DUT ----------------
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

  // ---------------- CLOCK ----------------
  initial clk_100 = 0;
  always #5 clk_100 = ~clk_100;

  // ---------------- INITIAL RESET ----------------
  initial begin
    reset = 1;
    subapetures_completed = 0;
    frame_complete = 0;
    rec_intensity = 0;
    x_intensity = 0;
    y_intensity = 0;

    #20;
    reset = 0;
  end

  // =========================================================
  // TASK: APPLY INPUTS
  // =========================================================
  task apply_input;
    input [7:0] subap;
    input [26:0] rec;
    input [19:0] xi;
    input [19:0] yi;
    begin
      subapetures_completed = subap;
      rec_intensity = rec;
      x_intensity = xi;
      y_intensity = yi;
    end
  endtask

  // =========================================================
  // TASK: PRINT OUTPUTS
  // =========================================================
  task print_outputs;
    input integer id;
    begin
      $display("------------------------------------------------");
      $display("TEST %0d", id);
      $display("Inputs:");
      $display("  subap = %0d rec = %0d x = %0d y = %0d",
                subapetures_completed, rec_intensity,
                x_intensity, y_intensity);

      $display("Outputs:");
      $display("  x_centroid = %0d", x_centroid);
      $display("  y_centroid = %0d", y_centroid);
      $display("  x_slope    = %0d", x_slope);
      $display("  y_slope    = %0d", y_slope);
      $display("  new_subap  = %0d", new_subapeture);
    end
  endtask

  // =========================================================
  // ALL TESTS (BEHAVIOR COVERAGE)
  // =========================================================
  task run_all_tests;
  begin
    $display("========== START TESTS ==========");

    // RESET BEHAVIOR
    reset = 1;
    apply_input(0, 0, 0, 0);
    #10;
    reset = 0;
    #10;
    print_outputs(1);

    // ZERO INPUT
    apply_input(1, 0, 0, 0);
    #10;
    print_outputs(2);

    // NORMAL SMALL VALUES
    apply_input(2, 100, 10, 20);
    #10;
    print_outputs(3);

    // SYMMETRY CASE
    apply_input(3, 200, 50, 50);
    #10;
    print_outputs(4);

    // MEDIUM VALUES
    apply_input(4, 500, 123, 456);
    #10;
    print_outputs(5);

    // LARGE VALUES STRESS
    apply_input(5, 5000, 10000, 20000);
    #10;
    print_outputs(6);

    // MAX EDGE CASE
    apply_input(6, 27'h7FFFFFF, 20'hFFFFF, 20'hFFFFF);
    #10;
    print_outputs(7);

    // LOW SIGNAL PRECISION TEST
    apply_input(7, 10, 1, 2);
    #10;
    print_outputs(8);

    // SUBAPERTURE EDGE TEST
    subapetures_completed = 0;
    apply_input(8, 100, 30, 40);
    #10;
    print_outputs(9);

    subapetures_completed = 1;
    #10;
    print_outputs(9);

    // DYNAMIC CHANGE TEST
    apply_input(9, 100, 10, 20);
    #10;
    apply_input(10, 200, 30, 60);
    #10;
    apply_input(11, 300, 90, 10);
    #10;
    print_outputs(10);

    $display("========== TESTS COMPLETE ==========");
  end
  endtask

  // ---------------- START ----------------
  initial begin
    #25; // wait for reset release
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
