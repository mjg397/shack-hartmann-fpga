`timescale 1ns/1ns

module slope_calculation_tb.v();
    reg        clk_100,
    reg        reset,
    reg [7:0] subapetures_completed,
    reg       frame_complete,
    reg [26:0] rec_intensity,
    reg [19:0] x_intensity,
    reg [19:0] y_intensity,
    wire signed [26:0] x_centroid,
    wire signed [26:0] y_centroid,
    wire signed [26:0] x_slope,
    wire signed [26:0] y_slope,
    wire         new_subapeture

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

  // Initialize the clk and reset
  initial begin
    clk_100 <= 0;
    reset <= 1;
    #10
    reset <= 0;
  end

  //Toggle the clock
  always begin
    #5
    clk_100  = !clk_100;
  end
 
endmodule


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
