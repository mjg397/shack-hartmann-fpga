// i can later on make the centroids be unsigned im stupid so i didnt realize till now and frame complete
`timescale 1ns/1ps

module slope_calculation (
    input wire        clk,
    input wire        rst,
    input wire [7:0] subapetures_completed,
    input wire       frame_complete,
    input wire [26:0] rec_intensity, // input 1 / sI
    input wire [19:0] x_intensity,   // s(x * I)
    input wire [19:0] y_intensity,   // s(y * I)
    output reg signed [26:0] x_centroid, // s(x * I) / sI
    output reg signed [26:0] y_centroid, // s(y * I) / sI
    output reg signed [26:0] x_slope,    // (s(x * I) / sI) - 7.5
    output reg signed [26:0] y_slope,    // (s(y * I) / sI) - 7.5
    output reg         new_subapeture,
    output reg         subap_valid      // high for one cycle *WITH* new_subapeture when the subap is in the pupil
);

  localparam integer SCALE = 8388608; // 2^23 for 4.23 signed fixed point
  localparam x_ref = 62914560;
  localparam y_ref = 62914560;

  wire [26:0] x_centroid_mult;
  wire [26:0] y_centroid_mult;

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
        x_centroid <= $signed(x_centroid_mult);
        y_centroid <= $signed(y_centroid_mult);

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

`timescale 1ns/1ps

module unsigned_mult (
  output signed [26:0] out,
  input signed  [19:0] a,  // in this case 20.0
  input signed  [26:0] b  // in this case 0.27
  );
  // intermediate full bit length
  wire signed [46:0] mult_out;
  assign mult_out = a * b;
  // select bits for 4.23 fixed point
  assign out = mult_out[30:4];
endmodule
