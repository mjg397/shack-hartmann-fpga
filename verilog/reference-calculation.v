module reference_calculation (
  input wire        clk,
  input wire        rst,
  input wire [7:0] subapetures_completed,
  input wire [7:0] frame_complete,
  input wire [26:0] rec_intensity,
  input wire [19:0] x_intensity,
  input wire [19:0] y_intensity,
  output reg [26:0] x_centroid,
  output reg [26:0] y_centroid,
  output reg [26:0] x_slope,
  output reg [26:0] y_slope
);
  
localparam integer SCALE = 8388608; // 2^23 for 4.23 fixed point
localparam x_ref = 7.5 * SCALE;
localparam y_ref = 7.5 * SCALE;

always @(posedge clk) begin
  if (reset) begin
    x_slope <= 0;
    y_slope <= 0;
    slope_vector <= 0;
    subapetures_completed <= 0;
  end else begin
    x_centroid <= signed_mult(x_intensity, rec_intensity);
    y_centroid <= signed_mult(y_intensity, rec_intensity);
    x_slopes <= x_centroid - x_ref;
    y_slopes <= y_centroid - y_ref;
  end
endmodule

module signed_mult (out, a, b);
  output signed [26:0] out;
  input signed  [19:0] a;  // in this case 20.0
  input signed  [26:0] b;  // in this case 0.27
  // intermediate full bit length
  wire signed [46:0] mult_out;
  assign mult_out = a * b;
  // select bits for 4.23 fixed point
  assign out = mult_out[31:4];
endmodule
