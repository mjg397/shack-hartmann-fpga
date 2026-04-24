module reference_calculation (
  input wire        clk,
  input wire        rst,
  input wire [7:0] subapetures_completed,
  input wire [19:0] rec_intensity,
  input wire [19:0] x_intensity,
  input wire [19:0] y_intensity,
  output reg [19:0] x_centroid,
  output reg [19:0] y_centroid,
  output reg [7:0] subapetures_completed,
  output reg [19:0] x_slope,
  output reg [19:0] y_slope
);

localparam x_ref = //7.5;
localparam y_ref = //7.5;

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

module reference_calculation (
  input wire        clk,
  input wire        rst,
  input wire [7:0] subapetures_completed,
  input wire [19:0] rec_intensity
  input wire [19:0] x_intensity,
  input wire [19:0] y_intensity,
  output reg [19:0] x_slope,
  output reg [19:0] y_slope,
  output reg [7:0] subapetures_completed
  output reg [1:0] x_centroid,
  output reg[19:0] y_centroid,
);

module signed_mult (out, a, b);
  output signed [19:0] out;
  input signed  [19:0] a;
  input signed  [19:0] b;
  // intermediate full bit length
  wire signed [19:0] mult_out;
  assign mult_out = a * b;
  // select bits for 7.20 fixed point
  assign out = {mult_out[53], mult_out[45:20]};
endmodule
