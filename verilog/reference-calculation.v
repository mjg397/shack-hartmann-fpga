module reference_calculation (
  input wire        clk,
  input wire        rst,
  input wire [7:0] subapetures_completed,
  input wire [19:0] rec_intensity,
  input wire [19:0] x_intensity,
  input wire [19:0] y_intensity,
  output reg [19:0] x_slope,
  output reg [19:0] y_slope,
  output reg [7:0] subapetures_completed,
  output reg [1:0] x_centroid,
  output reg[19:0] y_centroid,
);

localparam x_ref;
localparam y_ref;

  always @(posedge clk) begin
  if (reset) begin
    x_slope <= 0;
    y_slope <= 0;
    slope_vector <= 0;
  end
  else
    x_centroid = x_intensity * rec_intensity;
    y_centroid = y_intensity * rec_intensity;
    x_slopes = x_centroid - x_ref
    y_slopes = y_centroid - y_ref
  else
    x_slopes <= 0;
    y_slopes <= 0;
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



localparam x_ref = 8'd10;
localparam y_ref = 8'd10;
  always @(posedge clk) begin
  if (reset) begin
    x_slope <= 0;
    y_slope <= 0;
    slope_vector <= 0;
  end
  else
    x_centroid = x_intensity * rec_intensity;
    y_centroid = y_intensity * rec_intensity;
    x_slopes = x_centroid - x_ref
    y_slopes = y_centroid - y_ref
  else
    x_slopes <= 0;
    y_slopes <= 0;
endmodule


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
