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
localparam x_ref
localparam y_ref 
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
