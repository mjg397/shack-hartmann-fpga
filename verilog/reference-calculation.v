module reference_calculation (
  input wire        clk,
  input wire        rst,
  input wire [17:0] x_intensity,
  input wire [17:0] y_intensity,
  input wire []     centroid [0:NUM_CENTROids],
  output reg [17:0] x_slope,
  output reg [17:0] y_slope
);

  always @(posedge clk) begin
  if (reset) begin
    x_slope <= 0;
    y_slope <= 0;
    slope_vector <= 0;
  end
  else
    if frame_valid begin
        x_slopes[i] = x_intensity[i] - x_ref[i]
        y_slopes[i] = y_intensity[i] - y_ref[i]
        slope_vector[2i] = x_slopes[i]
        slope_vector[2i+1] = y_slopes[i]
    else
      x_slopes <= 0;
      y_slopes <= 0;
endmodule
