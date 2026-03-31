module centroid_divison (
  input wire        clk,
  input wire        rst,
  input wire        valid,
  input wire [17:0] intensity,
  input wire [17:0] x_intensity,
  input wire [17:0] y_intensity,
  output reg [17:0] x_centroid,
  output reg [17:0] y_centroid
);

if not valid
  return 0 for intensity
when valid:
  do the intensity[i] division
  x_intensity[i] * 1/intensity[i]
  y_intensity[i] * 1/intensity[i]
end

do by way of: 

stephens LUT:

stephen's newton's method:

endmodule
