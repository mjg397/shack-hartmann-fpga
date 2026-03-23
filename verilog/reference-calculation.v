module reference_calculation
  input wire [17:0] x_intensity [N*N:0]
  input wire [17:0] y_intensity
  output wire [17:0] x_slopes [N*N:0]
  output wire  [17:0] y_slopes [N*N:0]
  output wire [17:0] slope_vector[2N*N:0]


  generate // generate a bunch of subtractors for the centroids
  for i in range(N^2):
    x_slopes[i] = x_intensity[i] - x_ref[i]
    y_slopes[i] = y_intensity[i] - y_ref[i]
    slope_vector[2i] = x_slopes[i]
    slope_vector[2i+1] = y_slopes[i]
endmodule
