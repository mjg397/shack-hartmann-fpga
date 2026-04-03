//This next section is for a 3x3(3x3) shack hartmann, w/ beginning of CoG centroid calculations.

// check if in the circle, if not then each of the pixels in the subapeture is invalid, we can also count the number of total subarpetures to get correct number of valid for when we can calculate the matrix. we will figure out inspired by literature exactly what we should do
module streaming_centroid_accumulator (
  input wire raw_data
  input centroids
  output [17:0] centroids [8:0] // corresponding to eeach lenslet not yet generalized
);

always @(posedge clk) begin
  if rst()

  else
    check the new memory index, then some form of case statement with mux of if passed new range, can add numbers considering last point
    when fully pases, completed flag is set to 1 for the section
    for the input idxs within each range, add to existing.
    so if 3 at a time, first gets 1, 2-2, 3,3, 4,1
    think about how to do this more

  wires for finish flag and


  should return centroid values right when they come out

endmodule
'''
completed[i]
intensity[i]
x_intensity[i]
y_intensity[i]
sum_1
x_sum_1
y_sum_1
sum_2
x_sum_1
y_sum_1
.
sum_n^2
make state machine for rows and columns


  parameter NUM_ZERNIKE = 10;

input wire clk;
input wire rst;
input wire slopes_ready
input wire [17:0] slope_vector [0:2NUM_*N];
output reg zernike_complete;
output reg [17:0] zernike_outputs [0:NUM_ZERNIKES-1];

wire [17:0] A [0:NUM_ZERNIKE-1][0:2*NUM_SUAP*NUM_SUBAP]; // i think this is 92160 / 120000 of our memory on our registers 💀💀💀💀

reg [17:0] zernike_accumulation [0:NUM_ZERNIKE-1];


assign slope_x <= slope_vector[slope_idx_x];
assign slope_y <= slope_vector[slope_idx_y];
/////////////////
genvar i;

// generate each individual accumulator for each zernike polynomial
generate
  for(int i = 0; i < NUM_ZERNIKES; i++) begin : accumulator
    always @ (posedge clk) begin
      if (rst) begin
        zernike_accumulation[i] <= 0;
        slope_idx_x <= 0;
        slope_idx_y <= 1;
      end
      else begin
        if slopes_ready begin
          zernike_accumulation[i] <= (A[i][slope_x_idx] * slope_x) + (A[i][slope_y_idx] * slope_y)
          slope_x <= slope_y + 2;
          slope_y <= slope_y + 2;
          if (slope_x == NUM_SUBAP*NUM_SUBAP) begin // check to ensure this goes off on correct conditions
            slope_idx_x <= 0;
            slope_idx_y <= 1;
            zernike_accumulation[i] <= 0;
            zernike_outputs[i] <= zernike_accumulation[i];
            zernike_complete <= 1;
          end
        end
      end
  end
endgenerate

////////////////////
