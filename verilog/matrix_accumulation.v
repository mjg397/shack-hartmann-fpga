// think about doing an optimized least squares caluclation
// have an fsm do this all starting to the end , with 3 states, a reset stage, an intermediate stage, and a completed stage, which then returns something to trigger that the processing has completed
take the matrix, at each point where the waveslopes are calculated, can have matrix get multiplied and accumulated

input wire [17:0] slope_vector    [0:2NUM_*N];
output wire [17:0] zernike_outputs [0:NUM_ZERNIKES-1];

reg [17:0] A [0:NUM_ZERNIKE-1][0:2*NUM_SUAP*NUM_SUBAP]; // i think this is 92160 / 120000 of our memory on our registers 💀💀💀💀

reg [17:0] zernike_accumulation [0:NUM_ZERNIKE-1];
generate
for(int i = 0; i < NUM_ZERNIKES; i++) begin
  zernike_accumulation[i] += 
counter <= counter + 1;

when counter == NUM_SUBAP*NUM_SUBAP:
  counter <= 0;
  zernike_complete <= 1;
  assign zernike_outputs = zernike_accumulation;
end
