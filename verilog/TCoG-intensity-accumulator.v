module TCoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES = 16,
  parameter NUM_PIXELS_SUBAPETURE = 16,
  parameter NUM_CENTROIDS = 9
)(
  input wire clk,
  input wire reset,
  input wire valid,
  input wire [17:0] data_in,
  output reg        subap_valid [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0] = {(NUM_SUBAPETURES*NUMSUBAPETURES)1'b0}, \\set them default invalid until the streaming comes in (this allows for us to have feature of early complete detection)
  output reg [17:0] intensity   [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [17:0] x_intensity [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [17:0] y_intensity [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [$clog2(NUM_SUBAPETURES)-1:0] subap_col,
  output reg [$clog2(NUM_SUBAPETURES)-1:0] subap_row
);

reg [$clog2(NUM_SUBAPETURES)-1:0] subap_col;
reg [$clog2(NUM_SUBAPETURES)-1:0] subap_row;
-
assign subap_idx = subap_row * NUM_SUBAP_X + subap_col;

// control logic for TCoG
always @(posedge clk) begin
  if (reset) begin

  else begin
    if valid begin
      count_pixel_h <= count_pixel_h + 1;
      if (count_pixel_h == NUM_PIXELS_SUBAPETURE) begin
        count_pixel_h <= 0;
        subap_col <= subap_col + 1;
        if (subap_col_v == NUM_SUBAPETURE) begin
          subap_col <= 0;
          count_pixel_v <= count_pixel_v + 1;
          if (subap_pixel_v == NUM_PIXELS_SUBAPETURE) begin
            subap_row <= subap_row + 1;
            subap_col <= 0;
          end
          if (subap_row == NUM_SUBAPETURE && subap_col == NUM_SUBAPEUTRE && count_pixel_h == NUM_PIXELS_SUBAPEUTRE && count_pixel_v == NUM_PIXELS_SUBAPETURE) begin
            complete_frame <= 1;
          end
        end
      end
  end

  // accumulation of pixel data
  always @(posedge clk) begin
  if (reset) begin
  end
  else begin
    intensity[subap_row * NUM_SUBAP + subap_col] <= data_in
    x_intensity[subap_row * NUM_SUBAP + subap_col] <= data_in*count_pixel_h
    y_intensity[subap_row * NUM_SUBAP + subap_col] <= data_in*count_pixel_y
  end

  // valid data handling

  if at row NUM_SUBAP-1 and
    if (valid && ()) // if it is valid pixel and is in the frame

endmodule
