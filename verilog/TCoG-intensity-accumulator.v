module TCoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES = 16,
  parameter NUM_PIXELS_SUBAPETURE = 16,
  parameter NUM_CENTROIDS = 9
)(
  input wire clk,
  input wire reset,
  input wire valid,
  input wire [7:0] data_in,
  output reg [$clog2(NUM_PIXELS_SUBAPETURE)-1:0] subapeture // fix this
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE)-1:0] intensity   [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE)-1:0] x_intensity [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE)-1:0] y_intensity [NUM_SUBAPETURES*NUM_SUBAPETURES-1:0],
  output reg [$clog2(NUM_SUBAPETURES)-1:0] subap_col,
  output reg [$clog2(NUM_SUBAPETURES)-1:0] subap_row
);

reg [$clog2(NUM_SUBAPETURES)-1:0] subap_col;
reg [$clog2(NUM_SUBAPETURES)-1:0] subap_row;

assign subap_idx = subap_row * NUM_SUBAP_X + subap_col;

// Control logic for TCoG
always @(posedge clk) begin
  if (reset) begin
  end
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
  end

  // Intensity accumulation of valid pixel data
  always @(posedge clk) begin
    if (reset) begin
      for (i = 0; i < NUM_SUBAPETURES*NUM_SUBAPETURES-1; i = i + 1) begin
              intensity[i]   <= 8'b0; 
              x_intensity[i] <= 8'b0;
              y_intensity[i] <= 8'b0;
      end
    end
    else if (valid) begin
      intensity[subap_idx]   <= data_in;
      x_intensity[subap_idx] <= data_in*count_pixel_h;
      y_intensity[subap_idx] <= data_in*count_pixel_y;
    end
  end
endmodule
