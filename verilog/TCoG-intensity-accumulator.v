module TCoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES_SQRT = 16,
  parameter NUM_PIXELS_SUBAPETURE_SQRT = 16,
  parameter NUM_CENTROIDS = 9
)(
  input wire clk,
  input wire reset,
  input wire valid,
  input wire [7:0] data_in,
  output reg [$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] subapetures_completed // indexed with i=1 corresponding to centroid 0, 0 completed
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_intensity,
  output reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_col,
  output reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_row
);

reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_col;
reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_row;
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i   [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];

assign subap_idx = subap_row * NUM_SUBAP_X + subap_col;

//  TCoG indexing control logic
  always @(posedge clk) begin
    if (reset) begin
      subapetures_completed <= 0
      intensity <= 0;
      x_intensity <= 0;
      y_intensity <= 0;
      subap_col <= 0;
      subap_row <= 0;
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
              if (subap_row == NUM_SUBAPETURE && subap_col == NUM_SUBAPEUTRE && count_pixel_h == NUM_PIXELS_SUBAPEUTRE && count_pixel_v == NUM_PIXELS_SUBAPETURE) begin
                complete_full_frame <= 1;
                
              end
            end
          end
        end
      end
    end
  end

  // completion logic

  // Intensity accumulation of valid pixel data
  always @(posedge clk) begin
    if (reset) begin
      for (i = 0; i < NUM_SUBAPETURES*NUM_SUBAPETURES-1; i = i + 1) begin
              i[i]   <= 8'b0; 
              x_i[i] <= 8'b0;
              y_i[i] <= 8'b0;
      end
    end
    else if (valid) begin
      i[subap_idx]   <= i[subap_idx] + data_in;
      x_i[subap_idx] <= x_i[subap_idx] + data_in*count_pixel_h;
      y_i[subap_idx] <= x_i[subap_idx] + data_in*count_pixel_y;
    end
  end
endmodule
