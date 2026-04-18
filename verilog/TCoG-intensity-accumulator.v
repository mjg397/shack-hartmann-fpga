module TCoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES_SQRT = 16,
  parameter NUM_PIXELS_SUBAPETURE_SQRT = 16,
  parameter NUM_CENTROIDS = 9
)(
  input wire clk,
  input wire reset,
  input wire valid,
  input wire [7:0] data_in,
  output reg full_frame_complete;
  output reg [$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] subapetures_completed // indexed with i=1 corresponding to centroid 0, 0 completed
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_intensity,
);

wire [$clog2(NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT)-1:0] subap_idx;
wire last_pixel_h = (count_pixel_h == NUM_PIXELS_SUBAPETURE-SQRT-1);
wire last_pixel_v = (count_pixel_v == NUM_PIXELS_SUBAPETURE_SQRT-1);
wire last_subap_col = (subap_col == NUM_SUBAPETURE_SQRT-1);
wire last_subap_row = (subap_row == NUM_SUBAPETURE_SQRT-1);

reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_col;
reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_row;
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i   [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];

assign subap_idx = (subap_row * NUM_SUBAPETURES_SQRT + subap_col);

//  TCoG indexing control logic
  always @(posedge clk) begin
    if (reset) begin
      subapetures_completed <= 0;
      subap_col <= 0;
      subap_row <= 0;
    end
    else if (valid) begin
      full_frame_complete <= 0;

      if (last_pixel_h) begin
        count_pixel_h <= 0;
      end else begin
        count_pixel_h <= count_pixel_h + 1;
      end

      if (last_pixel_h) begin
        if (last_pixel_v) begin
          count_pixel_v <= 0;
        end else begin
          count_pixel_v <= count_pixel_v + 1;
        end
      end

      if (last_pixel_h && last_pixel_v) begin
          if (last_subap_col)
              subap_col <= 0;
          else
              subap_col <= subap_col + 1;
       end

      if (last_pixel_h && last_pixel_v && last_subap_col) begin
        if (last_subap_row) begin
          subap_row <= 0;
          full_frame_complete <= 1;
        end else begin
          subap_row <= subap_row + 1;
        end
      end
    end
  end
  
  // Completed subapeture indication
  always @(posedge clk) begin
    if (reset) begin
      intensity <= 0;
      x_intensity <= 0;
      y_intensity <= 0;
    end else begin
      if (last_pixel_h && last_pixel_v) begin
        subapetures_completed <= subapetures_completed + 1;
        intensity <= i[subap_idx];
        x_intensity <= x_i[subap_idx];
        y_intensity <= y_i[subap_idx];
      end
    end
  end


  // Intensity accumulation of valid pixel data
  always @(posedge clk) begin
    if (reset) begin
      for (i = 0; i < NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1; i = i + 1) begin
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
