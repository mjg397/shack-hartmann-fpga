`timescale 1ns/1ps

module TCoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES_SQRT = 16,
  parameter NUM_PIXELS_SUBAPETURE_SQRT = 16
)(
  input wire clk,
  input wire reset,
  input wire valid,
  input wire [7:0] data_in,
  output reg full_frame_complete,
  output reg [$clog2(NUM_PIXELS_SUBAPETURE_SQRT*NUM_PIXELS_SUBAPETURE_SQRT)-1:0] subapetures_completed, // indexed with i=1 corresponding to centroid 0, 0 completed
  output reg [$clog2(255*NUM_PIXELS_SUBAPETURE_SQRT*NUM_PIXELS_SUBAPETURE_SQRT)-1:0] intensity,
  output reg [19:0] x_intensity,
  output reg [19:0] y_intensity
);

reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_col;
reg [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_row;
reg [$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] count_pixel_h;
reg [$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] count_pixel_v;

wire [$clog2(NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT)-1:0] subap_idx;
wire last_pixel_h = (count_pixel_h == NUM_PIXELS_SUBAPETURE_SQRT-1);
wire last_pixel_v = (count_pixel_v == NUM_PIXELS_SUBAPETURE_SQRT-1);
wire last_subap_col = (subap_col == NUM_SUBAPETURES_SQRT-1);
wire last_subap_row = (subap_row == NUM_SUBAPETURES_SQRT-1);


reg [$clog2(255*NUM_PIXELS_SUBAPETURE_SQRT*NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i   [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [19:0] x_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [19:0] y_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];

reg subap_done_delay;
reg [$clog2(NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT)-1:0] subap_idx_delay;

assign subap_idx = (subap_row * NUM_SUBAPETURES_SQRT + subap_col);

//  TCoG indexing control logic
  always @(posedge clk) begin
    if (reset) begin
      subap_col <= 0;
      subap_row <= 0;
      count_pixel_h <= 0;
      count_pixel_v <= 0;
      full_frame_complete <= 0;
    end
    else if (valid) begin
      full_frame_complete <= 0;
      
      // If at last column of subapeture
      if (last_pixel_h) begin
        count_pixel_h <= 0;
      	 // Adjust the subapeture column
      	if (last_subap_col) begin
       	 subap_col <= 0;
     	end else begin
      	  subap_col <= subap_col + 1;
        end

      end else begin
        count_pixel_h <= count_pixel_h + 1;
      end
      
      // If at last column of subapeture and last subapeture of the apeture's column
      if (last_pixel_h && last_subap_col) begin
	// If at last row of subapeture
        if (last_pixel_v) begin
	  count_pixel_v <= 0;
	 // If at last subapeture of the apeture's row
	  if (last_subap_row) begin
	    subap_row <= 0;
	    full_frame_complete <= 1;
	  end else begin
	    subap_row <= subap_row + 1;
 	  end
	end else begin
          count_pixel_v <= count_pixel_v + 1;
        end 
      end
    end
  end
  
  // Delay subapeture done flag and subapeture index
  always @(posedge clk) begin
    if (reset) begin
      subap_done_delay <= 0;
      subap_idx_delay  <= 0;
    end else begin
      subap_done_delay <= valid && last_pixel_h && last_pixel_v;
      subap_idx_delay <= subap_idx;
    end
  end

  // Completed subapeture indication
  always @(posedge clk) begin
    if (reset) begin
      intensity <= 0;
      x_intensity <= 0;
      y_intensity <= 0;
      subapetures_completed <= 0;
    end else begin
      if (subap_done_delay) begin
        subapetures_completed <= subapetures_completed + 1;
        intensity <= i[subap_idx_delay];
        x_intensity <= x_i[subap_idx_delay];
        y_intensity <= y_i[subap_idx_delay];
      end
    end
  end


  // Intensity accumulation of valid pixel data
  integer j;
  always @(posedge clk) begin
    if (reset) begin
      for (j = 0; j < NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT; j = j + 1) begin
              i[j]   = 16'b0; 
              x_i[j] = 20'b0;
              y_i[j] = 20'b0;
      end
    end
    else if (valid) begin
      i[subap_idx]   <= i[subap_idx] + data_in;
      x_i[subap_idx] <= x_i[subap_idx] + data_in*count_pixel_h;
      y_i[subap_idx] <= y_i[subap_idx] + data_in*count_pixel_v;
    end
  end
endmodule
