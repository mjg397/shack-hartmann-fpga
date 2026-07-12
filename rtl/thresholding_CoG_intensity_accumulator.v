`timescale 1ns/1ps

// Thresholding Center of Gravity Intensity Accumulor
module thresholding_CoG_intensity_accumulator #(
  parameter NUM_SUBAPETURES_SQRT = 16,
  parameter NUM_PIXELS_SUBAPETURE_SQRT = 16,
  parameter PIXEL_THRESHOLD = 8'd5
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

wire [$clog2(NUM_SUBAPETURES_SQRT)-1:0] subap_idx;
wire last_pixel_h = (count_pixel_h == NUM_PIXELS_SUBAPETURE_SQRT-1);
wire last_pixel_v = (count_pixel_v == NUM_PIXELS_SUBAPETURE_SQRT-1);
wire last_subap_col = (subap_col == NUM_SUBAPETURES_SQRT-1);
wire last_subap_row = (subap_row == NUM_SUBAPETURES_SQRT-1);





// made these vectors of 16 to reduce fanout
reg [$clog2(255*NUM_PIXELS_SUBAPETURE_SQRT*NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i   [0:NUM_SUBAPETURES_SQRT-1];
reg [19:0] x_i [0:NUM_SUBAPETURES_SQRT-1];
reg [19:0] y_i [0:NUM_SUBAPETURES_SQRT-1];


wire subap_done_now = valid && last_pixel_h && last_pixel_v;

wire [$clog2(255*NUM_PIXELS_SUBAPETURE_SQRT*NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i_next;
wire [19:0] x_i_next;
wire [19:0] y_i_next;
wire [7:0] thresholded_data;

assign thresholded_data = (data_in < PIXEL_THRESHOLD) ? 8'd0 : data_in;

assign i_next   = i[subap_idx] + thresholded_data;
assign x_i_next = x_i[subap_idx] + thresholded_data * count_pixel_h;
assign y_i_next = y_i[subap_idx] + thresholded_data * count_pixel_v;



//assign subap_idx = (subap_row * NUM_SUBAPETURES_SQRT + subap_col);
assign subap_idx = subap_col;

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

  // Completed subapeture indication
  integer j;
  always @(posedge clk) begin
    if (reset) begin
      intensity <= '0;
      x_intensity <= '0;
      y_intensity <= '0;
      subapetures_completed <= '0;

      for (j = 0; j < NUM_SUBAPETURES_SQRT; j = j + 1) begin
        i[j]   <= '0;
        x_i[j] <= '0;
        y_i[j] <= '0;
      end
    end else if (valid) begin
      if (subap_done_now) begin
        // output includes final pixel
        intensity   <= i_next;
        x_intensity <= x_i_next;
        y_intensity <= y_i_next;
        subapetures_completed <= subapetures_completed + 1;

        // clear this accumulator for next subap row band
        i[subap_idx]   <= '0;
        x_i[subap_idx] <= '0;
        y_i[subap_idx] <= '0;
      end else begin
        // normal accumulation
        i[subap_idx]   <= i_next;
        x_i[subap_idx] <= x_i_next;
        y_i[subap_idx] <= y_i_next;
      end
    end
  end
endmodule
