`timescale 1ns/1ns

module test_TCoG_intensity_accumulator();
  
  localparam NUM_SUBAP_SQRT = 16;
  localparam NUM_PIX_SQRT = 16;
  localparam FRAME_SIZE = NUM_SUBAP_SQRT*NUM_SUBAP_SQRT*NUM_PIX_SQRT*NUM_PIX_SQRT;
  reg clk_100;
  reg reset;
  reg valid;
  reg [7:0] data_in;
  wire full_frame_complete;
  wire [$clog2(NUM_SUBAP_SQRT*NUM_SUBAP_SQRT)-1:0] subapetures_completed;
  wire [8+$clog2(NUM_PIX_SQRT)-1:0] intensity;
  wire [8+$clog2(NUM_PIX_SQRT)-1:0] x_intensity;
  wire [8+$clog2(NUM_PIX_SQRT)-1:0] y_intensity;

  TCoG_intensity_accumulator #(
    .NUM_SUBAPETURES_SQRT(NUM_SUBAP_SQRT),
    .NUM_PIXELS_SUBAPETURE_SQRT(NUM_PIX_SQRT)
  ) DUT (
    .clk(clk_100),
    .reset(reset),
    .valid(valid),
    .data_in(data_in),
    .full_frame_complete(full_frame_complete),
    .subapetures_completed(subapetures_completed),
    .intensity(intensity),
    .x_intensity(x_intensity),
    .y_intensity(y_intensity)
  );

  // Initialize the clk and reset
  initial begin
    clk_100 <= 0;
    reset <= 1;
    #10
    reset <= 0;
  end

  //Toggle the clock
  always begin
    #5
    clk_100  = !clk_100;
  end
 
 integer i;

  initial begin
    #20
    clk_100 = 0;
    reset = 1;
    valid = 0;
    data_in = 0;
  
    #20;
    reset = 0;
    valid = 1;
  
    for (i = 0; i < FRAME_SIZE; i = i + 1) begin
      @(posedge clk_100);
      data_in <= i[7:0];   // simple ramp pattern
    end

    @(posedge clk_100);
    valid = 0;

    #50;
    $finish;
  end
endmodule

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
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_intensity,
  output reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_intensity
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


reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] i   [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] x_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];
reg [8+$clog2(NUM_PIXELS_SUBAPETURE_SQRT)-1:0] y_i [NUM_SUBAPETURES_SQRT*NUM_SUBAPETURES_SQRT-1:0];

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
              i[j]   <= 8'b0; 
              x_i[j] <= 8'b0;
              y_i[j] <= 8'b0;
      end
    end
    else if (valid) begin
      i[subap_idx]   <= i[subap_idx] + data_in;
      x_i[subap_idx] <= x_i[subap_idx] + data_in*count_pixel_h;
      y_i[subap_idx] <= y_i[subap_idx] + data_in*count_pixel_v;
    end
  end
endmodule
