// i can later on make the centroids be unsigned im stupid so i didnt realize till now and frame complete
`timescale 1ns/1ps

module slope_calculation (
    input wire               clk,
    input wire               rst,
    input wire        [7:0]  subapetures_completed,
    input wire               frame_complete,
    input wire        [17:0] rec_intensity,
    input wire        [18:0] x_intensity,
    input wire        [18:0] y_intensity,
    output reg        [23:0] x_centroid,
    output reg        [23:0] y_centroid,
    output reg signed [24:0] x_slope,
    output reg signed [24:0] y_slope,
    output reg               new_subapeture,
    output reg               subap_valid 
    // subap_valid is high for one cycle *WITH* new_subapeture
    //when the subap is in the pupil
);

  reg [255:0] subap_bitmap_mem[0:0];
  wire [255:0] subap_bitmap; // row major ordered bitmap

  initial begin
    $readmemh("subaperture_bitmap.mem", subap_bitmap_mem);
  end

  assign subap_bitmap = subap_bitmap_mem[0];

  reg signed [24:0] slopes_ref_x_mem [0:255];
  reg signed [24:0] slopes_ref_y_mem [0:255];

  initial begin
    $readmemh("slopes_ref_x.mem", slopes_ref_x_mem);
    $readmemh("slopes_ref_y.mem", slopes_ref_y_mem);
  end

  wire [23:0] x_ref = slopes_ref_x_mem[subapetures_completed - 1][23:0];
  wire [23:0] y_ref = slopes_ref_y_mem[subapetures_completed - 1][23:0];

  wire [23:0] x_centroid_mult;
  wire [23:0] y_centroid_mult;

  unsigned_mult x_mult(
    .out(x_centroid_mult),
    .a(x_intensity),
    .b(rec_intensity)
  );
  
  unsigned_mult y_mult(
    .out(y_centroid_mult),
    .a(y_intensity),
    .b(rec_intensity)
  );


  wire signed [24:0] x_centroid_s = {1'b0, x_centroid_mult};
  wire signed [24:0] y_centroid_s = {1'b0, y_centroid_mult};
  wire signed [24:0] x_ref_s       = {1'b0, x_ref};
  wire signed [24:0] y_ref_s       = {1'b0, y_ref};

  wire signed [25:0] raw_x_slope = $signed({x_centroid_s[24], x_centroid_s}) - $signed({x_ref_s[24], x_ref_s});
  wire signed [25:0] raw_y_slope = $signed({y_centroid_s[24], y_centroid_s}) - $signed({y_ref_s[24], y_ref_s});

  always @(posedge clk) begin
    x_centroid <= x_centroid_mult;
    y_centroid <= y_centroid_mult;

    x_slope <= raw_x_slope;
    y_slope <= raw_y_slope;
  end

  // Basic runtime checks for the fixed-point handoff and sign handling.
  always @(posedge clk) begin
    if (!rst) begin
      if (subapetures_completed > 8'd255) $error("subapetures_completed out of range");
      if (subap_valid && subapetures_completed != 0 &&
          !subap_bitmap[subapetures_completed - 1]) begin
        $error("subap_valid asserted for masked subaperture");
      end
    end
  end

  // high every time a new subap appears
  reg [7:0] current_subap;
  always @(posedge clk) begin
    if (rst) begin
      current_subap <= 0;
      new_subapeture <= 0;
    end else begin
      if (subapetures_completed > current_subap) begin
        current_subap <= current_subap + 1;
        new_subapeture <= 1;
        subap_valid <= subap_bitmap[subapetures_completed - 1]; // check bitmap for last completed subap
      end else begin
        new_subapeture <= 0;
        subap_valid <= 1'b0;
      end
    end
  end

endmodule

`timescale 1ns/1ps

module unsigned_mult (
  output [23:0] out,
  input [18:0] a,
  input [17:0] b
  );
  wire [36:0] mult_out;
  assign mult_out = a * b;
  assign out = mult_out[35:12];
endmodule
