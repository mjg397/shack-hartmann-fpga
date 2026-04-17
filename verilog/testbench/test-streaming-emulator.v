`timescale 1ns/1ns

module testbench_streaming_emulator();

  reg clk_100;
  reg reset;
  wire [7:0] data;
  wire fv;
  wire lv;
  wire frame_complete;
  wire valid;


  streaming_emulator DUT 
  (
    .clk(clk_100),
    .reset(reset),
    .data(data),
    .fv(fv),
    .lv(lv),
    .frame_complete(frame_complete),
    .valid(valid)
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
    clk_100  = !clk_100;/testbench_streaming_emulator/DUT/mem
  end

  // --- Debug memory check ---
  integer row, col;
  initial begin
      // Check row 64 in memory
      row = 64;
      $display("Checking row %0d:", row);
      for (col = 0; col < 128; col = col + 1) begin
          $display("Row %0d, Col %0d: %h", row, col, DUT.mem[row*128 + col]);
      end
  end

// --- Automated testcases ---
integer r, c;
reg [7:0] expected_val;
reg test_failed;
reg [7:0] pattern[0:2]; // pattern array

initial begin
    pattern[0] = 0;
    pattern[1] = 100;
    pattern[2] = 255;
    test_failed = 0;

    @(negedge reset);
    $display("Starting automated testcases...");

    // first pixel
    @(posedge clk_100);
    expected_val = pattern[0];
    if (data !== expected_val) begin
        $display("FAIL: First pixel mismatch. Got %0h, expected %0h", data, expected_val);
        test_failed = 1;
    end

    // first row
    for (c = 0; c < 256; c = c + 1) begin
        @(posedge clk_100);
        expected_val = pattern[c % 3];
        if (data !== expected_val) begin
            $display("FAIL: Row 0, Col %0d: Got %0h, expected %0h", c, data, expected_val);
            test_failed = 1;
        end
    end

    // middle row
    r = 128;
    for (c = 0; c < 256; c = c + 1) begin
        @(posedge clk_100);
        expected_val = pattern[(r+c) % 3];
        if (data !== expected_val) begin
            $display("FAIL: Row %0d, Col %0d: Got %0h, expected %0h", r, c, data, expected_val);
            test_failed = 1;
        end
    end

    // last pixel
    r = 255; c = 255;
    @(posedge clk_100);
    expected_val = pattern[(r+c) % 3];
    if (data !== expected_val) begin
        $display("FAIL: Last pixel mismatch. Got %0h, expected %0h", data, expected_val);
        test_failed = 1;
    end

    if (!test_failed) $display("All automated testcases PASSED!");
    else $display("Some automated testcases FAILED. Check above messages.");

    $stop;
end

endmodule


module streaming_emulator #(
  parameter HSIZE  = 256, // Horozontal Frame Size
  parameter VSIZE  = 256, // Vertical Frame Size
  parameter HBLANK = 4,   // Horozontal Blanking
  parameter VBLANK = 152  // Vertical Blanking
)(
  input wire clk,
  input wire reset,
  output reg [7:0] data,
  output reg fv,
  output reg lv,
  output reg frame_complete,
  output wire valid
);

  assign valid = fv & lv;

  reg [$clog2(HSIZE)-1:0] line_counter;
  reg [$clog2(VSIZE)-1:0] row_counter;
  reg [$clog2(HBLANK)-1:0] h_blank_counter;
  reg [$clog2(VBLANK)-1:0] v_blank_counter;

  // Instantiate into ROM
  reg [7:0] mem [0:65535];

  initial begin
      $readmemh("C:/Users/mjg397/image_rotating.hex", mem);
  end

  // state machine for pixel input
  localparam STATE_FRAME_INIT          = 2'b00;
  localparam STATE_ACTIVE_FRAME        = 2'b01;
  localparam STATE_HOROZONTAL_BLANKING = 2'b10;
  localparam STATE_VERTICAL_BLANKING   = 2'b11;

  reg [1:0] state;

  always @(posedge clk) begin
    if (reset == 1) begin
      state <= STATE_FRAME_INIT;
      line_counter <= 0;
      row_counter <= 0;
      h_blank_counter <= 0;
      v_blank_counter <= 0;
      lv <= 0;
      fv <= 0;
      frame_complete <= 0;
    end

    else begin
      case (state)
        STATE_FRAME_INIT: begin
          fv <= 1;
          line_counter <= 0;
          row_counter <= 0;
          state <= STATE_ACTIVE_FRAME;
        end

        STATE_ACTIVE_FRAME: begin
          data <= mem[row_counter * HSIZE + line_counter];
          line_counter <= line_counter + 1;
          lv <= 1;
          if ((line_counter == HSIZE - 1) && (row_counter == VSIZE - 1)) begin
            state <= STATE_VERTICAL_BLANKING;
            frame_complete <= 1;
          end
          else if (line_counter == HSIZE - 1) begin
            state <= STATE_HOROZONTAL_BLANKING;
          end
        end

        STATE_HOROZONTAL_BLANKING: begin
          lv <= 0;
          line_counter <= 0;
          h_blank_counter = h_blank_counter + 1;
          if (h_blank_counter == HBLANK - 1) begin
            h_blank_counter <= 0;
            row_counter <= row_counter + 1;
            state <= STATE_ACTIVE_FRAME;
          end
        end

        STATE_VERTICAL_BLANKING : begin
          fv <= 0;
          lv <= 0;
          line_counter <= 0;
          row_counter <= 0;
          v_blank_counter <= v_blank_counter + 1;
          if (v_blank_counter == VBLANK - 1) begin
            v_blank_counter <= 0;
            state <= STATE_FRAME_INIT;
          end
        end
      endcase
    end
  end
endmodule
