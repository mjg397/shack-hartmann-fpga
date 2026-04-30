module streaming_emulator #(
  parameter HSIZE  = 256, // Horozontal Frame Size
  parameter VSIZE  = 256, // Vertical Frame Size
  parameter HBLANK = 4,   // Horozontal Blanking
  parameter VBLANK = 152  // Vertical Blanking
)(
  input wire clk,
  input wire reset,
  input wire [7:0] rdata,
  input wire start,
  
  output reg [15:0] raddr,
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

//  initial begin
//      $readmemh("C:/Users/sjbar/OneDrive/Desktop/ECE5760/shack-hartmann-fpga/image_rotating.hex", mem);
//  end

  // state machine for pixel input
  localparam STATE_FRAME_INIT          = 2'b00;
  localparam STATE_ACTIVE_FRAME        = 2'b01;
  localparam STATE_HOROZONTAL_BLANKING = 2'b10;
  localparam STATE_VERTICAL_BLANKING   = 2'b11;

  reg [1:0] state;

  reg start_latch;
  always @(posedge clk) begin
      if (reset)                              start_latch <= 0;
      else if (start)                         start_latch <= 1;
      else if (state == STATE_ACTIVE_FRAME)   start_latch <= 0;  // consume it
  end

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
		raddr <= 16'd0;
    end

    else begin
      case (state)
        STATE_FRAME_INIT: begin
          fv <= 1;
          line_counter <= 0;
          row_counter <= 0;
          state <= start_latch ? STATE_ACTIVE_FRAME : STATE_FRAME_INIT;
          frame_complete <= 0;
        end

        STATE_ACTIVE_FRAME: begin
			 data <= rdata;
			 raddr <= row_counter * HSIZE + line_counter;
//          data <= mem[row_counter * HSIZE + line_counter];
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
          h_blank_counter <= h_blank_counter + 1;
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

