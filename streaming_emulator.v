`timescale 1ns/1ps

module streaming_emulator #(
  parameter HSIZE  = 256,
  parameter VSIZE  = 256,
  parameter HBLANK = 4,
  parameter VBLANK = 152
)(
  input wire clk,
  input wire reset,

  // 32-bit BRAM read interface
  // This is a WORD address: 0..16383 for a 256x256x8 image packed 4 pixels/word.
  output reg  [13:0] bram_addr,
  output reg         bram_en,
  input  wire [31:0] bram_rddata,

  // Same outputs as before
  output reg [7:0] data,
  output reg       fv,
  output reg       lv,
  output reg       frame_complete,
  output wire      valid
);

  assign valid = fv & lv;

  localparam STATE_FRAME_INIT          = 3'd0;
  localparam STATE_LINE_PRIME          = 3'd1;
  localparam STATE_ACTIVE_FRAME        = 3'd2;
  localparam STATE_HOROZONTAL_BLANKING = 3'd3;
  localparam STATE_VERTICAL_BLANKING   = 3'd4;

  reg [2:0] state;

  reg [$clog2(HSIZE)-1:0]  line_counter;
  reg [$clog2(VSIZE)-1:0]  row_counter;
  reg [$clog2(HBLANK)-1:0] h_blank_counter;
  reg [$clog2(VBLANK)-1:0] v_blank_counter;

  wire [15:0] pixel_index;
  wire [15:0] pixel_index_next;

  assign pixel_index      = (row_counter << 8) + line_counter;
  assign pixel_index_next = pixel_index + 16'd1;

  wire [13:0] word_addr;
  wire [1:0]  byte_sel;

  assign word_addr = pixel_index[15:2];
  assign byte_sel  = pixel_index[1:0];

  reg [1:0] byte_sel_d;

  always @(posedge clk) begin
    if (reset) begin
      state           <= STATE_FRAME_INIT;
      line_counter    <= 0;
      row_counter     <= 0;
      h_blank_counter <= 0;
      v_blank_counter <= 0;

      bram_addr <= 14'd0;
      bram_en   <= 1'b0;

      data           <= 8'd0;
      fv             <= 1'b0;
      lv             <= 1'b0;
      frame_complete <= 1'b0;

      byte_sel_d <= 2'd0;
    end else begin
      fv             <= 1'b0;
      lv             <= 1'b0;
      frame_complete <= 1'b0;

      case (state)

        STATE_FRAME_INIT: begin
          line_counter    <= 0;
          row_counter     <= 0;
          h_blank_counter <= 0;
          v_blank_counter <= 0;

          bram_addr  <= 14'd0;
          bram_en    <= 1'b1;
          byte_sel_d <= 2'd0;

          state <= STATE_LINE_PRIME;
        end

        // Wait one cycle for BRAM data from bram_addr.
        STATE_LINE_PRIME: begin
          bram_en <= 1'b1;
          state   <= STATE_ACTIVE_FRAME;
        end

        STATE_ACTIVE_FRAME: begin
          case (byte_sel_d)
            2'd0: data <= bram_rddata[7:0];
            2'd1: data <= bram_rddata[15:8];
            2'd2: data <= bram_rddata[23:16];
            2'd3: data <= bram_rddata[31:24];
          endcase

          fv <= 1'b1;
          lv <= 1'b1;

          if (line_counter == HSIZE - 1) begin
            line_counter <= 0;
            bram_en      <= 1'b0;

            if (row_counter == VSIZE - 1) begin
              row_counter     <= 0;
              frame_complete  <= 1'b1;
              state           <= STATE_VERTICAL_BLANKING;
            end else begin
              state <= STATE_HOROZONTAL_BLANKING;
            end

          end else begin
            line_counter <= line_counter + 1'b1;

            bram_addr  <= pixel_index_next[15:2];
            byte_sel_d <= pixel_index_next[1:0];
            bram_en    <= 1'b1;
          end
        end

        STATE_HOROZONTAL_BLANKING: begin
          fv <= 1'b1;
          lv <= 1'b0;

          h_blank_counter <= h_blank_counter + 1'b1;

          if (h_blank_counter == HBLANK - 1) begin
            h_blank_counter <= 0;
            row_counter     <= row_counter + 1'b1;

            bram_addr  <= (((row_counter + 1'b1) << 8) >> 2);
            byte_sel_d <= 2'd0;
            bram_en    <= 1'b1;

            state <= STATE_LINE_PRIME;
          end
        end

        STATE_VERTICAL_BLANKING: begin
          fv <= 1'b0;
          lv <= 1'b0;

          v_blank_counter <= v_blank_counter + 1'b1;

          if (v_blank_counter == VBLANK - 1) begin
            v_blank_counter <= 0;
            state <= STATE_FRAME_INIT;
          end
        end

        default: begin
          state <= STATE_FRAME_INIT;
        end

      endcase
    end
  end

endmodule