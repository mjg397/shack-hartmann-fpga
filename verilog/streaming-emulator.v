// this emulates the streaming property of the shack-hartmann wavefront sensor from a preloaded image into FPGA memroy
// this intends to allow for our shack-hartmann processing pipeline to have an accurate representation of incoming data
// I'm not sure if it syntheizes i havent tried yet :\
parameter HSIZE  = 256; // Horozontal Frame Size
parameter VSIZE  = 256; // Vertical Frame Size
parameter HBLANK = 4;   // Horozontal Blanking
parameter VBLANK = 152; // Vertical Blanking

output reg fv;
output reg lv;
output reg frame_complete;
output wire valid;

assign valid = fv & lv;

reg [$clog2(HSIZE)-1:0] line_counter
reg [$clog2(VSIZE)-1:0] row_counter

// Instantiate into ROM
reg [7:0] mem [0:65535];

initial begin
    $readmemh("image.hex", mem);
end


// state machine for pixel input

localparam STATE_FRAME_INIT          = 2'b00;
localparam STATE_ACTIVE_FRAME;       = 2'b01;
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
  end

  else begin
    case (state) begin

      STATE_FRAME_INIT: begin
        fv <= 1;
        line_counter <= 0;
        row_counter <= 0;
        state <= STATE_ACTIVE_FRAME
      end

      STATE_ACTIVE_FRAME: begin
        data <= mem[row_counter * HSIZE + line_counter];
        line_counter <= line_counter + 1;
        lv <= 1;
        if ((line_counter == HSIZE - 1) && (row_counter == VSIZE - 1)) begin
          STATE <= STATE_VERTICAL_BLANKING;
          frame_complete <= 1;
        end
        else if (line_counter == HSIZE - 1) begin
          state <= STATE_HOROZONTAL_BLANKING
        end
      end
                 
      STATE_HOROZONTAL_BLANKING: begin
        lv <= 0;
        line_counter <= 0;
        h_blank_coutner = h_blank_counter + 1;
        if (h_blank_counter == HBLANK - 1)
          h_blank_counter <= 0;
          row_counter <= row_counter + 1
          state <= STATE_ACTIVE_FRAME;
        end
      end

      STATE_VERTICAL_BLANKING: begin
        fv <= 0;
        lv <= 0;
        line_counter <= 0;
        row_counter <= 0;
        v_blank_counter <= v_blank_counter + 1;
        if (v_blank_counter == VBLANK - 1) begin
          v_blank_counter <= 0;
          state <= STATE_FRAME_INIT
        end
      end
    endcase
