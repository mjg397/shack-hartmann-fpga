// this emulates the streaming property of the shack-hartmann wavefront sensor from a preloaded image into FPGA memroy
// this intends to allow for our shack-hartmann processing pipeline to have an accurate representation of incoming data
// currently this verilog is pseudo and does not synthesize
parameter WIDTH
parameter HEIGHT

module (parameter N) Shack_Hartmann_Emulator ( // N is the number of pixels per clk cycle
    input wire clk,
    input wire rst,
    output reg [26:0] raw_data [N:0]
);
  // memory is read from given from python spi code
  reg mem_idx = 0;

  always @(posedge_clk) begin
    if (rst) begin
      mem_idx <= 0;
      data_out <= 0;
    end
    else if (mem_idx == memory_size) begin
      mem_idx <= 0;
      complete_frame <= 1;
    end
    else begin
      mem_idx <= mem_idx + N;
      raw_data <= memory[mem_idx:N];
    end
  end

endmodule
