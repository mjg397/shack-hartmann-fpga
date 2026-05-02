`timescale 1ns/1ps

module tb;

    reg clk;
    reg key_reset;
    reg hps_reset;

    initial begin

        $dumpfile("waves/pipeline.vcd");
        $dumpvars(0, tb);

        clk = 1'b0;
        hps_reset = 1'b0;
        key_reset = 1'b0;
        #10;
        key_reset = 1'b1;
        #10;
        key_reset = 1'b0;
        #1;
    end

    integer i;

    initial begin
        for (i = 0; i < 67000; i = i + 1) begin
            clk = 1'b1;
            #1;
            clk = 1'b0;
            #1;
        end
    end

    DE1_SoC_Computer dut (
        .clk            (clk),
        .hps_reset      (hps_reset),
        .key_reset      (key_reset)
    );

endmodule
