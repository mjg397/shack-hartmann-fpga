`include "../shwfs_top.v"
`timescale 1ns/1ps

module shwfs_pipeline_tb ();

    reg clk_100;
    reg reset;

    reg [19:0] xI_reciprocal;
    reg [19:0] yI_reciprocal;
    reg [15:0] rI_reciprocal;
    reg [7:0] subaps_done_reciprocal;

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

    initial begin
        #10000
        $finish;
    end

    initial begin
        $dumpfile("shwfs_pipeline_tb.vcd");
        $dumpvars(0, shwfs_pipeline_tb);
    end

    shwfs_top DUT
    (
        .clk                    (clk_100),
        .reset                  (reset), 
        .xI_reciprocal          (xI_reciprocal),
        .yI_reciprocal          (yI_reciprocal),
        .rI_reciprocal          (rI_reciprocal),
        .subaps_done_reciprocal (subaps_done_reciprocal)
    );

endmodule
