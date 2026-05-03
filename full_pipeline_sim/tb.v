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

        key_reset = 1'b1;
        #1
        clk = 1'b1;
        #1
        clk = 1'b0;
        key_reset = 1'b0;

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

    // ---- Monitor: print Zernike coefficients when done pulses ----
    always @(posedge clk) begin
        if (dut.done) begin
            $display("Zernike Coefficients (Nanometers, Q4.23 raw):");
            $display("Mode 1  (Tilt X):    %d", $signed(dut.zernike_out[  26:  0]));
            $display("Mode 2  (Tilt Y):    %d", $signed(dut.zernike_out[  53: 27]));
            $display("Mode 3  (Defocus):   %d", $signed(dut.zernike_out[  80: 54]));
            $display("Mode 4  (Astig 45):  %d", $signed(dut.zernike_out[ 107: 81]));
            $display("Mode 5  (Astig 0):   %d", $signed(dut.zernike_out[ 134:108]));
            $display("Mode 6  (Coma X):    %d", $signed(dut.zernike_out[ 161:135]));
            $display("Mode 7  (Coma Y):    %d", $signed(dut.zernike_out[ 188:162]));
            $display("Mode 8  (Trefoil X): %d", $signed(dut.zernike_out[ 215:189]));
            $display("Mode 9  (Trefoil Y): %d", $signed(dut.zernike_out[ 242:216]));
            $display("Mode 10 (Sph.):      %d", $signed(dut.zernike_out[ 269:243]));
            $finish;
        end
    end

    // ---- Debug: print first 3 slopes fed to E-matrix accumulator ----
    always @(posedge clk) begin
        if (dut.subap_valid && dut.em.sub_counter < 3) begin
            $display("  slope[%0d] x=%0d  y=%0d",
                     dut.em.sub_counter,
                     $signed(dut.x_slopes),
                     $signed(dut.y_slopes));
        end
    end

endmodule
