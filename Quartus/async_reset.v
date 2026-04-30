module async_reset (
    input  wire reset_async,
    input  wire clk,
    output wire reset_sync
);

reg [1:0] sync_ff;
reg       sync_d;
reg [3:0] reset_count;

wire sync_level = sync_ff[1];
wire rising_edge = sync_level & ~sync_d;

always @(posedge clk) begin
    sync_ff     <= {sync_ff[0], reset_async}; // double flop
    sync_d      <= sync_level;

    if (rising_edge)
        reset_count <= 4'd8;
    else if (reset_count != 4'd0)
        reset_count <= reset_count - 4'd1;
end

assign reset_sync = (reset_count != 4'd0);

endmodule
