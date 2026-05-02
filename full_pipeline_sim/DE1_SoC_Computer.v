`timescale 1ns/1ps

module DE1_SoC_Computer (

	clk,
	hps_reset,
	key_reset

);

	//=======================================================
	//  PARAMETER declarations
	//=======================================================


	//=======================================================
	//  PORT declarations
	//=======================================================

	input						clk;
	input						hps_reset;
	input						key_reset;

	//=======================================================
	//  REG/WIRE declarations
	//=======================================================

	wire			[15: 0]	hex3_hex0=0;

	//========================================================
	//	Top Level Design
	//========================================================
	
	wire [31:0] result_rdata;
	reg  [31:0]	result_wdata;
	reg  [10:0]	result_addr;

	wire 			result_clken = 1'b1;
	reg				result_write;
	wire				result_chipsel = 1'b1;
	wire [3:0]		result_bytesel = 4'hF;
	
	// Pipeline start
	wire [7:0] streamer_data;
	wire streamer_fv;
	wire streamer_lv;
	wire frame_complete;
	wire streamer_valid;
	wire frame_complete_w;
	// assign clk = CLOCK_50;
	
	wire [7:0] ctrl_reg_h2f;
	wire [7:0] ctrl_reg_f2h;
	
	wire reset;

	assign frame_complete_w = frame_complete;

	reg [1:0] start_sync;
	always @(posedge clk)
		start_sync <= {start_sync[0], ctrl_reg_h2f[0]};

	wire start_synced = start_sync[1];
	

	streaming_emulator streamer 
	(
		.clk            	(clk),
		.reset          	(reset),
		.data           	(streamer_data),
		.fv             	(streamer_fv),
		.lv             	(streamer_lv),
		.frame_complete 	(frame_complete),
		.valid          	(streamer_valid)
	);

	wire full_frame_complete_accumulator;
	wire [7:0] subaps_done_accumulator;
	wire [15:0] intensity_accumulator;
	wire [19:0] xI_accumulator;
	wire [19:0] yI_accumulator;

	TCoG_intensity_accumulator accumulator
	(
		.clk            (clk),
		.reset          (reset),

		// Inputs from streamer
		.valid          (streamer_valid),
		.data_in        (streamer_data),

		// Outputs 
		.full_frame_complete    (full_frame_complete_accumulator),
		.subapetures_completed  (subaps_done_accumulator),
		.intensity              (intensity_accumulator),
		.x_intensity            (xI_accumulator),
		.y_intensity            (yI_accumulator)
	);

	wire [19:0] xI_reciprocal;
	wire [19:0] yI_reciprocal;
	wire [26:0] rI_reciprocal;
	wire [7:0] subaps_done_reciprocal;

	intensity_reciprocal reciprocal
	(
		.clk                (clk),
		.reset              (reset),
		.xI_in              (xI_accumulator),
		.yI_in              (yI_accumulator),
		.sI                 (intensity_accumulator),
		.centroids_done_in  (subaps_done_accumulator),

		.xI_out             (xI_reciprocal),
		.yI_out             (yI_reciprocal),
		.rI_out             (rI_reciprocal),
		.centroids_done_out (subaps_done_reciprocal)
	);

	(*keep*) wire [26:0] x_centroid; 
	(*keep*) wire [26:0] y_centroid; 
	(*keep*) wire [26:0] x_slopes; 
	(*keep*) wire [26:0] y_slopes; 
	wire new_subapeture;

	slope_calculation slopes
	(
		.clk                    (clk),
		.rst                    (reset),
		.subapetures_completed  (subaps_done_reciprocal),
		.frame_complete         (frame_complete),
		.rec_intensity          (rI_reciprocal),
		.x_intensity            (xI_reciprocal),
		.y_intensity            (yI_reciprocal),
		.x_centroid             (x_centroid),
		.y_centroid             (y_centroid),
		.x_slope                (x_slopes),
		.y_slope                (y_slopes),
		.new_subapeture         (new_subapeture)
	);

	(*keep*) wire [269:0] zernike_out;
	(*keep*) wire done; /* synthesis keep */
	
	assign ctrl_reg_f2h[0] = done;

	ematrix_accumulator em
	(
		.clk             (clk),
		.rst             (reset),

		.sub_valid       (new_subapeture),
		.x_slope         (x_slopes),   // Q4.23
		.y_slope         (y_slopes),   // Q4.23

		.zernike_out     (zernike_out),     // 10 × 27 bits, Q4.23 per mode
		.done            (done)
	);
	
	(*preserve, noprune*) reg [26:0] zernike_out_reg[9:0];
	(*preserve, noprune*) reg [26:0] x_centroid_out;
	(*preserve, noprune*) reg [26:0] y_centroid_out;
	(*preserve, noprune*) reg [26:0] x_slopes_out;
	(*preserve, noprune*) reg [26:0] y_slopes_out;
	(*preserve, noprune*) reg [10:0] resw_write_idx;
	
	localparam SLOPE_X_OFFSET = 11'd0;
	localparam SLOPE_Y_OFFSET = 11'd256;
	localparam CENTROID_X_OFFSET = 11'd512;
	localparam CENTROID_Y_OFFSET = 11'd768;
	localparam ZERNIKE_OFFSET = 11'd1024;
	
	localparam RESULT_W_WAIT = 5'd0;
	localparam RESULT_W_SX = 5'd1;
	localparam RESULT_W_SY = 5'd2;
	localparam RESULT_W_CX = 5'd3;
	localparam RESULT_W_CY = 5'd4;
	localparam RESULT_W_ZK = 5'd5;
	localparam RESULT_W_DONE = 5'd6;
	
	(*preserve, noprune*) reg  [4:0] resw_next_state;
	(*preserve, noprune*) reg	 		 resw_start;
	(*preserve, noprune*) reg  [4:0] resw_state;
	(*preserve, noprune*) reg 		 write_zernike_latch;
	(*preserve, noprune*) reg  [10:0] write_zernike_count;
	
	genvar i;
	generate
		for (i = 0; i < 10; i = i + 1) begin : zernike_assignment
			always @(posedge clk) begin
				if (done) begin
					zernike_out_reg[i] <= zernike_out[ ((i + 1) * 27) - 1 : i * 27];
				end
			end
		end
	endgenerate
	
	// Result writing FSM
	always @(*) begin
		case (resw_state)
			RESULT_W_WAIT: resw_next_state = resw_start ? RESULT_W_SX : RESULT_W_WAIT;
			RESULT_W_SX:	resw_next_state = RESULT_W_SY;
			RESULT_W_SY: 	resw_next_state = RESULT_W_CX;
			RESULT_W_CX:	resw_next_state = RESULT_W_CY;
			RESULT_W_CY:	resw_next_state = write_zernike_latch ? RESULT_W_ZK : RESULT_W_DONE;
			RESULT_W_ZK:	resw_next_state = write_zernike_count == 11'd9 ? RESULT_W_DONE : RESULT_W_ZK;
			RESULT_W_DONE:	resw_next_state = RESULT_W_WAIT;
			default:		resw_next_state = RESULT_W_WAIT;
		endcase
	end

	always @(posedge clk) begin
		if (reset) begin
			resw_state       <= RESULT_W_WAIT;
			resw_write_idx   <= 11'd0;
			resw_start       <= 1'b0;
			result_write     <= 1'b0;
			result_wdata     <= 32'd0;
			result_addr      <= 11'd0;
			write_zernike_latch <= 1'b0;
		end
		else begin
			resw_state <= resw_next_state;
			
			// resw_start
			resw_start <= new_subapeture ? 1'b1 : 1'b0;
			
			// Capture outputs
			x_centroid_out <= x_centroid;
			y_centroid_out <= y_centroid;
			x_slopes_out   <= x_slopes;
			y_slopes_out   <= y_slopes;

			if (done)
				write_zernike_latch <= 1'b1;
			else if (resw_state == RESULT_W_ZK)
				write_zernike_latch <= 1'b0;

			// FSM datapath
		case (resw_state)
				RESULT_W_WAIT: begin
					result_wdata <= 32'd0;
					result_addr  <= 11'd0;
					result_write <= 1'b0;
					write_zernike_count <= 11'd0;
				end
				RESULT_W_SX: begin
					result_wdata <= {5'd0, x_slopes};
					result_addr  <= resw_write_idx + SLOPE_X_OFFSET;
					result_write <= 1'b1;
				end
				RESULT_W_SY: begin
					result_wdata <= {5'd0, y_slopes};
					result_addr  <= resw_write_idx + SLOPE_Y_OFFSET;
					result_write <= 1'b1;
				end
				RESULT_W_CX: begin
					result_wdata <= {5'd0, x_centroid};
					result_addr  <= resw_write_idx + CENTROID_X_OFFSET;
					result_write <= 1'b1;
				end
				RESULT_W_CY: begin
					result_wdata <= {5'd0, y_centroid};
					result_addr  <= resw_write_idx + CENTROID_Y_OFFSET;
					result_write <= 1'b1;
				end
				RESULT_W_ZK: begin
					write_zernike_count <= write_zernike_count + 11'd1;
					result_wdata <= {5'd0, zernike_out_reg[write_zernike_count[3:0]]};
					result_addr <= write_zernike_count + ZERNIKE_OFFSET;
					result_write <= 1'b1;
				end
				RESULT_W_DONE: begin
					resw_write_idx <= resw_write_idx + 11'd1;
					result_write   <= 1'b0;
				end
				default: begin
					result_wdata <= 32'd0;
					result_addr  <= 11'd0;
					result_write <= 1'b0;
					write_zernike_count <= 11'd0;
				end
			endcase
		end
	end

	wire reset_from_key;
	wire reset_from_hps;

	async_reset key_reset_sync (
		.clk         (clk),
		.reset_async (key_reset),
		.reset_sync  (reset_from_key)
	);

	async_reset hps_reset_sync (
		.clk         (clk),
		.reset_async (hps_reset),  // bit 1 -> reset (bit 0 -> start)
		.reset_sync  (reset_from_hps)
	);

	assign reset = reset_from_key | reset_from_hps;

endmodule // end top level


/// end /////////////////////////////////////////////////////////////////////
