

module DE1_SoC_Computer (
	////////////////////////////////////
	// FPGA Pins
	////////////////////////////////////

	// Clock pins
	CLOCK_50,
	CLOCK2_50,
	CLOCK3_50,
	CLOCK4_50,

	// ADC
	ADC_CS_N,
	ADC_DIN,
	ADC_DOUT,
	ADC_SCLK,

	// Audio
	AUD_ADCDAT,
	AUD_ADCLRCK,
	AUD_BCLK,
	AUD_DACDAT,
	AUD_DACLRCK,
	AUD_XCK,

	// SDRAM
	DRAM_ADDR,
	DRAM_BA,
	DRAM_CAS_N,
	DRAM_CKE,
	DRAM_CLK,
	DRAM_CS_N,
	DRAM_DQ,
	DRAM_LDQM,
	DRAM_RAS_N,
	DRAM_UDQM,
	DRAM_WE_N,

	// I2C Bus for Configuration of the Audio and Video-In Chips
	FPGA_I2C_SCLK,
	FPGA_I2C_SDAT,

	// 40-Pin Headers
	GPIO_0,
	GPIO_1,
	
	// Seven Segment Displays
	HEX0,
	HEX1,
	HEX2,
	HEX3,
	HEX4,
	HEX5,

	// IR
	IRDA_RXD,
	IRDA_TXD,

	// Pushbuttons
	KEY,

	// LEDs
	LEDR,

	// PS2 Ports
	PS2_CLK,
	PS2_DAT,
	
	PS2_CLK2,
	PS2_DAT2,

	// Slider Switches
	SW,

	// Video-In
	TD_CLK27,
	TD_DATA,
	TD_HS,
	TD_RESET_N,
	TD_VS,

	// VGA
	VGA_B,
	VGA_BLANK_N,
	VGA_CLK,
	VGA_G,
	VGA_HS,
	VGA_R,
	VGA_SYNC_N,
	VGA_VS,

	////////////////////////////////////
	// HPS Pins
	////////////////////////////////////
	
	// DDR3 SDRAM
	HPS_DDR3_ADDR,
	HPS_DDR3_BA,
	HPS_DDR3_CAS_N,
	HPS_DDR3_CKE,
	HPS_DDR3_CK_N,
	HPS_DDR3_CK_P,
	HPS_DDR3_CS_N,
	HPS_DDR3_DM,
	HPS_DDR3_DQ,
	HPS_DDR3_DQS_N,
	HPS_DDR3_DQS_P,
	HPS_DDR3_ODT,
	HPS_DDR3_RAS_N,
	HPS_DDR3_RESET_N,
	HPS_DDR3_RZQ,
	HPS_DDR3_WE_N,

	// Ethernet
	HPS_ENET_GTX_CLK,
	HPS_ENET_INT_N,
	HPS_ENET_MDC,
	HPS_ENET_MDIO,
	HPS_ENET_RX_CLK,
	HPS_ENET_RX_DATA,
	HPS_ENET_RX_DV,
	HPS_ENET_TX_DATA,
	HPS_ENET_TX_EN,

	// Flash
	HPS_FLASH_DATA,
	HPS_FLASH_DCLK,
	HPS_FLASH_NCSO,

	// Accelerometer
	HPS_GSENSOR_INT,
		
	// General Purpose I/O
	HPS_GPIO,
		
	// I2C
	HPS_I2C_CONTROL,
	HPS_I2C1_SCLK,
	HPS_I2C1_SDAT,
	HPS_I2C2_SCLK,
	HPS_I2C2_SDAT,

	// Pushbutton
	HPS_KEY,

	// LED
	HPS_LED,
		
	// SD Card
	HPS_SD_CLK,
	HPS_SD_CMD,
	HPS_SD_DATA,

	// SPI
	HPS_SPIM_CLK,
	HPS_SPIM_MISO,
	HPS_SPIM_MOSI,
	HPS_SPIM_SS,

	// UART
	HPS_UART_RX,
	HPS_UART_TX,

	// USB
	HPS_CONV_USB_N,
	HPS_USB_CLKOUT,
	HPS_USB_DATA,
	HPS_USB_DIR,
	HPS_USB_NXT,
	HPS_USB_STP
);

//=======================================================
//  PARAMETER declarations
//=======================================================


//=======================================================
//  PORT declarations
//=======================================================

////////////////////////////////////
// FPGA Pins
////////////////////////////////////

// Clock pins
input						CLOCK_50;
input						CLOCK2_50;
input						CLOCK3_50;
input						CLOCK4_50;

// ADC
inout						ADC_CS_N;
output					ADC_DIN;
input						ADC_DOUT;
output					ADC_SCLK;

// Audio
input						AUD_ADCDAT;
inout						AUD_ADCLRCK;
inout						AUD_BCLK;
output					AUD_DACDAT;
inout						AUD_DACLRCK;
output					AUD_XCK;

// SDRAM
output 		[12: 0]	DRAM_ADDR;
output		[ 1: 0]	DRAM_BA;
output					DRAM_CAS_N;
output					DRAM_CKE;
output					DRAM_CLK;
output					DRAM_CS_N;
inout			[15: 0]	DRAM_DQ;
output					DRAM_LDQM;
output					DRAM_RAS_N;
output					DRAM_UDQM;
output					DRAM_WE_N;

// I2C Bus for Configuration of the Audio and Video-In Chips
output					FPGA_I2C_SCLK;
inout						FPGA_I2C_SDAT;

// 40-pin headers
inout			[35: 0]	GPIO_0;
inout			[35: 0]	GPIO_1;

// Seven Segment Displays
output		[ 6: 0]	HEX0;
output		[ 6: 0]	HEX1;
output		[ 6: 0]	HEX2;
output		[ 6: 0]	HEX3;
output		[ 6: 0]	HEX4;
output		[ 6: 0]	HEX5;

// IR
input						IRDA_RXD;
output					IRDA_TXD;

// Pushbuttons
input			[ 3: 0]	KEY;

// LEDs
output		[ 9: 0]	LEDR;

// PS2 Ports
inout						PS2_CLK;
inout						PS2_DAT;

inout						PS2_CLK2;
inout						PS2_DAT2;

// Slider Switches
input			[ 9: 0]	SW;

// Video-In
input						TD_CLK27;
input			[ 7: 0]	TD_DATA;
input						TD_HS;
output					TD_RESET_N;
input						TD_VS;

// VGA
output		[ 7: 0]	VGA_B;
output					VGA_BLANK_N;
output					VGA_CLK;
output		[ 7: 0]	VGA_G;
output					VGA_HS;
output		[ 7: 0]	VGA_R;
output					VGA_SYNC_N;
output					VGA_VS;



////////////////////////////////////
// HPS Pins
////////////////////////////////////
	
// DDR3 SDRAM
output		[14: 0]	HPS_DDR3_ADDR;
output		[ 2: 0]  HPS_DDR3_BA;
output					HPS_DDR3_CAS_N;
output					HPS_DDR3_CKE;
output					HPS_DDR3_CK_N;
output					HPS_DDR3_CK_P;
output					HPS_DDR3_CS_N;
output		[ 3: 0]	HPS_DDR3_DM;
inout			[31: 0]	HPS_DDR3_DQ;
inout			[ 3: 0]	HPS_DDR3_DQS_N;
inout			[ 3: 0]	HPS_DDR3_DQS_P;
output					HPS_DDR3_ODT;
output					HPS_DDR3_RAS_N;
output					HPS_DDR3_RESET_N;
input						HPS_DDR3_RZQ;
output					HPS_DDR3_WE_N;

// Ethernet
output					HPS_ENET_GTX_CLK;
inout						HPS_ENET_INT_N;
output					HPS_ENET_MDC;
inout						HPS_ENET_MDIO;
input						HPS_ENET_RX_CLK;
input			[ 3: 0]	HPS_ENET_RX_DATA;
input						HPS_ENET_RX_DV;
output		[ 3: 0]	HPS_ENET_TX_DATA;
output					HPS_ENET_TX_EN;

// Flash
inout			[ 3: 0]	HPS_FLASH_DATA;
output					HPS_FLASH_DCLK;
output					HPS_FLASH_NCSO;

// Accelerometer
inout						HPS_GSENSOR_INT;

// General Purpose I/O
inout			[ 1: 0]	HPS_GPIO;

// I2C
inout						HPS_I2C_CONTROL;
inout						HPS_I2C1_SCLK;
inout						HPS_I2C1_SDAT;
inout						HPS_I2C2_SCLK;
inout						HPS_I2C2_SDAT;

// Pushbutton
inout						HPS_KEY;

// LED
inout						HPS_LED;

// SD Card
output					HPS_SD_CLK;
inout						HPS_SD_CMD;
inout			[ 3: 0]	HPS_SD_DATA;

// SPI
output					HPS_SPIM_CLK;
input						HPS_SPIM_MISO;
output					HPS_SPIM_MOSI;
inout						HPS_SPIM_SS;

// UART
input						HPS_UART_RX;
output					HPS_UART_TX;

// USB
inout						HPS_CONV_USB_N;
input						HPS_USB_CLKOUT;
inout			[ 7: 0]	HPS_USB_DATA;
input						HPS_USB_DIR;
input						HPS_USB_NXT;
output					HPS_USB_STP;

//=======================================================
//  REG/WIRE declarations
//=======================================================

wire			[15: 0]	hex3_hex0=0;
//wire			[15: 0]	hex5_hex4;

//assign HEX0 = ~hex3_hex0[ 6: 0]; // hex3_hex0[ 6: 0]; 
//assign HEX1 = ~hex3_hex0[14: 8];
//assign HEX2 = ~hex3_hex0[22:16];
//assign HEX3 = ~hex3_hex0[30:24];
assign HEX4 = 7'b1111111;
assign HEX5 = 7'b1111111;

HexDigit Digit0(HEX0, hex3_hex0[3:0]);
HexDigit Digit1(HEX1, hex3_hex0[7:4]);
HexDigit Digit2(HEX2, hex3_hex0[11:8]);
HexDigit Digit3(HEX3, hex3_hex0[15:12]);

//========================================================
//	Top Level Design
//========================================================

// Wires for interfacing with onchip sram for intensity
 wire [7:0] 	intensity_rdata;
 reg  [7:0]		intensity_wdata;
 wire [15:0]	intensity_addr;

 wire 			intensity_clken = 1'b1;
 reg				intensity_write;
 wire				intensity_chipsel = 1'b1;
 
 wire [31:0] 	result_rdata;
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
 wire clk;
 assign clk = CLOCK_50;
 
 wire [7:0] ctrl_reg_h2f;
 wire [7:0] ctrl_reg_f2h;


 assign frame_complete_w = frame_complete;

 streaming_emulator streamer 
 (
	  .clk            (clk),
	  .reset          (reset),
	  .rdata				(intensity_rdata),
	  .start				(ctrl_reg_h2f[0]),
	  
	  .raddr				(intensity_addr),
	  .data           (streamer_data),
	  .fv             (streamer_fv),
	  .lv             (streamer_lv),
	  .frame_complete (frame_complete),
	  .valid          (streamer_valid)
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

 (*keep*) wire [27:0] x_centroid; 
 (*keep*) wire [27:0] y_centroid; 
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
 (*preserve, noprune*) reg [27:0] x_centroid_out;
 (*preserve, noprune*) reg [27:0] y_centroid_out;
 (*preserve, noprune*) reg [26:0] x_slopes_out;
 (*preserve, noprune*) reg [26:0] y_slopes_out;
 (*preserve, noprune*) reg [7:0]  resw_write_idx;
 
 localparam SLOPE_X_OFFSET = 0;
 localparam SLOPE_Y_OFFSET = 256;
 localparam CENTROID_X_OFFSET = 512;
 localparam CENTROID_Y_OFFSET = 768;
 localparam ZERNIKE_OFFSET = 1024;
 
 localparam RESULT_W_WAIT = 4'd0;
 localparam RESULT_W_SX = 4'd1;
 localparam RESULT_W_SY = 4'd2;
 localparam RESULT_W_CX = 4'd3;
 localparam RESULT_W_CY = 4'd4;
 localparam RESULT_W_DONE = 4'd5;
 
 reg  [4:0] resw_next_state;
 reg	 		resw_start;
 reg  [4:0] resw_state;
 
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
		RESULT_W_CY:	resw_next_state = RESULT_W_DONE;
		RESULT_W_DONE:	resw_next_state = RESULT_W_WAIT;
	endcase
 end
 
 always @(posedge clk) begin
	if (resw_state == RESULT_W_WAIT) begin
		result_wdata <= 32'd0;
		result_addr <= 11'd0;
		result_write <= 1'b0;
		resw_write_idx <= 8'd0;
	end
	else if (resw_state == RESULT_W_SX) begin
		result_wdata <= x_slopes;
		result_addr <= resw_write_idx + SLOPE_X_OFFSET;
		result_write <= 1'b1;
	end
	else if (resw_state == RESULT_W_SY) begin
		result_wdata <= y_slopes;
		result_addr <= resw_write_idx + SLOPE_Y_OFFSET;
		result_write <= 1'b1;
	end
	else if (resw_state == RESULT_W_CX) begin
		result_wdata <= x_centroid;
		result_addr <= resw_write_idx + CENTROID_X_OFFSET;
		result_write <= 1'b1;
	end
	else if (resw_state == RESULT_W_CY) begin
		result_wdata <= y_centroid;
		result_addr <= resw_write_idx + CENTROID_Y_OFFSET;
		result_write <= 1'b1;
	end
	else if (resw_state == RESULT_W_DONE) begin
		resw_write_idx <= resw_write_idx + 8'd1;
		result_write <= 1'b0;
	end
 end

 // Intensity reading logic
 always @(posedge clk) begin
	if (reset) begin
		intensity_write <= 1'b0;
		intensity_wdata <= 8'd0;
	end
	else begin
		
		if (new_subapeture) begin
			resw_start <= 1'b1;
		end
		else begin
			resw_start <= 1'b0;
		end
		
		x_centroid_out <= x_centroid;
		y_centroid_out <= y_centroid;
		x_slopes_out <= x_slopes;
		y_slopes_out <= y_slopes;
		
	end
 end


//=======================================================
//  Structural coding
//=======================================================
// From Qsys

Computer_System The_System (
	////////////////////////////////////
	// FPGA Side
	////////////////////////////////////

	// Global signals
	.system_pll_ref_clk_clk					(CLOCK_50),
	.system_pll_ref_reset_reset			(1'b0),
	
	
	.intensity_sram_address			(intensity_addr),            	//       intensity_sram.address
	.intensity_sram_chipselect		(intensity_chipsel),          //                     .chipselect
	.intensity_sram_clken			(intensity_clken),          	//                     .clken
	.intensity_sram_write			(intensity_write),            //                     .write
	.intensity_sram_readdata		(intensity_rdata),            //                     .readdata
	.intensity_sram_writedata		(intensity_wdata),           	//                     .writedata
	
	.result_sram_address			(result_addr),            			//          result_sram.address
	.result_sram_chipselect		(result_chipsel),          		//                     .chipselect
	.result_sram_clken			(result_clken),          			//                     .clken
	.result_sram_write			(result_write),            		//                     .write
	.result_sram_readdata		(result_rdata),            		//                     .readdata
	.result_sram_writedata		(result_wdata),           			//                     .writedata
	.result_sram_byteenable		(result_bytesel),						//							  .byteenable
	
	.ctrl_reg_h2f_export			(ctrl_reg_h2f),
	.ctrl_reg_f2h_export			(ctrl_reg_f2h),
	
	////////////////////////////////////
	// HPS Side
	////////////////////////////////////
	// DDR3 SDRAM
	.memory_mem_a			(HPS_DDR3_ADDR),
	.memory_mem_ba			(HPS_DDR3_BA),
	.memory_mem_ck			(HPS_DDR3_CK_P),
	.memory_mem_ck_n		(HPS_DDR3_CK_N),
	.memory_mem_cke		(HPS_DDR3_CKE),
	.memory_mem_cs_n		(HPS_DDR3_CS_N),
	.memory_mem_ras_n		(HPS_DDR3_RAS_N),
	.memory_mem_cas_n		(HPS_DDR3_CAS_N),
	.memory_mem_we_n		(HPS_DDR3_WE_N),
	.memory_mem_reset_n	(HPS_DDR3_RESET_N),
	.memory_mem_dq			(HPS_DDR3_DQ),
	.memory_mem_dqs		(HPS_DDR3_DQS_P),
	.memory_mem_dqs_n		(HPS_DDR3_DQS_N),
	.memory_mem_odt		(HPS_DDR3_ODT),
	.memory_mem_dm			(HPS_DDR3_DM),
	.memory_oct_rzqin		(HPS_DDR3_RZQ),
		  
	// Ethernet
	.hps_io_hps_io_gpio_inst_GPIO35	(HPS_ENET_INT_N),
	.hps_io_hps_io_emac1_inst_TX_CLK	(HPS_ENET_GTX_CLK),
	.hps_io_hps_io_emac1_inst_TXD0	(HPS_ENET_TX_DATA[0]),
	.hps_io_hps_io_emac1_inst_TXD1	(HPS_ENET_TX_DATA[1]),
	.hps_io_hps_io_emac1_inst_TXD2	(HPS_ENET_TX_DATA[2]),
	.hps_io_hps_io_emac1_inst_TXD3	(HPS_ENET_TX_DATA[3]),
	.hps_io_hps_io_emac1_inst_RXD0	(HPS_ENET_RX_DATA[0]),
	.hps_io_hps_io_emac1_inst_MDIO	(HPS_ENET_MDIO),
	.hps_io_hps_io_emac1_inst_MDC		(HPS_ENET_MDC),
	.hps_io_hps_io_emac1_inst_RX_CTL	(HPS_ENET_RX_DV),
	.hps_io_hps_io_emac1_inst_TX_CTL	(HPS_ENET_TX_EN),
	.hps_io_hps_io_emac1_inst_RX_CLK	(HPS_ENET_RX_CLK),
	.hps_io_hps_io_emac1_inst_RXD1	(HPS_ENET_RX_DATA[1]),
	.hps_io_hps_io_emac1_inst_RXD2	(HPS_ENET_RX_DATA[2]),
	.hps_io_hps_io_emac1_inst_RXD3	(HPS_ENET_RX_DATA[3]),

	// Flash
	.hps_io_hps_io_qspi_inst_IO0	(HPS_FLASH_DATA[0]),
	.hps_io_hps_io_qspi_inst_IO1	(HPS_FLASH_DATA[1]),
	.hps_io_hps_io_qspi_inst_IO2	(HPS_FLASH_DATA[2]),
	.hps_io_hps_io_qspi_inst_IO3	(HPS_FLASH_DATA[3]),
	.hps_io_hps_io_qspi_inst_SS0	(HPS_FLASH_NCSO),
	.hps_io_hps_io_qspi_inst_CLK	(HPS_FLASH_DCLK),

	// Accelerometer
	.hps_io_hps_io_gpio_inst_GPIO61	(HPS_GSENSOR_INT),

	//.adc_sclk                        (ADC_SCLK),
	//.adc_cs_n                        (ADC_CS_N),
	//.adc_dout                        (ADC_DOUT),
	//.adc_din                         (ADC_DIN),

	// General Purpose I/O
	.hps_io_hps_io_gpio_inst_GPIO40	(HPS_GPIO[0]),
	.hps_io_hps_io_gpio_inst_GPIO41	(HPS_GPIO[1]),

	// I2C
	.hps_io_hps_io_gpio_inst_GPIO48	(HPS_I2C_CONTROL),
	.hps_io_hps_io_i2c0_inst_SDA		(HPS_I2C1_SDAT),
	.hps_io_hps_io_i2c0_inst_SCL		(HPS_I2C1_SCLK),
	.hps_io_hps_io_i2c1_inst_SDA		(HPS_I2C2_SDAT),
	.hps_io_hps_io_i2c1_inst_SCL		(HPS_I2C2_SCLK),

	// Pushbutton
	.hps_io_hps_io_gpio_inst_GPIO54	(HPS_KEY),

	// LED
	.hps_io_hps_io_gpio_inst_GPIO53	(HPS_LED),

	// SD Card
	.hps_io_hps_io_sdio_inst_CMD	(HPS_SD_CMD),
	.hps_io_hps_io_sdio_inst_D0	(HPS_SD_DATA[0]),
	.hps_io_hps_io_sdio_inst_D1	(HPS_SD_DATA[1]),
	.hps_io_hps_io_sdio_inst_CLK	(HPS_SD_CLK),
	.hps_io_hps_io_sdio_inst_D2	(HPS_SD_DATA[2]),
	.hps_io_hps_io_sdio_inst_D3	(HPS_SD_DATA[3]),

	// SPI
	.hps_io_hps_io_spim1_inst_CLK		(HPS_SPIM_CLK),
	.hps_io_hps_io_spim1_inst_MOSI	(HPS_SPIM_MOSI),
	.hps_io_hps_io_spim1_inst_MISO	(HPS_SPIM_MISO),
	.hps_io_hps_io_spim1_inst_SS0		(HPS_SPIM_SS),

	// UART
	.hps_io_hps_io_uart0_inst_RX	(HPS_UART_RX),
	.hps_io_hps_io_uart0_inst_TX	(HPS_UART_TX),

	// USB
	.hps_io_hps_io_gpio_inst_GPIO09	(HPS_CONV_USB_N),
	.hps_io_hps_io_usb1_inst_D0		(HPS_USB_DATA[0]),
	.hps_io_hps_io_usb1_inst_D1		(HPS_USB_DATA[1]),
	.hps_io_hps_io_usb1_inst_D2		(HPS_USB_DATA[2]),
	.hps_io_hps_io_usb1_inst_D3		(HPS_USB_DATA[3]),
	.hps_io_hps_io_usb1_inst_D4		(HPS_USB_DATA[4]),
	.hps_io_hps_io_usb1_inst_D5		(HPS_USB_DATA[5]),
	.hps_io_hps_io_usb1_inst_D6		(HPS_USB_DATA[6]),
	.hps_io_hps_io_usb1_inst_D7		(HPS_USB_DATA[7]),
	.hps_io_hps_io_usb1_inst_CLK		(HPS_USB_CLKOUT),
	.hps_io_hps_io_usb1_inst_STP		(HPS_USB_STP),
	.hps_io_hps_io_usb1_inst_DIR		(HPS_USB_DIR),
	.hps_io_hps_io_usb1_inst_NXT		(HPS_USB_NXT)
);
endmodule // end top level


/// end /////////////////////////////////////////////////////////////////////