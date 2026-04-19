///////////////////////////////////////
/// Plot RGB image from gnuplot script on VGA
/// Based on graphics_video_16bit.c
/// Reads blutux.rgb (128x128 RGB image) and displays on VGA
/// compile with
/// gcc plot_rgb_vga.c -o plot_rgb -O2 -lm
///
///////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <math.h>

// video display
#define SDRAM_BASE            0xC0000000
#define SDRAM_END             0xC3FFFFFF
#define SDRAM_SPAN			  0x04000000
// characters
#define FPGA_CHAR_BASE        0xC9000000
#define FPGA_CHAR_END         0xC9001FFF
#define FPGA_CHAR_SPAN        0x00002000
/* Cyclone V FPGA devices */
#define HW_REGS_BASE          0xff200000
#define HW_REGS_SPAN          0x00005000

// graphics primitives
void VGA_text (int, int, char *);
void VGA_text_clear();
void VGA_box (int, int, int, int, short);
void VGA_rect (int, int, int, int, short);
void VGA_line(int, int, int, int, short) ;
void VGA_Vline(int, int, int, short) ;
void VGA_Hline(int, int, int, short) ;
void VGA_disc (int, int, int, short);
void VGA_circle (int, int, int, int);

// pixel macro
#define VGA_PIXEL(x,y,color) do{\
	int  *pixel_ptr ;\
	pixel_ptr = (int*)((char *)vga_pixel_ptr + (((y)*640+(x))<<1)) ; \
	*(short *)pixel_ptr = (color);\
} while(0)

// the light weight buss base
void *h2p_lw_virtual_base;

// pixel buffer
volatile unsigned int * vga_pixel_ptr = NULL ;
void *vga_pixel_virtual_base;

// character buffer
volatile unsigned int * vga_char_ptr = NULL ;
void *vga_char_virtual_base;

// /dev/mem file id
int fd;

// Convert 24-bit RGB to 16-bit VGA color
// R bits 11-15, G bits 5-10, B bits 0-4
short rgb_to_16bit(unsigned char r, unsigned char g, unsigned char b) {
    unsigned short rr = (r >> 3) & 0x1F;  // 5 bits
    unsigned short gg = (g >> 2) & 0x3F;  // 6 bits
    unsigned short bb = (b >> 3) & 0x1F;  // 5 bits
    return (short)(bb | (gg << 5) | (rr << 11));
}

int main(void)
{

	// === need to mmap: =======================
	// FPGA_CHAR_BASE
	// FPGA_ONCHIP_BASE
	// HW_REGS_BASE

	// === get FPGA addresses ==================
    // Open /dev/mem
	if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) 	{
		printf( "ERROR: could not open \"/dev/mem\"...\n" );
		return( 1 );
	}

    // get virtual addr that maps to physical
	h2p_lw_virtual_base = mmap( NULL, HW_REGS_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HW_REGS_BASE );
	if( h2p_lw_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap1() failed...\n" );
		close( fd );
		return(1);
	}


	// === get VGA char addr =====================
	// get virtual addr that maps to physical
	vga_char_virtual_base = mmap( NULL, FPGA_CHAR_SPAN, ( 	PROT_READ | PROT_WRITE ), MAP_SHARED, fd, FPGA_CHAR_BASE );
	if( vga_char_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap2() failed...\n" );
		close( fd );
		return(1);
	}

    // Get the address that maps to the FPGA LED control
	vga_char_ptr =(unsigned int *)(vga_char_virtual_base);

	// === get VGA pixel addr ====================
	// get virtual addr that maps to physical
	vga_pixel_virtual_base = mmap( NULL, SDRAM_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, SDRAM_BASE);
	if( vga_pixel_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap3() failed...\n" );
		close( fd );
		return(1);
	}

    // Get the address that maps to the FPGA pixel buffer
	vga_pixel_ptr =(unsigned int *)(vga_pixel_virtual_base);

	// ===========================================

	/* create a message to be displayed on the VGA
          and LCD displays */
	char text_top_row[40] = "RGB File Display\0";
	char text_bottom_row[40] = "Cornell ece5760\0";

	// clear the screen
	VGA_box (0, 0, 639, 479, 0x0000);
	// clear the text
	VGA_text_clear();
	// write text
	VGA_text (10, 1, text_top_row);
	VGA_text (10, 2, text_bottom_row);

	// Read the RGB file directly
	FILE *fp = fopen("md.rgb", "rb");
	if (fp == NULL) {
		printf("ERROR: could not open blutux.rgb\n");
		VGA_text (10, 4, "Error: blutux.rgb not found");
		return(1);
	}

	// Image size 128x128
	int width = 400;
	int height = 400;
	int x_offset = 50;  // position on screen
	int y_offset = 50;

	unsigned char rgb[3];
	int x, y;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (fread(rgb, 1, 3, fp) != 3) {
				printf("ERROR: reading RGB data\n");
				fclose(fp);
				return(1);
			}
			short color = rgb_to_16bit(rgb[0], rgb[1], rgb[2]);  // assuming RGB order
			VGA_PIXEL(x + x_offset, y + y_offset, color);
		}
	}

	fclose(fp);

	VGA_text (10, 4, "RGB file displayed successfully");

	// Keep the program running to view the image
	while(1) {
		sleep(1);
	}

} // end main

/****************************************************************************************
 * Subroutine to send a string of text to the VGA monitor
****************************************************************************************/
void VGA_text(int x, int y, char * text_ptr)
{
  	volatile char * character_buffer = (char *) vga_char_ptr ;	// VGA character buffer
	int offset;
	/* assume that the text string fits on one line */
	offset = (y << 7) + x;
	while ( *(text_ptr) )
	{
		// write to the character buffer
		*(character_buffer + offset) = *(text_ptr);
		++text_ptr;
		++offset;
	}
}

/****************************************************************************************
 * Subroutine to clear text to the VGA monitor
****************************************************************************************/
void VGA_text_clear()
{
  	volatile char * character_buffer = (char *) vga_char_ptr ;	// VGA character buffer
	int offset, x, y;
	for (x=0; x<79; x++){
		for (y=0; y<59; y++){
	/* assume that the text string fits on one line */
			offset = (y << 7) + x;
			// write to the character buffer
			*(character_buffer + offset) = ' ';
		}
	}
}

/****************************************************************************************
 * Draw a filled rectangle on the VGA monitor
****************************************************************************************/
#define SWAP(X,Y) do{int temp=X; X=Y; Y=temp;}while(0)

void VGA_box(int x1, int y1, int x2, int y2, short pixel_color)
{
	char  *pixel_ptr ;
	int row, col;

	/* check and fix box coordinates to be valid */
	if (x1>639) x1 = 639;
	if (y1>479) y1 = 479;
	if (x2>639) x2 = 639;
	if (y2>479) y2 = 479;
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (x2<0) x2 = 0;
	if (y2<0) y2 = 0;
	if (x1>x2) SWAP(x1,x2);
	if (y1>y2) SWAP(y1,y2);
	for (row = y1; row <= y2; row++)
		for (col = x1; col <= x2; ++col)
		{
			VGA_PIXEL(col,row,pixel_color);
		}
}

/****************************************************************************************
 * Draw a outline rectangle on the VGA monitor
****************************************************************************************/
#define SWAP(X,Y) do{int temp=X; X=Y; Y=temp;}while(0)

void VGA_rect(int x1, int y1, int x2, int y2, short pixel_color)
{
	int row, col;

	/* check and fix box coordinates to be valid */
	if (x1>639) x1 = 639;
	if (y1>479) y1 = 479;
	if (x2>639) x2 = 639;
	if (y2>479) y2 = 479;
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (x2<0) x2 = 0;
	if (y2<0) y2 = 0;
	if (x1>x2) SWAP(x1,x2);
	if (y1>y2) SWAP(y1,y2);
	// left edge
	col = x1;
	for (row = y1; row <= y2; row++){
		VGA_PIXEL(col,row,pixel_color);
	}

	// right edge
	col = x2;
	for (row = y1; row <= y2; row++){
		VGA_PIXEL(col,row,pixel_color);
	}

	// top edge
	row = y1;
	for (col = x1; col <= x2; ++col){
		VGA_PIXEL(col,row,pixel_color);
	}

	// bottom edge
	row = y2;
	for (col = x1; col <= x2; ++col){
		VGA_PIXEL(col,row,pixel_color);
	}
}

/****************************************************************************************
 * Draw a horizontal line on the VGA monitor
****************************************************************************************/
#define SWAP(X,Y) do{int temp=X; X=Y; Y=temp;}while(0)

void VGA_Hline(int x1, int y1, int x2, short pixel_color)
{
	int row, col;

	/* check and fix box coordinates to be valid */
	if (x1>639) x1 = 639;
	if (y1>479) y1 = 479;
	if (x2>639) x2 = 639;
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (x2<0) x2 = 0;
	if (x1>x2) SWAP(x1,x2);
	// line
	row = y1;
	for (col = x1; col <= x2; ++col){
		VGA_PIXEL(col,row,pixel_color);
	}
}

/****************************************************************************************
 * Draw a vertical line on the VGA monitor
****************************************************************************************/
#define SWAP(X,Y) do{int temp=X; X=Y; Y=temp;}while(0)

void VGA_Vline(int x1, int y1, int y2, short pixel_color)
{
	int row, col;

	/* check and fix box coordinates to be valid */
	if (x1>639) x1 = 639;
	if (y1>479) y1 = 479;
	if (y2>479) y2 = 479;
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (y2<0) y2 = 0;
	if (y1>y2) SWAP(y1,y2);
	// line
	col = x1;
	for (row = y1; row <= y2; row++){
		VGA_PIXEL(col,row,pixel_color);
	}
}

/****************************************************************************************
 * Draw a filled circle on the VGA monitor
****************************************************************************************/

void VGA_disc(int x, int y, int r, short pixel_color)
{
	int row, col, rsqr, xc, yc;

	rsqr = r*r;

	for (yc = -r; yc <= r; yc++)
		for (xc = -r; xc <= r; xc++)
		{
			col = xc;
			row = yc;
			// add the r to make the edge smoother
			if(col*col+row*row <= rsqr+r){
				col += x; // add the center point
				row += y; // add the center point
				//check for valid 640x480
				if (col>639) col = 639;
				if (row>479) row = 479;
				if (col<0) col = 0;
				if (row<0) row = 0;
				VGA_PIXEL(col,row,pixel_color);
			}

		}
}

/****************************************************************************************
 * Draw a circle on the VGA monitor
****************************************************************************************/

void VGA_circle(int x, int y, int r, int pixel_color)
{
	int row, col, rsqr, xc, yc;
	int col1, row1;
	rsqr = r*r;

	for (yc = -r; yc <= r; yc++){
		col1 = (int)sqrt((float)(rsqr + r - yc*yc));
		// right edge
		col = col1 + x;
		row = yc + y;
		if (col>639) col = 639;
		if (row>479) row = 479;
		if (col<0) col = 0;
		if (row<0) row = 0;
		VGA_PIXEL(col,row,pixel_color);
		// left edge
		col = -col1 + x;
		if (col>639) col = 639;
		if (row>479) row = 479;
		if (col<0) col = 0;
		if (row<0) row = 0;
		VGA_PIXEL(col,row,pixel_color);
	}
	for (xc = -r; xc <= r; xc++){
		row1 = (int)sqrt((float)(rsqr + r - xc*xc));
		// bottom edge
		col = xc + x;
		row = row1 + y;
		if (col>639) col = 639;
		if (row>479) row = 479;
		if (col<0) col = 0;
		if (row<0) row = 0;
		VGA_PIXEL(col,row,pixel_color);
		// top edge
		row = -row1 + y;
		if (col>639) col = 639;
		if (row>479) row = 479;
		if (col<0) col = 0;
		if (row<0) row = 0;
		VGA_PIXEL(col,row,pixel_color);
	}
}

// =============================================
// === Draw a line
// =============================================
//plot a line
//at x1,y1 to x2,y2 with color
//Code is from David Rodgers,
//"Procedural Elements of Computer Graphics",1985
void VGA_line(int x1, int y1, int x2, int y2, short c) {
	int e;
	signed int dx,dy,j, temp;
	signed int s1,s2, xchange;
     signed int x,y;

	/* check and fix line coordinates to be valid */
	if (x1>639) x1 = 639;
	if (y1>479) y1 = 479;
	if (x2>639) x2 = 639;
	if (y2>479) y2 = 479;
	if (x1<0) x1 = 0;
	if (y1<0) y1 = 0;
	if (x2<0) x2 = 0;
	if (y2<0) y2 = 0;

	x = x1;
	y = y1;

	//take absolute value
	if (x2 < x1) {
		dx = x1 - x2;
		s1 = -1;
	}

	else if (x2 == x1) {
		dx = 0;
		s1 = 0;
	}

	else {
		dx = x2 - x1;
		s1 = 1;
	}

	if (y2 < y1) {
		dy = y1 - y2;
		s2 = -1;
	}

	else if (y2 == y1) {
		dy = 0;
		s2 = 0;
	}

	else {
		dy = y2 - y1;
		s2 = 1;
	}

	xchange = 0;

	if (dy>dx) {
		temp = dx;
		dx = dy;
		dy = temp;
		xchange = 1;
	}

	e = ((int)dy<<1) - dx;

	for (j=0; j<=dx; j++) {
		VGA_PIXEL(x,y,c);

		if (e>=0) {
			if (xchange==1) x = x + s1;
			else y = y + s2;
			e = e - ((int)dx<<1);
		}

		if (xchange==1) y = y + s2;
		else x = x + s1;

		e = e + ((int)dy<<1);
	}
}