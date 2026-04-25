///////////////////////////////////////
/// memory testing M10k, MLAB, Qsys RAM
// compile with
// gcc mem_test_1.c -o mem1 -lm
///////////////////////////////////////
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <sys/mman.h>
#include <sys/time.h> 
#include <math.h> 

//#include "address_map_arm_brl4.h"
#define axi_master  0xc0000000 
#define axi_ram_base 0xc0000000
#define axi_master_span 0x00004000

#define lw_axi_master 0xff200000
#define lw_axi_master_span  0x00005000 

// i/o prot addresses
#define fp_arg_0 0x0070
#define fp_arg_1 0x0060
#define fp_result 0x0080

unsigned int floatToReg27(float) ;
float reg27ToFloat(unsigned int) ;


// the light weight buss base
void *h2p_lw_virtual_base;
volatile unsigned int *h2p_arg1_addr=NULL;
volatile unsigned int *h2p_arg0_addr=NULL;
volatile unsigned int *h2p_out_addr=NULL;

// RAM fp buffer
volatile unsigned int * fp_ram_ptr = NULL ;
void *fp_ram_virtual_base;

// /dev/mem file id
int fd;
	
int main(void)
{
  
	// === get FPGA addresses ==================
    // Open /dev/mem
	if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) 	{
		printf( "ERROR: could not open \"/dev/mem\"...\n" );
		return( 1 );
	}
    
    // get virtual addr that maps to physical
	// for light weight bus
	h2p_lw_virtual_base = mmap( NULL, lw_axi_master_span, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, lw_axi_master );	
	if( h2p_lw_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap1() failed...\n" );
		close( fd );
		return(1);
	}
	// floating point readout
    h2p_arg1_addr=(volatile unsigned int *)(h2p_lw_virtual_base + fp_arg_1);
	h2p_arg0_addr=(volatile unsigned int *)(h2p_lw_virtual_base + fp_arg_0);
	h2p_out_addr =(volatile unsigned int *)(h2p_lw_virtual_base + fp_result);
    
	
	// === get RAM float parameter addr =========
	fp_ram_virtual_base = mmap( NULL, axi_master_span, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, axi_master); //fp	
	
	if( fp_ram_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap3() failed...\n" );
		close( fd );
		return(1);
	}
    // Get the address that maps to the RAM buffer
	fp_ram_ptr =(unsigned int *)(fp_ram_virtual_base);
	
	// ===========================================
	
	while(1) 
	{
		float in0, in1, result_io, result_mem;
		int addr ;

		// and in qsys_ram
		*(fp_ram_ptr) = e_matrix_array;
		
		// trigger FPGA (fpga resets this when done)
		*(fp_ram_ptr) = 1;
		
		// wait for FPGA done
		while(*(fp_ram_ptr) == 1) {}; 
		
		// read result in i/o register
		result_io = reg27ToFloat(*h2p_out_addr);
		// read result from RAM (after copy to/from internal M10k)
		result_mem = reg27ToFloat(*(fp_ram_ptr+3));
		
		// print out inputs, HPS result, HW result
		printf("for fpga: sw[0]==mult, sw[1]==add \n\r") ;
		printf("HPS_mult=%f, HPS_add=%f, \n\r fpga_io=%f, fpga_mem=%f\n\r", in0*in1, in0+in1, result_io, result_mem);
		// print RAM contents
		for(addr=0; addr<4; addr++) {
			printf("RAM addr=%d contents=%f \n\r", addr, reg27ToFloat(*(fp_ram_ptr+addr)));	
		}
		
	} // end while(1)
} // end main

/**************************************************************************
 * Mark Eiding mje56                                                      *
 * ECE 5760                                                               *
 * Modified IEEE single precision FP                                      *
 * bit 26:      Sign     (0: pos, 1: neg)                                 *
 * bits[25:18]: Exponent (unsigned)                                       *
 * bits[17:0]:  Fraction (unsigned)                                       *
 *  (-1)^SIGN * 2^(EXP-127) * (1+.FRAC)                                   *
 * (http://en.wikipedia.org/wiki/Single-precision_floating-point_format)  *
 * Adapted from Skyler Schneider ss868                                    *
 *************************************************************************/
// Convert a C floating point into a 27-bit register floating point.
unsigned int floatToReg27(float f) {
    int f_f = (*(int*)&f);
    int f_sign = (f_f >> 31) & 0x1;
    int f_exp = (f_f >> 23) & 0xFF;
    int f_frac = f_f & 0x007FFFFF;
    int r_sign;
    int r_exp;
    int r_frac;
    r_sign = f_sign;
    if((f_exp == 0x00) || (f_exp == 0xFF)) {
        // 0x00 -> 0 or subnormal
        // 0xFF -> infinity or NaN
        r_exp = 0;
        r_frac = 0;
    } else {
        r_exp = (f_exp) & 0xFF;
        r_frac = ((f_frac >> 5)) & 0x0003FFFF;
    }
    return (r_sign << 26) | (r_exp << 18) | r_frac;
}

// Convert a 27-bit register floating point into a C floating point.
float reg27ToFloat(unsigned int r) {
    int sign = (r & 0x04000000) >> 26;
    unsigned int exp = (r & 0x03FC0000) >> 18;
    int frac = (r & 0x0003FFFF);
    float result = pow(2.0, (float) (exp-127.0));
    result = (1.0+(((float)frac) / 262144.0)) * result;
    if(sign) result = result * (-1);
    return result;
}


/// /// ///////////////////////////////////// 
/// end /////////////////////////////////////