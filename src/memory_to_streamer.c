///////////////////////////////////////
// DMA test
// Send data from HPS on-chip memory
// to FPGA SRAM
// compile with
// gcc DMA_1.c -o dma  -O3
//
// based on a design from
// https://github.com/robertofem/CycloneVSoC-examples/blob/master/Linux-applications/DMA_transfer_FPGA_DMAC/README.md
//
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
#include <stdint.h>

// ======================================
#define H2F_AXI_MASTER_BASE   0xC0000000
// main bus; scratch RAM
#define FPGA_ONCHIP_BASE      0xC8000000
#define FPGA_ONCHIP_SPAN      0x00010000
// h2f bus
// RAM FPGA port s2
// main bus addess 0x0800_0000
volatile unsigned int * sram_ptr = NULL ;
void *sram_virtual_base;

// ======================================
// lw_bus; DMA  addresses
#define HW_REGS_BASE        0xff200000
#define HW_REGS_SPAN        0x00005000 
#define DMA					0xff200000
#define DMA_STATUS_OFFSET	0x00
#define DMA_READ_ADD_OFFSET	0x04 // DATASHEET says 1!
#define DMA_WRT_ADD_OFFSET	0x08	
#define DMA_LENGTH_OFFSET	0x012
#define DMA_CNTL_OFFSET		0x024	
// the h2f light weight bus base
void *h2p_lw_virtual_base;
// HPS_to_FPGA DMA address = 0
volatile unsigned int * DMA_status_ptr = NULL ;	
volatile unsigned int * DMA_read_ptr = NULL ;	
volatile unsigned int * DMA_write_ptr = NULL ;	
volatile unsigned int * DMA_length_ptr = NULL ;	
volatile unsigned int * DMA_cntl_ptr = NULL ;	

// ======================================
// HPS onchip memory base/span
// 2^16 bytes at the top of memory
#define HPS_ONCHIP_BASE		0xffff0000
#define HPS_ONCHIP_SPAN		0x00010000
// HPS onchip memory (HPS side!)
volatile unsigned int * hps_onchip_ptr = NULL ;
void *hps_onchip_virtual_base;
// ======================================
// HPS linux MMU memory
//int test_array[];
uint8_t data[65536] ;
// ======================================
		  
// WAIT looks nicer than just braces
#define WAIT {}

// /dev/mem file id
int fd;	

// timer variables
struct timeval t1, t2;
double elapsedTime;

void DMA_transfer_bytes(uint8_t *data, int N, unsigned int *DMA_status_ptr, unsigned int *DMA_read_ptr, 
						unsigned int *DMA_write_ptr, unsigned int *DMA_length_ptr, unsigned int *DMA_cntl_ptr) 
	{
		// === DMA transfer HPS->FPGA 
		// set up DMA
		// from https://www.altera.com/en_US/pdfs/literature/ug/ug_embedded_ip.pdf
		// section 25.4.3 Tables 224 and 225
		*(DMA_status_ptr) = 0;
		// read bus-master gets data from HPS addr=0xffff0000
		*(DMA_status_ptr+1) = HPS_ONCHIP_BASE ;
		// write bus_master for fpga sram is mapped to 0x08000000 
		*(DMA_status_ptr+2) = 0x08000000 ;
		// copy N bytes (65536 for coeff array)
		*(DMA_status_ptr+3) = N ;
		// set bit 2 for WORD transfer
		// set bit 3 to start DMA
		// set bit 7 to stop on byte-coun	t
		// start the timer because DMA will start

		*(DMA_status_ptr+6) = 0b10001100;
		while ((*(DMA_status_ptr) & 0x010) == 0) WAIT;
	}
	
// int main(void)
// {

// 	// Declare volatile pointers to I/O registers (volatile 	
// 	// means that IO load and store instructions will be used 	
// 	// to access these pointer locations, 
// 	// instead of regular memory loads and stores)  
  
// 	// === get FPGA addresses ==================
//     // Open /dev/mem
// 	if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) 	{
// 		printf( "ERROR: could not open \"/dev/mem\"...\n" );
// 		return( 1 );
// 	}
    
// 	//============================================
//     // get virtual addr that maps to physical
// 	// for light weight bus
// 	// DMA status register
// 	h2p_lw_virtual_base = mmap( NULL, HW_REGS_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HW_REGS_BASE );	
// 	if( h2p_lw_virtual_base == MAP_FAILED ) {
// 		printf( "ERROR: mmap1() failed...\n" );
// 		close( fd );
// 		return(1);
// 	}
// 	// the DMA registers
// 	DMA_status_ptr = (unsigned int *)(h2p_lw_virtual_base);
// 	DMA_read_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_READ_ADD_OFFSET);
// 	DMA_write_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_WRT_ADD_OFFSET);
// 	DMA_length_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_LENGTH_OFFSET);
// 	DMA_cntl_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_CNTL_OFFSET);
	
	
// 	//============================================
	
// 	//  RAM FPGA parameter addr 
// 	sram_virtual_base = mmap( NULL, FPGA_ONCHIP_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, FPGA_ONCHIP_BASE); 	
	
// 	if( sram_virtual_base == MAP_FAILED ) {
// 		printf( "ERROR: mmap3() failed...\n" );
// 		close( fd );
// 		return(1);
// 	}
//     // Get the address that maps to the RAM buffer
// 	sram_ptr =(unsigned int *)(sram_virtual_base);
	
// 	// ===========================================

// 	// HPS onchip ram
// 	hps_onchip_virtual_base = mmap( NULL, HPS_ONCHIP_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HPS_ONCHIP_BASE); 	
	
// 	if( hps_onchip_virtual_base == MAP_FAILED ) {
// 		printf( "ERROR: mmap3() failed...\n" );
// 		close( fd );
// 		return(1);
// 	}
//     // Get the address that maps to the HPS ram
// 	hps_onchip_ptr =(unsigned int *)(hps_onchip_virtual_base);
	
	
// 	//============================================
// 	int N = 65536;
// 	//int data[16384] ;
// 	int i ;
// 	int temp1, temp2;
	
// 	while(1) 
// 	{
// 		// generate a sequence and write fpga memory
// 		for (i=0; i<N; i++){
// 			*(sram_ptr+i) = i ;
// 		}
		
// 		// === read back fpga memory
// 		// start the timer
// 		gettimeofday(&t1, NULL);
// 		temp1 = 0 ;
// 		for (i=0; i<N; i++){
// 			temp1 += *(sram_ptr+i) ;
// 		}
		
// 		// === DMA transfer HPS->FPGA 
// 		// set up DMA
// 		// from https://www.altera.com/en_US/pdfs/literature/ug/ug_embedded_ip.pdf
// 		// section 25.4.3 Tables 224 and 225
// 		*(DMA_status_ptr) = 0;
// 		// read bus-master gets data from HPS addr=0xffff0000
// 		*(DMA_status_ptr+1) = HPS_ONCHIP_BASE ;
// 		// write bus_master for fpga sram is mapped to 0x08000000 
// 		*(DMA_status_ptr+2) = 0x08000000 ;
// 		// copy N bytes (65536 for coeff array)
// 		*(DMA_status_ptr+3) = N ;
// 		// set bit 2 for WORD transfer
// 		// set bit 3 to start DMA
// 		// set bit 7 to stop on byte-coun	t
// 		// start the timer because DMA will start
// 		gettimeofday(&t1, NULL);
// 		*(DMA_status_ptr+6) = 0b10001100;
// 		while ((*(DMA_status_ptr) & 0x010) == 0) WAIT;
// 		// finish timing the transfer
// 		gettimeofday(&t2, NULL);
// 		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000000.0;      // sec to us
// 		elapsedTime += (t2.tv_usec - t1.tv_usec) ;   // us to 
// 		printf("DMA write T=%.0f uSec  n/sec=%2.3e\n\r", elapsedTime, N*1e6/elapsedTime);
		
// 		// === read back fpga memory that was loaded using DMA
// 		// start the timer
// 		gettimeofday(&t1, NULL);
// 		temp1 = 0 ;
// 		for (i=0; i<N; i++){
// 			temp1 += *(sram_ptr+i) ;
// 		}
// 		// finish timing the transfer
// 		gettimeofday(&t2, NULL);
// 		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000000.0;      // sec to us
// 		elapsedTime += (t2.tv_usec - t1.tv_usec) ;   // us to 
// 		printf("fpga read T=%.0f uSec  n/sec=%2.3e\n\r", elapsedTime, N*1e6/elapsedTime);
// 		printf("fpga memory sum=%d \n\r", temp1);
// 		// time the transaction
		
// 	} // end while(1)
// } // end main
	
//////////////////////////////////////////////////////////////////
/// end /////////////////////////////////////