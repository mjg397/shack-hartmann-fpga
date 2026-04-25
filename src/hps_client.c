#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h> // bzero()
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h> // read(), write(), close()
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <sys/mman.h>
#include <sys/time.h> 
#include <math.h> 

#define MAX 4096
#define PORT 80
#define SA struct sockaddr
#define DUMP_DIR "debug_dumps"
#define ENABLE_DUMPS 0

char filename[32];

// C Arrays to hold input data
uint8_t coeffs[65536]; // 256x256(x8) coefficient array
uint32_t e_matrix[3360]; // 10 zernike modes x 336 slopes (18 bit values - Q1.16 fixed point)

// C Arrays to hold results
uint32_t centroids[512];
uint32_t slopes[512];
uint32_t zernike_coeffs[10];

// ======================================
#define H2F_AXI_MASTER_BASE   0xC0000000
// main bus; scratch RAM
#define FPGA_ONCHIP_BASE      0xC8000000
#define FPGA_ONCHIP_SPAN      0x00010000
// h2f bus
// RAM FPGA port s2
// main bus addess 0x0800_0000
volatile uint8_t * sram_ptr = NULL ;
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

// DMA helper functions
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

void fabricate_results() {
    for (uint32_t i = 0; i < 512; i++) {
        centroids[i] = i;
        slopes[i] = i;
    }

    for (uint32_t i = 0; i < 10; i++) {
        zernike_coeffs[i] = i;
    }
}

// Debug helpers
static void ensure_dump_dir(void)
{
#if ENABLE_DUMPS
    mkdir(DUMP_DIR, 0777);
#endif
}

static void dump_buffer(const char *base_name, const void *buf, size_t len, size_t elem_size, size_t elems_per_row)
{
#if !ENABLE_DUMPS
    (void)base_name;
    (void)buf;
    (void)len;
    (void)elem_size;
    (void)elems_per_row;
    return;
#else
    char bin_path[256];
    char txt_path[256];
    FILE *fb;
    FILE *ft;

    snprintf(bin_path, sizeof(bin_path), "%s/%s.bin", DUMP_DIR, base_name);
    snprintf(txt_path, sizeof(txt_path), "%s/%s.txt", DUMP_DIR, base_name);

    fb = fopen(bin_path, "wb");
    if (fb != NULL) {
        fwrite(buf, 1, len, fb);
        fclose(fb);
    }

    ft = fopen(txt_path, "w");
    if (ft == NULL) {
        return;
    }

    if (elem_size == 4 && (len % 4) == 0) {
        const uint32_t *w = (const uint32_t *)buf;
        size_t n = len / 4;
        for (size_t i = 0; i < n; i++) {
            if (i > 0 && i % elems_per_row == 0) {
                fprintf(ft, "\n");
            }
            fprintf(ft, "%08X ", w[i]);
        }
        fprintf(ft, "\n");
    } else {
        const uint8_t *b = (const uint8_t *)buf;
        size_t row = elems_per_row > 0 ? elems_per_row : 16;
        for (size_t i = 0; i < len; i++) {
            if (i > 0 && i % row == 0) {
                fprintf(ft, "\n");
            }
            fprintf(ft, "%02X ", b[i]);
        }
        fprintf(ft, "\n");
    }

    fclose(ft);
#endif
}

// Socket helpers
static int send_all(int sockfd, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    size_t sent = 0;

    while (sent < len) {
        ssize_t n = write(sockfd, p + sent, len - sent);
        if (n <= 0) {
            return -1;
        }
        sent += (size_t)n;
    }

    return 0;
}

static int recv_all(int sockfd, void *buf, size_t len)
{
    char *p = (char *)buf;
    size_t recvd = 0;

    while (recvd < len) {
        ssize_t n = read(sockfd, p + recvd, len - recvd);
        if (n <= 0) {
            return -1;
        }
        recvd += (size_t)n;
    }

    return 0;
}

static int recv_line(int sockfd, char *out, size_t out_sz)
{
    size_t used = 0;

    if (out_sz < 2) {
        return -1;
    }

    while (1) {
        char ch;
        ssize_t n = read(sockfd, &ch, 1);

        if (n == 0) {
            // Peer closed connection.
            return 0;
        }

        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Read timeout.
                return -2;
            }
            return -1;
        }

        if (ch == '\r') {
            continue;
        }

        if (ch == '\n') {
            out[used] = '\0';
            return 1;
        }

        if (used + 1 >= out_sz) {
            return -3;
        }

        out[used++] = ch;
    }
}

static int wait_for_ack(int sockfd, const char *expected_ack, int max_lines)
{
    char line[256];

    for (int i = 0; i < max_lines; i++) {
        int rc = recv_line(sockfd, line, sizeof(line));

        if (rc <= 0) {
            if (rc == 0) {
                printf("Server closed connection while waiting for ack '%s'\n", expected_ack);
            } else if (rc == -2) {
                printf("Timeout while waiting for ack '%s'\n", expected_ack);
            } else {
                printf("Failed while waiting for ack '%s' (rc=%d)\n", expected_ack, rc);
            }
            return -1;
        }

        if (strcmp(line, expected_ack) == 0) {
            return 0;
        }

        printf("Unexpected line while waiting for '%s': '%s'\n", expected_ack, line);
    }

    printf("Did not receive ack '%s' within %d lines\n", expected_ack, max_lines);
    return -1;
}

void receive_shwfs(int sockfd)
{
    char buff[MAX];
    int n;

    // send
    bzero(buff, sizeof(buff));

    if (send_all(sockfd, "start\n", 6) != 0) {
        printf("Failed to send command\n");
        return;
    }

    if (recv_all(sockfd, coeffs, sizeof(coeffs)) != 0) {
        printf("Failed to receive coeffs\n");
        return;
    }
    printf("Coeffs received!\n");
    dump_buffer("c_recv_coeffs", coeffs, sizeof(coeffs), 1, 256);
    send_all(sockfd, "coeffs_done\n", 12);
    
    // FILE *f;
    // f = fopen("received_coeffs.txt", "w"); 

    // if (f == NULL) {
    //     printf("Failed to open output file\n");
    //     return;
    // }

    // for (int i = 0; i < 65536; i++) {
    //     if (i % 256 == 0 && i > 0)
    //         fprintf(f, "\n");
    //     fprintf(f, "%3u ", coeffs[i]);
    // }
    // fprintf(f, "\n");
    // printf("wrote output file\n");
    // fclose(f);

    printf("Done receiving\n");

    // Now received data gets written into M10K on avalon bus - or transmitted some other way

    // When FPGA notifies ARM that computation is done - we're here
    // Result data from M10K gets written into HPS memory

}

void send_shwfs(int sockfd) {
    fabricate_results();

    // Notify server that compute stage is done before sending results.
    send_all(sockfd, "compute_done\n", 13);

    printf("Sending centroid vector. \n");
    dump_buffer("c_sent_centroids", centroids, sizeof(centroids), 4, 64);
    send_all(sockfd, centroids, sizeof(centroids));

    // wait for ack from pyserver
    if (wait_for_ack(sockfd, "centroids_done", 8) != 0) {
        printf("    Missing centroid ack; continuing.\n");
    }

    
    printf("    Centroids received.");

    printf("Sending slope vector.\n");
    dump_buffer("c_sent_slopes", slopes, sizeof(slopes), 4, 64);
    send_all(sockfd, slopes, sizeof(slopes));

    // wait for ack from pyserver
    if (wait_for_ack(sockfd, "slopes_done", 8) != 0) {
        printf("    Missing slope ack; continuing.\n");
    }

    printf("    Slopes received\n");
     
    printf("Sending zernike coefficients\n");
    dump_buffer("c_sent_zernike", zernike_coeffs, sizeof(zernike_coeffs), 4, 10);
    send_all(sockfd, zernike_coeffs, sizeof(zernike_coeffs));

    // wait for ack from server
    if (wait_for_ack(sockfd, "zernike_done", 8) != 0) {
        printf("    Missing zernike ack; continuing.\n");
    }

    printf("    Zernike coeffs received. \n");
}

// entry point
int main(int argc, char **argv)
{
	// Declare volatile pointers to I/O registers (volatile 	
	// means that IO load and store instructions will be used 	
	// to access these pointer locations, 
	// instead of regular memory loads and stores)  

	// === get FPGA addresses ==================
    // Open /dev/mem
	if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) 	{
		printf( "ERROR: could not open \"/dev/mem\"...\n" );
		return( 1 );
	}

    //============================================
    // get virtual addr that maps to physical
	// for light weight bus
	// DMA status register
	h2p_lw_virtual_base = mmap( NULL, HW_REGS_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HW_REGS_BASE );	
	if( h2p_lw_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap1() failed...\n" );
		close( fd );
		return(1);
	}
	// the DMA registers
	DMA_status_ptr = (unsigned int *)(h2p_lw_virtual_base);
	DMA_read_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_READ_ADD_OFFSET);
	DMA_write_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_WRT_ADD_OFFSET);
	DMA_length_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_LENGTH_OFFSET);
	DMA_cntl_ptr = (unsigned int *)(h2p_lw_virtual_base + DMA_CNTL_OFFSET);


	//============================================

	//  RAM FPGA parameter addr 
	sram_virtual_base = mmap( NULL, FPGA_ONCHIP_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, FPGA_ONCHIP_BASE); 	

	if( sram_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap3() failed...\n" );
		close( fd );
		return(1);
	}
    // Get the address that maps to the RAM buffer
	sram_ptr = (volatile uint8_t *)(sram_virtual_base);

	// ===========================================

	// HPS onchip ram
	hps_onchip_virtual_base = mmap( NULL, HPS_ONCHIP_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HPS_ONCHIP_BASE); 	

	if( hps_onchip_virtual_base == MAP_FAILED ) {
		printf( "ERROR: mmap3() failed...\n" );
		close( fd );
		return(1);
	}
    // Get the address that maps to the HPS ram
	hps_onchip_ptr =(unsigned int *)(hps_onchip_virtual_base);


	//============================================
	int N = 65536;
	//int data[16384] ;
	int i ;

    // TCP SOCKET

    int sockfd, connfd;
    struct sockaddr_in servaddr, cli;

    if (argc > 1) {
        strcpy(filename, argv[1]);
    } else {
        strcpy(filename, "output.bin");
    }

    ensure_dump_dir();

    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("10.48.143.64");
    servaddr.sin_port = htons(PORT);

    // connect the client socket to server socket
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connected to the server..\n");

    {
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    }

    // Handle Data Transfers

    // function for chat
    receive_shwfs(sockfd);

    printf("Starting DMA transfer\n");
    // Coefficients should now be in array
    DMA_transfer_bytes(coeffs, sizeof(coeffs), DMA_status_ptr, DMA_read_ptr, DMA_write_ptr, DMA_length_ptr, DMA_cntl_ptr);
    printf("DMA Transfer complete.\n");
    // send ack to fpga? start compute
    // wait for results

    FILE *output = fopen("read_out_coeffs.hex", "w");
    if (output == NULL) {
        printf("Failed to open file output.\n");
        return 1;
    } else {
        printf("Output file opened successfully.\n");
    }
    // read results back from FPGA mem

    for (i = 0; i < 65536; i++) {
        if (fprintf(output, "0x%02X\n", sram_ptr[i]) < 0) {
            printf("Write failed at byte %d\n", i);
            break;
        }
    }

    fclose(output);

    // close the socket
    close(sockfd);
} 