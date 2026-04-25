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

void fabricate_results() {
    for (uint32_t i = 0; i < 512; i++) {
        centroids[i] = i;
        slopes[i] = i;
    }

    for (uint32_t i = 0; i < 10; i++) {
        zernike_coeffs[i] = i;
    }
}

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

// static int send_line(int sockfd, const void *buf) {
//     const char *p = (const char *) buf;
    
//     while (p[0] != "\n") {
//         write(sockfd, p, 1);
//         p++;
//     }

//     if (p >= buf + MAX) return -1;
//     return 0;
// }

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

void func(int sockfd)
{
    char buff[MAX];
    int n;

    // send
    bzero(buff, sizeof(buff));
    printf("Enter the string : ");
    n = 0;
    while ((buff[n++] = getchar()) != '\n');
    if (send_all(sockfd, buff, (size_t)n) != 0) {
        printf("Failed to send command\n");
        return;
    }

    // Assuming "start" was sent - receive e_matrix then coeffs
    if (recv_all(sockfd, e_matrix, sizeof(e_matrix)) != 0) {
        printf("Failed to receive coeffs\n");
        return;
    }
    printf("Matrix received!\n");
    dump_buffer("c_recv_e_matrix", e_matrix, sizeof(e_matrix), 4, 336);

    send_all(sockfd, "matrix_done\n", 12);

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

int main(int argc, char **argv)
{
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
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
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

    // function for chat
    func(sockfd);

    // close the socket
    close(sockfd);
} 