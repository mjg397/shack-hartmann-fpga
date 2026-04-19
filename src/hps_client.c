#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h> // bzero()
#include <sys/socket.h>
#include <unistd.h> // read(), write(), close()

#define MAX 4096
#define PORT 80
#define SA struct sockaddr

char filename[32];

// C Arrays to hold input data
uint8_t coeffs[65536]; // 256x256(x8) coefficient array
uint32_t e_matrix[3360]; // 10 zernike modes x 336 slopes

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

    FILE *fm = fopen("received_e_matrix.txt", "w");
    if (fm == NULL) {
        printf("Failed to open e_matrix output file\n");
        return;
    }
    for (int i = 0; i < 3360; i++) {
        if (i > 0 && i % 336 == 0)
            fprintf(fm, "\n");
        fprintf(fm, "%08X ", e_matrix[i]);
    }
    fprintf(fm, "\n");
    fclose(fm);
    printf("wrote received_e_matrix.txt\n");

    send_all(sockfd, "matrix_done\n", 12);

    if (recv_all(sockfd, coeffs, sizeof(coeffs)) != 0) {
        printf("Failed to receive coeffs\n");
        return;
    }
    printf("Coeffs received!\n");
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
    servaddr.sin_addr.s_addr = inet_addr("192.168.86.45");
    servaddr.sin_port = htons(PORT);

    // connect the client socket to server socket
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connected to the server..\n");

    // function for chat
    func(sockfd);

    // close the socket
    close(sockfd);
} 