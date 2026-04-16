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
#define PORT 1234
#define SA struct sockaddr

char filename[32];

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

    FILE *f;
    f = fopen(filename, "wb"); 

    if (f == NULL) {
        printf("Failed to open output file\n");
        return;
    }

    uint32_t net_size = 0;
    if (recv_all(sockfd, &net_size, sizeof(net_size)) != 0) {
        printf("Failed to receive file size\n");
        fclose(f);
        return;
    }

    uint32_t remaining = ntohl(net_size);
    while (remaining > 0) {
        size_t chunk = (remaining > MAX) ? MAX : remaining;
        n = read(sockfd, buff, chunk);
        if (n <= 0) {
            printf("Connection closed while receiving file\n");
            fclose(f);
            return;
        }

        fwrite(buff, 1, (size_t)n, f);
        remaining -= (uint32_t)n;
    }

    fclose(f);
    printf("Done receiving\n");

    // send image back

    printf("Sending back\n");
    if (send_all(sockfd, "returning\n", 10) != 0) {
        printf("Failed to send returning command\n");
        return;
    }
    printf("Sent write\n");

    f = fopen(filename, "rb");

    if (f == NULL) {
        printf("Failed to open input file\n");
        return;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        printf("Failed to seek input file\n");
        fclose(f);
        return;
    }
    long file_size = ftell(f);
    if (file_size < 0 || file_size > 0xFFFFFFFFL) {
        printf("Invalid input file size\n");
        fclose(f);
        return;
    }
    rewind(f);

    net_size = htonl((uint32_t)file_size);
    if (send_all(sockfd, &net_size, sizeof(net_size)) != 0) {
        printf("Failed to send file size\n");
        fclose(f);
        return;
    }

    int bytes_read;

    do {
        bytes_read = fread(buff, 1, sizeof(buff), f); // read one buffer worth of bytes

        if (bytes_read > 0) {
            if (send_all(sockfd, buff, (size_t)bytes_read) != 0) {
                printf("Failed while sending file\n");
                fclose(f);
                return;
            }
        }

    } while(bytes_read > 0);

    printf("Done sending\n");
    fclose(f);
    
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
    servaddr.sin_addr.s_addr = inet_addr("10.48.157.28");
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