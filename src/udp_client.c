// udp client driver program
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>

#define PORT 31337
#define MAXLINE 1000

// Driver code
int main()
{   
    char buffer[2048];
    char *message = "Hello Server";
    int sockfd, n;
    struct sockaddr_in servaddr;
    
    // clear servaddr
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(PORT);
    servaddr.sin_family = AF_INET;
    
    // create datagram socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
    {
        perror("socket");
        exit(1);
    }

    struct timeval timeout;
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0)
    {
        perror("setsockopt");
        close(sockfd);
        exit(1);
    }
    
    // connect to server
    if(connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        printf("\n Error : Connect Failed \n");
        exit(0);
    }

    // request to send datagram
    // no need to specify server address in sendto
    // connect stores the peers IP and port
    sendto(sockfd, message, strlen(message), 0, (struct sockaddr*)NULL, sizeof(servaddr));

    FILE *img_rcvd;
    img_rcvd = fopen("received.png", "wb");
    if (img_rcvd == NULL) {
        perror("fopen");
        close(sockfd);
        exit(1);
    }
    
    while(1) {
        // waiting for response
        n = recvfrom(sockfd, buffer, sizeof(buffer), 0, NULL, NULL);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                printf("Receive timeout, stopping.\n");
                break;
            }
            perror("recvfrom");
            break;
        }

        if (n == 3 && memcmp(buffer, "EOF", 3) == 0) {
            printf("Saw EOF\n");
            break;
        }

        if (n >= 6 && memcmp(buffer, "DATA: ", 6) == 0) {
            fwrite(buffer + 6, 1, n - 6, img_rcvd);
        }

    }
    // close the descriptor

    printf("Done\n");
    fflush(img_rcvd);
    fclose(img_rcvd);
    close(sockfd);
}