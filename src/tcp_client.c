#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> // bzero()
#include <sys/socket.h>
#include <unistd.h> // read(), write(), close()
#define MAX 4096
#define PORT 1234
#define SA struct sockaddr

char filename[32];

void func(int sockfd)
{
    char buff[MAX];
    int n;

    // send
    bzero(buff, sizeof(buff));
    printf("Enter the string : ");
    n = 0;
    while ((buff[n++] = getchar()) != '\n')
        ;
    write(sockfd, buff, sizeof(buff));

    FILE *f;
    f = fopen(filename, "wb"); 

    for (;;) {

        // receive
        bzero(buff, sizeof(buff));
        n = read(sockfd, buff, sizeof(buff));

        if ((strncmp(buff, "exit", 4)) == 0) {
            printf("Client Exit...\n");
            break;
        }

        // printf("fwrite: %d bytes\n", n);
        fwrite(buff, 1, n, f);

    }

    fflush(f);
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

    // function for chat
    func(sockfd);

    // close the socket
    close(sockfd);
} 