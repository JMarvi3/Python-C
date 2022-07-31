#include "../../include/http/http.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>


int HttpRequest_send_port(
    HttpRequest * request_object,
    byte_t * response_buffer,
    int port
) {
    int sock;
    int done_bytes, request_msg_len, request_msg_len_sent, response_msg_len_got;
    byte_t * request_msg;
    struct hostent * server;
    struct sockaddr_in serv_addr;

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if(sock < 0) {
        return ERROR_SEND_SOCKET_CREATE;
    }

    server = gethostbyname(request_object->host);
    if(server == NULL) {
        return ERROR_SEND_NO_HOST;
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);

    if(connect(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return ERROR_SEND_SOCKET_CONNECT;
    }

    request_msg = HttpRequest_build(request_object);
    request_msg_len = strlen((const char *) request_msg);
    request_msg_len_sent = 0;
    do {
        done_bytes = write(sock, request_msg + request_msg_len_sent, request_msg_len - request_msg_len_sent);
        if (done_bytes < 0) {
            FREE(request_msg);
            return ERROR_SEND_WRITE_REQUEST;
        } else if (done_bytes == 0) {
            break;
        }
        request_msg_len_sent += done_bytes;
    } while (request_msg_len_sent < request_msg_len);
    FREE(request_msg);

    response_msg_len_got = 0;
    while(true) {
        done_bytes = read(sock, response_buffer + response_msg_len_got, HTTP_RESPONSE_BLOCK);
        if (done_bytes < 0) {
            return ERROR_SEND_READ_RESPONSE;
        } else if (done_bytes == 0) {
            break;
        }
        response_msg_len_got += done_bytes;
    }
    response_buffer[response_msg_len_got] = '\0';

    return ERROR_NO;
}

int HttpRequest_send(
    HttpRequest * request_object,
    byte_t * response_buffer
) {
    return HttpRequest_send_port(request_object, response_buffer, HTTP_PORT);
}
