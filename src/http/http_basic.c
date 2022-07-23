#include "../../include/http/http.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


HttpRequest * HttpRequest_alloc(
    int method_code,
    char * host,
    char * url,
    char * body
) {
    HttpRequest * request_object = MALLOC1(HttpRequest);
    int err = HttpRequest_init(request_object, method_code, host, url, body);
    if(err == ERROR_NO) {
        return request_object;
    }
    FREE(request_object);
    return NULL;
}

static int check_method(int method_code) {
    if(GET_HTTP_METHOD_STR(method_code) == NULL) {
        return ERROR_INIT_METHOD;
    }
    return ERROR_NO;
}

static int check_host(char * host) {
    if(host == NULL || strlen(host) == 0) {
        return ERROR_INIT_HOST;
    }
    return ERROR_NO;
}

static int check_url(char * url) {
    if(url == NULL || strlen(url) == 0) {
        return ERROR_INIT_URL;
    }
    if(url[0] != '/') {
        return ERROR_INIT_URL;
    }
    return ERROR_NO;
}

int HttpRequest_init(
    HttpRequest * request_object,
    int method_code,
    char * host,
    char * url,
    char * body
) {
    request_object->method_code = HTTP_METHOD_NULL;
    request_object->url = NULL;
    request_object->body = NULL;
    request_object->headers = NULL;
    if(check_method(method_code) != ERROR_NO) {
        return ERROR_INIT_METHOD;
    }
    if(check_host(host) != ERROR_NO) {
        return ERROR_INIT_HOST;
    }
    if(check_url(url) != ERROR_NO) {
        return ERROR_INIT_URL;
    }
    request_object->method_code = method_code;
    request_object->host = strdup(host);
    request_object->url = strdup(url);
    if(body != NULL) {
        request_object->body = strdup(body);
    }
    request_object->headers = MALLOC1(StringVector);
    StringVector_init(request_object->headers);
    char len_str[POW2(5)];
    sprintf(
        len_str, "%ld", IF_ELSE(body == NULL, 0, strlen(request_object->body)
    ));
    HttpRequest_add_header(request_object, "Host", host);
    HttpRequest_add_header(request_object, "Content-Length", len_str);
    return ERROR_NO;
}

int HttpRequest_add_header(
    HttpRequest * request_object,
    char * key,
    char * value
) {
    char * buffer = MALLOC(char, strlen(key) + strlen(value) + 2);
    sprintf(buffer, "%s:%s", key, value);
    StringVector_push(request_object->headers, buffer);
    FREE(buffer);
    return ERROR_NO;
}

static int msg_max_length(HttpRequest * request_object) {
    const int line_rl_len = strlen(HTTP_PROTOCOL_LB);
    int max_size = 0;
    max_size += strlen(GET_HTTP_METHOD_STR(request_object->method_code));
    max_size += 1;
    max_size += strlen(request_object->url);
    max_size += 1;
    max_size += strlen(HTTP_VERSION);
    max_size += line_rl_len;
    for(int i = 0; i < StringVector_size(request_object->headers); i++) {
        max_size += strlen(StringVector_get(request_object->headers, i));
        max_size += line_rl_len;
    }
    max_size += line_rl_len;
    if(request_object->method_code != HTTP_METHOD_HEAD && request_object->body != NULL) {
        max_size += strlen(request_object->body);
    }
    return max_size;
}

byte_t * HttpRequest_build(HttpRequest * request_object) {
    const char * method_str = GET_HTTP_METHOD_STR(request_object->method_code);
    char * request_msg = MALLOC(char, msg_max_length(request_object) + 1);
    int index = sprintf(request_msg, "%s %s %s%s", method_str, request_object->url, HTTP_VERSION, HTTP_PROTOCOL_LB);
    for(int i = 0; i < StringVector_size(request_object->headers); i++) {
        const char * s = StringVector_get(request_object->headers, i);
        index += sprintf(request_msg + index, "%s%s", s, HTTP_PROTOCOL_LB);
    }
    index += sprintf(request_msg + index, "%s", HTTP_PROTOCOL_LB);
    request_msg[index] = '\0';
    return (byte_t *) request_msg;
}

int HttpRequest_destroy(HttpRequest * request_object) {
    FREE(request_object->url);
    FREE(request_object->host);
    if(request_object->body != NULL) {
        FREE(request_object->body);
    }
    StringVector_destroy(request_object->headers);
    FREE(request_object->headers);
    return ERROR_NO;
}
