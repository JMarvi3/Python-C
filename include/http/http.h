#include "http_errors.h"
#include "../util/vector.h"
#include "../util/macros.h"
#include <stddef.h>


#define HTTP_PORT           80
#define HTTPS_PORT          443

#define HTTP_METHOD_NULL    1000
#define HTTP_METHOD_GET     1001
#define HTTP_METHOD_POST    1002
#define HTTP_METHOD_HEAD    1003
#define HTTP_METHOD_PUT     1004
#define HTTP_METHOD_DELETE  1005

#define HTTP_VERSION        "HTTP/1.1"

#define HTTP_DEFAULT_METHOD HTTP_METHOD_GET
#define HTTP_DEFAULT_HOST   "localhost"
#define HTTP_DEFAULT_URL    "/"

#define HTTP_PROTOCOL_LB    "\r\n"

#define HTTP_RESPONSE_BLOCK POW2(10)    // 1KB
#define HTTP_RESPONSE_SIZE  POW2(14)    // 16KB

#define GET_HTTP_METHOD_STR(code) \
    IF_ELSE(code == HTTP_METHOD_GET, "GET", \
        IF_ELSE(code == HTTP_METHOD_POST, "POST", \
            IF_ELSE(code == HTTP_METHOD_HEAD, "HEAD", \
                IF_ELSE(code == HTTP_METHOD_PUT, "PUT", \
                    IF_ELSE(code == HTTP_METHOD_DELETE, "DELETE", NULL \
    )))))


typedef unsigned char byte_t;


typedef struct {
    int method_code;
    char * url;
    char * host;
    char * body;
    StringVector * headers;
} HttpRequest;


HttpRequest * HttpRequest_alloc(int method_code, char * host, char * url, char * body);

int HttpRequest_init(HttpRequest *, int method_code, char * host, char * url, char * body);

int HttpRequest_add_header(HttpRequest *, char * key, char * value);

byte_t * HttpRequest_build(HttpRequest *);

int HttpRequest_send(HttpRequest *, byte_t * response_buffer);

int HttpRequest_destroy(HttpRequest *);
