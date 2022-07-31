#include <Python.h>
#include "include/http/http.h"
#include <stdbool.h>
#include <string.h>


#define NO_CONF             0
#define EXCLUDE_BODY_CONF   1

#define LOCALHOST           "localhost"

#define ENCODING_UTF8       "UTF-8"

#define IF_THEN_NULL(cond) \
    if(cond) return NULL;

#define IF_NULL_THEN_NULL(ptr) \
    IF_THEN_NULL(ptr == NULL)


static PyObject * py_http_request(PyObject *, PyObject *, int, int);


static PyObject * py_get_request(PyObject *self, PyObject *args) {
    return py_http_request(self, args, HTTP_METHOD_GET, NO_CONF);
}

static PyObject * py_post_request(PyObject *self, PyObject *args) {
    return py_http_request(self, args, HTTP_METHOD_POST, NO_CONF);
}

static PyObject * py_head_request(PyObject *self, PyObject *args) {
    return py_http_request(self, args, HTTP_METHOD_HEAD, EXCLUDE_BODY_CONF);
}

/*
 * < Private functions >
 */

static bool check_error_init(int err_code) {
    switch(err_code) {
        case ERROR_INIT_METHOD:
            PyErr_SetString(PyExc_ValueError, "Invalid Request Method");
            break;
        case ERROR_INIT_HOST:
            PyErr_SetString(PyExc_ValueError, "Invalid Request Host");
            break;
        case ERROR_INIT_URL:
            PyErr_SetString(PyExc_ValueError, "Invalid Request URL");
            break;
    }
    return err_code != ERROR_NO;
}

static bool check_error_send(int err_code) {
    switch(err_code) {
        case ERROR_SEND_SOCKET_CREATE:
            PyErr_SetString(PyExc_ConnectionError, "Socket creation failed");
            break;
        case ERROR_SEND_NO_HOST:
            PyErr_SetString(PyExc_ConnectionError, "Host not found");
            break;
        case ERROR_SEND_SOCKET_CONNECT:
            PyErr_SetString(PyExc_ConnectionError, "Socket connection failed");
            break;
        case ERROR_SEND_WRITE_REQUEST:
            PyErr_SetString(PyExc_ConnectionError, "Can't send bytes");
            break;
        case ERROR_SEND_READ_RESPONSE:
            PyErr_SetString(PyExc_ConnectionError, "Can't receive bytes");
            break;
    }
    return err_code != ERROR_NO;
}

static bool is_localhost(char * value, int * port) {
    const char * localhost_str = LOCALHOST;
    int index = 0;
    for(; localhost_str[index] != '\0'; index++) {
        if(localhost_str[index] != value[index]) {
            return false;
        }
    }
    if(value[index] == ':' && isdigit(value[index + 1])) {
        *port = atoi(value + index + 1);
    } else {
        *port = -1;
    }
    return true;
}

/*
 * </ Private functions >
 */

static PyObject * py_http_request(
    PyObject *self,
    PyObject *args,
    int method_code,
    int conf
) {
    char * host, * url, * body = NULL;
    PyObject * dict = NULL;

    int parsed_result = IF_ELSE(
        conf == EXCLUDE_BODY_CONF,
        PyArg_ParseTuple(args, "ss|O", &host, &url, &dict),
        PyArg_ParseTuple(args, "ssz|O", &host, &url, &body, &dict)
    );

    IF_THEN_NULL(!parsed_result);

    int port = -1;
    bool is_local = is_localhost(host, &port);

    if(port < 0) {
        port = HTTP_PORT;
    }
    if(is_local) {
        host = LOCALHOST;
    }

    HttpRequest * request = MALLOC1(HttpRequest);
    int init_code = HttpRequest_init(
        request, method_code, host, url, body
    );
    if(check_error_init(init_code)) {
        return NULL;
    }

    if(dict != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while(PyDict_Next(dict, &pos, &key, &value)) {
            PyObject * encoded_key = PyUnicode_AsEncodedString(
                key, ENCODING_UTF8, "strict"
            );
            IF_NULL_THEN_NULL(encoded_key);
            PyObject * value_str = IF_ELSE(
                PyUnicode_Check(value), value, PyObject_Str(value)
            );
            IF_NULL_THEN_NULL(value_str);
            PyObject * encoded_value = PyUnicode_AsEncodedString(
                value_str, ENCODING_UTF8, "strict"
            );
            IF_NULL_THEN_NULL(encoded_value);
            HttpRequest_add_header(
                request,
                PyBytes_AsString(encoded_key),
                PyBytes_AsString(encoded_value)
            );
            printf("%s: %s\n", PyBytes_AsString(encoded_key), PyBytes_AsString(encoded_value));
            Py_DECREF(encoded_key);
            Py_DECREF(encoded_value);
        }
    }

    byte_t * response = MALLOC_EMPTY(byte_t, 2 * HTTP_RESPONSE_SIZE);
    int send_code = IF_ELSE(
        is_local,
        HttpRequest_send_port(request, response, port),
        HttpRequest_send(request, response)
    );
    HttpRequest_destroy(request);

    if(check_error_send(send_code)) {
        FREE(response);
        return NULL;
    }

    PyObject * response_py_str = PyUnicode_FromString((const char *) response);
    FREE(response);
    FREE(request);
    return response_py_str;
}


static PyMethodDef myhttp_methods[] = {
    {"get", py_get_request, METH_VARARGS, "Get Request"},
    {"post", py_post_request, METH_VARARGS, "Post Request"},
    {"head", py_head_request, METH_VARARGS, "Head Request"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myhttp_module = {
    PyModuleDef_HEAD_INIT,
    "MyHttp",                              
    "Very simple Http client",  
    -1,                                   
    myhttp_methods                          
};

PyMODINIT_FUNC PyInit_myhttp() {
    return PyModule_Create(&myhttp_module);
};
