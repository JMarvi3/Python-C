#include <Python.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyObject *py_test(PyObject *self, PyObject *args) {
    printf("Hello from C!\n");
    return PyUnicode_FromString("Hello. I'm a C function.");
}
// def decode_uv_delta(f, data_offsets, num_times, num_wavelengths):
//     uint_unpack = struct.Struct('<I').unpack
//     int_unpack = struct.Struct('<i').unpack
//     short_unpack = struct.Struct('<h').unpack

//     f.seek(data_offsets["data_start"])
//     times = np.empty(num_times, dtype=np.uint32)
//     data = np.empty((num_times, num_wavelengths), dtype=np.int64)
//     for i in range(num_times):
//         f.read(4)
//         times[i] = uint_unpack(f.read(4))[0]
//         f.read(14)
//         # If the next short is equal to -0x8000
//         #     then the next absorbance value is the next integer.
//         # Otherwise, the short is a delta from the last absorbance value.
//         absorb_accum = 0
//         for j in range(num_wavelengths):
//             check_int = short_unpack(f.read(2))[0]
//             if check_int == -0x8000:
//                 absorb_accum = int_unpack(f.read(4))[0]
//             else:
//                 absorb_accum += check_int
//             data[i, j] = absorb_accum

//     return times, data

PyObject *py_decode_uv_delta(PyObject *self, PyObject *args) {
    // takes f, data_offests, num_times, num_wavelengths
    uint32_t data_offset, num_times, num_wavelengths;
    PyObject *f;
    if (!PyArg_ParseTuple(args, "OIII", &f, &data_offset, &num_times, &num_wavelengths)) {
        return NULL;
    }

    PyObject *f_fileno = PyObject_CallNoArgs(PyObject_GetAttrString(f, "fileno"));
    int fd = PyLong_AsLong(f_fileno);

    // Get the file size with stat
    struct stat st;
    fstat(fd, &st);
    off_t file_size = st.st_size;
    // mmap the file
    void *file = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ssize_t pos = data_offset;
    
    npy_intp dims[2] = {num_times, num_wavelengths};
    PyObject *times = PyArray_SimpleNew(1, dims, NPY_UINT32);

    npy_int64 *data_array = malloc(num_times * num_wavelengths * sizeof(npy_int64));

    for (uint32_t i=0; i<num_times; i++) {
        pos += 4;
        *((uint32_t *)PyArray_GETPTR1((PyArrayObject *)times, i)) = *(uint32_t *)(file+pos);
        pos += 4 + 14;

        int64_t absorb_accum = 0;
        for (uint32_t j=0; j<num_wavelengths; j++) {
            int16_t check_int;
            check_int = *((int16_t *)(file+pos));
            pos += 2;
            if (check_int == -0x8000) {
                absorb_accum = *((int32_t *)(file + pos));
                pos += 4;
            } else {
                absorb_accum += check_int;
            }
            data_array[i * num_wavelengths + j] = absorb_accum;
        }
    }
    munmap(file, file_size);
    PyObject *data = PyArray_SimpleNewFromData(2, dims, NPY_INT64, (void *)data_array);
    
    free(data_array);

    return PyTuple_Pack(2, times, data);
}

PyObject *py_create_ndarray(PyObject *self, PyObject *args) {
    int nd = 2;
    int dim1, dim2;
    PyObject *f;
    if (!PyArg_ParseTuple(args, "(ii)O", &dim1, &dim2, &f)) {
        return NULL;
    }
    PyObject *f_fileno = PyObject_CallNoArgs(PyObject_GetAttrString(f, "fileno"));
    int fd = PyLong_AsLong(f_fileno);
    // Get current position
    off_t pos = lseek(fd, 0, SEEK_CUR);
    printf("fileno: %d\n", fd);
    char buffer[1024];
    int n = read(fd, buffer, 1000);
    printf("read %d bytes\n", n);
    buffer[n] = 0;
    printf("buffer: %s\n", buffer);
    // Restore the position
    lseek(fd, pos, SEEK_SET);
    npy_intp dims[2] = {dim1, dim2};
    PyObject *arr = PyArray_SimpleNew(nd, dims, NPY_INT);
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            *((npy_int *)PyArray_GETPTR2((PyArrayObject *)arr, i, j)) = i * 100 + j;
        }
    }
    return arr;
}

static PyMethodDef myexample_methods[] = {
    {"test", py_test, METH_NOARGS, "Test"},
    {"decode_uv_delta", py_decode_uv_delta, METH_VARARGS, "Decode UV Delta"},
    {"create_ndarray", py_create_ndarray, METH_VARARGS, "Create ndarray"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myexample_module = {
    PyModuleDef_HEAD_INIT,
    "MyExample",                              
    "Very simple C-API example",  
    -1,                                   
    myexample_methods
};

PyMODINIT_FUNC PyInit_myexample() {
    import_array();
    return PyModule_Create(&myexample_module);
};
