#include <Python.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <arpa/inet.h> // For ntohs and company

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
    // takes f, data_offest, num_times, num_wavelengths
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

uint16_t read_big_short(void *buff, ssize_t *pos) {
    uint16_t val = ntohs(*(uint16_t *)(buff + *pos));
    *pos += 2;
    return val;
}

uint32_t read_big_int(void *buff, ssize_t *pos) {
    uint32_t val = ntohl(*(uint32_t *)(buff + *pos));
    *pos += 4;
    return val;
}

uint16_t bisect(uint16_t *array, uint16_t size, uint16_t value) {
    uint16_t low = 0;
    uint16_t high = size;
    while (low < high) {
        uint16_t mid = (low + high) / 2;
        if (array[mid] < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

uint64_t decode_intensity(uint16_t intensity) {
    uint32_t mantissa = intensity & 0x3FFF;
    uint32_t exponent = intensity >> 14;
    return ((uint64_t)1 << (3 * exponent)) * mantissa;
}

int compare_uint16_t(const void* a, const void* b) {
    uint16_t arg1 = *(const uint16_t*) a;
    uint16_t arg2 = *(const uint16_t*) b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

PyObject *py_decode_ms(PyObject *self, PyObject *args, PyObject *kwargs) {
    // takes f, data_offest
    uint32_t data_offset, num_times;
    PyObject *f;
    static char *kwlist[] = {"file", "data_offset", "num_times", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OII", kwlist, &f, &data_offset, &num_times)) {
        PyErr_SetString(PyExc_TypeError, "Invalid arguments");
        return NULL;
    }
    PyObject *f_fileno = PyObject_CallNoArgs(PyObject_GetAttrString(f, "fileno"));
    int fd = PyLong_AsLong(f_fileno);
    printf("fd: %d, data_offset: %d, num_times: %d\n", fd, data_offset, num_times);
    struct stat st;
    fstat(fd, &st);
    off_t file_size = st.st_size;
    // mmap the file
    void *file = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ssize_t pos = data_offset;
    ssize_t start_pos = read_big_short(file, &pos) * 2 - 2;
    pos = start_pos;

    uint32_t times[num_times];
    uint16_t pair_counts[num_times];
    ssize_t pair_locs[num_times];

    uint32_t total_pair_count = 0;
    // Read just the times and pair counts
    for(uint32_t i=0; i<num_times; i++) {
        pos += 2;
        times[i] = read_big_int(file, &pos);
        pos += 6;
        uint16_t pair_count = read_big_short(file, &pos);
        pair_locs[i] = pos;
        // We'll read the data at this pos later
        pair_counts[i] = pair_count;
        total_pair_count += pair_count;
        pos += 4 + pair_count * 4 + 10;
    }

    uint16_t mzs[total_pair_count];
    uint64_t intensities[total_pair_count];

    for(uint32_t i=0; i<num_times; i++) {
        pos = pair_locs[i];
        uint16_t pair_count = pair_counts[i];
        for(int j=0; j<pair_count; j++) {
            uint16_t index = i * pair_count + j;
            mzs[index] = read_big_short(file, &pos);
            intensities[index] = decode_intensity(read_big_short(file, &pos));
        }
    }
    uint16_t sorted_mzs[total_pair_count];
    memcpy(sorted_mzs, mzs, total_pair_count * sizeof(uint16_t));
    qsort(sorted_mzs, total_pair_count, sizeof(uint16_t), compare_uint16_t);
    uint16_t unique_mzs[total_pair_count], unique_mzs_count = 0;

    unique_mzs[0] = sorted_mzs[0];
    for (uint32_t i=1; i<total_pair_count; i++) {
        if (sorted_mzs[i] != sorted_mzs[i-1]) {
            unique_mzs[++unique_mzs_count] = sorted_mzs[i];
        }
    }

    uint64_t data[num_times * unique_mzs_count];
    for (uint32_t i=0; i<num_times * unique_mzs_count; i++) {
        data[i] = 0;
    }
    uint16_t curr_index = 0;
    for(uint32_t i=0; i<num_times; i++) {
        uint16_t stop_index = curr_index + pair_counts[i];
        for(int j=curr_index; j<stop_index; j++) {
            int mz_index = bisect(unique_mzs, unique_mzs_count, mzs[j]);
            data[i * unique_mzs_count + mz_index] += intensities[j];
        }
    }
    npy_intp dims[2] = {num_times, unique_mzs_count};
    PyObject *data_array = PyArray_SimpleNewFromData(2, dims, NPY_INT64, (void *)data);
    PyObject *times_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT32, (void *)times);
    dims[1] = unique_mzs_count;
    PyObject *mzs_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT16, (void *)unique_mzs);

    // Return a tuple of times, ylabels, and data
    return PyTuple_Pack(3, times_array, mzs_array, data_array);
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
    {"decode_ms", py_decode_ms, METH_KEYWORDS, "Decode MS"},
    {"create_ndarray", py_create_ndarray, METH_VARARGS, "Create ndarray"},
    {NULL, NULL, 0, NULL},
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
