"""
pybart: Python bindings for BART (https://github.com/mrirecon/bart)
        written in Cython

Created on Mon Jan  9 18:10:39 2023

@author: Philipp Ehses (philipp.ehses@dzne.de)
"""

# cimport the Cython declarations for numpy
from libcpp cimport bool
cimport numpy as np
np.import_array()

cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8(object unicode)

cdef extern from "misc/io.h":
    void io_reserve_input(const char* name)
    void io_memory_cleanup()

cdef extern from "misc/memcfl.h":
    float complex* memcfl_load(const char* name, int D, long dims[])
    bool memcfl_unmap(const float complex* p)
    void memcfl_unlink(const char* name)
    void memcfl_register(const char* name, int D, const long dims[], float complex* data, bool managed)
    bool memcfl_exists(const char* name)

cdef extern from "bart_embed_api.h":
    int bart_command(int len_, char* out, int argc, char* argv[])


def call_bart(bart_cmd, input_data, outfiles):

    # define how memory output arrays should be managed
    # True: numpy manages data; False: bart deletes data after copy to numpy object
    # False will "indirectly" leak a very small amount of memory (memcfl_list properties like name&dims?)
    let_numpy_manage_arrays = False

    nargs = len(bart_cmd)
    cdef unsigned int DIMS = 16
    cdef np.npy_intp nd = DIMS
    cdef np.npy_intp size
    cdef long c_dims[16]
    cdef float complex* c_array_in[8]
    cdef float complex* c_array_out
    cdef char stdout[4096]
    cdef char *argv[256]
    for i in xrange(nargs):
        argv[i] = PyUnicode_AsUTF8(bart_cmd[i])

    for k, (name, data) in enumerate(input_data.items()):
        # create memcfl for input
        # initialize c_dims
        for d in xrange(DIMS):
            if d < data.ndim:
                c_dims[d] = data.shape[d]
            else:
                c_dims[d] = 1
        c_array_in[k] = <float complex*> np.PyArray_DATA(data)
        memcfl_register(name, DIMS, c_dims, c_array_in[k], False)

    errcode = bart_command(4096, stdout, nargs, argv)

    # clean up the input data
    for k, name in enumerate(input_data):
        memcfl_unmap(c_array_in[k])
        memcfl_unlink(name)

    output = list()
    for name in outfiles:
        io_reserve_input(name)
        c_array_out = memcfl_load(name, DIMS, c_dims)

        # determine number of elements and dimensions
        size = DIMS
        dims = np.PyArray_ZEROS(1, &size, np.NPY_LONG, 0)
        size = 1
        ndim = 1
        for k in xrange(DIMS):
            dims[k] = c_dims[k]
            size *= dims[k]
            if dims[k] > 1:
                ndim = k+1

        ndarray = np.PyArray_SimpleNewFromData(1, &size,
                    np.NPY_COMPLEX64, <void *> c_array_out)
        
        if let_numpy_manage_arrays:
            np.PyArray_UpdateFlags(ndarray, ndarray.flags.num | np.NPY_OWNDATA)
            output.append(ndarray.reshape(dims[:ndim], order='F'))
        else:
            output.append(ndarray.reshape(dims[:ndim], order='F').copy())
            memcfl_unmap(c_array_out)
            memcfl_unlink(name)  # this will destroy the output data

    io_memory_cleanup()

    # print('infiles:')
    # for name in input_data:
    #     print(name.decode('utf-8'), memcfl_exists(name))
    # print('outfiles:')
    # for name in outfiles:
    #     print(name.decode('utf-8'), memcfl_exists(name))

    return output, errcode, stdout.decode('utf-8')
