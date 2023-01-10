"""
cybart: Python bindings for BART (https://github.com/mrirecon/bart)
        written in Cython

Created on Mon Jan  9 18:10:39 2023

@author: Philipp Ehses (philipp.ehses@dzne.de)
"""

# cimport the Cython declarations for numpy
from libcpp cimport bool
cimport numpy as c_np
from libc.stdlib cimport malloc, free
import sys
import numpy as np

c_np.import_array()

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

def random_name(N=12):
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=N))


def call_bart(bart_cmd, input_data, outfiles):

    # define how memory output arrays should be managed
    # True: numpy manages data; False: bart deletes data after copy to numpy object
    # False will "indirectly" leak a very small amount of memory (memcfl_list properties like name&dims?)
    let_numpy_manage_arrays = False

    nargs = len(bart_cmd)
    cdef c_np.npy_intp nd = 16
    cdef c_np.npy_intp size
    cdef long c_dims[16]
    cdef float complex* c_array_in[8]
    cdef float complex* c_array_out
    cdef char *argv[256]
    cdef char stdout[4096]
    cdef char **string_buf = <char **>malloc(nargs * sizeof(char*))
    string_buf[0] = '\0'
    stdout[0] = '\0'

    for key, item in enumerate(bart_cmd):
        string_buf[key] = PyUnicode_AsUTF8(item)

    for k, (name, data) in enumerate(input_data.items()):
        # create memcfl for input
        dims = np.ones((16), dtype=np.int64)
        dims[:data.ndim] = data.shape
        for key, item in enumerate(dims):
            c_dims[key] = item
        if data.dtype != np.complex64:
            data = data.astype(np.complex64)
        data = np.asfortranarray(data)
        c_array_in[k] = <float complex*> c_np.PyArray_DATA(data)
        memcfl_register(name, 16, c_dims, c_array_in[k], False)

    errcode = bart_command(4096, stdout, nargs, string_buf)
    # errcode = bart_command(0, NULL, nargs, string_buf)
    print('errcode: ', errcode)
    free(string_buf)

    # clean up the input data
    for k, name in enumerate(input_data):
        memcfl_unmap(c_array_in[k])
        memcfl_unlink(name)

    output = list()
    for name in outfiles:
        io_reserve_input(name)
        c_array_out = memcfl_load(name, 16, c_dims)

        dims = c_np.PyArray_SimpleNewFromData(1, &nd,
                    c_np.NPY_LONG, <void *> c_dims)
        c_np.PyArray_UpdateFlags(dims, dims.flags.num | c_np.NPY_OWNDATA)

        # determine number of dimensions
        for k in range(nd-1, -1, -1):
            if dims[k] > 1:
                break

        # determine number of elements
        size = dims.prod()

        ndarray = c_np.PyArray_SimpleNewFromData(1, &size,
                    c_np.NPY_COMPLEX64, <void *> c_array_out)
        
        if let_numpy_manage_arrays:
            c_np.PyArray_UpdateFlags(ndarray, ndarray.flags.num | c_np.NPY_OWNDATA)
            output.append(ndarray.reshape(dims[:k+1], order='F'))
        else:
            output.append(ndarray.reshape(dims[:k+1], order='F').copy())
            memcfl_unmap(c_array_out)
            memcfl_unlink(name)  # this will destroy the output data

    io_memory_cleanup()

    return output, errcode, stdout.decode('utf-8')


def bart(nargout, cmd, *args, **kwargs):

    input_data = {}
    bart_cmd = ['bart']  # empty cmd string will output list of available commands

    cmd = cmd.strip()
    if len(cmd)>0:
        bart_cmd = cmd.split(' ')  # not really necessary, just for consistency

    for key, item in kwargs.items():
        kw = ("--" if len(key)>1 else "-") + key
        name = random_name() + '.mem'
        input_data[name.encode('utf-8')] = item
        bart_cmd.append(kw)
        bart_cmd.append(name)

    for item in args:
        name = random_name() + '.mem'
        input_data[name.encode('utf-8')] = item
        bart_cmd.append(name)
    
    outfiles = []
    for _ in range(nargout):
        # create memcfl names for output
        name = random_name() + '.mem'
        outfiles.append(name.encode('utf-8'))
        bart_cmd.append(name)

    output, errcode, stdout = call_bart(bart_cmd, input_data, outfiles)

    if len(stdout)>0:
        print(stdout)

    if errcode:
        print(f"Command exited with error code {errcode}.")
        return

    # print('infiles:')
    # for name in input_data:
    #     print(name.decode('utf-8'), memcfl_exists(name))
    # print('outfiles:')
    # for name in outfiles:
    #     print(name.decode('utf-8'), memcfl_exists(name))
    
    if nargout == 0:
        return
    elif nargout == 1:
        return output[0]
    else:
        return output
