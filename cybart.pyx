"""
cybart: Python bindings for BART (https://github.com/mrirecon/bart)
        written in Cython

Created on Mon Jan  9 18:10:39 2023

@author: Philipp Ehses (philipp.ehses@dzne.de)
"""

# cimport the Cython declarations for numpy
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import numpy as np
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
    cdef unsigned int DIMS = 16
    cdef np.npy_intp nd = DIMS
    cdef np.npy_intp size
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
        # initialize c_dims
        for d in range(DIMS):
            if d < data.ndim:
                c_dims[d] = data.shape[d]
            else:
                c_dims[d] = 1
        c_array_in[k] = <float complex*> np.PyArray_DATA(data)
        memcfl_register(name, DIMS, c_dims, c_array_in[k], False)

    errcode = bart_command(4096, stdout, nargs, string_buf)
    free(string_buf)

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
        for k in range(DIMS):
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


def bart(nargout, cmd, *args, **kwargs):

    input_data = {}
    bart_cmd = ['bart']  # empty cmd string will output list of available commands

    cmd = cmd.strip()
    if len(cmd)>0:
        bart_cmd = cmd.split(' ')

    for key, item in (*kwargs.items(), *zip([None]*len(args), args)):
        if key is not None:
            kw = ("--" if len(key)>1 else "-") + key
            bart_cmd.append(kw)
        name = random_name() + '.mem'
        bart_cmd.append(name)
        if item.dtype != np.complex64:
            item = item.astype(np.complex64)
        item = np.asfortranarray(item)
        input_data[name.encode('utf-8')] = item
    
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

    if nargout == 0:
        return
    elif nargout == 1:
        return output[0]
    else:
        return output
