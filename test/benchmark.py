import timeit

import_module="""
import pybart
import _pybart
import bart
"""

testcode="""
def bench(pkg):
    sig = pkg.bart(1, 'phantom -x1024 -s4')
    sig = pkg.bart(1, 'fft 7', sig)
    sig = pkg.bart(1, 'fft -i 7', sig)
    return sig
"""

print("bart: ", timeit.repeat(stmt=testcode + "bench(bart)",  setup=import_module, repeat=5, number=1))
print("pybart: ", timeit.repeat(stmt=testcode + "bench(pybart)",  setup=import_module, repeat=5, number=1))
print("_pybart: ", timeit.repeat(stmt=testcode + "bench(_pybart)",  setup=import_module, repeat=5, number=1))
