import timeit

import_module="""
import cybart
import bart
"""

testcode="""
def bench(pkg):
    sig = pkg.bart(1, 'phantom -x 1024')
    sig = pkg.bart(1, 'fft 7', sig)
    sig = pkg.bart(1, 'fft -i 7', sig)
    return sig
"""

print("bart: ", timeit.repeat(stmt=testcode + "bench(bart)",  setup=import_module, repeat=5, number=1))
print("cybart: ", timeit.repeat(stmt=testcode + "bench(cybart)",  setup=import_module, repeat=5, number=1))
