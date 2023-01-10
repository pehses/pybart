#!/usr/bin/env python

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='pybart',
    version='0.1',
    description='Python bindings for BART',
    author="Philipp Ehses",
    author_email="philipp.ehses@dzne.de",
    url="https://github.com/pehses",
    cmdclass={'build_ext': build_ext},
    packages=['pybart'],
    ext_modules=[Extension('_pybart',
                 sources=['_pybart.pyx'],
                 include_dirs=[numpy.get_include(), '/opt/src/bart/src/'],
                 library_dirs=['/usr/lib'],
                 libraries=['bart'],
                 dependencies=['cython', 'numpy'])],
)
