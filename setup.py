#!/usr/bin/env python

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='cybart',
    version='0.1',
    description='Python bindings for BART',
    author="Philipp Ehses",
    author_email="philipp.ehses@dzne.de",
    url="https://github.com/pehses",
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('cybart',
                 sources=['cybart.pyx'],
                 include_dirs=[numpy.get_include(), '/opt/src/bart/src/'],
                 library_dirs=['/usr/local/lib'],
                 libraries=['bart'],
                 dependencies=['cython', 'numpy'])],
)
