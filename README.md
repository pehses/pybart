# cybart

Python bindings for [BART](https://github.com/mrirecon/bart) written in Cython. Data is passed in memory to BART using its memcfl C-interface.


## Installation

First, make sure to compile bart as a shared library (follow the general instructions for compiling bart and then type 'make shared-lib').

After cloning this repository, adjust 'include_dirs' and 'library_dirs' within setup.py, so that they point to the location of BART's source code and its compiled library, respectively. Finally, run the following command within the base cybart folder:

    python setup.py install