# pybart

Python bindings for [BART](https://github.com/mrirecon/bart) written in Cython. Data is passed in memory to BART using its memcfl C-interface.


## Installation

First, make sure to compile bart as a shared library (follow the general instructions for compiling bart and then type 'make shared-lib').

After cloning this repository, adjust 'include_dirs' and 'library_dirs' within setup.py, so that they point to the location of BART's source code and its compiled library, respectively. Finally, run the following command within the base pybart folder:

    python setup.py install

## Usage

pybart is meant as a drop-in replacement of bart.py from the BART project. Just change the import from `from bart import bart` to `from pybart import bart` and everything should (ideally) work the same.
