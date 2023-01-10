# pybart

Python bindings for [BART](https://github.com/mrirecon/bart) written in Cython. Data is passed in memory to BART using its memcfl C-interface.


## Installation

First, make sure to compile bart as a shared library (follow the general instructions for compiling bart and then type 'make shared-lib'). Copy the shared library to a folder within your 'LD_LIBRARY_PATH' (/usr/lib should always work on Linux/Unix systems), and then run 'sudo ldconfig'.

After cloning this repository, adjust 'include_dirs' and 'library_dirs' within setup.py if necessary, so that they point to the location of BART's source code and its compiled library, respectively. Install cython in your python environment (`conda install cython` if you use anaconda, else `pip install Cython`).

Finally, run the following command within the base pybart folder:

    python setup.py install

## Usage

pybart is meant as a drop-in replacement for bart.py from the BART project. Just change the import from `from bart import bart` to `from pybart import bart` and everything should (ideally) work the same.
