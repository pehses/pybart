from pybart import bart
import numpy as np


data = np.random.randn(1024)
bart(1, 'fft 1', data)
