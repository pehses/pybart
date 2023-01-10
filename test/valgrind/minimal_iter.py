import cybart
import numpy as np


data = np.random.randn(1024)
for i in range(8):
    cybart.bart(1, 'fft 1', data)
