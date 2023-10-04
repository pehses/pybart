from pybart import bart
import numpy as np
import matplotlib.pyplot as plt

sig = bart(1, 'phantom -S8')
a = bart(1, 'cc -G -M', sig)
b = bart(1, 'ccapply -G -p4', sig, a)
print(b.shape)
