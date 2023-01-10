from cybart import bart
import numpy as np
import matplotlib.pyplot as plt


sig = bart(1, 'phantom -S 8')
print(sig.shape)
#sens = bart(1, 'caldir 24', sig)
sens = bart(1, 'ecalib', sig)  # -> "free(): invalid next size (fast)"
print(sens.shape)

reco = bart(1, 'pics', sig, sens)
print(reco.shape)

