import numpy as np
import matplotlib.pyplot as plt
import pybart
import bart

# d = np.arange(16)
# pybart.bart(0, "show", d)

cy = pybart.bart(1, "phantom -x 32")
print(cy.shape)

rev = bart.bart(1, "phantom -x 32")

print(cy.shape, rev.shape)

print(np.allclose(cy, rev))

plt.figure()
plt.subplot(121)
plt.imshow(abs(rev))
plt.title('bart')
plt.axis('off')

plt.subplot(122)
plt.imshow(abs(cy))
plt.title('pybart')
plt.axis('off')
plt.show()
