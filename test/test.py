import numpy as np
import matplotlib.pyplot as plt
import cybart
import bart

# d = np.arange(16)
# cybart.bart(0, "show", d)

cy = cybart.bart(1, "phantom -x 32")
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
plt.title('cybart')
plt.axis('off')
plt.show()
