import NPFields.ScalarField
from matplotlib import pyplot as plt
import numpy as np

field = np.zeros((100,100))

xx,yy = np.meshgrid(np.arange(field.shape[0]),np.arange(field.shape[1]))

#Line
xx,yy = np.meshgrid(np.arange(field.shape[0]),np.arange(field.shape[1]))
temp = np.abs(xx-yy)
temp = 1 * (temp<3)
field = temp

plt.title("Original Field")
plt.imshow(field.T,'gray')

grad = NPFields.ScalarField.gradient(field)
plt.figure()
plt.title("X Component (Finite Differences)")
plt.imshow(grad[:,:,0].T)
plt.figure()
grad = NPFields.ScalarField.gradient_fourier(field)
plt.title("X Component (Fourier)")
plt.imshow(grad[:,:,0].T)

plt.figure()
temp = np.zeros(((field.shape[0]+20),(field.shape[1]+20)))
temp[10:-10,10:-10] = field
grad = NPFields.ScalarField.gradient_fourier(temp)[10:-10,10:-10]
plt.title("X Component (Fourier - padded with zeros)")
plt.imshow(grad[:,:,0].T)

plt.show()