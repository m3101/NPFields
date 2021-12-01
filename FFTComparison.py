import NPFields.ScalarField
from matplotlib import pyplot as plt
import numpy as np

#Four sections of 200X200
field = np.zeros((800,800))

xx,yy = np.meshgrid(np.arange(field.shape[0]),np.arange(field.shape[1]))

#Square
field[50:150,50:150] = 1
#Points and tiny square
field[400+199:400+201,199:201] = 1
field[400+10,10] = 1
field[400+390,10] = 1
field[400+10,390] = 1
field[400+390,390] = 1
#Gaussian kernel
sigma = 50
temp = (xx[:400,:400]-(399/2))**2+(yy[:400,:400]-(399/2))**2
temp = (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*temp/(sigma**2))
temp = temp/temp.max()
field[:400,400:] = temp
#Line
temp = np.abs(xx[:400,:400]-yy[:400,:400])
temp = 1 * (temp<3)
field[400:,400:] = temp

plt.title("Original Field")
plt.imshow(field.T,'gray')

grad = NPFields.ScalarField.gradient_fourier(field)
plt.figure()
plt.title("X Component")
plt.imshow(grad[:,:,0].T)
plt.figure()
plt.title("Y Component")
plt.imshow(grad[:,:,1].T)

plt.show()