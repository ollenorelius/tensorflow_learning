from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = misc.imread('2.png')
img = img[:,:,1]
print(img)
stuff = plt.imshow(img)
plt.plot(1)
plt.show()
