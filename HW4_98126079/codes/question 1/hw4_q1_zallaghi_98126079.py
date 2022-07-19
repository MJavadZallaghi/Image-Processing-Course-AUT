# DIP Course - fall 2020 - HW: 4
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# importing tools module for help develpoed by TA
from tools import *

# redaing noise motion-blured image in gray (it is actually gray...)
retina_motionblurred = cv2.cvtColor(cv2.imread('retina_motionblurred.jpg'),cv2.COLOR_BGR2GRAY)
# reading retina image
retina = cv2.cvtColor(cv2.imread('Retina.jpg'),cv2.COLOR_BGR2GRAY)

# step a:

kernel = (1/np.eye(13).shape[0])*np.eye(13)

# step b:

# importin wiener restoration algorithm
from skimage.restoration import wiener

# applying wiener algorithm on motion blured image
balance = 0.05
restored = n_range(wiener(normal(retina_motionblurred), kernel, balance))

# step c:

# showing results
fig0 = plt.figure('Motion blured image and Wiener algrithm')
fig0.add_subplot(2,2,1)
plt.imshow(retina_motionblurred, cmap = 'gray', vmin=0, vmax=255)
plt.title('retina motion-blured image')
fig0.add_subplot(2,2,2)
plt.imshow(restored, cmap = 'gray', vmin=0, vmax=255)
plt.title('restored image, balance ='+str(balance))
fig0.add_subplot(2,2,3)
plt.imshow(logmagnitude(retina_motionblurred), cmap = 'gray')
plt.title('motion-blured dft spectrun')
fig0.add_subplot(2,2,4)
plt.imshow(logmagnitude(restored), cmap = 'gray')
plt.title('restored dft spectrum')
plt.show()
