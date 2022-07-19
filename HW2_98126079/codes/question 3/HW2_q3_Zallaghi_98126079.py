# DIP Course - fall 2020 - HW: 2
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# َa: defining a function for calculating normalized intesity number levels of an image
def norma_intensity_num(img):
  M = img.shape[0]
  N = img.shape[1]
  levels_vec = np.zeros((256,1), dtype=float)
  for i in range(256):
    levels_vec[i] = (img == i).sum()/float(M*N)
  return levels_vec

# b: defining a function for implementing Histogram Equalization using section a
def histogram_equalization(img):
  img_op = img.copy()
  lm1 = 255
  for k in range(0,lm1+1):
    s_k = lm1 * norma_intensity_num(img)[0:k+1,0].sum()
    img_op[img == k] = s_k
  return img_op


# c: 

# reading images
Lowcontrast = cv2.cvtColor(cv2.imread('Lowcontrast.tif'), cv2.COLOR_BGR2GRAY)
Dark = cv2.cvtColor(cv2.imread('Dark.tif'), cv2.COLOR_BGR2GRAY)
Bright = cv2.cvtColor(cv2.imread('Bright.tif'), cv2.COLOR_BGR2GRAY)

# applying histogram equalization function
Lowcontrast_equ = histogram_equalization(Lowcontrast)
Dark_equ = histogram_equalization(Dark)
Bright_equ = histogram_equalization(Bright)

# showing results
# low contrast image
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Dark_equ, cmap = "gray", vmin = 0, vmax = 255)
ax[0,0].set_title('Equalized Dark image')
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)

ax[1,0].hist(Dark_equ, range = [0,255])
ax[1,0].set_title("Equalized Histogram")
ax[1,0].get_xaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(False)

ax[0,1].imshow(Dark, cmap = "gray", vmin = 0, vmax = 255)
ax[0,1].set_title('Original Dark image')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[1,1].hist(Dark, range = [0,255])
ax[1,1].set_title("Original Histogram")
ax[1,1].get_xaxis().set_visible(True)
ax[1,1].get_yaxis().set_visible(False)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Lowcontrast_equ, cmap = "gray", vmin = 0, vmax = 255)
ax[0,0].set_title('Equalized Low Cont. image')
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)

ax[1,0].hist(Lowcontrast_equ, range = [0,255])
ax[1,0].set_title("Equalized Histogram")
ax[1,0].get_xaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(False)

ax[0,1].imshow(Lowcontrast, cmap = "gray", vmin = 0, vmax = 255)
ax[0,1].set_title('Original Low Cont. image')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[1,1].hist(Lowcontrast, range = [0,255])
ax[1,1].set_title("Original Histogram")
ax[1,1].get_xaxis().set_visible(True)
ax[1,1].get_yaxis().set_visible(False)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(Bright_equ, cmap = "gray", vmin = 0, vmax = 255)
ax[0,0].set_title('Equalized Bright image')
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)

ax[1,0].hist(Bright_equ, range = [0,255])
ax[1,0].set_title("Equalized Histogram")
ax[1,0].get_xaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(False)

ax[0,1].imshow(Bright, cmap = "gray", vmin = 0, vmax = 255)
ax[0,1].set_title('Original Bright image')
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[1,1].hist(Bright, range = [0,255])
ax[1,1].set_title("Original Histogram")
ax[1,1].get_xaxis().set_visible(True)
ax[1,1].get_yaxis().set_visible(False)

plt.show()
