# DIP Course - fall 2020 - HW: 4
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a) 

# reading image in gray mode
fingerprint_noisy = cv2.cvtColor(cv2.imread('fingerprint.png'), cv2.COLOR_BGR2GRAY)

# noise removing using opening and closing
fingerprint_opened = cv2.morphologyEx(fingerprint_noisy, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
fingerprint_opened_closed = cv2.morphologyEx(fingerprint_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

# showing results
fig0 = plt.figure('Removing noise in fingreprint image')
fig0.add_subplot(1,3,1)
plt.imshow(fingerprint_noisy, cmap='gray')
plt.title('Noisy fingerprint')
fig0.add_subplot(1,3,2)
plt.imshow(fingerprint_opened, cmap='gray')
plt.title('opened fingerprint using opening')
fig0.add_subplot(1,3,3)
plt.imshow(fingerprint_opened_closed, cmap='gray')
plt.title('closing opened fingerprint - denoised')
plt.show()

# b)

# reading image in gray mode
img = cv2.cvtColor(cv2.imread('headCT.png'), cv2.COLOR_BGR2GRAY)
# images dilation
img_dil_3 = cv2.dilate(img, np.ones((3,3)), iterations = 1)
img_dil_7 = cv2.dilate(img, np.ones((7,7)), iterations = 1)
# images erosions
img_ero_3 = cv2.erode(img, np.ones((3,3)), iterations = 1)
img_ero_7 = cv2.erode(img, np.ones((7,7)), iterations = 1)
# gradiantes
gradiant_3 = img_dil_3 - img_ero_3
gradiant_7 = img_dil_7 - img_ero_7

# showing results
fig1 = plt.figure('image gradiants using diff in dilation and erosion')
fig1.add_subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('main image')
fig1.add_subplot(1,3,2)
plt.imshow(gradiant_3, cmap='gray')
plt.title('morph. gradiant with 3-element')
fig1.add_subplot(1,3,3)
plt.imshow(gradiant_7, cmap='gray')
plt.title('morph. gradiant with 7-element')
plt.show()

# c)

# reading image in gray mode
rice = cv2.cvtColor(cv2.imread('rice.tif'), cv2.COLOR_BGR2GRAY)
# opening image
rice_opened = cv2.morphologyEx(rice, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50)))
# top hat tranformation
rice_top_hat = rice - rice_opened
# thresholing
ret,rice_threshold = cv2.threshold(rice_top_hat,50,255,cv2.THRESH_BINARY)
# noise eroding
rice_eroded = cv2.erode(rice_threshold, np.ones((5,5)), iterations = 1)
# boundary and hole filling eroding
rice_dil = cv2.dilate(rice_eroded, np.ones((5,5)), iterations = 1)

# showing results
fig2 = plt.figure('rice etraction')
fig2.add_subplot(2,3,1)
plt.imshow(rice, cmap='gray')
plt.title('main image')
fig2.add_subplot(2,3,2)
plt.imshow(rice_opened, cmap='gray')
plt.title('opened')
fig2.add_subplot(2,3,3)
plt.imshow(rice_top_hat, cmap='gray')
plt.title('top hat: f - foB')
fig2.add_subplot(2,3,4)
plt.imshow(rice_threshold, cmap='gray')
plt.title('threshold after top hat')
fig2.add_subplot(2,3,5)
plt.imshow(rice_eroded, cmap='gray')
plt.title('eroded for noise removing')
fig2.add_subplot(2,3,6)
plt.imshow(rice_dil, cmap='gray')
plt.title('dilated for hole filling')
plt.show()
