# DIP Course - fall 2020 - HW: 2
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 2 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# importting image in gray mode (it is actually gray...)
kidney_gray = cv2.cvtColor(cv2.imread('kidney.tif'),cv2.COLOR_BGR2GRAY)

# creating associated vectoriez functions to given intensity transformations

def transform_intensity_a(r, lm1):
  s = 0
  # lm1 mean l-1
  if r>=160 and r<240:
    s = 150
  else:
    s = 20
  return s

def transform_intensity_b(r, lm1, lpm1):
  s = 0
  # lm1 mean l-1 and lpm1 mean l'-1
  if r>=100 and r<165:
    s = 200
  else:
    s = r
  return s

# making scaler functions vectorize
vec_transform_intensity_a = np.vectorize(transform_intensity_a)
vec_transform_intensity_b = np.vectorize(transform_intensity_b)

# applying defined intensity transformations on image
kidney_transfored_a = vec_transform_intensity_a(kidney_gray, lm1 = np.max(kidney_gray))
kidney_transfored_b = vec_transform_intensity_b(kidney_gray, lm1 = np.max(kidney_gray), lpm1 = 255)

# showing all images
fig = plt.figure("images under defined intensity transformation")
fig.add_subplot(1,3,1)
plt.imshow(kidney_gray, cmap='gray')
plt.title('Original image')
fig.add_subplot(1,3,2)
plt.imshow(kidney_transfored_a, cmap='gray')
plt.title('left function intensity transformation')
fig.add_subplot(1,3,3)
plt.imshow(kidney_transfored_b, cmap='gray')
plt.title('right function intensity trransformation')
plt.show()
