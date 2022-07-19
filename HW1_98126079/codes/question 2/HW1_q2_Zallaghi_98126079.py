# DIP Course - fall 2020 - HW: 1
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 2 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a)

# reading images in gray mode (actually they are gray...)
dental_xray = cv2.cvtColor(cv2.imread("dental_xray.tif"), cv2.COLOR_BGR2GRAY)
dental_xray_mask = cv2.cvtColor(cv2.imread("dental_xray_mask.tif"), cv2.COLOR_BGR2GRAY)

# extracting wanted zones from image using mask
dental_xray_wanted_zone = np.zeros_like(dental_xray)
for i in range(dental_xray.shape[0]):
  for j in range(dental_xray.shape[1]):
    if dental_xray_mask[i,j] != 0 :
      dental_xray_wanted_zone[i,j] = dental_xray[i,j]


#dental_xray_wanted_zone = np.multiply(dental_xray, dental_xray_mask)

# showing images

fig0 = plt.figure("2:a)")
fig0.add_subplot(1,3,1)
plt.imshow(dental_xray, cmap="gray")
plt.title("dental X-ray image")
fig0.add_subplot(1,3,2)
plt.imshow(dental_xray_mask, cmap="gray")
plt.title("Mask for applying")
fig0.add_subplot(1,3,3)
plt.imshow(dental_xray_wanted_zone, cmap="gray")
plt.title("Picture after appliyng mask")
plt.show()

# b)

# reading image in gray mode
partial_body_scan = cv2.cvtColor(cv2.imread("partial_body_scan.tif"), cv2.COLOR_BGR2GRAY)

# defining a function for image supplement finder
def imgSupplement(img):
  imgsupplement = np.zeros_like(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      imgsupplement[i,j] = 255 - img[i,j]
  return imgsupplement

# finidig suuplement of partial_body_scan
partial_body_scan_supplement = imgSupplement(partial_body_scan)

# body partial and its supplement gathering
body_assembly = partial_body_scan + partial_body_scan_supplement

#showing images ...
fig1 = plt.figure("showing image with its supplement and their assembly")
fig1.add_subplot(1,3,1)
plt.imshow(partial_body_scan, cmap = "gray")
plt.title("partial body scan")
fig1.add_subplot(1,3,2)
plt.imshow(partial_body_scan_supplement, cmap = "gray")
plt.title("its supplement")
fig1.add_subplot(1,3,3)
plt.imshow(body_assembly, cmap = "gray_r")
plt.title("body assembly")
plt.show()

# c)

# radding images from files in gray mode (they are actually gray...)
angiography_live = cv2.cvtColor(cv2.imread("angiography_live.tif"), cv2.COLOR_BGR2GRAY)
angiography_mask = cv2.cvtColor(cv2.imread("angiography_mask.tif"), cv2.COLOR_BGR2GRAY)

# calculating difference of angiography images
angiography_diff = angiography_live - angiography_mask

# finding supplement of the dif image
angiography_diff_supplement = imgSupplement(angiography_diff)

# normalization of pixel values
angiography_diff_supplement_normalized = cv2.normalize(angiography_diff_supplement, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

# showing the results
fig2 = plt.figure("angiography images analysis")
fig2.add_subplot(1,3,1)
plt.imshow(angiography_diff, cmap = "gray")
plt.title("difference image")
fig2.add_subplot(1,3,2)
plt.imshow(angiography_diff_supplement, cmap = "gray")
plt.title("supplement of diff image")
fig2.add_subplot(1,3,3)
plt.imshow(angiography_diff_supplement_normalized, cmap = "gray")
plt.title("normaliozed supplement")
plt.show()
