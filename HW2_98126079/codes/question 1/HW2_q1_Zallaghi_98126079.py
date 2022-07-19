# DIP Course - fall 2020 - HW: 2
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# reading image in gray mode (it is actually gray...)
brains_gray = cv2.cvtColor(cv2.imread("brains.png"), cv2.COLOR_BGR2GRAY)

# defining function for power intensity transformation
def powerIntensityTransform(img, gama):
  c = (255) / ((np.max(img))**(gama))
  img_transformed = c * np.power(img, gama)
  img_transformed_unit_8 = np.array(img_transformed, dtype = np.uint8)
  return img_transformed_unit_8

# defining function for log intensity transformation
def logIntensityTransform(img):
  c = (255)/(np.log10(np.max(img)+1))
  img_transformed = c * np.log10(img+1)
  img_transformed_unit_8 = np.array(img_transformed, dtype = np.uint8)
  return img_transformed_unit_8

# making transformed images under power transformations
brains_power_intensity_transfrom = powerIntensityTransform(brains_gray, 0.7)
brains_log_intensity_transfrom = logIntensityTransform(brains_gray)

# showing results with associated histograsms

fig, ax = plt.subplots(2,3)

ax[0,0].imshow(brains_power_intensity_transfrom, cmap = "gray", vmin = 0, vmax = 255)
ax[0,0].set_title("Power Intensity Transformation\ngama = " + str(0.7))
ax[0,0].get_xaxis().set_visible(False)
ax[0,0].get_yaxis().set_visible(False)

ax[1,0].hist(brains_power_intensity_transfrom, range = [0,255])
ax[1,0].set_title("Image associated Histogram in power Tran.")
ax[1,0].get_xaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(False)

ax[0,1].imshow(brains_log_intensity_transfrom, cmap = "gray", vmin = 0, vmax = 255)
ax[0,1].set_title("Log10 Intensity Transformation")
ax[0,1].get_xaxis().set_visible(False)
ax[0,1].get_yaxis().set_visible(False)

ax[1,1].hist(brains_log_intensity_transfrom, range = [0,255])
ax[1,1].set_title("Image associated Histogram in Log Tran.")
ax[1,1].get_xaxis().set_visible(True)
ax[1,1].get_yaxis().set_visible(False)

ax[0,2].imshow(brains_gray, cmap = "gray", vmin = 0, vmax = 255)
ax[0,2].set_title("Original image")
ax[0,2].get_xaxis().set_visible(False)
ax[0,2].get_yaxis().set_visible(False)

ax[1,2].hist(brains_gray, range = [0,255])
ax[1,2].set_title("Image associated Histogram without Tran.")
ax[1,2].get_xaxis().set_visible(True)
ax[1,2].get_yaxis().set_visible(False)


plt.show()
