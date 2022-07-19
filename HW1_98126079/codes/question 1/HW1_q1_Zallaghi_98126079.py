# DIP Course - fall 2020 - HW: 1
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

img0 = cv2.imread("mandrill.jpg")

# a)

# reading image dimensions
dim0 = img0.shape
# image oixel data types
dt0 = img0.dtype

# b)

img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
cv2.imwrite("mandrill_gray.jpg", img1)
# cv2.imshow("gray mandriil :)", img1 )
img1.shape

# c)
 
# function for stretching gray levels
def contrastStretching(img,wantedRange):
  converted_img = np.zeros_like(img)
  a = np.amin(img)
  b = np.amax(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      converted_img[i,j] = wantedRange[0] + round(((wantedRange[1]-wantedRange[0])/(b-a))*(img[i,j]-a))
  return converted_img
# images in wanted gray levels
img1_6bit = contrastStretching(img1, (0,63)) #64 gray level image
img1_4bit = contrastStretching(img1, (0,15)) #16 gray level image
img1_2bit = contrastStretching(img1, (0,1))  #2  gray level image
# plotting (Showing) images
imagse = [img1, img1_6bit, img1_4bit, img1_2bit]
fig = plt.figure("shaowing changed gray level image", (3,1))
for i in range(4):
  fig.add_subplot(1,4,i+1)
  plt.title("array: " + str(imagse[i]))
  plt.imshow(imagse[i], cmap="gray")
plt.show()

# d)

# defining an function for croping images between a(x,y) and b(x',y')
def imgCrop(img, cropRange):
  x = cropRange[0]
  y = cropRange[1]
  xp = cropRange[2]
  yp = cropRange[3]
  cropedImg = np.zeros_like(img[x:xp,y:yp])
  for i in range(x,xp):
    for j in range(y,yp):
      cropedImg[i-x,j-y] = img[i,j]
  return cropedImg
# applying function on the image
img0_crop_l = imgCrop(img0,(0,0,int(img0.shape[0]),int(img0.shape[1]/2)))
img0_crop_l = cv2.cvtColor(img0_crop_l, cv2.COLOR_BGR2RGB)
img0_crop_r = imgCrop(img0,(0,int(img0.shape[1]/2),int(img0.shape[0]),int(img0.shape[1])))
img0_crop_r = cv2.cvtColor(img0_crop_r, cv2.COLOR_BGR2RGB)
# showing croped images along-side
croped_images = [img0_crop_l, img0_crop_r]
fig2 = plt.figure("Wnidow: showing croped images together")
for i in range(2):
  fig2.add_subplot(1,2,i+1)
  plt.title("image number: "+str(i))
  plt.imshow(croped_images[i])

plt.show()

# e)

# for fliping images in cv2, we can use flip function.

img0_fliped_u2d = cv2.flip(img0, 0)
img0_fliped_u2d = cv2.cvtColor(img0_fliped_u2d, cv2.COLOR_BGR2RGB)
img0_fliped_r2l = cv2.flip(img0, 1)
img0_fliped_r2l = cv2.cvtColor(img0_fliped_r2l, cv2.COLOR_BGR2RGB)
fig3 = plt.figure("showing fliped images")
fig3.add_subplot(1,3,1)
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.title("Original image")
fig3.add_subplot(1,3,2)
plt.imshow(img0_fliped_u2d)
plt.title("Up to Down fliped image")
fig3.add_subplot(1,3,3)
plt.imshow(img0_fliped_r2l)
plt.title("Right to Left fliped image")
plt.show()

# f)

# savimg images :)
cv2.imwrite("Up_to_Down_fliped_image.png", cv2.cvtColor(img0_fliped_u2d, cv2.COLOR_BGR2RGB))

# g)

# aspect ratio = 3
img1_biLinInterpol_3 = cv2.resize(img1, None, fx = 3, fy = 3, interpolation = cv2.INTER_LINEAR)
img1_areaInterpol_3 = cv2.resize(img1, None, fx = 3, fy = 3, interpolation = cv2.INTER_AREA)
img1_nearesInterpol_3 = cv2.resize(img1, None, fx = 3, fy = 3, interpolation = cv2.INTER_NEAREST)
# aspect ration = 1/3
img1_biLinInterpol_1_3 = cv2.resize(img1, None, fx = 1/3, fy = 1/3, interpolation = cv2.INTER_LINEAR)
img1_areaInterpol_1_3 = cv2.resize(img1, None,  fx = 1/3, fy = 1/3, interpolation = cv2.INTER_AREA)
img1_nearesInterpol_1_3 = cv2.resize(img1, None, fx = 1/3, fy = 1/3, interpolation = cv2.INTER_NEAREST)
# showing images in 1 window
fig4 = plt.figure("images with different scale and interpolation method")
fig4.add_subplot(2,3,1)
plt.imshow(img1_biLinInterpol_3, cmap="gray")
plt.title("scale: 3 - bilinear")
fig4.add_subplot(2,3,2)
plt.imshow(img1_areaInterpol_3, cmap="gray")
plt.title("scale: 3 - area")
fig4.add_subplot(2,3,3)
plt.imshow(img1_nearesInterpol_3, cmap="gray")
plt.title("scale: 3 - near")
fig4.add_subplot(2,3,4)
plt.imshow(img1_biLinInterpol_1_3, cmap="gray")
plt.title("scale: 1/3 - bilinear")
fig4.add_subplot(2,3,5)
plt.imshow(img1_areaInterpol_1_3, cmap="gray")
plt.title("scale: 1/3 - area")
fig4.add_subplot(2,3,6)
plt.imshow(img1_nearesInterpol_1_3, cmap="gray")
plt.title("scale: 1/3 - near")
plt.show()
