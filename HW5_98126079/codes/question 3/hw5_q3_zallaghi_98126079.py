# DIP Course - fall 2020 - HW: 5
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a: tools detection in surgery image

# reading image in gray mode 
surg_img = cv2.imread('surgery.jpeg', 0)
surg_img_copy = cv2.cvtColor(cv2.imread('surgery.jpeg'), cv2.COLOR_BGR2RGB)

# edged of image using canny algorithm
surg_img_edges = cv2.Canny(surg_img,100,100,apertureSize = 3)

# appliying Hough Transform Algorith for finding lines
lines = cv2.HoughLinesP(surg_img_edges,1,np.pi/180,140,minLineLength=110, maxLineGap=20)
# drawing lines on the copy of image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(surg_img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Showing results
fig0 = plt.figure('a: Surgery Tools detction in image')
fig0.add_subplot(1,3,1)
plt.imshow(surg_img, cmap = 'gray')
plt.title('main image')
fig0.add_subplot(1,3,2)
plt.imshow(surg_img_edges, cmap = 'gray')
plt.title('Edges from Canny algorithm')
fig0.add_subplot(1,3,3)
plt.imshow(surg_img_copy)
plt.title('marks on tools')
plt.show()

# b: detection of white globule and red global

# reading image in gray mode 
red_cell_img = cv2.imread('redcell.jpeg', 0)
red_cell_img_copy = cv2.cvtColor(cv2.imread('redcell.jpeg'), cv2.COLOR_BGR2RGB)

# Blur the image to reduce noise
img_blur = cv2.medianBlur(red_cell_img, 5)

# Apply hough transform on the image
# whites
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img_blur.shape[0]/64, param1=250, param2=20, minRadius=20, maxRadius=30)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(red_cell_img_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(red_cell_img_copy, (i[0], i[1]), 2, (0, 0, 255), 3)
# reds
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img_blur.shape[0]/64, param1=225, param2=24, minRadius=35, maxRadius=50)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(red_cell_img_copy, (i[0], i[1]), i[2], (255, 0, 0), 2)
        # Draw inner circle
        cv2.circle(red_cell_img_copy, (i[0], i[1]), 2, (0, 0, 255), 3)

# Showing results
fig1 = plt.figure('b: detection of white globule and red globule')
fig1.add_subplot(1,3,1)
plt.imshow(red_cell_img, cmap = 'gray')
plt.title('main image')
fig1.add_subplot(1,3,2)
plt.imshow(img_blur, cmap = 'gray')
plt.title('blured image')
fig1.add_subplot(1,3,3)
plt.imshow(red_cell_img_copy)
plt.title('red globule: Green\nwhite globule: red')
plt.show()
