# DIP Course - fall 2020 - HW: 4
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 2 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a) erosion and dialation

# reading image in binary mode
img = cv2.cvtColor(cv2.imread('noisy_rectangle.png'),cv2.COLOR_BGR2GRAY)

# eroded image with circular structural element - r = 15
img_eroded = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)), iterations = 1)

# dialated image with circular structural element - r = 15
img_dialated = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)), iterations = 1)

# showing results 

# eroded image figure
fig0 = plt.figure('Image and its eroded version')
fig0.add_subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('main image')
fig0.add_subplot(1,2,2)
plt.imshow(img_eroded, cmap='gray', vmin=0, vmax=255)
plt.title('eroded image')
plt.show()

# showing results 

# eroded image figure
fig1 = plt.figure('Image and its dialated version')
fig1.add_subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('main image')
fig1.add_subplot(1,2,2)
plt.imshow(img_dialated, cmap='gray', vmin=0, vmax=255)
plt.title('dialated image')
plt.show()

# b) outer noise reduction using opening

img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (44,44)))

# showing results 

# eroded image figure
fig2 = plt.figure('Image and its opened version')
fig2.add_subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('main image')
fig2.add_subplot(1,2,2)
plt.imshow(img_opened, cmap='gray', vmin=0, vmax=255)
plt.title('opened image')
plt.show()

# c) interior noise reduction using closing

img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (32,32)))

# showing results 

# eroded image figure
fig3 = plt.figure('opened version and its closed version')
fig3.add_subplot(1,2,1)
plt.imshow(img_opened, cmap='gray', vmin=0, vmax=255)
plt.title('image b: just opening')
fig3.add_subplot(1,2,2)
plt.imshow(img_closed, cmap='gray', vmin=0, vmax=255)
plt.title('image c: 1- opening 2- closing')
plt.show()
