# DIP Course - fall 2020 - HW: 5
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 4 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# reading image in gray mode
gray = cv2.imread('sonography.jpg',0)

# Sobel Edge Detector
scale = 1
delta = 0
ddepth = cv2.CV_16S
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Showing Sobel results
fig0 = plt.figure('Sobel')
fig0.add_subplot(1,4,1)
plt.imshow(gray, cmap = 'gray')
plt.title('main image')
fig0.add_subplot(1,4,2)
plt.imshow(grad_x, cmap = 'gray')
plt.title('Sobel: X')
fig0.add_subplot(1,4,3)
plt.imshow(grad_y, cmap = 'gray')
plt.title('Sobel: Y')
fig0.add_subplot(1,4,4)
plt.imshow(grad, cmap = 'gray')
plt.title('Sobel: (X^2+Y^2)^0.5')
plt.show()

# Prewitt Edge Detector
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(gray, -1, kernelx)
img_prewitty = cv2.filter2D(gray, -1, kernely)
abs_img_prewittx= cv2.convertScaleAbs(img_prewittx)
abs_img_prewittx = cv2.convertScaleAbs(img_prewitty)
abs_img_prewitt = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Showing Prewitt results
fig1 = plt.figure('Prewitt')
fig1.add_subplot(1,4,1)
plt.imshow(gray, cmap = 'gray')
plt.title('main image')
fig1.add_subplot(1,4,2)
plt.imshow(img_prewittx, cmap = 'gray')
plt.title('Prewitt: X')
fig1.add_subplot(1,4,3)
plt.imshow(img_prewitty, cmap = 'gray')
plt.title('Prewitt: Y')
fig1.add_subplot(1,4,4)
plt.imshow(abs_img_prewitt, cmap = 'gray')
plt.title('Prewitt: (X^2+Y^2)^0.5')
plt.show()

#LoG
ddepth = cv2.CV_16S
kernel_size = 3
# Remove noise by blurring with a Gaussian filter
src_gray = cv2.GaussianBlur(gray, (3, 3), 0)
# Apply Laplace function
dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
abs_dst = cv2.convertScaleAbs(dst)

# Showing results
fig2 = plt.figure('LoG')
fig2.add_subplot(1,3,1)
plt.imshow(gray, cmap = 'gray')
plt.title('main image')
fig2.add_subplot(1,3,2)
plt.imshow(src_gray, cmap = 'gray')
plt.title('blured image: Gussian')
fig2.add_subplot(1,3,3)
plt.imshow(abs_dst, cmap = 'gray')
plt.title('Laplacian of Gussian: abs')
plt.show()

# Canny
img_canny = cv2.Canny(gray,30,120)

# Showing results
fig3 = plt.figure('Canny')
fig3.add_subplot(1,2,1)
plt.imshow(gray, cmap = 'gray')
plt.title('main image')
fig3.add_subplot(1,2,2)
plt.imshow(img_canny, cmap = 'gray')
plt.title('Edges by canny algorithm')
plt.show()

# Roberts
roberts_cross_x = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )
roberts_cross_y = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )
img_robertsx = cv2.filter2D(gray, -1, roberts_cross_x)
img_robertsy = cv2.filter2D(gray, -1, roberts_cross_y)
abs_img_robertsx= cv2.convertScaleAbs(img_robertsx)
abs_img_robertsy = cv2.convertScaleAbs(img_robertsy)
abs_img_roberts = cv2.addWeighted(abs_img_robertsx, 0.5, abs_img_robertsy, 0.5, 0)

# Showing Roberts results
fig4 = plt.figure('Roberts')
fig4.add_subplot(1,4,1)
plt.imshow(gray, cmap = 'gray')
plt.title('main image')
fig4.add_subplot(1,4,2)
plt.imshow(img_robertsx, cmap = 'gray')
plt.title('Roberts: X')
fig4.add_subplot(1,4,3)
plt.imshow(img_robertsy, cmap = 'gray')
plt.title('Roberts: Y')
fig4.add_subplot(1,4,4)
plt.imshow(abs_img_roberts, cmap = 'gray')
plt.title('Roberts: (X^2+Y^2)^0.5')
plt.show()
