# DIP Course - fall 2020 - HW: 1
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt
# importing math module
import math

# reading image in gray mode (actually it is gray...)
T = cv2.cvtColor(cv2.imread("T.jpg"), cv2.COLOR_BGR2GRAY)

# a) image scaling

# scaling image
T_scaled = cv2.resize(T, None, fx = 1/2 , fy = 2)
# showing result
fig0 = plt.figure("Scaling")
fig0.add_subplot(2,1,1)
plt.imshow(T, cmap = "gray")
plt.title("Original image")
fig0.add_subplot(2,1,2)
plt.imshow(T_scaled, cmap = "gray")
plt.title("Scaled image: fx = 1/2    fy = 2")
plt.show()

# b) image Translation

# defining an function for image translation
def imgTrans(img,dx,dy):
  # Homogenous matrix for transformation
  mat = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
  img_out = np.zeros_like(img)
  for m in range(img.shape[0]):
    for n in range(img.shape[1]):
      p = np.dot(mat[0,:],np.array([m,n,1]))
      q = np.dot(mat[1,:],np.array([m,n,1]))
      if p<img_out.shape[0] and q<img_out.shape[1]:
        img_out[p,q] = img[m,n]
  return img_out

# appliyng function on image 
T_translated = imgTrans(T,dx=150,dy=100)

# showing result
fig1 = plt.figure("Translating")
fig1.add_subplot(2,1,1)
plt.imshow(T, cmap = "gray")
plt.title("Original image")
fig1.add_subplot(2,1,2)
plt.imshow(T_translated, cmap = "gray")
plt.title("Translated image: dx = 150    dy = 100")
plt.show()

# c) Horizontal shear

# defining an function for Horizontal shear transformation
def imgHShear(img,s_h):
  # Homogenous matrix for transformation
  mat = np.array([[1,0,0],[s_h,1,0],[0,0,1]])
  img_out = np.zeros_like(img)
  for m in range(img.shape[0]):
    for n in range(img.shape[1]):
      p = int(round(np.dot(mat[0,:],np.array([m,n,1]))))
      q = int(round(np.dot(mat[1,:],np.array([m,n,1]))))
      if p<img_out.shape[0] and q<img_out.shape[1]:
        img_out[p,q] = img[m,n]
  return img_out

# appliyng function on image 
T_shear_hor = imgHShear(T,s_h = 0.3)

# showing result
fig2 = plt.figure("Horizontal Shear")
fig2.add_subplot(2,1,1)
plt.imshow(T, cmap = "gray")
plt.title("Original image")
fig2.add_subplot(2,1,2)
plt.imshow(T_shear_hor, cmap = "gray")
plt.title("Horizontal Shear image: s_h = 0.3")
plt.show()

# d) Vertical shear

# defining an function for Vertical shear transformation
def imgVShear(img,s_v):
  # Homogenous matrix for transformation
  mat = np.array([[1,s_v,0],[0,1,0],[0,0,1]])
  img_out = np.zeros_like(img)
  for m in range(img.shape[0]):
    for n in range(img.shape[1]):
      p = int(round(np.dot(mat[0,:],np.array([m,n,1]))))
      q = int(round(np.dot(mat[1,:],np.array([m,n,1]))))
      if p<img_out.shape[0] and q<img_out.shape[1]:
        img_out[p,q] = img[m,n]
  return img_out

# appliyng function on image 
T_shear_ver = imgVShear(T,s_v = 0.3)

# showing result
fig3 = plt.figure("Vertical Shear")
fig3.add_subplot(2,1,1)
plt.imshow(T, cmap = "gray")
plt.title("Original image")
fig3.add_subplot(2,1,2)
plt.imshow(T_shear_ver, cmap = "gray")
plt.title("Vertical Shear image: s_v = 0.3")
plt.show()

# e) Forward and Inverse Rotation


# defining an function for Forward and Inverse Rotation
def imgRot(img,theta,operationType):
  img_out = np.zeros_like(img)
  # Homogenous matrix for transformation
  mat = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
  if operationType == "Forward":
    for m in range(img.shape[0]):
      for n in range(img.shape[1]):
        p = int(round(np.dot(mat[0,:],np.array([m,n,1]))))
        q = int(round(np.dot(mat[1,:],np.array([m,n,1]))))
        #print("p: ",p,"    q:",q, "    img shape 0: ", img_out.shape[0], "    img shape 1: ", img_out.shape[1])
        if p<img_out.shape[0] and q<img_out.shape[1]:
          img_out[p,q] = img[m,n]
  if operationType == "Inverse":
    mat = np.linalg.inv(mat)
    for p in range(img.shape[0]):
      for q in range(img.shape[1]):
        m = int(round(np.dot(mat[0,:],np.array([p,q,1]))))
        n = int(round(np.dot(mat[1,:],np.array([p,q,1])))) 
        # print("p: ",p,"    q:",q, "    img shape 0: ", img_out.shape[0], "    img shape 1: ", img_out.shape[1])      
        if m<img_out.shape[0] and n<img_out.shape[1]:
          img_out[p,q] = img[m,n]
  return img_out


# appliyng function on image 
T_Rot_Forward = imgRot(T,theta = 0.2, operationType = "Forward")
T_Rot_Inverse = imgRot(T,theta = 0.2, operationType = "Inverse")

# showing result
fig4 = plt.figure("Image Rotation")
fig4.add_subplot(1,3,1)
plt.imshow(T, cmap = "gray")
plt.title("Original image")
fig4.add_subplot(1,3,2)
plt.imshow(T_Rot_Forward, cmap = "gray")
plt.title("Forward rotation: theta = 0.2")
fig4.add_subplot(1,3,3)
plt.imshow(T_Rot_Inverse, cmap = "gray")
plt.title("Inverse rotation: theta = 0.2")
plt.show()
