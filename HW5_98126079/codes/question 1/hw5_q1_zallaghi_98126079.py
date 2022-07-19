# DIP Course - fall 2020 - HW: 5
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# reading images in gray mode
MRIF = cv2.imread('MRIF.png',0)
MRIS = cv2.imread('MRIS.png',0)
# resizing secon mri image
MRIS = cv2.resize(MRIS, (MRIF.shape[1],MRIF.shape[0]))

# defining function for geting tie points from user

x_from_user = []
y_from_user = []

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(0,0,255),-1)
        x_from_user.append(y)
        y_from_user.append(x)
        mouseX,mouseY = x,y
        
img = MRIF.copy()
cv2.namedWindow('First MRI')
cv2.setMouseCallback('First MRI',draw_circle)

print('Selection method: cursore left button double click\n')
print('Hint: please first select n tie pint from first mri.\nAfter selection press Escape. \nThen select n point from second mri and after selection, press Escape.')

while(True):
    cv2.imshow('First MRI',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print('\n',str(len(x_from_user)), 'point from first mri has been selected.\nplease select same numer of similar points from second mri.')

img = MRIS.copy()
cv2.namedWindow('Second MRI')
cv2.setMouseCallback('Second MRI',draw_circle)

while(True):
    cv2.imshow('Second MRI',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
print('\nall points has been selected. thanks :)\n')

# registration process

# points before affine transformation
x = np.array(x_from_user[0:int(len(x_from_user)/2)])
y = np.array(y_from_user[0:int(len(y_from_user)/2)])
# points after affine transformation
xp = np.array(x_from_user[int(len(x_from_user)/2):])
yp = np.array(y_from_user[int(len(y_from_user)/2):])

A = np.ones((x.shape[0],3))
for i in range(x.shape[0]):
    A[i,0] = x[i]
    A[i,1] = y[i]

a_1 = np.dot(np.linalg.pinv(A), xp)
a_2 = np.dot(np.linalg.pinv(A), yp)

print('afine transformation matrix is:\n',a_1,'\n',a_2,'\n',[0,0,1])
