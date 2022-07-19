# DIP Course - fall 2020 - HW: 2
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 4 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a) 

# difining function for convolving a mask on an image
def maskConvoler(img,mask):
  # making image ready for operation
  img_copy = img.copy()
  img_convolved = np.zeros_like(img)
  # padding source image with 1 pixel in all borders
  img_copy = cv2.copyMakeBorder(img_copy, 1, 1, 1, 1, cv2.BORDER_REFLECT)
  # applying mask
  if type(mask)==np.ndarray:
    if mask.shape == (3,3):
      for i in range(img_convolved.shape[0]):
        for j in range(img_convolved.shape[1]):
          val = np.multiply(img_copy[i:i+3,j:j+3],mask).sum()
          if val>=0 and val<=255:
            img_convolved[i,j] = val
          elif val<0:
            img_convolved[i,j] = 0
          else:
            img_convolved[i,j] = 255

      return img_convolved
    else:
      return False
  elif type(mask)==str:
    if mask=='median':
      for i in range(img_convolved.shape[0]):
        for j in range(img_convolved.shape[1]):
          unsorted_array = img_copy[i:i+3,j:j+3]
          sorted_array = np.sort(unsorted_array, axis=None)
          img_convolved[i,j] = sorted_array[5]
      return img_convolved
    else:
      return False

# b)

# reading image in gray mode (it is actually gray...)
img = cv2.cvtColor(cv2.imread('bone-scan.png'), cv2.COLOR_BGR2GRAY)

# image under median mask
img_med_masked = maskConvoler(img, mask='median')

# showing image and its median masked type
fig0 = plt.figure('bone image and its median masked type')
fig0.add_subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('bone image')
fig0.add_subplot(1,2,2)
plt.imshow(img_med_masked, cmap='gray')
plt.title('bone image under median mask')
plt.show()

# c)

# image under simple mean mask
img_mean_masked = maskConvoler(img, mask=(1/9)*np.ones((3,3)))

# showing image mean-meidan masked type
fig0 = plt.figure('comparison of median and simple mean mask')
fig0.add_subplot(1,2,1)
plt.imshow(img_med_masked, cmap='gray')
plt.title('bone image under median mask')
fig0.add_subplot(1,2,2)
plt.imshow(img_mean_masked, cmap='gray')
plt.title('bone image under simple mean mask')
plt.show()

# d)

# image under simple mean mask
img_lapla_masked = maskConvoler(img, mask=np.array([[0,1,0],[1,-4,1],[0,1,0]]))
img_up = img + img_lapla_masked
# same laplacian using cv2 function
laplaciancv = cv2.filter2D(img, -1, kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]]), borderType = cv2.BORDER_REFLECT)

# showing bone image and its laplacian masked type
fig0 = plt.figure('bone image under laplacian mask')
fig0.add_subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('bone image')
fig0.add_subplot(1,3,2)
plt.imshow(img_lapla_masked, cmap='gray')
plt.title('filter valuse on image')
fig0.add_subplot(1,3,3)
plt.imshow(img_up, cmap='gray')
plt.title('bone image under laplacian mask')
plt.show()

print('Checking defined function for convolving laplacian masks:\n')
print(np.all(laplaciancv==img_lapla_masked))


# e)
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
c0 = 0
src0 = img + c0 * img_lapla_masked

l = plt.imshow(src0, cmap = 'gray')
ax.set_title('image + c * laplacian')


# this is an inset axes over the main axes
right_inset_ax = fig.add_axes([0.1, 0.4, .25, .25], facecolor='b')
right_inset_ax.hist(src0, range=[0,255],log=True)
right_inset_ax.set(title='Histogram')



ax.margins(x=1)
axcolor = 'lightgoldenrodyellow'
axc = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
sc = Slider(axc, 'C', -20.0, 20.0, valinit=c0, valstep=0.5)

def update(val):
    c = sc.val
    src = img + c * img_lapla_masked
    src[src>255] = 255
    src[src<0] = 0
    l.set_data(src)
    fig.canvas.draw_idle()
    right_inset_ax.cla()
    right_inset_ax.hist(src, range=[0,255],log=True)
    plt.draw()


sc.on_changed(update)

plt.show()


