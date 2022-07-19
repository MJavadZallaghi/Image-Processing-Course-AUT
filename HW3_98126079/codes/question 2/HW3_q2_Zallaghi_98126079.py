# DIP Course - fall 2020 - HW: 3
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 2 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# defining function for filtering images in  frequency domain

def freq_filtering(img, filterr, parameters):
  f = img.copy()
  A, B = f.shape
  D0 = parameters[0]
  # cacluating padding size
  P = 2 * A
  Q = 2 * B
  # padded image
  f_p = np.zeros((P,Q))
  f_p[0:A,0:B] = f
  # centering padded image
  f_p_centered = f_p.copy()
  for x in range(P):
    for y in range(Q):
      f_p_centered[x,y] = (-1)**(x+y) * f_p[x,y]
  # computing dft of centered padded image
  F = np.fft.fft2(f_p_centered)
  # Making H (filter)
  p_h = int(P/2)
  q_h = int(Q/2)
  d0_h = int(D0/2)
  if filterr == 'LP_Ideal':
    H = np.zeros((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = 1
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice
  if filterr == 'HP_Ideal':
    H = np.ones((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = 0
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice 
  if filterr == 'LP_Butterworth':
    n = parameters[1]
    H = np.zeros((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = 1 / (1 + D/D0)**(2*n)
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice
  if filterr == 'HP_Butterworth':
    n = parameters[1]
    H = np.ones((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = 1 - 1 / (1 + D/D0)**(2*n)
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice
  if filterr == 'LP_Gaussian':
    e = 2.71828
    H = np.zeros((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = e**(-50*(D**2)/(2*(D0**2)))
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice
  if filterr == 'HP_Gaussian':
    e = 2.71828
    H = np.ones((P,Q))
    square_slice = H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h]
    for u in range(square_slice.shape[0]):
      for v in range(square_slice.shape[1]):
        D = ((u-int(square_slice.shape[0]/2))**2 + (v-int(square_slice.shape[1]/2))**2)**0.5
        if D<=d0_h:
          square_slice[u,v] = 1 - e**(-50*(D**2)/(2*(D0**2)))
    H[p_h-d0_h:p_h+d0_h, q_h-d0_h:q_h+d0_h] = square_slice


  # applying filter on image in frequency domain
  G = np.multiply(H,F)
  # computing idft of centered padded image after filter
  g_p_centered = np.real(np.fft.ifft2(G))
  g_p = g_p_centered.copy()
  for x in range(P):
    for y in range(Q):
      g_p[x,y] = (-1)**(x+y) * g_p_centered[x,y]
  # unpadded g with int intensities
  g = np.array(g_p[:A,:B], dtype=np.uint8)
  return g

# reading chest image in gray mode (it is actualy gray...)
img = cv2.cvtColor(cv2.imread('a.tif'), cv2.COLOR_BGR2GRAY)

# appliying low pass filters 
img_lp_ideal_50 = freq_filtering(img, 'LP_Ideal', (50,))
img_lp_ideal_100 = freq_filtering(img, 'LP_Ideal', (100,))
img_lp_ideal_200 = freq_filtering(img, 'LP_Ideal', (200,))
img_lp_bw_50 = freq_filtering(img, 'LP_Butterworth', (50,2))
img_lp_bw_100 = freq_filtering(img, 'LP_Butterworth', (100,2))
img_lp_bw_200 = freq_filtering(img, 'LP_Butterworth', (200,2))
img_lp_gussian_50 = freq_filtering(img, 'LP_Gaussian', (50,))
img_lp_gussian_100 = freq_filtering(img, 'LP_Gaussian', (100,))
img_lp_gussian_200 = freq_filtering(img, 'LP_Gaussian', (200,))

# appliying high pass filters 
img_hp_ideal_50 = freq_filtering(img, 'HP_Ideal', (50,))
img_hp_ideal_100 = freq_filtering(img, 'HP_Ideal', (100,))
img_hp_ideal_200 = freq_filtering(img, 'HP_Ideal', (200,))
img_hp_bw_50 = freq_filtering(img, 'HP_Butterworth', (50,2))
img_hp_bw_100 = freq_filtering(img, 'HP_Butterworth', (100,2))
img_hp_bw_200 = freq_filtering(img, 'HP_Butterworth', (200,2))
img_hp_gussian_50 = freq_filtering(img, 'HP_Gaussian', (50,))
img_hp_gussian_100 = freq_filtering(img, 'HP_Gaussian', (100,))
img_hp_gussian_200 = freq_filtering(img, 'HP_Gaussian', (200,))

# showing results of low pass filters
fig0 = plt.figure('Low pass filters on image')
fig0.add_subplot(3,3,1)
plt.imshow(img_lp_ideal_50, cmap='gray')
plt.title('Ideal, D0 = 50')
fig0.add_subplot(3,3,2)
plt.imshow(img_lp_bw_50, cmap='gray')
plt.title('BW, D0 = 50, n=2')
fig0.add_subplot(3,3,3)
plt.imshow(img_lp_gussian_50, cmap='gray')
plt.title('Gussian, D0 = 50')
fig0.add_subplot(3,3,4)
plt.imshow(img_lp_ideal_100, cmap='gray')
plt.title('Ideal, D0 = 100')
fig0.add_subplot(3,3,5)
plt.imshow(img_lp_bw_100, cmap='gray')
plt.title('BW, D0 = 100, n=2')
fig0.add_subplot(3,3,6)
plt.imshow(img_lp_gussian_100, cmap='gray')
plt.title('Gussian, D0 = 100')
fig0.add_subplot(3,3,7)
plt.imshow(img_lp_ideal_200, cmap='gray')
plt.title('Ideal, D0 = 200')
fig0.add_subplot(3,3,8)
plt.imshow(img_lp_bw_200, cmap='gray')
plt.title('BW, D0 = 200, n=2')
fig0.add_subplot(3,3,9)
plt.imshow(img_lp_gussian_200, cmap='gray')
plt.title('Gussian, D0 = 200')
plt.show()

# showing results of high pass filters
fig1 = plt.figure('High pass filters on image')
fig1.add_subplot(3,3,1)
plt.imshow(img_hp_ideal_50, cmap='gray')
plt.title('Ideal, D0 = 50')
fig1.add_subplot(3,3,2)
plt.imshow(img_hp_bw_50, cmap='gray')
plt.title('BW, D0 = 50, n=2')
fig1.add_subplot(3,3,3)
plt.imshow(img_hp_gussian_50, cmap='gray')
plt.title('Gussian, D0 = 50')
fig1.add_subplot(3,3,4)
plt.imshow(img_hp_ideal_100, cmap='gray')
plt.title('Ideal, D0 = 100')
fig1.add_subplot(3,3,5)
plt.imshow(img_hp_bw_100, cmap='gray')
plt.title('BW, D0 = 100, n=2')
fig1.add_subplot(3,3,6)
plt.imshow(img_hp_gussian_100, cmap='gray')
plt.title('Gussian, D0 = 100')
fig1.add_subplot(3,3,7)
plt.imshow(img_hp_ideal_200, cmap='gray')
plt.title('Ideal, D0 = 200')
fig1.add_subplot(3,3,8)
plt.imshow(img_hp_bw_200, cmap='gray')
plt.title('BW, D0 = 200, n=2')
fig1.add_subplot(3,3,9)
plt.imshow(img_hp_gussian_200, cmap='gray')
plt.title('Gussian, D0 = 200')
plt.show()
