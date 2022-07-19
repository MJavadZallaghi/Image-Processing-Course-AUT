# DIP Course - fall 2020 - HW: 3
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# reading images in gray mode
clown_pos_domain = cv2.cvtColor(cv2.imread('clown.tif'),cv2.COLOR_BGR2GRAY)
mandrill_pos_domain = cv2.cvtColor(cv2.imread('mandrill.tif'),cv2.COLOR_BGR2GRAY)
# Computing the 2-dimensional discrete Fourier Transform of images using numpy
clown_freq_domain = np.fft.fft2(clown_pos_domain)
mandrill_freq_domain = np.fft.fft2(mandrill_pos_domain)

# defining function for finding amplitude and phase of complex arrays
def amp_phase(x):
  return np.abs(x), np.angle(x)
# defining function for creating complex array by amplitude and phase of complex arrays
def complex_array(amp,phase):
  return np.multiply(amp, np.exp(1j*phase))

# phase and amplitude of images dtfs
clown_amp, clown_phase = amp_phase(clown_freq_domain)
mandrill_amp, mandrill_phase = amp_phase(mandrill_freq_domain)

# changing phase of images in frequncy domain and makeing new complex arrays
clown_new_freq = complex_array(clown_amp, mandrill_phase)
mandrill_new_freq = complex_array(mandrill_amp, clown_phase)

# finding inverse of new images
clown_new_pos = np.fft.ifft2(clown_new_freq)
clown_new_pos = np.real(clown_new_pos)
mandrill_new_pos = np.fft.ifft2(mandrill_new_freq)
mandrill_new_pos = np.real(mandrill_new_pos)

# showing results
fig0 = plt.figure('Images with reversed phases')
fig0.add_subplot(2,2,1)
plt.imshow(clown_pos_domain, cmap='gray')
plt.title('Original Clown')
fig0.add_subplot(2,2,2)
plt.imshow(mandrill_pos_domain, cmap='gray')
plt.title('Original Mandrill')
fig0.add_subplot(2,2,3)
plt.imshow(clown_new_pos, cmap='gray')
plt.title('Clown amp with Mandrill phase')
fig0.add_subplot(2,2,4)
plt.imshow(mandrill_new_pos, cmap='gray')
plt.title('Mandrill amp with Clown phase')
plt.show()
