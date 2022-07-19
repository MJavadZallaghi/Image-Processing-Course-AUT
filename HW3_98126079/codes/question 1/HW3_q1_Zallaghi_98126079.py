# DIP Course - fall 2020 - HW: 3
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 1 code


# using numpy for working with arrays
import numpy as np
# using openCv module
import cv2
# using matplotlib for multiple image show in 1 fig
import matplotlib.pyplot as plt

# a)

# reading chest image in gray mode (it is actualy gray...)
chest_img_pos_domain = cv2.cvtColor(cv2.imread('chest.tif'),cv2.COLOR_BGR2GRAY)
# Computing the 2-dimensional discrete Fourier Transform of image using numpy
chest_img_freq_domain = np.fft.fft2(chest_img_pos_domain)
# centering the image in frequency domain using numpy: Shift the zero-frequency component to the center of the spectrum
chest_img_freq_domain_centered = np.fft.fftshift(chest_img_freq_domain)

# phase spectrum centered and non-centered image
phase_spectrum_uncentered = np.angle(chest_img_freq_domain)
phase_spectrum_centered = np.angle(chest_img_freq_domain_centered)
# magnitude spectrum
magnitude_spectrum_uncentered = np.abs(chest_img_freq_domain)
magnitude_spectrum_uncentered = np.array(magnitude_spectrum_uncentered/np.max(magnitude_spectrum_uncentered)*255, dtype=np.uint8)
magnitude_spectrum_centered = np.abs(chest_img_freq_domain_centered)
magnitude_spectrum_centered = np.array(magnitude_spectrum_centered/np.max(magnitude_spectrum_centered)*255, dtype=np.uint8)

# showing results
fig0 = plt.figure('Magnitude and Phase plots')
fig0.add_subplot(2,2,1)
plt.imshow(magnitude_spectrum_uncentered, cmap='gray')
plt.title('Mag. Spectrum: un-centered')
fig0.add_subplot(2,2,2)
plt.imshow(phase_spectrum_uncentered, cmap='gray')
plt.title('Phase. Spectrum: un-centered')
fig0.add_subplot(2,2,3)
plt.imshow(magnitude_spectrum_centered, cmap='gray')
plt.title('Mag. Spectrum: centered')
fig0.add_subplot(2,2,4)
plt.imshow(phase_spectrum_centered, cmap='gray')
plt.title('Phase. Spectrum: centered')
plt.show()

# b)

# applying inverse dft on genrated dft

img_from_dft_uncentered = np.fft.ifft2(chest_img_freq_domain)
img_from_dft_uncentered = np.real(img_from_dft_uncentered)

img_from_dft_centered = np.fft.ifft2(chest_img_freq_domain_centered)
img_from_dft_centered = np.real(img_from_dft_centered)

# showing results
fig1 = plt.figure('Inverse DFT')
fig1.add_subplot(3,1,1)
plt.imshow(chest_img_pos_domain, cmap='gray')
plt.title('Original image')
fig1.add_subplot(3,1,2)
plt.imshow(img_from_dft_uncentered, cmap='gray')
plt.title('Image from un-centered dft')
fig1.add_subplot(3,1,3)
plt.imshow(img_from_dft_centered, cmap='gray')
plt.title('Image from centered dft')
plt.show()

# c)

# mirroring about center
mirrored_about_cented_freq = np.conj(chest_img_freq_domain_centered)
# shifting to corner
mirrored_about_cented_freq = np.fft.ifftshift(mirrored_about_cented_freq)
# frequency dmaoin to pose domain
mirrored_about_center_pos = np.fft.ifft2(mirrored_about_cented_freq)
mirrored_about_center_pos = np.real(mirrored_about_center_pos)

# showing results
fig2 = plt.figure('Image mirroring')
fig2.add_subplot(1,2,1)
plt.imshow(chest_img_pos_domain, cmap='gray')
plt.title('Original image')
fig2.add_subplot(1,2,2)
plt.imshow(mirrored_about_center_pos, cmap='gray')
plt.title('Mirrored image')
plt.show()
