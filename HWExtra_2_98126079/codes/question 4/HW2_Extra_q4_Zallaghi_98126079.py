# DIP Course - fall 2020 - HW: Extra 2
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 4 code


# Motion detection using phase correlation algorithm in frequency domain

# importing numpy for working with arrays
import numpy as np
# importing cv2 as image processing tools !
import cv2 


# def a function for reading images from webcam online and detecting motion

def motionDetector_phase_correlation():
    # making an video capturing object
    vidObj = cv2.VideoCapture(0)
    # reading first image before loop
    ret, old_liveImg = vidObj.read()
    # gray_old_liveImg = cv2.cvtColor(old_liveImg, cv2.COLOR_BGR2GRAY)
    old_liveImg = old_liveImg[:,:,0]
    # cropping to sqare image
    crop_size = 500
    old_liveImg = old_liveImg[:crop_size,:crop_size]
    # gray_old_liveImg_blu = cv2.GaussianBlur(gray_old_liveImg ,(5,5),cv2.BORDER_DEFAULT)
    motion_counter = 0
    while True:
        # reading new image
        ret, new_liveImg = vidObj.read()
        # gray_new_liveImg = cv2.cvtColor(new_liveImg, cv2.COLOR_BGR2GRAY)
        # redaing red channel of images
        new_liveImg = new_liveImg[:,:,0]
        # cropping to sqare image
        new_liveImg = new_liveImg[:crop_size,:crop_size]
        # gray_new_liveImg_blu = cv2.GaussianBlur(gray_new_liveImg,(5,5),cv2.BORDER_DEFAULT)
        # phase correlation algorithm
        # dft calculations
        f1 = np.fft.fft2(old_liveImg)
        f1_shf_cplx = np.fft.fftshift(f1)
        f2 = np.fft.fft2(new_liveImg)
        f2_shf_cplx = np.fft.fftshift(f2)
        # core of phase correlation
        f1_shf_abs = np.abs(f1_shf_cplx)
        f2_shf_abs = np.abs(f2_shf_cplx)
        total_abs = f1_shf_abs * f2_shf_abs
        R_real = (np.real(f1_shf_cplx)*np.real(f2_shf_cplx) +
                  np.imag(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
        R_imag = (np.imag(f1_shf_cplx)*np.real(f2_shf_cplx) +
                  np.real(f1_shf_cplx)*np.imag(f2_shf_cplx))/total_abs
        R_complex = R_real + 1j*R_imag
        # inverse dft and displacement of object
        R_inverse = np.real(np.fft.ifft2(R_complex))
        #max_id = [0, 0]
        #max_val = 0
        #for idy in range(crop_size):
        #    for idx in range(crop_size):
        #        if R_inverse[idy,idx] > max_val:
        #            max_val = R_inverse[idy,idx]
        #            max_id = [idy, idx]
        #shift_x = crop_size - max_id[0]
        #shift_y = crop_size - max_id[1]
        #print(shift_x, shift_y)
        
        # updating old image
        old_liveImg = new_liveImg
        # gray_old_liveImg_blu = gray_new_liveImg_blu
        # showing online images from webcam and motion detection message
        cv2.imshow("live webcam image changes for motion detection. Hit Escape to Stop.",R_inverse)
        if cv2.waitKey(1) & 0xFF==27:
            break
        # finding motion and recordimg its frame

    cv2.destroyAllWindows()
    vidObj.release()

# running function with Thereshold value = 1
motionDetector_phase_correlation()
