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
        F1 = np.fft.fft2(old_liveImg)
        F1_centered = np.fft.fftshift(F1)
        F2 = np.fft.fft2(new_liveImg)
        F2_centered = np.fft.fftshift(F2)
        
        # updating old image
        old_liveImg = new_liveImg
        # gray_old_liveImg_blu = gray_new_liveImg_blu
        # showing online images from webcam and motion detection message
        cv2.imshow("live webcam image changes for motion detection. Hit Escape to Stop.",old_liveImg)
        if cv2.waitKey(1) & 0xFF==27:
            break
        # finding motion and recordimg its frame

    cv2.destroyAllWindows()
    vidObj.release()

# running function with Thereshold value = 1
motionDetector_phase_correlation()
