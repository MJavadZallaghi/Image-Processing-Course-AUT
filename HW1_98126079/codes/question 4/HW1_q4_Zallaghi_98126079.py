# DIP Course - fall 2020 - HW: 1
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 4 code


# simple Motion detector using averagaing in difference image

# importing numpy for working with arrays
import numpy as np
# importing cv2 as image tools !
import cv2 


# def a function for reading images from webcam online and detecting motion

def motionDetector(thereshold):
    # making an video capturing object
    vidObj = cv2.VideoCapture(0)
    # reading first image for finding changes in lixels
    ret, old_liveImg = vidObj.read()
    gray_old_liveImg = cv2.cvtColor(old_liveImg, cv2.COLOR_BGR2GRAY)
    gray_old_liveImg_blu = cv2.GaussianBlur(gray_old_liveImg ,(5,5),cv2.BORDER_DEFAULT)
    motion_counter = 0
    while True:
        # reading new image
        ret, new_liveImg = vidObj.read()
        gray_new_liveImg = cv2.cvtColor(new_liveImg, cv2.COLOR_BGR2GRAY)
        gray_new_liveImg_blu = cv2.GaussianBlur(gray_new_liveImg,(5,5),cv2.BORDER_DEFAULT)
        # understanding motion using mean of changes in pixel values of blured image
        gray_changed_Img = cv2.absdiff(gray_old_liveImg,gray_new_liveImg)
        gray_changed_Img_blu = cv2.absdiff(gray_old_liveImg_blu, gray_new_liveImg_blu)
        changed_mean = np.mean(gray_changed_Img_blu)
        # updating old image
        gray_old_liveImg = gray_new_liveImg
        gray_old_liveImg_blu = gray_new_liveImg_blu
        # showing online images from webcam and motion detection message
        cv2.imshow("live webcam image changes for motion detection. Hit Escape to Stop.",gray_changed_Img_blu)
        if cv2.waitKey(1) & 0xFF==27:
            break
        # finding motion and recordimg its frame
        if changed_mean > thereshold :
            motion_counter += 1
            print("Motion has been detected - # of Excitation: ", motion_counter)
    cv2.destroyAllWindows()
    vidObj.release()

# running function with Thereshold value = 1
motionDetector(1)
