# DIP Course - fall 2020 - HW: 0
# Student: M. J. Zallaghi    -    ID: 98126079


# Question: 3 code

# importing of useful librarys
import numpy as np
import cv2

# function definition
def convertor(inputArray):
    " Input array is an array with float data type members: members between [a,b]"
    " Output array is corrosponding array with member unit8 data type: members between [0,255] "

    # first thing: finding a and b
    a = np.amin(inputArray)
    b = np.amax(inputArray)
    # element to element mapper function
    def mapper(x):
        xprime = int(((255-0)/(b-a))*(x-a))
        return xprime
    # appliying defined function on numpy array object
    vectorMapper = np.vectorize(mapper)
    outPut =  vectorMapper(inputArray)
    outPut = outPut.astype('uint8')
    return outPut

# Generating a random digital image !
random_image_array_ab = np.random.uniform(low=-3.2, high=9.3, size=(50,40,3))
random_image_array_uint8 = convertor(random_image_array_ab)
cv2.imwrite('color_img.jpg', random_image_array_uint8)
cv2.imshow("image", random_image_array_uint8)
cv2.waitKey()



        
