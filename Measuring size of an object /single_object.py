import cv2
import numpy as np
import pandas as pd

#loading the image
img = cv2.imread('images.jpg')

#converting to grayscale
grayed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#to separate the background and foreground
ret, thresh = cv2.threshold(grayed_img, 127, 255, 0)

#finding contours of the image
contours, hierarchy = cv2.findContours(thresh,
                                       cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)

#drawing contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

#get the are of the object under pixel
area = cv2.contourArea(contours[0])

#convert the arewa to real word unit
scaling_fact = 0.1 
size = area * scaling_fact

print(f"The size of the object is: {size} square units")

#print the size of the object
print('Size:', size)

#display the image with contours
cv2.imshow('Image with Contours', img)
cv2.waitKey(0)  

#save the image with contours
cv2.imwrite('vodka_with_contours.png', img)
