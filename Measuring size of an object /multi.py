import cv2
import numpy as np

#loading the image
img = cv2.imread('house.jpg')

#converting to grayscale
grayed_img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#applying thresholding to separate the background and foreground
ret, thresh = cv2.threshold(grayed_img,
                            0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#finding the contours of the image
contours, hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

#loop through each contour and draw it on the original image
for cnt in contours:

    area = cv2.contourArea(cnt) 

    #draw a rect for each
    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    cv2.putText(img, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 234), 2)   

#display the image with contours
cv2.imshow('Image with Contours', img)
cv2.waitKey(0)

#save the image with contours
cv2.imwrite('donuts_with_contours.png', img)

cv2.destroyAllWindows()


