import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

filePath = os.path.join("Images", "Game5.jpg")
img = cv.imread(filePath)
bwimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
bwimg = cv.medianBlur(bwimg,5)

cimg = cv.cvtColor(bwimg,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=30,maxRadius=40)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv.imshow('detected circles',img)
cv.waitKey(0)
cv.destroyAllWindows()
#grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#edges = cv.Canny(grayscale, 30, 100)
#lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
#for line in lines:
#    for x1, y1, x2, y2 in line:
#        cv.line(img, (x1, y1), (x2, y2), (20, 220, 20), 3)
