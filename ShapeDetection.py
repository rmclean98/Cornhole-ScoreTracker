import cv2 as cv
import numpy as np
import os
import sys

drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
count = 0
boxCords = []

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, count, boxCords
    if event == cv.EVENT_LBUTTONDOWN:
        count += 1
        drawing = True
        ix,iy = x,y
        if mode and count <= 4:
            boxCords.append([x,y])

#img = np.zeros((512,512,3), np.uint8)
filePath = os.path.join("Images", "Game5.jpg")
img = cv.imread(filePath)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    print(boxCords)
    if len(boxCords) == 4:
        cv.rectangle(img, boxCords[0], boxCords[3], (255,255,255), -1)
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv.destroyAllWindows()
"""
filePath = os.path.join("Images", "Game5.jpg")
img = cv.imread(filePath)

imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret , thrash = cv.threshold(imgGry, 240 , 255, cv.CHAIN_APPROX_NONE)
contours , hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

maxcnt = 0
index = 0
count = 0
board = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4 :
        x, y , w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if not aspectRatio >= 0.95 and aspectRatio < 1.05:
            area = cv.contourArea(contour)
            print(area)
            if area > maxcnt and area < 90000:
                    maxcnt = area
                    index = count
                    board = approx
            #cv.putText(img, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            #cv.drawContours(img, [approx], 0, (0, 0, 0), 5)
    count += 1

print(maxcnt)
print(index)
approx = cv.approxPolyDP(contours[index], 0.01* cv.arcLength(contours[index], True), True)
x = approx.ravel()[0]
y = approx.ravel()[1] - 5
cv.putText(img, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))
cv.drawContours(img, approx, 0, (255,255,255), 5)
"""

cv.namedWindow('detected circles')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('detected circles',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = True
    elif k == 27:
        break
cv.destroyAllWindows()

"""
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


    def FindCircle(self, img):
        global circlePoints
        bwimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        bwimg = cv.medianBlur(bwimg,5)

        cimg = cv.cvtColor(bwimg,cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,1,100,
                                    param1=50,param2=50,minRadius=30,maxRadius=40)
        circlePoints = np.uint16(np.around(circles))

                    if sendCircle:
                        self.FindCircle(cv_img)
                        if circlePoints.size > 1:
                            for i in circlePoints[0,:]:
                                # draw the outer circle
                                cv.circle(cv_img,(i[0],i[1]),i[2],(0,255,0),2)
                                # draw the center of the circle
                                cv.circle(cv_img,(i[0],i[1]),2,(0,0,255),3)
                                """

#grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#edges = cv.Canny(grayscale, 30, 100)
#lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)
#for line in lines:
#    for x1, y1, x2, y2 in line:
#        cv.line(img, (x1, y1), (x2, y2), (20, 220, 20), 3)
