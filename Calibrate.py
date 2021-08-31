import cv2 as cv
import numpy as np
import os
import sys

class Calibrate():
    """Takes a picture from the camera to use to find the hole and the board"""

    def __init__(self):
        super().__init__()
        self.minRadius = 30
        self.maxRadius = 50
        self.windowName = "Calibrate"
        self.img = None
        self.drawing = False
        self.circlePoints = []
        self.rectPoints = []
        self.boardContour = None
        self.getPoints()

    def _getCirclePoints(self):
        return self.circlePoints

    def _getRectPoints(self):
        return self.rectPoints

    def _getContour(self):
        return self.boardContour

    def _maxRadius(self, value):
        self.maxRadius = value
        cimg = self.img.copy()
        bwimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
        bwimg = cv.medianBlur(bwimg,5)
        circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,2,100,
                            param1=200,param2=100,minRadius=self.minRadius,maxRadius=value)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.circlePoints = circles
            print(self.circlePoints)
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv.imshow(self.windowName, cimg)


    def _minRadius(self, value):
        self.minRadius = value
        cimg = self.img.copy()
        bwimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
        bwimg = cv.medianBlur(bwimg,5)
        circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,2,100,
                            param1=200,param2=100,minRadius=value,maxRadius=self.maxRadius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.circlePoints = circles
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv.imshow(self.windowName, cimg)

    def _selectPoints(self, value):
        cimg = self.img.copy()
        if value == 0:
            self.drawing = False
        else:
            self.drawing = True
            if len(self.rectPoints) > 0:
                for i in self.rectPoints:
                    cv.circle(cimg, (i[0], i[1]), radius=10, color=(255, 255, 255), thickness=-1)
            cv.imshow(self.windowName, cimg)

    def PolyArea(self,x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def _closestContour(self, value):
        if value == 1:
            if len(self.rectPoints) == 4:
                boardX = []
                boardY = []
                boardArea = cv.contourArea(np.array(self.rectPoints))
                print("board area: ")
                print(boardArea)
                cimg = self.img.copy()
                imgray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
                ret, thresh = cv.threshold(imgray, 240, 255, 0)
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                maxcnt = 0
                index = 0
                count = 0
                for contour in contours:
                    if len(contour) == 4 :
                        cnt = np.reshape(contour, [4,2])
                        cnt = cnt.tolist()
                        print(cnt)
                        print("countor area: ")
                        cntX = []
                        cntY = []
                        for point in cnt:
                            cntX.append(point[0])
                            cntY.append(point[1])
                        area = self.PolyArea(cntX, cntY)
                        #area = cv.contourArea(cnt)
                        print(area)
                        if area > maxcnt and area < boardArea:
                            maxcnt = area
                            index = count
                    count += 1
                        #cv.putText(img, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv.drawContours(cimg, contours[index], 0, (0, 0, 0), 5)
                cv.imshow(self.windowName, cimg)

    def rectPrem(self,event,x,y,flags,param):
        cimg = self.img.copy()
        if event == cv.EVENT_LBUTTONDOWN:
            if self.drawing and len(self.rectPoints) <= 4:
                self.rectPoints.append([x,y])
                cv.circle(cimg, (x,y), radius=10, color=(255, 255, 255), thickness=-1)
                cv.imshow(self.windowName, cimg)

    def getPoints(self):
        filePath = os.path.join("Images", "vid1.mp4")
        cam = cv.VideoCapture(filePath)
        self.img = cam.read()[1]
        img = self.img
        cv.namedWindow(self.windowName)
        #cv.imshow(self.windowName, img)
        cv.createTrackbar("Hole Max Radius", self.windowName, 0, 100, self._maxRadius)
        cv.createTrackbar("Hole Min Radius", self.windowName, 0, 100, self._minRadius)
        cv.createTrackbar("Record Board Points", self.windowName, 0, 1, self._selectPoints)
        cv.createTrackbar("Get Contour of Points", self.windowName, 0, 1, self._closestContour)
        cv.imshow(self.windowName, img)
        cv.setMouseCallback(self.windowName,self.rectPrem)
        cv.waitKey(0)
        cv.destroyAllWindows()
