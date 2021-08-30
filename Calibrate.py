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
        self.circlePoints = []
        self.rectPoints = []
        self.getPoints()

    def _getCirclePoints(self):
        return self.circlePoints

    def _getRectPoints(self):
        return self.rectPoints

    def _maxRadius(self, value):
        self.maxRadius = value
        cimg = self.img.copy()
        bwimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
        bwimg = cv.medianBlur(bwimg,5)
        circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,2,100,
                            param1=200,param2=100,minRadius=self.minRadius,maxRadius=value)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            self.circlePoints = circles
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
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            self.circlePoints = circles
        cv.imshow(self.windowName, cimg)

    def getPoints(self):
        filePath = os.path.join("Images", "vid1.mp4")
        cam = cv.VideoCapture(filePath)
        self.img = cam.read()[1]
        img = self.img

        cv.imshow(self.windowName, img)
        cv.createTrackbar("Hole Max Radius", self.windowName, 0, 100, self._maxRadius)
        cv.createTrackbar("Hole Min Radius", self.windowName, 0, 100, self._minRadius)
        print(self.circlePoints)
        cv.waitKey(0)
        cv.destroyAllWindows()
