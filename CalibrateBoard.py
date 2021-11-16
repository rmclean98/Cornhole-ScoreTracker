import cv2 as cv
import numpy as np
import math
import os
import sys
import torch

class CalibrateBoard():
    """Takes a picture from the camera to use to find the hole and the board"""

    def __init__(self):
        super().__init__()
        self.windowName = "Calibrate"
        self.img = None
        self.circlePoints = []
        self.rectPoints = []
        self.boardContour = None
        self.detection = None
        self.getPoints()

    def _getCirclePoints(self):
        return self.circlePoints

    def _getRectPoints(self):
        return self.rectPoints

    def _getContour(self):
        return self.boardContour

    def _closestContour(self):
        for index, row in self.detection.iterrows():
            if row['name'] == "Cornhole Board" and row['confidence'] > .7:
                print(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                if len(self.rectPoints) == 0:
                    self.rectPoints.append([int(row['xmin']), int(row['ymin'])])
                    self.rectPoints.append([int(row['xmax']), int(row['ymin'])])
                    self.rectPoints.append([int(row['xmax']), int(row['ymax'])])
                    self.rectPoints.append([int(row['xmin']), int(row['ymax'])])
        if len(self.rectPoints) == 4:
            board = np.array(self.rectPoints)
            boardArea = cv.contourArea(board)
            print("board area: ")
            print(boardArea)
            cimg = self.img.copy()
            imgray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
            imgray = cv.medianBlur(imgray,5)
            edge = cv.Canny(imgray, 60, 180)
            mask = np.zeros_like(edge)
            cv.fillPoly(mask, pts = [board], color =(255,255,255))
            masked = cv.bitwise_and(edge, mask)
            ret, thresh = cv.threshold(imgray, 144, 255, 0)
            contours, hierarchy = cv.findContours(masked, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            maxcnt = 0
            index = 0
            count = 0
            for contour in contours:
                if len(contour) >= 4 :
                    area = cv.contourArea(contour)
                    print("countor area: ")
                    print(area)
                    if area > maxcnt and area < boardArea:
                        maxcnt = area
                        index = count
                count += 1
            cnt = contours[index]
            self.boardContour = cnt

    def _cirlcepoints(self):
        for index, row in self.detection.iterrows():
            if row['name'] == "Cornhole Hole" and row['confidence'] > .7:
                print(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                if len(self.circlePoints) == 0:
                    self.circlePoints.append([int(row['xmin']), int(row['ymin'])])
                    self.circlePoints.append([int(row['xmax']), int(row['ymin'])])
                    self.circlePoints.append([int(row['xmax']), int(row['ymax'])])
                    self.circlePoints.append([int(row['xmin']), int(row['ymax'])])
        if len(self.circlePoints) == 4:
            circle = np.array(self.circlePoints)
            circleArea = cv.contourArea(circle)
            print("circle area: ")
            print(circleArea)
            cimg = self.img.copy()
            imgray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
            imgray = cv.medianBlur(imgray,5)
            edge = cv.Canny(imgray, 60, 180)
            mask = np.zeros_like(edge)
            cv.fillPoly(mask, pts = [circle], color =(255,255,255))
            masked = cv.bitwise_and(edge, mask)
            circleRadius = math.sqrt((circleArea / math.pi))
            circles = cv.HoughCircles(masked,cv.HOUGH_GRADIENT,2,100,
                                param1=200,param2=100,minRadius=0,maxRadius=int(circleRadius))
            if circles is not None:
                circles = np.uint16(np.around(circles))
                self.circlePoints = circles
                print("circle Points:")
                print(self.circlePoints)

    def getPoints(self):
        filePath = os.path.join("Images", "vid1.mp4")
        #filePath = os.path.join("Images", "Game1.jpg")
        cam = cv.VideoCapture(filePath)
        cam.set(3, 1280)
        cam.set(4, 720)
        self.img = cam.read()[1]
        img = self.img
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='CornholeTrackerv6.pt')
        model.classes = [1, 2]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        results = model(img)
        self.detection = results.pandas().xyxy[0]
        self._closestContour()
        self._cirlcepoints()
