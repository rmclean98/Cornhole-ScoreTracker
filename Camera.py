import numpy as np
import cv2 as cv
import CornholeScoreKeeper
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPalette, QPixmap, QImage, QCursor
from PyQt5.QtWidgets import *
import sys
import os

rect = []
circlePoints = []
sendCircle = False


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def FindCircle(self, img):
        global circlePoints
        bwimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        bwimg = cv.medianBlur(bwimg,5)

        cimg = cv.cvtColor(bwimg,cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(bwimg,cv.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=30,minRadius=30,maxRadius=40)
        circlePoints = np.uint16(np.around(circles))


    def run(self):
        global circlePoints, sendCircle
        filePath = os.path.join("Images", "vid1.mp4")
        # capture from web cam
        cap = cv.VideoCapture(filePath)
        while self._run_flag:
            ret, cv_img = cap.read()
            if sendCircle:
                self.FindCircle(cv_img)
                if circlePoints.size > 0:
                    for i in circlePoints[0,:]:
                        # draw the outer circle
                        cv.circle(cv_img,(i[0],i[1]),i[2],(0,255,0),2)
                        # draw the center of the circle
                        cv.circle(cv_img,(i[0],i[1]),2,(0,0,255),3)
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class Camera(QWidget):
    def __init__(self):
        super().__init__()
        self.main = CornholeScoreKeeper.MainWindow
        self.setWindowTitle('Camera')
        self.disply_width = 1280
        self.display_height = 720
        self.setFixedSize(self.disply_width, self.display_height)
        self.image_label = QLabel(self)
        self.drawCircle = QPushButton('Draw Cirlce')
        self.drawCircle.setCheckable(True)
        self.drawRect = QPushButton('Draw Rectangle')
        self.drawRect.setCheckable(True)
        #self.image_label.size(self.disply_width, self.display_height)
        vbox = QVBoxLayout()
        vbox.addWidget(self.drawCircle)
        vbox.addWidget(self.drawRect)
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)
        self.drawRect.clicked.connect(self._calibrate)
        self.drawCircle.clicked.connect(self._calibrate)
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()


    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def _calibrate(self):
        global rect, sendCircle, circlePoints
        if self.drawCircle.isChecked() & self.drawRect.isChecked():
            self.drawRect.setChecked(False)
            self.drawCircle.setChecked(False)
            pass
        if self.drawCircle.isChecked():
            sendCircle = True
        if not self.drawCircle.isChecked():
            sendCircle = False
        if self.drawRect.isChecked():
            print(circlePoints)
