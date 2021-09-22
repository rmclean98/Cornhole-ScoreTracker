import numpy as np
import cv2 as cv
import CornholeScoreKeeper
from CalibrateBoard import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPalette, QPixmap, QImage, QCursor
from PyQt5.QtWidgets import *
import sys
import os

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.draw = False
        self.circlePoints = []
        self.boardPoints = None

    def setPoints(self, circle, board):
        self.circlePoints = circle
        self.boardPoints = board

    def setDraw(self, value):
        if value == 0:
            self.draw = False
        else:
            self.draw = True

    def run(self):
        filePath = os.path.join("Images", "vid1.mp4")
        # capture from web cam
        cap = cv.VideoCapture(filePath)
        while self._run_flag:
            ret, cv_img = cap.read()
            if self.draw:
                cv.drawContours(cv_img, [self.boardPoints], 0, (0, 255, 0), 5)
                for i in self.circlePoints[0,:]:
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
        self.setWindowFlag(Qt.WindowTitleHint)
        self.disply_width = 1280
        self.display_height = 720
        self.drawing = False
        self.boardPoints = None
        self.circlePoints = None
        self.setFixedSize(self.disply_width, self.display_height)
        self.image_label = QLabel(self)
        self.showContours = QPushButton("Show Contours")
        self.showContours.setCheckable(True)
        self.showContours.clicked.connect(self._drawContours)
        #self.image_label.size(self.disply_width, self.display_height)
        vbox = QVBoxLayout()
        vbox.addWidget(self.showContours)
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def _drawContours(self):
        if self.showContours.isChecked():
            self.drawing = True
        else:
            self.drawing = False

    def _setPoints(self, circle, board):
        self.boardPoints = board
        self.circlePoints = circle

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        if self.drawing:
            #img = cv_img.copy()
            if self.boardPoints is not None:
                cv.drawContours(cv_img, [self.boardPoints], 0, (0, 255, 0), 5)
            if self.circlePoints is not None:
                for i in self.circlePoints[0,:]:
                    # draw the outer circle
                    cv.circle(cv_img,(i[0],i[1]),i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv.circle(cv_img,(i[0],i[1]),2,(0,0,255),3)
        cv_img = self.trackingBags(cv_img)
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def trackingBags(self, img):
        roi = img[340: 720,500: 800]
        return img
