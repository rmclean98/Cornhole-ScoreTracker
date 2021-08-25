import numpy as np
import cv2 as cv
import CornholeScoreKeeper
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPalette, QPixmap, QImage, QCursor
from PyQt5.QtWidgets import *
import sys
import os

rect = []
circle = []
circleBool = False

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def findCircle(self, img):
        global circleBool
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_blurred = cv.blur(gray, (3, 3))
        detected_circles = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, 20,
        param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv.circle(img, (a, b), r, (0, 255, 0), 2)
            cv.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv.imshow("Detected Circle", img)
            cv.waitKey(0)

    def run(self):
        global circleBool
        # capture from web cam
        cap = cv.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if(circleBool):
                self.findCircle(cv_img)
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
        global rect, findCircle
        if self.drawCircle.isChecked() & self.drawRect.isChecked():
            self.drawRect.setChecked(False)
            self.drawCircle.setChecked(False)
        if(self.drawCircle.isChecked() == False):
            circleBool = True


    def mousePressEvent(self, QMouseEvent):
        if self.drawCircle.isChecked():
            print(self.QMouseEvent.globalPos())
            circle.append(self.QMouseEvent.globalPos())
