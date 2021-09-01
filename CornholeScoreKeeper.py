import numpy as np
import cv2 as cv
import Camera
from Calibrate import *
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import *
import sys
import os

img = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cornhole Score Tracker')
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self.generalLayout = QGridLayout()
        self._centralWidget.setLayout(self.generalLayout)
        self._createPlayers()
        self._createButtons()
        self.circlePoints = []
        self.rectPoints = []
        self.boardPoints = None
        self.cameraWindow = Camera.Camera()
        #self.cameraWindow.show()

    def _createPlayers(self):
        playersLayout = QGridLayout()
        self.playersLabel = QLabel('Players:')
        self.scoreLabel = QLabel('Score:')
        self.player1 = QLineEdit()
        self.player2 = QLineEdit()
        self.score1 = QLineEdit()
        self.score2 = QLineEdit()
        self.player1.setFixedHeight(25)
        #self.player1.setFixedWidth(100)
        self.score1.setFixedHeight(25)
        self.score1.setFixedWidth(25)
        self.score1.setText('0')
        self.score1.setReadOnly(True)
        self.score1.setAlignment(Qt.AlignCenter)
        self.player2.setFixedHeight(25)
        #self.player2.setFixedWidth(100)
        self.score2.setFixedHeight(25)
        self.score2.setFixedWidth(25)
        self.score2.setText('0')
        self.score1.setReadOnly(True)
        self.score2.setAlignment(Qt.AlignCenter)
        playersLayout.addWidget(self.playersLabel, 0, 2)
        playersLayout.addWidget(self.scoreLabel, 0, 3)
        playersLayout.addWidget(self.player1, 1, 2)
        playersLayout.addWidget(self.score1, 1, 3)
        playersLayout.addWidget(self.player2, 2, 2)
        playersLayout.addWidget(self.score2, 2, 3)
        self.generalLayout.addLayout(playersLayout, 0, 1)

    def _createButtons(self):
        buttonsLayout = QVBoxLayout()
        self.showHideButton = QPushButton('Show/Hide Camera')
        self.calibrateButton = QPushButton('Calibrate')
        self.showHideButton.clicked.connect(self._showCamera)
        self.calibrateButton.clicked.connect(self._calibrate)
        buttonsLayout.addWidget(self.showHideButton)
        buttonsLayout.addWidget(self.calibrateButton)
        self.generalLayout.addLayout(buttonsLayout, 0, 0)


    def _showCamera(self):
        if self.cameraWindow.isVisible():
            self.cameraWindow.hide()
        else:
            self.cameraWindow.show()

    def _calibrate(self):
        calibrate = Calibrate()
        self.circlePoints = calibrate._getCirclePoints()
        print("circle Point")
        print(self.circlePoints)
        self.rectPoints = calibrate._getRectPoints()
        print("Rectangle Points")
        print(self.rectPoints)
        self.boardPoints = calibrate._getContour()
        print("contour:")
        print(self.boardPoints)
        self.cameraWindow._setPoints(self.circlePoints, self.boardPoints)



if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Fusion')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
