import numpy as np
import cv2 as cv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import *
import sys
import os

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
        self.score1.setAlignment(Qt.AlignCenter)
        self.player2.setFixedHeight(25)
        #self.player2.setFixedWidth(100)
        self.score2.setFixedHeight(25)
        self.score2.setFixedWidth(25)
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
        self.calButton = QPushButton('Calabreate Camera')
        self.scanBagsButton = QPushButton('Scan Bags')
        self.calButton.clicked.connect(self._showCamera)
        buttonsLayout.addWidget(self.calButton)
        buttonsLayout.addWidget(self.scanBagsButton)
        self.generalLayout.addLayout(buttonsLayout, 0, 0)

    def _showCamera(self):
        vid = cv.VideoCapture(0)

        while(True):
            ret, frame = vid.read()
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()



if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
