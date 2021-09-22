import cv2 as cv
import numpy as np

model = cv.dnn.readNetFromONNX('CornholeBBBestV1.onnx')
