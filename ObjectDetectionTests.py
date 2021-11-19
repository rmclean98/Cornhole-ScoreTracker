import cv2 as cv
import numpy as np
import os
import torch
""""
ObjectDetectionTest plays a video on the screen and runs the model on every frame and displays the results.
If the user has cuda installed then cuda will be used, if not then the cpu will be used. This file is mainly
used to visually see the results of the model when its done training.
"""

weightfilepath = os.path.join("CornholeTrackerv7.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightfilepath)
model = model.to(device)
#model.conf = 0.40
#filePathimg  = os.path.join("Images", "vid1.mp4")
filePathimg  = os.path.join("Images", "IMG_7494.MOV")
tracker = cv.TrackerKCF_create()
cap = cv.VideoCapture(filePathimg)
#img = cv.imread(filePathimg)
while(True):
    ret, img = cap.read()
    """
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    image = cv.flip(resized, 0)
    resized = cv.flip(image, 1)
    img = resized
    """
    timer = cv.getTickCount()
    #cimg = img.copy()
    newx = img.shape[1]/640
    newy = img.shape[0]/640
    imgresized = cv.resize(img, (640,640), interpolation = cv.INTER_AREA)
    cimg = img.copy()
    img = cv.cvtColor(imgresized, cv.COLOR_BGR2RGB)
    results = model(img)
    if not results.pandas().xyxy[0].empty:
        for index, row in results.pandas().xyxy[0].iterrows():
            if row['confidence'] > .5:
                #bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                start = (int(row['xmin']*newx), int(row['ymin']*newy))
                end = (int(row['xmax']*newx), int(row['ymax']*newy))
                strText = row['name'] + " - " + str(row['confidence'])
                cv.rectangle(cimg, start, end, (255, 255, 255), 2)
                cv.putText(cimg, strText, start, cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    cv.putText(cimg, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv.imshow("game", cimg)
    key = cv.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
