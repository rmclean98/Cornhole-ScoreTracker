import cv2 as cv
import numpy as np
import os
import torch

weightfilepath = os.path.join("best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightfilepath)
model.classes = [0]
model = model.to(device)
#model.conf = 0.40
filePathimg  = os.path.join("Images", "vid3.mp4")
cap = cv.VideoCapture(filePathimg)
#img = cv.imread(filePathimg)
while(True):
    ret, img = cap.read()
    cimg = img.copy()
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = model(img)
    #results.print()
    if not results.pandas().xyxy[0].empty:
        for index, row in results.pandas().xyxy[0].iterrows():
            if row['confidence'] > .5:
                start = (int(row['xmin']), int(row['ymin']))
                end = (int(row['xmax']), int(row['ymax']))
                strText = row['name'] + " - " + str(row['confidence'])
                cv.rectangle(cimg, start, end, (255, 255, 255), 2)
                cv.putText(cimg, strText, start, cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv.imshow("game", cimg)
    key = cv.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
"""
results.print()
results.show()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]
"""
