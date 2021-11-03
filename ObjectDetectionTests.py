import cv2 as cv
import numpy as np
import os
import torch

def detectBoxes(img, model):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = model(img)
    return results
weightfilepath = os.path.join("best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightfilepath)
model.classes = [0]
model = model.to(device)
#model.conf = 0.40
<<<<<<< HEAD
filePathimg  = os.path.join("Images", "vid3.mp4")
=======
filePathimg  = os.path.join("Images", "vid1.mp4")
tracker = cv.TrackerKCF_create()
>>>>>>> 34dddb0938ea9f61a394135e2f51085c5ea63021
cap = cv.VideoCapture(filePathimg)
#img = cv.imread(filePathimg)
bbox = [287, 23, 86, 320]
count = 50
while(True):
    ret, img = cap.read()
    timer = cv.getTickCount()
    cimg = img.copy()
<<<<<<< HEAD
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = model(img)
=======
    if count == 50:
        results = detectBoxes(cimg, model)
        count = 0
>>>>>>> 34dddb0938ea9f61a394135e2f51085c5ea63021
    #results.print()
    if not results.pandas().xyxy[0].empty:
        for index, row in results.pandas().xyxy[0].iterrows():
            if row['confidence'] > .5:
                #bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                start = (int(row['xmin']), int(row['ymin']))
                end = (int(row['xmax']), int(row['ymax']))
                strText = row['name'] + " - " + str(row['confidence'])
                cv.rectangle(cimg, start, end, (255, 255, 255), 2)
                cv.putText(cimg, strText, start, cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    ok, bbox = tracker.update(cimg)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(cimg, p1, p2, (255,0,0), 2, 1)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    cv.putText(cimg, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    cv.imshow("game", cimg)
    count += 1
    key = cv.waitKey(10)
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
