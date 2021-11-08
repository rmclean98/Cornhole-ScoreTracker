import os
import torch
import numpy as np
import cv2 as cv
import pafy
def nothing():
    pass

def videoCap():
<<<<<<< HEAD
    weightfilepath = os.path.join("CornholeTrackerv4.pt")
=======
    weightfilepath = os.path.join("CornholeTrackerv5.pt")
>>>>>>> 434ce4608339a6e276258600a9d3672c523a168b
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weightfilepath)
    model = model.to(device)
<<<<<<< HEAD
    #filePath  = os.path.join("Images", "vid4.mp4")
    filePath  = os.path.join("Images", "IMG_7491.MOV")
=======
    filePath  = os.path.join("Images", "vid5.mp4")
>>>>>>> 434ce4608339a6e276258600a9d3672c523a168b
    cap = cv.VideoCapture(filePath)
    count = 0
    imgCount = 0
    cv.namedWindow("Game")
    cv.createTrackbar("pic", "Game", 0, 1, nothing)
    while True:
        count += 1
        ret, img = cap.read()
        cimg = img.copy()
        pimg = img.copy()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        takePic = cv.getTrackbarPos("pic", "Game")
        if count == 2000 or takePic == 1:
            cv.setTrackbarPos("pic", "Game", 0)
            imgCount += 1
            count = 0
            results = model(img)
            labelLines = []
            if not results.pandas().xyxy[0].empty:
                for index, row in results.pandas().xyxy[0].iterrows():
                    if row['confidence'] > .4:
                        x = round((row['xmax'] - ((row['xmax']-row['xmin'])/2))/img.shape[1], 6)
                        y = round((row['ymax'] - ((row['ymax']-row['ymin'])/2))/img.shape[0], 6)
                        height = round((row['ymax'] - row['ymin'])/img.shape[0], 6)
                        width = round((row['xmax'] - row['xmin'])/img.shape[1], 6)
                        if row['class'] == 0:
                            labelLines.append("1 " + str(x) + " " + str(y) + " " + str(width) + " " + str(height))
                        elif row['class'] == 1:
                            labelLines.append("0 " + str(x) + " " + str(y) + " " + str(width) + " " + str(height))
                        else:
                            labelLines.append("2 " + str(x) + " " + str(y) + " " + str(width) + " " + str(height))
                        start = (int(row['xmin']), int(row['ymin']))
                        end = (int(row['xmax']), int(row['ymax']))
                        strText = row['name'] + " - " + str(row['confidence'])
                        cv.rectangle(cimg, start, end, (255, 255, 255), 2)
                        cv.putText(cimg, strText, start, cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            print(labelLines)
            if len(labelLines) >= 0:
                vidName = os.path.split(filePath)
                fileName = str(imgCount) + str(vidName[1])
                fileNameimg  = os.path.join("labelImg", fileName)
                cv.imwrite(fileNameimg + '.png', pimg)
                fileNameLabel  = os.path.join("labeltxt", fileName)
                with open(fileNameLabel + '.txt', 'w') as f:
                    f.writelines('\n'.join(labelLines))
                f.close()
        cv.imshow("Game", cimg)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    videoCap()
