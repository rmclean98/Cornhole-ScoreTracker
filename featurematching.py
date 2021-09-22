import cv2 as cv
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

filePath = os.path.join("Images", "vid1.mp4")
cap = cv.VideoCapture(filePath)
"""
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

while(True):
    ret, img = cap.read()
    imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    mask = cv.inRange(imgHSV,lowerBound,upperBound)

    maskOpen = cv.morphologyEx(mask,cv.MORPH_OPEN,kernelOpen)
    maskClose = cv.morphologyEx(maskOpen,cv.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose
    conts,h=cv.findContours(maskFinal.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    cv.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv.boundingRect(conts[i])
        cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv.putText(img, str(i+1),(x,y+h),cv.FONT_HERSHEY_SIMPLEX,0.75, (0,255,255), 2)
    cv.imshow("mask",mask)
    cv.imshow("cam",img)
    cv.imshow("maskClose",maskClose)
    cv.imshow("maskOpen",maskOpen)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


tracker = cv.TrackerGOTURN_create()

if not cap.isOpened():
    print("Could not open video")
    sys.exit()
ok,frame = cap.read()

if not ok:
    print("Cannot read video file")
    sys.exit()

bbox = (59, 33, 360, 617)
ok = tracker.init(frame,bbox)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    timer = cv.getTickCount()
    ok, bbox = tracker.update(frame)
    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    cv.putText(frame, "GOTURN Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    cv.imshow("Tracking", frame)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break


"""
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
back_sub = cv.createBackgroundSubtractorMOG2(history=50,
        varThreshold=25, detectShadows=True)

    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
kernel = np.ones((30,30),np.uint8)
kernelOpen=np.ones((5,5))
lowerBound1=np.array([33,80,40])
upperBound1=np.array([102,255,255])
lowerBound2=np.array([190,170,120])
upperBound2=np.array([140,115,40])
while(True):

        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
    ret, frame = cap.read()
    timer = cv.getTickCount()
    mask = np.zeros_like(frame)
    board = np.array([[59, 33], [369, 45], [360, 617], [74, 617]])
    cv.fillPoly(mask, pts = [board], color =(255,255,255))
    masked = cv.bitwise_and(frame, mask)
    #cv.imshow("masked", masked)
    imgHSV = cv.cvtColor(masked,cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(imgHSV,lowerBound1,upperBound1)
    mask2 = cv.inRange(imgHSV,lowerBound2,upperBound2)

        # Use every frame to calculate the foreground mask and update
        # the background
    fg_mask1 = back_sub.apply(mask1)
    fg_mask2 = back_sub.apply(mask2)

        # Close dark gaps in foreground object using closing
    #fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
    maskOpen1 = cv.morphologyEx(fg_mask1,cv.MORPH_OPEN,kernelOpen)
    fg_mask1 = cv.morphologyEx(maskOpen1,cv.MORPH_CLOSE,kernel)
    maskOpen2 = cv.morphologyEx(fg_mask2,cv.MORPH_OPEN,kernelOpen)
    fg_mask2 = cv.morphologyEx(maskOpen2,cv.MORPH_CLOSE,kernel)

        # Remove salt and pepper noise with a median filter
    fg_mask1 = cv.medianBlur(fg_mask1, 5)
    cv.imshow('mask with blur1', fg_mask1)
    fg_mask2 = cv.medianBlur(fg_mask2, 5)
    cv.imshow('mask with blur2', fg_mask2)
        # Threshold the image to make it either black or white
    _, fg_mask1 = cv.threshold(fg_mask1,127,255,cv.THRESH_BINARY)
    _, fg_mask2 = cv.threshold(fg_mask2,127,255,cv.THRESH_BINARY)


        # Find the index of the largest contour and draw bounding box
    fg_mask_bb1 = fg_mask1
    fg_mask_bb2 = fg_mask2
    contours1, hierarchy1 = cv.findContours(fg_mask_bb1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2:]
    contours2, hierarchy2 = cv.findContours(fg_mask_bb2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2:]
    areas1 = [cv.contourArea(c) for c in contours1]
    areas2 = [cv.contourArea(c) for c in contours2]

        # If there are no countours
    if len(areas1) < 1 and len(areas2):
        cv.imshow('frame',frame)

    elif len(areas1) > 0:
        if np.argmax(areas1) < 5000:
            max_index1 = np.argmax(areas1)
        else:
            max_index1 = (np.abs(np.asarray(areas1) - 4000)).argmin()
        cnt1 = contours1[max_index1]
        cv.drawContours(frame,cnt1,-1,(255,0,0),3)

    elif len(areas2) > 0:
        if np.argmax(areas2) < 5000:
            max_index2 = np.argmax(areas2)
        else:
            max_index2 = (np.abs(np.asarray(areas2) - 4000)).argmin()
        cnt2 = contours2[max_index2]
        cv.drawContours(frame,cnt2,-1,(255,0,0),3)

    fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
    cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        # Draw the bounding box
    cv.imshow('frame',frame)

        # If "q" is pressed on the keyboard,
        # exit this loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # Close down the video stream
cap.release()
cv.destroyAllWindows()