import cv2
import time
import os
import mediapipe as mp
import HandTrackingModule as htm
import re

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
cTime = 0
pTime = 0
folderPath = "FingerImages"
imageList = os.listdir(folderPath)
imageList.sort()
print(imageList)
overlayList = []
for imPath in imageList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

while True:
    count = 0
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, 5):
            if lmList[tipIds[i]][2] < lmList[tipIds[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        img[0:211, 0:80] = overlayList[totalFingers-1]
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_DUPLEX, 10, (25, 0, 0), 8)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), 3, cv2.FONT_HERSHEY_DUPLEX, 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
