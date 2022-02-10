import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cTime = 0
pTime = 0
wScr, hScr = autopy.screen.size()
print(wScr)
detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find hand landmarks...
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get tip if index and middle fingers...
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which finger is up...
        fingers = detector.fingersUp()
        print(fingers)

        # 4. Only index finger : Moving mode...
        # if(fingers[1]==1 and fingers[2]==0):
        #
        #     # 5. Convert coordinates...
        #     x3 = np.interp(x1, (0, wCam), (0, wScr))
        #     y3 = np.interp(y1, (0, hCam), (0, hScr))

    # 6. Smoothen values...
    # 7. Move mouse...
    # autopy.mouse.move(wScr-x3, y3)
    # 8. Both index and middle fingers are up : Clicking mode...
    # 9. Find distance between fingers...
    # 10. Click mouse if distance short...
    # 11. Frame rate...
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    # 12. Display...
    cv2.imshow("Image", img)
    cv2.waitKey(1)
