import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import subprocess

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cTime = 0
pTime = 0
smooth = 20
plocX, plocY = 0, 0
clocX, clocY = 0, 0
# Screen wScr and hScr
cmd = ['xrandr']
cmd2 = ['grep', '*']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
p.stdout.close()
resolution_string, junk = p2.communicate()
resolution = resolution_string.split()[0]
resolution = resolution.decode("utf-8")
print(resolution)
wScr, hScr = resolution.split('x')
hScr = int(hScr)
wScr = int(wScr)
print(type(wScr))
# wScr = float(wScr)
# hScr = float(hScr)
# Frame Reduction
frameR = 100
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
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 255, 255), 2)

        # 4. Only index finger : Moving mode...
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert coordinates...
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen values...
            clocX = plocX + (x3 - plocX) / smooth
            clocY = plocY + (y3 - plocY) / smooth

            # 7. Move mouse...
            autopy.mouse.move(int(clocX), int(clocY))
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX = clocX
            plocY = clocY

    # 8. Both index and middle fingers are up : Clicking mode...
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find distance between fingers...
            length, img, info = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (info[4], info[5]), 10, (0, 255, 0), cv2.FILLED)

            # 10. Click mouse if distance short...
            autopy.mouse.click()

    # 11. Frame rate...
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    # 12. Display...
    cv2.imshow("Image", img)
    cv2.waitKey(1)
