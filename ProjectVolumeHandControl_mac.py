import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
import subprocess  # for mac

# ______________
w_cam, h_cam = 640, 480
# ______________

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

previous_time = 0

detector = htm.HandDetector(detection_confidence=0.7)

vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x1 - x2, y1 - y2)

        # # Hand range 50 - 300
        # # Volume range 0 - 100 for mac
        # vol = np.interp(length, [50, 300], [0, 100])  # for mac
        # vol_bar = np.interp(length, [50, 300], [400, 150])
        # vol_per = np.interp(length, [50, 300], [0, 100])  # percent
        # print(int(length), vol)

        # Hand range 50 - 200
        # Volume range 0 - 100 for mac
        vol = np.interp(length, [50, 200], [0, 100])  # for mac
        vol_bar = np.interp(length, [50, 200], [400, 150])
        vol_per = np.interp(length, [50, 200], [0, 100])  # percent
        print(int(length), vol)

        # Call AppleScript to change the volume
        script = f'set volume output volume {vol}'
        subprocess.run(["osascript", "-e", script])

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{int(vol_per)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0), 1)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
