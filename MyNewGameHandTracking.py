import cv2
import mediapipe as mp
import time
import HandTrackingMin as htm


cap = cv2.VideoCapture(0)
previous_time = 0
current_time = 0
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)  # draw=False
    lm_list = detector.find_position(img)  # draw=False
    if len(lm_list) != 0:
        print(lm_list[4])

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
