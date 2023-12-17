import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folder_path = 'Header'
my_list = os.listdir(folder_path)
print(my_list)

# Сортуємо список за номерами файлів (враховуючи лише файли з розширенням .png)
sorted_list = sorted([file for file in my_list if file.endswith(".jpg")])
print(sorted_list)

over_lay_list = []

for im_path in sorted_list:
    image = cv2.imread(f"{folder_path}/{im_path}")
    over_lay_list.append(image)

print(len(over_lay_list))

header = over_lay_list[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detection_confidence=0.85)


while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Yanh Landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        # print(lm_list)

        # tip of inde[ and middle fingers
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)

        # 4. If Selections Mode - Two finger are up
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv2.FILLED)
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = over_lay_list[0]
                elif 550 < x1 < 750:
                    header = over_lay_list[1]
                elif 800 < x1 < 950:
                    header = over_lay_list[2]
                elif 1050 < x1 < 1200:
                    header = over_lay_list[3]

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 25, (255, 0, 255), cv2.FILLED)
            print("Drawing Mode")

    # Setting the header image
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)


