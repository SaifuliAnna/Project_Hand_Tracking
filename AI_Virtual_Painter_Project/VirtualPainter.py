import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


# _______________________________

brush_thickness = 15
eraser_thickness = 200

# _______________________________


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
draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detection_confidence=0.85)
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

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
            xp, yp = 0, 0
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = over_lay_list[0]
                    draw_color = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = over_lay_list[1]
                    draw_color = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = over_lay_list[2]
                    draw_color = (255, 255, 0)
                elif 1050 < x1 < 1200:
                    header = over_lay_list[3]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

        # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 25, draw_color, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

            cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

            xp, yp = x1, y1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Image Canvas", img_canvas)
    cv2.imshow("Inv", img_inv)
    cv2.waitKey(1)


