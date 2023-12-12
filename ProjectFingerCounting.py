import cv2
import time
import os
import HandTrackingModule as htm

# ______________
# w_cam, h_cam = 640, 480
w_cam, h_cam = 840, 680
# ______________

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

folder_path = "finger_images_right"
my_list = os.listdir(folder_path)
print(my_list)

# Сортуємо список за номерами файлів (враховуючи лише файли з розширенням .png)
sorted_list = sorted([file for file in my_list if file.endswith(".jpeg")])
print(sorted_list)

over_lay_list = []

for img_path in sorted_list:
    image = cv2.imread(f"{folder_path}/{img_path}")
    # print(f"{folder_path}/{img_path}")
    over_lay_list.append(image)

print(len(over_lay_list))

previous_time = 0

detector = htm.HandDetector(detection_confidence=0.75)

tip_index = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, False)
    # print(lm_list)

    if len(lm_list) != 0:
        fingers = []

        # Thumb
        if lm_list[tip_index[0]][1] > lm_list[tip_index[0] - 1][1]:
            # print("Index finger open")
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lm_list[tip_index[id]][2] < lm_list[tip_index[id] - 2][2]:
                # print("Index finger open")
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)

        height, weight, chanel = over_lay_list[total_fingers-1].shape
        img[0:height, 0:weight] = over_lay_list[total_fingers-1]

    # img[100:420, 100:280] = over_lay_list[0]  # screen offset proportional to size
    # img[0:320, 0:180] = over_lay_list[0]

    # height, weight, chanel = over_lay_list[0].shape
    # img[0:height, 0:weight] = over_lay_list[0]

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f"FPS: {int(fps)}", (400, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
