import cv2  # OpenCV is used to work with the video stream and image processing.
import mediapipe as mp  # Mediapipe is used to detect and track key points on the hands.
import time

# Video camera initialization:
# Create a VideoCapture object to read a video stream from a webcam (0 points to the first available camera).
cap = cv2.VideoCapture(0)

# Initializing Mediapipe for hand detection:
# Creating objects for working with hands using Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initializing variables for time tracking:
# Variables for calculating frames per second (FPS).
previous_time = 0
current_time = 0

# Endless loop for video stream processing:
while True:
    # A loop that endlessly reads frames from a webcam.
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts the BGR color space to RGB
    results = hands.process(img_rgb)  # Call the process method to detect hands in an image.
    # print(results.multi_hand_landmarks)

    # Processing hand detection results:
    # Checks if hand detection results are available.
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lms.landmark):
                # print(id, lm)
                height, weight, chanel = img.shape
                chanel_x, chanel_y = int(lm.x * weight), int(lm.y * height)
                print(id, chanel_x, chanel_y)
                # If so, the index and coordinates of each key point are displayed for each hand.
                # A circle is drawn on the first key point (with index 0) of the hand.
                if id == 0:
                    cv2.circle(img, (chanel_x, chanel_y), 25, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    # Calculate and display frames per second in the image.
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # Image display:
    # Displays an image with tracked hands and other information.
    # cv2.waitKey (1) - waits for pressing the delay key for correct operation of the display window.
    cv2.imshow("Image", img)
    cv2.waitKey(1)
