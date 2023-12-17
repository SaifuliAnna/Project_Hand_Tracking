import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode  # still image mode on (True) or off (False)
        self.max_hands = max_hands  # maximum number of hands a model can detect in one frame
        # complexity of the model. This is usually a number from 0 to 2, where 0 - smallest complexity and 2 - largest
        self.model_complexity = model_complexity
        #  Specifies the trust threshold for hand detection.
        #  Hands that are detected with less confidence can be discarded
        self.detection_confidence = detection_confidence
        #  Specifies the trust threshold for hand tracking.
        #  Hands that cannot be tracked with sufficient confidence may be lost.
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_confidence,
                                        self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_index = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        - Converts the color format of a BGR image to RGB.
        - Calls the process method for processing hands on an image and stores the results
          in the results attribute.
        - If draw is set to True, the method also draws key points and the lines connecting them.
        :param img:
        :param draw:
        :return: img
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number=0, draw=True):
        """
        - Returns a list of coordinates of the key points of the hand (lm_list).
        - Each point is represented as [id, chanel_x, chanel_y], where id is the index of the point,
          chanel_x and chanel_y are the coordinates of the point in the image.
        - If draw is set to True, then the method also draws circles in the image at key point locations.
        :param img:
        :param hand_number:
        :param draw:
        :return: lm_list
        """
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                height, weight, chanel = img.shape
                chanel_x, chanel_y = int(lm.x * weight), int(lm.y * height)
                self.lm_list.append([id, chanel_x, chanel_y])
                if draw:
                    cv2.circle(img, (chanel_x, chanel_y), 15, (255, 0, 255), cv2.FILLED)

        return self.lm_list

    def fingers_up(self):
        fingers = []

        # Thumb
        if self.lm_list[self.tip_index[0]][1] < self.lm_list[self.tip_index[0] - 1][1]:
            # print("Index finger open")
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_index[id]][2] < self.lm_list[self.tip_index[id] - 2][2]:
                # print("Index finger open")
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list) != 0:
            print(lm_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
