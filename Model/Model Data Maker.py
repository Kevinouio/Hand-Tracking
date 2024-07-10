import cv2
import numpy as np
import mediapipe as mp
import csv


def get_gestures():
    files = open('gestures.csv', 'r')
    gestures = csv.DictReader(files)
    return gestures

#Defining video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640*1.5)
cap.set(4, 480*1.5)

gesture_name = "Fist"
file = open(f"{gesture_name}_data.txt", "a")  # Append mode

#Defines the drawing each hand connection and making the hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_Landmarks(landmarks):
    return [coord for landmark in landmarks.landmark for coord in (landmark.x, landmark.y)]


while True:
    ret, frame = cap.read()
    if ret:
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = get_Landmarks(hand_landmarks)
                file.write(",".join(map(str, landmarks)) + f",{gesture_name}\n")
                print(",".join(map(str, landmarks)) + f",{gesture_name}\n")




        cv2.imshow('frame', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
file.close()