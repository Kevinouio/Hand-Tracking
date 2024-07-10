import cv2
import numpy as np
import mediapipe as mp
import joblib
import csv




#Reads in the gestures that are possible that the model has trained with
def get_gestures():
    files = open('gestures.csv', 'r')
    gestures = csv.DictReader(files)
    return gestures

#gets the data points for each landmark in the hand
def get_Landmarks(landmarks):
    return [coord for landmark in landmarks.landmark for coord in (landmark.x, landmark.y)]

def get_bounds_of_hands(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_gesture(image, prediction, x, y):
    cv2.putText(image, "Gesture: " + str(prediction), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



#draws a box around the hand
def draw_box(img, brect):
    cv2.rectangle(img, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 255), 1)


# Load the model and scaler
model = joblib.load('../Model/gesture_model.pkl')
scaler = joblib.load('../Model/scaler.pkl')

# Save the trained model and scaler
joblib.dump(model, '../Model/gesture_model.pkl')
joblib.dump(scaler, '../Model/scaler.pkl')

#Gets the list of gestures the program can detect
gestures = get_gestures()

#Defining video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640 * 1.5)
cap.set(4, 480 * 1.5)

#Defines the drawing each hand connection and making the hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#Loads in the hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if ret:
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Check if all 21 landmarks are present for this hand
                if len(hand_landmarks.landmark) == 21:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = get_Landmarks(hand_landmarks)

                    # Convert list to numpy array
                    landmarks_array = np.array(landmarks)

                    # Bounding box calculation and draws the box around the hand
                    brect = get_bounds_of_hands(frame, hand_landmarks)
                    landmarks_2d = landmarks_array.reshape(1, -1)
                    prediction = model.predict(scaler.transform(landmarks_2d))

                    draw_box(frame, brect)
                    draw_gesture(frame, prediction, brect[0], brect[1])






        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
