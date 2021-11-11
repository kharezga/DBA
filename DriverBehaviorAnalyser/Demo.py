# This file contains all demo functions
import numpy as np
import mediapipe as mp
import cv2 as cv
import os


def faceRecognition():
    actions = np.array(['left', 'right', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values')

    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        for action in actions:
            for file in os.listdir('Dataset2/' + action):
                cap = cv.VideoCapture('Dataset2/' + action + "/" + file)
                frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Count the number of frames in each video
                for frameNumber in range(frameCount):

                    ret, frame = cap.read()  # Read the data frame

                    if not ret:  # Error handling
                        print("Can't receive frame!")
                        break

                    frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                                      interpolation=cv.INTER_LINEAR)  # Resize the image for better performance

                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Color conversion
                    image.flags.writeable = False  # Save a little bit of memory
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Color conversion

                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                     circle_radius=1))  # Draw the landmarks on the image

                    cv.imshow('Video', image)

                    if cv.waitKey(10) == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()

def createDirectories(actions, numberOfSequences, pathData):
    for action in actions:
        for sequence in range(numberOfSequences):
            try:
                os.makedirs(os.path.join(pathData, action, str(sequence)))
            except:
                pass