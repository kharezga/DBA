import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import mediapipe as mp


def faceDetection():
    cap = cv.VideoCapture('testVideo/test_vid_1.mp4')
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()  # Read the data frame

            if not ret:  # Error handling
                print("Can't receive frame!")
                break
            frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                              interpolation=cv.INTER_LINEAR)  # Resize the image for better performance

            image, results = mediapipe_detection(frame, holistic)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                             circle_radius=1))  # Draw the landmarks on the image

            cv.imshow('Video', image)

            if cv.waitKey(20) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()


def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False  # Save a little bit of memory
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Color conversion
    return image, results


if __name__ == '__main__':
    faceDetection()
