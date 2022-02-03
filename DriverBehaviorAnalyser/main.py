import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import shutil

#from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.layers.wrappers import TimeDistributed
from keras.utils.vis_utils import plot_model

import DriverAnalyser
import Model
import DriverPrediction


def probabilityBar(res, actions, frame):
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    out_frame = frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(out_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv.putText(out_frame, actions[num], (0, 85 + num * 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                   cv.LINE_AA)
    return out_frame


def makePredictionOnWebcam():
    # TODO: Poprawić implementacje Real-Time. Caly czs wyrzuca rezultat że jade w lewo nie jadąc w lewo

    sequence = []
    actions = np.array(['left', 'right', 'straight'])
    mp_holistic = mp.solutions.holistic

    model = Model.createModel('Adamax', actions)  # New LSTM model
    model.load_weights('action.h5')  # Load weights into the network

    cap = cv.VideoCapture(0)  # Capture the video from a webcam

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = DriverAnalyser.faceFeaturesDetection(frame, holistic)  # Detect face landmarks of a driver
            DriverAnalyser.drawCustomLandmarks(image, results, mp_holistic)  # Customize the landmarks

            # Prediction logic
            keypoints = DriverAnalyser.extractKeypoints(results)  # Extract the keypoints of the face
            sequence.insert(0, keypoints)
            sequence = sequence[:80]  # Load last 150 values

            if len(sequence) == 80:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            image = probabilityBar(res, actions, image)

            cv.imshow('Image', image)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()


def makePredictions(testVideo):
    sequence = []
    actions = np.array(['left', 'right', 'straight'])

    mp_holistic = mp.solutions.holistic  # Holistic model

    model = Model.createModel('Adamax', actions)  # New LSTM model
    model.load_weights('action.h5')  # Load weights into the network

    cap = cv.VideoCapture(testVideo)

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()  # Read the data frame

            if not ret:  # Error handling
                print("Can't receive frame!")
                break

            image, results = DriverAnalyser.faceFeaturesDetection(frame, holistic)
            DriverAnalyser.drawCustomLandmarks(image, results, mp_holistic)

            # Prediction logic
            keypoints = DriverAnalyser.extractKeypoints(results)
            sequence.append(keypoints)
            sequence = sequence[:150]

            if len(sequence) == 150:
                # res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            image = probabilityBar(res, actions, image)

            cv.imshow('Image', image)
            cv.waitKey(10)
    cap.release()
    cv.destroyAllWindows()


def extractAvi():
    """
    Extraction algorithm which extracts .avi videos from the raw dataset
    """
    actions = np.array(['rturn', 'rchange', 'lturn', 'lchange', 'straight'])
    os.chdir('Full_Dataset')
    for action in actions:
        dest = '/Users/kacpe/OneDrive/Pulpit/DBA/DriverBehaviorAnalyser/Extracted_Videos_Avi2/' + str(action)
        os.chdir(action)
        for folder in os.listdir():
            os.chdir(folder)
            for file in os.listdir():
                if file == 'Extracted_Videos_Avi':
                    pass
                else:
                    shutil.copy(file, dest)
            os.chdir('..')
        os.chdir('..')


if __name__ == '__main__':
    actions = np.array(['lchange', 'rchange', 'straight'])

    model = Sequential()  # Model instance

    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(150, 1434)))
    model.add(TimeDistributed(Dropout(0.1)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(TimeDistributed(Dropout(0.1)))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.summary()
    plot_model(model, to_file='test.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

    # makePredictions('test_vid_left.avi')
    # makePredictionOnWebcam()
    # extractAvi()
    # DriverAnalyser.collectDataWithAccurateFace()
    # DriverAnalyser.collectData2()
    # DriverPrediction.preprocessData2()
