import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

import DriverAnalyser
import Model
import DriverPrediction


def preprocessData():
    actions = np.array(['left', 'right', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values')
    label_map = {label: num for num, label in enumerate(actions)}
    sequenceLength = 80  # Reduce to the 80 frames due to the different video length

    files, labels = [], []
    for action in actions:
        for file in os.listdir('Dataset2/' + action):
            window = []
            for frame in range(sequenceLength):
                res = np.load(os.path.join(DATA_PATH, action, str(file), "{}.npy".format(frame)))
                window.append(res)
            files.append(window)
            labels.append(label_map[action])

    X = np.array(files)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    logDir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=logDir)

    model = Sequential()  # Model instance

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(80, 1404)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

    # Save Weights
    model.save('action.h5')
    model.load_weights('action.h5')

    yhat = model.predict(X_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(accuracy_score(ytrue, yhat))


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
            keypoints = DriverAnalyser.extractKeypoints(results)    # Extract the keypoints of the face
            sequence.insert(0, keypoints)
            sequence = sequence[:80]   # Load last 150 values

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
    # TODO: Opreacja Real-Time
    # TODO: Zoptymalizować model
    # TODO: Implementacja facemesh
    # TODO: Dodanie refine landmakrs

     makePredictions('test_vid_left.avi')
    #makePredictionOnWebcam()
    # extractAvi()
    # DriverAnalyser.collectData2()
    # DriverPrediction.preprocessData2()
