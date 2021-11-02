import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def drawCustomLandmarks(image, results, holistic):
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    mp_drawing.draw_landmarks(image, results.face_landmarks, holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                     circle_radius=1))


def extractKeypoints(result):
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() \
        if result.face_landmarks else np.zeros(1404)
    return face


def collectData():
    DATA_PATH = os.path.join('Extracted_Values')
    actions = np.array(['left', 'right', 'straight'])
    numberOfSequences = 30  # TODO: Zwryfikowac to z ilosci nagran w dtasecie
    lengthOfSequence = 30  # Frames to be capture

    for action in actions:
        for sequence in range(numberOfSequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv.VideoCapture('testVideo/test_vid_1.mp4')
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through actions
        for action in actions:
            for sequence in range(numberOfSequences):
                for frameNumber in range(lengthOfSequence):

                    ret, frame = cap.read()  # Read the data frame

                    if not ret:  # Error handling
                        print("Can't receive frame!")
                        break
                    frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                                      interpolation=cv.INTER_LINEAR)  # Resize the image for better performance

                    image, results = mediapipeDetection(frame, holistic)
                    drawCustomLandmarks(image, results, mp_holistic)

                    cv.imshow('Video', image)
                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(sequence), str(frameNumber))
                    np.save(npyPath, keypoints)

            if cv.waitKey(20) == ord('q'):
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


def collectData2():
    actions = np.array(['left', 'right', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values')

    for action in actions:
        for sequence in os.listdir('Dataset2/' + action):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

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

                    image, results = mediapipeDetection(frame, holistic)
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                     circle_radius=1))  # Draw the landmarks on the image

                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(DATA_PATH, action, str(file), str(frameNumber))
                    np.save(npyPath, keypoints)

                    cv.imshow('Video', image)

                    if cv.waitKey(10) == ord('q'):
                        break
        cap.release()
        cv.destroyAllWindows()


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

    model = Sequential()  # Instance model

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(80, 1404)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])


def mediapipeDetection(image, model):
    """Performs face detection with use of mediapipe holistic model

     Parameters:
     image: Image on which the detection is performed
     model: Mediaholistic model

     Returns:
     image: Processed frame
     results: Landmarks

    """

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False  # Save a little bit of memory
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # Color conversion
    return image, results


if __name__ == '__main__':
    # collectData2()
    preprocessData()
