from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
import os

def preprocessData2():
    actions = np.array(['lchange', 'rchange', 'straight'])
    DATA_PATH = os.path.join('Extracted_Values_Sorted')
    label_map = {label: num for num, label in enumerate(actions)}
    sequenceLength = 150  # Reduce to the 80 frames due to the different video length

    files, labels = [], []
    for action in actions:
        for file in os.listdir('Extracted_Values_Sorted/' + action):
            window = []
            for frame in range(sequenceLength):
                res = np.load(os.path.join(DATA_PATH, action, str(file), "{}.npy".format(frame)))
                window.append(res)
            files.append(window)
            labels.append(label_map[action])

    print(np.array(files).shape)
    X = np.array(files)
    y = to_categorical(labels).astype(int) # converts initial label to the code representation

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) # 5% of our dataset is separated

    logDir = os.path.join('Logs_Sorted2')
    tb_callback = TensorBoard(log_dir=logDir)

    model = Sequential()  # Model instance

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(150, 1404)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, 98-y_train, epochs=1500, callbacks=[tb_callback])

    # Save Weights
    model.save('action_sorted2.h5')
    model.load_weights('action_sorted2.h5')

    yhat = model.predict(X_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(accuracy_score(ytrue, yhat))
    print("Confusion Matrix")
    print(multilabel_confusion_matrix(ytrue, yhat))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$TestDATA$$$$$$$$$$$")

    yhat2 = model.predict(X_test)
    ytrue2 = np.argmax(y_test, axis=1).tolist()
    yhat2 = np.argmax(yhat2, axis=1).tolist()

    print(accuracy_score(ytrue2, yhat2))
    print(multilabel_confusion_matrix(ytrue2, yhat2))



