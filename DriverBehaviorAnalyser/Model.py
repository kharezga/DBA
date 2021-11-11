from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def createModel(optimizer, actions):
    """Creates LSTM model
     Parameters:
     optimizer: model optimizer
     actions: np array with defined labels

     Returns:
     model: LSTM model
    """

    if optimizer is None:
        optimizer = 'Adamax'

    model = Sequential()  # Model instance

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(80, 1404)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
