from keras.layers import *
from keras.models import Sequential, model_from_json, load_model


def load_model(self, filepath):
    print('[Model] Loading model from file %s' % filepath)
    self.model = load_model(filepath)


def lstm(seq_len, input_dim):
    model = Sequential()
    if input_dim is None:
        print("error: input_dim is None! model build failed!")
        return

    model.add(CuDNNLSTM(256, return_sequences=True,
                        input_shape=(seq_len, input_dim)))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(CuDNNLSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model_json = model.to_json()
    with open("lstm%d.json" % seq_len, "w") as json_file:
        json_file.write(model_json)

    return model


def dnn(seq_len, input_dim):

    model = Sequential([
        Flatten(input_shape=(seq_len, input_dim)),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model_json = model.to_json()
    with open("dnn%d.json" % seq_len, "w") as json_file:
        json_file.write(model_json)

    return model


def cnn(seq_len, input_dim):
    model = Sequential([
        Conv1D(256, kernel_size=3, strides=1, padding='valid',
               activation='relu', input_shape=(seq_len, input_dim)),
        BatchNormalization(),
        Conv1D(256, kernel_size=3, strides=1, padding='valid',
               activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1),
    ])

    model_json = model.to_json()
    with open("cnn%d.json" % seq_len, "w") as json_file:
        json_file.write(model_json)

    return model


def build_from_json(json_file):
    # load json and create model
    json_file = open(json_file, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model


if __name__ == '__main__':

    model = dnn(10, 135)
    model.summary()