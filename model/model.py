import os
import numpy as np
from numpy import newaxis
from model.utils import Timer
from keras.layers import Dense, Dropout, LSTM, Input
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class Model:

    def __init__(self, configs=None):
        self.model = Sequential()
        self.configs = configs

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self):
        timer = Timer()
        timer.start()
        input_timesteps = self.configs[
            'input_timesteps'] if 'input_timesteps' in self.configs else None
        input_dim = self.configs['input_dim'] if 'input_dim' in self.configs else None
        if input_dim is None:
            print("error: input_dim is None! model build failed!")
            return
        elif input_timesteps is None:
            self.model.add(Input(shape=(input_dim, )))
        else:
            self.model.add(Input(shape=(input_timesteps, input_dim)))

        for layer in self.configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None

            if layer['type'] == 'lstm':
                self.model.add(
                    LSTM(neurons, return_sequences=return_seq))

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))

            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=self.configs['model']['loss'],
                           optimizer=self.configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir, val_split):

        name = self.configs['name'] if 'name' in self.configs else 'Model'
        print('[%s] Training Started' % name)
        print('[%s] %s epochs, %s batch size' % (name, epochs, batch_size))

        save_fname = os.path.join(save_dir,
                                  '%s_{epoch:02d}-{val_acc:.2f}.hdf5' % name)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss',
                            save_best_only=True),
            TensorBoard()
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=val_split,
        )

    def train_generator(self, data_gen, val_data, epochs, batch_size, steps_per_epoch,
                        save_dir):
        name = self.configs['name'] if 'name' in self.configs else 'Model'
        print('[%s] Training Started' % name)

        print('[%s] %s epochs, %s batch size, %s batches per epoch' % (name,
              epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir,
                                  '%s_{epoch:02d}-{val_acc:.2f}.hdf5' % name)

        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss',
                            save_best_only=True),
            TensorBoard()
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_data,
            validation_steps=100,
            workers=1
        )

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(
                    self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2],
                                       predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(
                self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1],
                                   axis=0)
        return predicted