import tensorflow as tf
from tensorflow._api.v1.keras import layers
from tensorflow._api.v1.keras import regularizers
from tensorflow._api.v1.keras import activations
from config import *

DNN = tf.keras.Sequential([
    layers.BatchNormalization(axis=1, input_shape=(NUM_FEATURE, )),
    layers.Dense(1000, activation='relu'),
    layers.Dense(1)
])

LSTM = tf.keras.Sequential([
    layers.CuDNNLSTM(units=HIDDEN_SIZE,
                     input_shape=(TIME_STEP, NUM_FEATURE),
                     return_sequences=False,
                     kernel_regularizer=regularizers.l2(RR),
                     bias_regularizer=regularizers.l2(RR),
                     ),
    layers.Dense(NUM_LABEL, activation=activations.tanh,
                 kernel_regularizer=regularizers.l2(RR))
])


