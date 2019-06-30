import numpy as np
from data_utils.data_loader import get_generator
from keras.models import load_model
import pandas as pd


def predict_sequences_multiple(input_dir, seq_len, batch_size):
    generator = get_generator(input_dir, seq_len, batch_size)
    model = load_model('checkpoints/dnn1.05-2.00.hdf5')
    x, y_true = next(generator)
    y_pred = model.predict_on_batch(x)
    y = np.concatenate((y_true, y_pred), axis=1)
    df = pd.DataFrame(y, dtype=np.float32)
    df.to_csv('pred.csv')


if __name__ == '__main__':
    from config import *
    import os
    input_dir = os.path.join(DATA_PATH, "nodrop\\test")
    predict_sequences_multiple(input_dir, 1, 1000)




