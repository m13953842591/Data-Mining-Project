import numpy as np
import glob
import os


class DataLoader:

    def __init__(self, input_dir, split):
        self.files = glob.glob(os.path.join(input_dir, "file_format"))
        self.split = split
        self.len_train_windows = None
        self.n = int(len(self.files) * split) + 1

    def get_test_data(self, seq_len, future_n):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_x = []
        data_y = []

        for file in self.files[self.n:]:
            data = np.load(file)
            for i in range(data.shape[0] - seq_len - future_n):
                x = data[i:i + seq_len]
                y = data[i + seq_len + future_n - 1][0] + \
                    data[i + seq_len + future_n - 1][1] - data[i + seq_len - 1][
                        0] - data[i + seq_len - 1][1]
                data_x.append(x)
                data_y.append(y)
            return np.array(data_x), np.array(data_y)

    def get_train_data(self, seq_len, future_n):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []

        for file in self.files[:self.n]:
            data = np.load(file)
            for i in range(data.shape[0] - seq_len - future_n):
                x = data[i:i + seq_len]
                y = data[i + seq_len + future_n - 1][0] + \
                    data[i + seq_len + future_n - 1][1] - data[i + seq_len - 1][
                        0] - data[i + seq_len - 1][1]
                data_x.append(x)
                data_y.append(y)
            return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, future_n, batch_size):
        '''
        Yield a generator of training data from filename on
        given list of cols split for train/test
        '''
        while True:
            for file in self.files[:self.n]:
                data = np.load(file)
                for i in range(0, data.shape[0] - seq_len - future_n,
                               batch_size):
                    batch_x = []
                    batch_y = []
                    bz = batch_size
                    if i + batch_size >= data.shape[0] - seq_len - future_n:
                        bz = data.shape[0] - seq_len - future_n - i
                    for j in range(bz):
                        b = i + j
                        x = data[b:b + seq_len]
                        y = data[b + seq_len + future_n - 1][0] + \
                            data[b + seq_len + future_n - 1][1] - \
                            data[b + seq_len - 1][0] - data[b + seq_len - 1][1]
                        batch_x.append(x)
                        batch_y.append(y)
                    yield batch_x, batch_y
