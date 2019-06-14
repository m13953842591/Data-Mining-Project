import numpy as np
import os
from config import *


class BatchGenerator(object):

    def __init__(self, input_path, time_steps, batch_size, x_units, y_units,
                 skip_step):
        self.input_path = input_path
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.x_units = x_units
        self.y_units = y_units
        self.current_idx = 0
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.skip_step = skip_step
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.files = os.listdir(self.input_path)
        self.files = [os.path.join(self.input_path, self.files[i]) for i in range(len(self.files))]
        self.total_batch_size = (FILE_SIZE - self.time_steps) // skip_step * len(self.files)
        self.steps_per_epoch = self.total_batch_size // self.batch_size

    def generate_sequential(self):
        batch_x = np.zeros((self.batch_size, self.time_steps, self.x_units),
                           dtype=np.float32)
        batch_y = np.zeros((self.batch_size, self.y_units),
                           dtype=np.float32)
        while True:
            for file in self.files:
                data = np.load(file)
                x = data['features']
                y = data['labels']
                # TODO: first read csv file
                for i in range(self.time_steps, x.shape[0], self.skip_step):
                    batch_x[self.current_idx] = x[i-self.time_steps: i]
                    batch_y[self.current_idx] = y[i-1]
                    self.current_idx += 1
                    if self.current_idx == self.batch_size:
                        yield batch_x, batch_y
                        self.current_idx = 0

    def generate(self):
        while True:
            for file in self.files:
                data = np.load(file)
                x = data['features']
                y = data['labels']
                for i in range(0, x.shape[0] - self.batch_size, self.batch_size):
                    batch_x = x[i: i+self.batch_size]
                    batch_y = y[i: i+self.batch_size]
                    yield batch_x, batch_y

