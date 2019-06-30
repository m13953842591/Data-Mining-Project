import numpy as np
import glob
import os


def get_generator(input_dir, seq_len, batch_size):

    files = glob.glob(os.path.join(input_dir, "*.npz"))
    if not files:
        raise Exception("there is no npz file in %s" % input_dir)

    while True:
        for file in files:
            data = np.load(file)
            xs, ys = data['x'], data['y']
            for base in range(0, xs.shape[0] - seq_len, batch_size):
                batch_x = []
                batch_y = []
                bz = batch_size
                if base + batch_size >= xs.shape[0] - seq_len:
                    bz = xs.shape[0] - seq_len - base
                for i in range(bz):
                    b = base + i
                    x = xs[b: b + seq_len]
                    y = ys[b + seq_len - 1]
                    batch_x.append(x)
                    batch_y.append(y)
                yield np.array(batch_x, dtype=np.float32), \
                      np.array(batch_y, dtype=np.float32)

