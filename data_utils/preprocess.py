import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
from config import *
import glob


def get_features_and_labels(dataframe):
    l1 = dataframe['AskPrice1'] + dataframe['BidPrice1']
    l2 = l1.shift(-FUTURE_N)
    label = l2 - l1
    label.dropna(axis=0, how='any', inplace=True)
    label = label.to_numpy(dtype=np.float32, copy=False)\
        .reshape(label.shape[0], 1)

    dataframe.drop(axis=1,
                   labels=['Unnamed: 0', 'UpperLimitPrice', 'LowerLimitPrice'],
                   inplace=True)
    x = dataframe.values
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)

    min_len = min(x_scaled.shape[0], label.shape[0])
    return x_scaled[:min_len], label[:min_len]


def clear(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def split_break_save(raw_data_path, save_path_root, file_size):
    # we split the dataset into train set and test set
    # then we break train set and test set into small batches
    # finally we save those small files
    print("reading data from %s" % raw_data_path)
    dataframe = pd.read_csv(raw_data_path)
    print("extracting features and labels...")
    features, labels = get_features_and_labels(dataframe)
    print("splitting...")
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=TEST_SIZE, shuffle=False)
    # shuffle = None so that we don't discard the sequential information

    num_files_train, num_files_test = \
        features_train.shape[0]//file_size, features_test.shape[0]//file_size

    train_path_dir = os.path.join(save_path_root, "train")
    test_path_dir = os.path.join(save_path_root, "test")
    print("clear file directories, hold on")
    clear(train_path_dir)
    clear(test_path_dir)
    print("file directory cleared, prepare to save...")

    for i in range(num_files_train):
        save_path = os.path.join(train_path_dir, "batches%d" % i)
        print("saving train data to %s..." % save_path)
        np.savez(save_path, features=features[i * file_size: (i+1) * file_size],
                 labels=labels[i * file_size: (i+1) * file_size])

    for i in range(num_files_test):
        save_path = os.path.join(test_path_dir, "batches%d" % i)
        print("saving test data to %s..." % save_path)
        np.savez(save_path, features=features[i * file_size: (i+1) * file_size],
                 labels=labels[i * file_size: (i+1) * file_size])


def get_numpy_dataset(input_dir, output_dir, split=0.3):

    files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not files:
        raise Exception("there is no csv file in directory: %s" % input_dir)
    if not os.path.exists(os.path.join(output_dir, "train")):
        os.makedirs(os.path.join(output_dir, "train"))
    if not os.path.exists(os.path.join(output_dir, "test")):
        os.makedirs(os.path.join(output_dir, "test"))

    n = int((1-split) * len(files))

    m_train, m_test = 0, 0
    for file in files[:n]:
        print("processing %s" % file)
        name = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        xs, ys = get_features_and_labels(df)
        save_path = os.path.join(output_dir, "train", name + ".npz")
        np.savez(save_path, x=xs, y=ys)
        m_train += ys.shape[0]

    if n == len(files):
        raise Exception("error: no test dataset!")

    for file in files[n:]:
        print("processing %s" % file)
        name = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        xs, ys = get_features_and_labels(df)
        save_path = os.path.join(output_dir, "test", name + ".npz")
        np.savez(save_path, x=xs, y=ys)
        m_test += ys.shape[0]

    print("finish processing, check new dataset in %s" % output_dir)
    print("generated training samples = %d, testing samples = %d" % (m_train,
                                                                     m_test))


if __name__ == '__main__':
    # nodrop: train 1219467 test: 501660
    # split: train 399323 test: 176320

    input_dir = "D:\\zchen\\下载\\data\\split"
    output_dir = os.path.join(DATA_PATH, "split")
    get_numpy_dataset(input_dir, output_dir, 0.3)
