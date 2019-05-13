import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import shutil
from parameters import *

def get_features_and_labels(dataframe):
    # TODO: to be overridden on stage2
    l1 = dataframe['AskPrice1'] + dataframe['BidPrice1']
    l2 = l1.shift(-FUTURE_N)
    label = l2 - l1
    label.dropna(axis=0, how='any', inplace=True)
    label = label.to_numpy(dtype=np.float32, copy=False)\
        .reshape((label.shape[0], 1))

    dataframe.drop(axis=1, labels=['UpdateTime', 'UpdateMillisec'], inplace=True)
    dataframe = dataframe.to_numpy(dtype=np.float32, copy=False)
    min_len = min(dataframe.shape[0], label.shape[0])
    return dataframe[:min_len], label[:min_len]


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


if __name__ == '__main__':
    raw_data_path = DATA_PATH + "/data.csv"
    split_break_save(raw_data_path, save_path_root=DATA_PATH,
                     file_size=FILE_SIZE)
    print("data management succeed!")