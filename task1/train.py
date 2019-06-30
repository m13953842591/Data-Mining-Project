#encoding: utf-8
#file: train.py
#author: shawn233
#date: 19-06-26

from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_torch_util import *
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


DATA_DIR = "D:\\zchen\\下载\\data\\nodrop\\"


class DNN(nn.Module):

    def __init__(self, n_in):
        super(DNN, self).__init__()
        self.n_in = n_in
        self.net = nn.Sequential(
            nn.Linear(n_in, 2048),
            nn.Tanh(),
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)



def calculate_labels(df, future_n=10):
    askprice1 = df.loc[:, "AskPrice1"]
    bidprice1 = df.loc[:, "BidPrice1"]
    n1 = (askprice1[future_n:]+bidprice1[future_n:])
    n2 = (askprice1[:len(askprice1)-future_n]+bidprice1[:len(bidprice1)-future_n])
    labels = (n1.values-n2.values)/2
    return labels


def main():
    future_n = 10
    learning_rate = 1e-2
    batch_size = 1024
    n_epochs = 10

    for root, dirs, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            # fn = "data_nodrop_18.csv"
            print("file:", fn)
            df = pd.read_csv(DATA_DIR+fn)
            print("raw df shape", df.shape)
            y = calculate_labels(df, future_n)
            print("y shape", y.shape)
            start_ind = 1 if df.columns[0] == "Unnamed: 0" else 0
            df.drop(labels=["UpperLimitPrice","LowerLimitPrice"], axis=1, inplace=True)
            x = df.values[:len(df)-future_n, start_ind:]
            print("x shape", x.shape)
            x = normalize(x)
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
            net = DNN(x.shape[1])
            train_model(net, train_x, train_y, test_x, test_y, \
                learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs)
            test_x = torch.from_numpy(test_x).float()
            pred = net(test_x).cpu().detach().numpy()
            pd.DataFrame(pred).to_csv("pred.csv")
            pd.DataFrame(test_y).to_csv("test_y.csv")


if __name__ == "__main__":
    main()