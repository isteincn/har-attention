#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:29:42 2017

@author: haoxiang
"""

import random
import numpy as np
import pandas as pd
import os

FEATURE_NAMES = ["Hand_" + str(i) for i in range(17)] + ["Chest_" + str(i) for i in range(17)] + ["Ankle_" + str(i) for i in range(17)]
LABEL_NAME = "activity"
COL_NAMES = ["timestamp"] + FEATURE_NAMES + [LABEL_NAME]

NUM_CLASSES = 12


def read_data(path="data/pamap2/train.dat", filter_act=True, down_sample=True, from_cache=True):
    df = pd.read_csv(path, sep=",", lineterminator="\n", header=None)
    df = df.dropna()
    return df

def one_hot_encode(y):
    n = y.shape[0]
    y_one_hot = np.zeros([n, NUM_CLASSES])
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot


def get_next_batch_internal(df, idx, batch_size, window=64, overlap=0.5, one_hot=False, return_last=True):
    n = df.shape[0]
    x_batch = []
    y_batch = []
    for _ in xrange(batch_size):
        if idx >= n:
            idx = 0
        data = df[idx:idx + 1].as_matrix()
        y = data[:,-1]
        y = y.astype(int)
        data = np.reshape(data[:,:-1], (-1, 52))
        x = data[:,1:]
        x_batch.append(x)
        if one_hot:
            y_batch.append(one_hot_encode(y))
        else:
            y_batch.append(y)
        idx += 1

    if return_last:
        y_last_batch = [y[-1] for y in y_batch]
        if one_hot:
            return idx, x_batch, np.vstack(y_last_batch)
        else:
            return idx, x_batch, np.array(y_last_batch)
    else:
        return idx, x_batch, y_batch


def get_next_batch(ds="train", batch_size=1024, window=64, overlap=0.5, one_hot=False, return_last=True):
    global train_idx
    global validate_idx
    global test_idx
    global df_train
    global df_validate
    global df_test

    if ds == "train":
        next_idx, x_batch, y_batch = get_next_batch_internal(df_train, train_idx, batch_size, window, overlap,
                                                             one_hot, return_last)
        train_idx = next_idx
        return x_batch, y_batch
    elif ds == "test":
        next_idx, x_batch, y_batch = get_next_batch_internal(df_test, test_idx, batch_size, window, overlap,
                                                             one_hot, return_last)
        test_idx = next_idx
        return x_batch, y_batch
    elif ds == "validate":
        next_idx, x_batch, y_batch = get_next_batch_internal(df_validate, validate_idx, batch_size, window, overlap,
                                                             one_hot, return_last)
        validate_idx = next_idx
        return x_batch, y_batch
    else:
        raise ValueError("data set must be train or validate or test")


df_train = read_data("data/pamap2_ming/train.dat")
df_validate = read_data("data/pamap2_ming/validate.dat")
df_test = read_data("data/pamap2_ming/test.dat")

train_idx = 0
validate_idx = 0
test_idx = 0

#
# df = read_data("data/small.csv")
#
#
# x_train, y_train, x_test, y_test = data_preprocess(df, resample=False)
#
# labels = [np.argmax(s[-1, :]) for s in y_train]
#
# counter = Counter(labels)
# print(counter)
#
# batch_x, batch_y = get_random_batch(x_train, y_train, 2, False)
#
# from data.pamap2_ming_data import *
# x, y = get_next_batch(ds="train", one_hot=True)