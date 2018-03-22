#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:29:42 2017

@author: haoxiang
"""

import random
import numpy as np
import pandas as pd


COL_NAMES = ["timestamp", "activity", "hand_x", "hand_y", "hand_z", "chest_x", "chest_y", "chest_z", "ankle_x",
             "ankle_y", "ankle_z"]
FEATURE_NAMES = ["hand_x", "hand_y", "hand_z", "chest_x", "chest_y", "chest_z", "ankle_x",
                 "ankle_y", "ankle_z"]
LABEL_NAME = "activity"

NUM_CLASSES = 25


def read_data(path="data/pamap2/train.dat", filter_act=True):
    df = pd.read_csv(path, sep=" ", lineterminator="\n", header=None, names=COL_NAMES)
    df = df.dropna()
    if filter_act:
        df = filter_activities(df)
    return df


# filter certain activities
def filter_activities(df, act_ids=None):
    # default activities to filter in the paper
    if act_ids is None:
        act_ids = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    # fitler act_ids
    df = df[df["activity"].apply(lambda x: x in act_ids)]
    # transform activities
    df["activity"] = df["activity"].apply(lambda x: act_ids.index(int(x)))
    global NUM_CLASSES
    NUM_CLASSES = len(act_ids)
    return df


def one_hot_encode(y):
    n = y.shape[0]
    y_one_hot = np.zeros([n, NUM_CLASSES])
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot


def get_random_batch_internal(df, batch_size, window=64, one_hot=False, return_last=True):
    n = df.shape[0]
    x_batch = []
    y_batch = []

    for _ in xrange(batch_size):
        start_idx = random.randint(0, n - window)
        x = df[start_idx:start_idx + window][FEATURE_NAMES].as_matrix()
        y = df[start_idx:start_idx + window][LABEL_NAME].as_matrix()
        x_batch.append(x)
        if one_hot:
            y_batch.append(one_hot_encode(y))
        else:
            y_batch.append(y)

    if return_last:
        y_last_batch = [y[-1] for y in y_batch]
        if one_hot:
            return x_batch, np.vstack(y_last_batch)
        else:
            return x_batch, np.array(y_last_batch)
    else:
        return x_batch, y_batch


def get_random_batch(ds="train", batch_size=1024, window=64, one_hot=False, return_last=True):
    if ds == "train":
        return get_random_batch_internal(df_train, batch_size=batch_size, window=window, one_hot=one_hot, return_last=return_last)
    elif ds == "test":
        return get_random_batch_internal(df_test, batch_size=batch_size, window=window, one_hot=one_hot, return_last=return_last)
    elif ds == "validate":
        return get_random_batch_internal(df_validate, batch_size=batch_size, window=window, one_hot=one_hot, return_last=return_last)
    else:
        raise ValueError("data set must be train or validate or test")
    return


df_train = read_data("data/pamap2/train.dat", filter_act=True)
df_validate = read_data("data/pamap2/validate.dat", filter_act=True)
df_test = read_data("data/pamap2/test.dat", filter_act=True)

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
