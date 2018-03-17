#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:29:34 2018
LSTM on pamap2 dataset
@author: haoxiang
"""

import data.pamap2_data as pamap2_data
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras_model_factory import *

import numpy as np

# hyper params
num_steps = 256
num_layers = 2
num_features = len(pamap2_data.FEATURE_NAMES)
num_classes = pamap2_data.NUM_CLASSES
num_hidden_units = 256
num_epochs = 1000
batch_size = 1024
epoch_size = 5


def train(model, data):
    for epoch in range(num_epochs):
        tr_losses = []
        tr_accs = []
        for _ in xrange(epoch_size):
            x, y = data.get_random_batch("train", batch_size, num_steps, one_hot=True)
            x = np.array(x)
            tr_loss, tr_acc = model.train_on_batch(x, y)
            tr_losses.append(tr_loss)
            tr_accs.append(tr_acc)

        print("=====" * 20)
        print("\t".join(["Epoch Train: ", str(epoch), " Loss:", str(sum(tr_losses) / epoch_size), " Acc: ",  str(sum(tr_accs) / epoch_size)]))
        # print acc on val data set
        x, y = data.get_random_batch("validate", batch_size, num_steps, one_hot=True)
        x_arr = np.array(x)
        ts_loss, ts_acc = model.test_on_batch(x_arr, y)
        print("\t".join(["Epoch Val: ", str(epoch), " Loss:", str(ts_loss), " Acc: ", str(ts_acc)]))
        print("=====" * 20)


# model_lstm = create_lstm_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model_lstm, pamap2_data)

model_att_hidden = create_attention_time_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
train(model_att_hidden, pamap2_data)
