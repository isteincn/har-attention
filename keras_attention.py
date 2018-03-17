#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:29:34 2018

@author: ubuntu
"""

from legacy.HAR_utils2 import *
from keras.models import Sequential
from keras.layers import Dense, LSTM
from layers.attention_time_layer import Attention

import numpy as np

num_steps = 64
num_layers = 2
num_features = 3
num_classes = 6
num_hidden_units = 64
num_epochs = 1000
batch_size = 1024
epoch_size = 10


df = read_data("data/big.csv")
x_train, y_train, x_test, y_test = data_preprocess(df, resample=True)

model = Sequential()
model.add(Attention((batch_size, num_steps, num_hidden_units), input_shape=(num_steps, num_features), batch_size=batch_size))
model.add(LSTM(num_hidden_units, input_shape=(num_steps, num_hidden_units), return_sequences=False, stateful=False))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(num_epochs):
    x, y = get_random_batch_last(x_train, y_train, batch_size, True)
    x_arr = np.array(x)
    y_arr = np.array(y)
    tr_loss, tr_acc = model.train_on_batch(x_arr, y_arr)
    print("Epoch Train: ", epoch, " Loss:", tr_loss, " Acc: ", tr_acc)
    
    if epoch % 10 == 0:
        x, y = get_random_batch_last(x_test, y_test, batch_size, True)
        x_arr = np.array(x)
        y_arr = np.array(y)
        ts_loss, ts_acc = model.test_on_batch(x_arr, y_arr)
        print("=====" * 10)
        print("Epoch Test: ", epoch, " Loss:", ts_loss, " Acc: ", ts_acc)
        print("=====" * 10)


