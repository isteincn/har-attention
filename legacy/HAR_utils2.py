#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:29:42 2017

@author: haoxiang
"""
from collections import Counter

import random
import numpy as np
import pandas as pd
from sklearn import preprocessing

def read_data(path="data/big.csv"):
  col_names=["timestamp", "activity", "hand_x", "hand_y", "hand_z", "chest_x", "chest_y", "chest_z", "ankle_x", "ankle_y", "ankle_z"]
  df = pd.read_csv(path, sep=" ", lineterminator="\n", header=None, col_names=col_names)
  return df

def data_preprocess(df, train_fraction = 0.8, stride = 64, window=64, resample=True):
  # one_hot encoding of labels
  labels = df["activity"].as_matrix()
  le = preprocessing.LabelEncoder()
  le.fit(labels)
  y = le.transform(labels)
  y = y.reshape(-1, 1)
  encoder = preprocessing.OneHotEncoder()
  encoder.fit(y)
  y = encoder.transform(y).toarray()
  # x shape: 1098204 * 3
  x = df[["x-axis", "y-axis", "z-axis"]].as_matrix()
  x_samples, y_samples = generate_samples(x, y, stride, window, resample)
  
  rand_perm = np.random.permutation(len(x_samples)).tolist()
  
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  
  for i, idx in enumerate(rand_perm):
    if i < len(rand_perm) * train_fraction:
      x_train.append(x_samples[idx])
      y_train.append(y_samples[idx])
    else:
      x_test.append(x_samples[idx])
      y_test.append(y_samples[idx])
  return x_train, y_train, x_test, y_test


def generate_samples(x, y, stride=64, window=64, resample=True):
  x_samples = []
  y_samples = []  
  n = x.shape[0]
  for i in range(0, n - stride, stride):
    x_samples.append(x[i : i + window, :])
    y_samples.append(y[i : i + window, :])
  
  if resample:
    labels = [np.argmax(y_sample[-1, :]) for y_sample in y_samples]
    counter = Counter(labels)
    max_num = max( counter.values() )
    classes = counter.keys()
    for c in classes:
      num_to_draw = max_num - counter[c]
      print("num_to_draw")
      print num_to_draw
      indexes_to_draw = []
      for i in range(len(labels)):
        if labels[i] == c:
          indexes_to_draw.append(i)
      print len(indexes_to_draw)
      for _ in range(num_to_draw):
        idx = indexes_to_draw[random.randrange(0, len(indexes_to_draw))]
        x_samples.append( x_samples[idx] )
        y_samples.append( y_samples[idx] )
  return x_samples, y_samples
  


def is_single_activity(y):
  mat = y.tolist()
  labels = [row.index(1.) for row in mat]
  return  len(set(labels)) == 1


def convert_one_hot(ys):
  ys = ys.tolist()
  res = [y.index(1.) for y in ys]
  res = np.array(res)
  return res

def get_random_batch(x, y, batch_size, one_hot=True):
  n = len(x)
  batch_x = []
  batch_y = []
  for _ in range(batch_size):
    idx = random.randrange(0, n)
    batch_x.append(x[idx])
    if one_hot:
      batch_y.append(y[idx])
    else:
      batch_y.append(convert_one_hot(y[idx]))
  return batch_x, batch_y


def get_random_batch_last(x, y, batch_size, one_hot=True):
  n = len(x)
  batch_x = []
  batch_y = []
  for _ in range(batch_size):
    idx = random.randrange(0, n)
    batch_x.append(x[idx])
    if one_hot:
      batch_y.append(y[idx][-1])
    else:
      batch_y.append(convert_one_hot(y[idx])[-1])
  return batch_x, batch_y

def get_num_batches(x, batch_size, num_steps):
  return (len(x) - num_steps) // batch_size

#
#df = read_data("data/small.csv") 
#  
#
#x_train, y_train, x_test, y_test = data_preprocess(df, resample=False)
#
#labels = [np.argmax(s[-1, :]) for s in y_train]
#
#counter = Counter(labels)
#print(counter)
#
#batch_x, batch_y = get_random_batch(x_train, y_train, 2, False)