#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:29:42 2017

@author: haoxiang
"""


import random
import numpy as np
import pandas as pd
from sklearn import preprocessing

def read_data(path="data/big.csv"):
  columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
  df = pd.read_csv(path, header = None, names = columns )
  df = df.dropna()
  return df

def data_preprocess(df, train_faction = 0.8):
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
  
  cols = y.shape[0]
  test_idx = int(cols * train_faction)
  x_train = x[0 : test_idx, :]
  y_train = y[0 : test_idx, :]
  
  x_test = x[test_idx : , :]
  y_test = y[test_idx : , :]
  
  return x_train, y_train, x_test, y_test

def is_single_activity(y):
  mat = y.tolist()
  labels = [row.index(1.) for row in mat]
  return  len(set(labels)) == 1


def convert_one_hot(ys):
  ys = ys.tolist()
  res = [y.index(1.) for y in ys]
  res = np.array(res)
  return res

def get_random_batch(x, y, batch_size, num_steps):
  batch_x = []
  batch_y = []
  while len(batch_y) < batch_size:
    idx = random.randrange(0, y.shape[0] - num_steps)
    if is_single_activity(y[idx:idx + num_steps]):
      batch_x.append(x[idx:idx + num_steps,:])
      batch_y.append(y[idx + num_steps / 2 - 1])
  batch_x = np.stack(batch_x)
  batch_y = np.stack(batch_y)
  return batch_x, batch_y

def get_random_batch2(x, y, batch_size, num_steps):
  batch_x = []
  batch_y = []
  while len(batch_y) < batch_size:
    idx = random.randrange(0, y.shape[0] - num_steps)
    if is_single_activity(y[idx:idx + num_steps]):
      batch_x.append(x[idx:idx + num_steps,:])
      batch_y.append(convert_one_hot(y[idx:idx + num_steps, :]))
  batch_x = np.stack(batch_x)
  batch_y = np.stack(batch_y)
  return batch_x, batch_y

def get_batch(x, y, batch_id, batch_size, num_steps):
  batch_x = []
  batch_y = []
  for i in xrange(batch_size):
    start = batch_id * batch_size + i
    end = start + num_steps
    batch_x.append( x[start : end, :] )
    batch_y.append( y[end - 1] )
  
  batch_x = np.stack(batch_x)
  batch_y = np.stack(batch_y)
  return batch_x, batch_y


def get_num_batches(x, batch_size, num_steps):
  return (len(x) - num_steps) // batch_size 
  

