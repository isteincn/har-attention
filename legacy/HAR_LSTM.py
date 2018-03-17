#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:35:44 2017

@author: haoxiang
"""

import tensorflow as tf
from HAR_utils2 import *
import sys
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing


num_steps = 64
num_layers = 1
num_features = 3
num_classes = 6
num_hidden_units = 256
num_epochs = 20
batch_size = 1024
epoch_size = 10

#


#def read_data(path="data/big.csv"):
#  columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
#  df = pd.read_csv(path, header = None, names = columns )
#  df = df.dropna()
#  return df
#
#def data_preprocess(df, train_faction = 0.8):
#  # one_hot encoding of labels
#  labels = df["activity"].as_matrix()
#  le = preprocessing.LabelEncoder()
#  le.fit(labels)
#  y = le.transform(labels)
#  y = y.reshape(-1, 1)
#  encoder = preprocessing.OneHotEncoder()
#  encoder.fit(y)
#  y = encoder.transform(y).toarray()
#  # x shape: 1098204 * 3
#  x = df[["x-axis", "y-axis", "z-axis"]].as_matrix()
#  
#  cols = y.shape[0]
#  test_idx = int(cols * train_faction)
#  x_train = x[0 : test_idx, :]
#  y_train = y[0 : test_idx, :]
#  
#  x_test = x[test_idx : , :]
#  y_test = y[test_idx : , :]
#  
#  return x_train, y_train, x_test, y_test
#
#def get_random_batch(x, y):
#  batch_x = []
#  batch_y = []
#  for i in xrange(batch_size):
#    idx = random.randrange(0, y.shape[0] - num_steps)
#    batch_x.append(x[idx:idx + num_steps,:])
#    batch_y.append(y[idx + num_steps - 1])
#  batch_x = np.stack(batch_x)
#  batch_y = np.stack(batch_y)
#  return batch_x, batch_y
#
#def get_batch(x, y, batch_id):
#  batch_x = []
#  batch_y = []
#  
#  for i in xrange(batch_size):
#    start = batch_id * batch_size + i
#    end = start + num_steps
#    batch_x.append( x[start, end] )
#    batch_y.append( y[start, end] )
#  
#  batch_x = np.stack(batch_x)
#  batch_y = np.stack(batch_y)
#  return batch_x, batch_y
#  

def build_LSTM():
  global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False)
  with tf.name_scope("input"):
    input = tf.placeholder(tf.float32, [batch_size, num_steps, num_features], name="input")
    target = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes], name="target")
    
  with tf.variable_scope("hidden"):
    W = tf.Variable(tf.random_normal([num_features, num_hidden_units]))
    b=  tf.Variable(tf.random_normal([num_hidden_units]))
    
    x = tf.reshape(input, [-1, num_features])
    hidden = tf.nn.relu(tf.matmul(x, W) + b)
    hidden = tf.reshape(hidden, [batch_size, num_steps, num_hidden_units ])
    
    
  with tf.name_scope("lstm"):
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(num_hidden_units) for _ in range(num_layers)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
    
    hidden = tf.transpose(hidden, [1, 0, 2])
    hidden = tf.reshape(hidden, [-1, num_hidden_units])
    hidden = tf.split(hidden, num_steps, 0)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)
    output = outputs[-1]
    
  with tf.variable_scope("softmax"):
    W = tf.Variable(tf.random_normal([num_hidden_units, num_classes]))
    b=  tf.Variable(tf.random_normal([num_classes]))
    logits = tf.nn.xw_plus_b(output, W, b)
    
  with tf.name_scope("loss"):
    last_target = tf.slice(target, [0, num_steps - 1, 0], [batch_size, 1, num_classes])
    last_target = tf.reshape(last_target, [batch_size, num_classes])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=last_target))
    pred = tf.argmax(logits, 1)
    label = tf.argmax(last_target, 1)
    correct_pred = tf.equal(pred, tf.argmax(last_target, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    
  with tf.name_scope("Summary"):
    confusion_mat = tf.confusion_matrix(label, pred,
                                             num_classes=num_classes,
                                             name='batch_confusion')
    loss_summary = tf.summary.scalar('cross_entropy', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)
    summary_op = tf.summary.merge_all()
    
  with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    grad_op = optimizer.minimize(loss, global_step=global_step)

  return input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat

#x_train, y_train, x_test, y_test = data_preprocess(df)
#batch_x, batch_y = get_random_batch(x_train, y_train)
df = read_data("data/big.csv")
x_train, y_train, x_test, y_test = data_preprocess(df, resample=True)
input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat = build_LSTM()
with tf.Session() as sess:
  train_writer = tf.summary.FileWriter("./lstm_train", graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter("./lstm_test", graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  for epoch_id in xrange(num_epochs):
    print("Epoch " + str(epoch_id))
    # train accuracy
    accs = []
    for batch_id in xrange(epoch_size):
      batch_x, batch_y = get_random_batch(x_train, y_train, batch_size)
      res, res2, _ , summary, step = sess.run([loss, acc, grad_op, summary_op, global_step], feed_dict={input: batch_x, target: batch_y})
      train_writer.add_summary(summary, step)
      accs.append(res2)
    print("Train Losses: " + str(res) + " Acc:" + str(np.average(np.array(accs))))
    
    # Test accuracy
    accs = []
    for batch_id in xrange(epoch_size):
      batch_x, batch_y = get_random_batch(x_train, y_train, batch_size)
      res, res2, summary, step, confusion_matrix = sess.run([loss, acc, summary_op, global_step, confusion_mat], feed_dict={input: batch_x, target: batch_y})
      test_writer.add_summary(summary, step)
      accs.append(res2)
    print("Test  Losses: " + str(res) + " Acc:" + str(np.average(np.array(accs))))
    print("Confusion Matrix")
    print(confusion_matrix)
      



