#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 18:14:37 2017

@author: haoxiang
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:35:44 2017

@author: haoxiang
"""

import tensorflow as tf
import sys
import math
import random
import numpy as np
import pandas as pd
from HAR_utils2 import *
from sklearn import preprocessing


num_steps = 64
num_features = 3
num_classes = 6
num_hidden_units = [1024, 30]
num_epochs = 200
batch_size = 1024
epoch_size = 100


def build_CNN(conv_k=4, conv_s=1, max_pooling_k=4, max_pooling_s=1):
  global_step = tf.get_variable('global_step', [], 
                            initializer = tf.constant_initializer(0), 
                            trainable = False)
  with tf.name_scope("input"):
    input = tf.placeholder(tf.float32, [batch_size, num_steps, num_features], name="input")
    target = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes], name="target")
    
  with tf.variable_scope("conv1d"):  
    W_conv = tf.get_variable("W", [conv_k, num_features, num_features])
    b_conv = tf.get_variable("b", [num_features])
    conv_output = tf.nn.relu(tf.nn.conv1d(input, W_conv, conv_s, "VALID") + b_conv)
    conv_output = tf.reshape(conv_output, [batch_size, -1, 1, num_features])
  
  with tf.variable_scope("max_pooling"):
    mp_output = tf.nn.max_pool(conv_output, ksize=[1, max_pooling_k, 1, 1], strides=[1, max_pooling_s, 1, 1], padding='VALID')
    mp_output = tf.reshape(mp_output, [batch_size, -1])
    
  dim_mp_output = int(mp_output.shape[1])
  
  with tf.variable_scope("hidden1"):
    W = tf.get_variable("W", [dim_mp_output, num_hidden_units[0]])
    b = tf.get_variable("b", [num_hidden_units[0]])
    h1_output = tf.nn.relu(tf.nn.xw_plus_b(mp_output, W, b))
    
  with tf.variable_scope("hidden2"):
    W = tf.get_variable("W", [num_hidden_units[0], num_hidden_units[1]])
    b = tf.get_variable("b", [num_hidden_units[1]])
    h2_output = tf.nn.relu(tf.nn.xw_plus_b(h1_output, W, b))
    
  with tf.variable_scope("softmax"):
    W = tf.get_variable("W", [num_hidden_units[1], num_classes])
    b=  tf.get_variable("b", [num_classes])
    logits = tf.nn.xw_plus_b(h2_output, W, b)
    
  with tf.name_scope("cross_entropy"):
    L2_LOSS = 0.0015
    l2 = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    
    last_target = tf.slice(target, [0, num_steps - 1, 0], [batch_size, 1, num_classes])
    last_target = tf.reshape(last_target, [batch_size, num_classes])
    
    
    pred = tf.argmax(logits, 1)
    label = tf.argmax(last_target, 1)
    
    confusion_mat = tf.confusion_matrix(label, pred,
                                         num_classes=num_classes,
                                         name='batch_confusion')
    
    loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(last_target, logits) ) + l2 
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(last_target, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    loss_summary = tf.summary.scalar('cross_entropy', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)
    summary_op = tf.summary.merge_all()
    
  with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
    grad_op = optimizer.minimize(loss, global_step=global_step)

  return input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat


def train_random_batches():
  #x_train, y_train, x_test, y_test = data_preprocess(df)
  #batch_x, batch_y = get_random_batch(x_train, y_train)
  df = read_data()
  x_train, y_train, x_test, y_test = data_preprocess(df)
  input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat = build_CNN()
  with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("./cnn_train", graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter("./cnn_test", graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    for epoch_id in xrange(num_epochs):
      print("Epoch " + str(epoch_id))
      # train accuracy
      accs = []
      for batch_id in xrange(epoch_size):
        batch_x, batch_y = get_random_batch(x_train, y_train, batch_size)
        res, res2, _, summary, step = sess.run([loss, acc, grad_op, summary_op, global_step], feed_dict={input: batch_x, target: batch_y})
        train_writer.add_summary(summary, step)
        accs.append(res2)
      print("Train Losses: " + str(res) + " Acc:" + str(res2))
      # Test accuracy
      accs = []
      for batch_id in xrange(epoch_size):
        batch_x, batch_y = get_random_batch(x_test, y_test, batch_size)
        res, res2, summary, step, confusion_matrix = sess.run([loss, acc, summary_op, global_step, confusion_mat], feed_dict={input: batch_x, target: batch_y})
        test_writer.add_summary(summary, step)        
        accs.append(res2)
      print("Test  Losses: " + str(res) + " Acc:" + str(np.average(np.array(accs))))
      print("Confusion Matrix:")
      print(confusion_matrix)
      np.savetxt("CNN conf matrix", confusion_matrix)

      
  
train_random_batches()


