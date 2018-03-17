#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:50:19 2017

@author: haoxiang
"""

from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem_hparams
import tensorflow as tf
import numpy as np
from HAR_utils2 import *



num_steps = 64
num_features = 3
num_classes = 6
num_epochs = 1000
batch_size = 128
epoch_size = 10


def getTransformerModel(hparams, mode=tf.estimator.ModeKeys.TRAIN):
  hparams.hidden_size = 8
  hparams.filter_size = 32
  hparams.num_heads = 1
  hparams.layer_prepostprocess_dropout = 0.0
  p_hparams = problem_hparams.test_problem_hparams(0, num_classes)
  p_hparams.input_modality = {"inputs":  ("real", 0)}
  hparams.problems = [p_hparams]
  return transformer.Transformer(hparams, mode)
  
#def generate_random_features():
#  #inputs = -1 + np.random.random_integers(VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
#  inputs = np.random.random(size=(BATCH_SIZE, INPUT_LENGTH, 3, 1)) * 10 
#  targets = -1 + np.random.random_integers(VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
#  features = {
#        "inputs": tf.constant(inputs, dtype=tf.float32, name="inputs"),
#        "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
#        "target_space_id": tf.constant(1, dtype=tf.int32)
#  }
#  return features


def build_model():
  global_step = tf.get_variable('global_step', [], 
                              initializer = tf.constant_initializer(0), 
                              trainable = False)
  model = getTransformerModel(transformer.transformer_small())
  
  with tf.name_scope("input"):
    input = tf.placeholder(tf.float32, [batch_size, num_steps, num_features], name="input")
    target = tf.placeholder(tf.int32, [batch_size, num_steps], name="target")
    input2 = tf.reshape(input, [batch_size, num_steps, num_features, 1])
    target2 = tf.reshape(target, [batch_size, num_steps, 1, 1])
  
  features = {
      "inputs": input2,
      "targets": target2,
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  
  out_logits, _ = model(features)
  out_logits = tf.squeeze(out_logits, axis=[2, 3])
  
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=tf.reshape(out_logits, [-1, num_classes]),
    labels=tf.reshape(features["targets"], [-1]))
  loss = tf.reduce_mean(loss)
  
  last_predicted = tf.split(tf.cast(tf.argmax(out_logits, 2), tf.int32), num_steps, 1)[-1]
  last_target = tf.split(target, num_steps, 1)[-1]
  

  confusion_mat = tf.confusion_matrix(tf.reshape(last_target, [batch_size]), tf.reshape(last_predicted, [batch_size]), num_classes=num_classes,name='batch_confusion')

  acc = tf.reduce_mean(tf.cast(tf.equal(last_predicted, last_target), tf.float32))
  grad_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
  
  
  loss_summary = tf.summary.scalar('cross_entropy', loss)
  acc_summary = tf.summary.scalar('accuracy', acc)
  summary_op = tf.summary.merge_all()
  return input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat

#x_train, y_train, x_test, y_test = data_preprocess(df)
#batch_x, batch_y = get_random_batch(x_train, y_train)
df = read_data()
x_train, y_train, x_test, y_test = data_preprocess(df, resample=True)

total_conf = np.zeros((6,6))
input, target, loss, acc, grad_op, summary_op, global_step, confusion_mat = build_model()
with tf.Session() as sess:
  train_writer = tf.summary.FileWriter("./trans_train", graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter("./trans_test", graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  for epoch_id in xrange(num_epochs):
    print("Epoch " + str(epoch_id))
    # train accuracy
    accs = []
    for batch_id in xrange(epoch_size):
      batch_x, batch_y = get_random_batch(x_train, y_train, batch_size, one_hot=False)
      res, res2, _ , summary, step = sess.run([loss, acc, grad_op, summary_op, global_step], feed_dict={input: batch_x, target: batch_y})
      train_writer.add_summary(summary, step)
      accs.append(res2)
    print("Train Losses: " + str(res) + " Acc:" + str(res2))
    # Test accuracy
    accs = []
   
    for batch_id in xrange(epoch_size):
      batch_x, batch_y = get_random_batch(x_test, y_test, batch_size, one_hot=False)
      res, res2, summary, step, confusion_matrix = sess.run([loss, acc, summary_op, global_step, confusion_mat], feed_dict={input: batch_x, target: batch_y})
      
      if res2 > 0.9:
         total_conf += confusion_matrix
      test_writer.add_summary(summary, step)
      accs.append(res2)
    print("Test  Losses: " + str(res) + " Acc:" + str(np.average(np.array(accs))))
    print("Confusion Matrix")
    print(total_conf)
    

  