#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:29:34 2018
LSTM on pamap2 dataset
@author: haoxiang
"""
import os
import data.pamap2_data as pamap2_data
import numpy as np
import matplotlib.pyplot as plt
from keras_model_factory import *
from sklearn.metrics import f1_score

# hyper params
num_steps = 128
num_features = len(pamap2_data.FEATURE_NAMES)
num_classes = pamap2_data.NUM_CLASSES
num_hidden_units = 64
batch_size = 512
num_heads = 12



def train(model, data, random_batch=False, overlap=0.5, num_epochs=20, load=True):
    # calc num of batches per epoch
    epoch_size = data.df_train.shape[0] / (num_steps / 2) / batch_size

    # load model if model file exists
    if load and os.path.exists(model.name + ".h5"):
        model.load_weights(model.name + '.h5')

    print("Training Model: " + model.name)
    for epoch in range(num_epochs):
        tr_losses = []
        tr_accs = []
        for _ in xrange(epoch_size):
            if random_batch:
                x, y = data.get_random_batch("train", batch_size, num_steps, one_hot=True)
            else:
                x, y = data.get_next_batch("train", batch_size, num_steps, overlap=overlap, one_hot=True)
            x = np.array(x)
            tr_loss, tr_acc = model.train_on_batch(x, y)
            tr_losses.append(tr_loss)
            tr_accs.append(tr_acc)
        print("Epoch " + str(epoch))
        print("Train Acc: %f\tLoss: %f\t" % (sum(tr_accs) / epoch_size, sum(tr_losses) / epoch_size))
        test(model, data, overlap)
        print("")
        model.save(model.name + '.h5')
    return


def test(model, data, overlap=0.5):
    epoch_size = data.df_test.shape[0] / (num_steps / 2) / batch_size # num of batch per epoch
    ts_losses, ts_accs = [], []
    y_pred = np.array([0])
    y_true = np.array([0])
    for _ in xrange(epoch_size):
        x, y = data.get_next_batch("test", batch_size, num_steps, overlap=overlap, one_hot=True)
        x = np.array(x)
        ts_loss, ts_acc = model.test_on_batch(x, y)
        ts_losses.append(ts_loss)
        ts_accs.append(ts_acc)
        y_pred = np.append(y_pred, np.argmax(model.predict_on_batch(x), axis=1))
        y_true = np.append(y_true, np.argmax(y, axis=1))

    print("Test Acc: %f\tLoss: %f\tF1 Score: %f" % (
        sum(ts_accs) / epoch_size, sum(ts_losses) / epoch_size,
        f1_score(y_true, y_pred, average="macro")))
    return

def plot_raw_signals(x):
    x = x[:, [1, 18, 35]]
    plt.plot(x)
    plt.show()
    return



def plot_att(att):
    if len(att.shape) == 1: # plot 1-d time series
        plt.plot(att)
        plt.show()
    elif len(att.shape) == 2: # plot 2d as heatmap
        plt.imshow(att, cmap='hot', interpolation='nearest')
        plt.show()
    else:
        raise ValueError("Attention dimension should be 1d or 2d")


def visualize(model, data, activity):
    # load model if model file exists
    if os.path.exists(model.name + ".h5"):
        model.load_weights(model.name + '.h5')
    else:
        raise ValueError("Please train and save the model first")

    epoch_size = data.df_train.shape[0] / (num_steps / 2) / batch_size

    for _ in xrange(epoch_size):
        x, y = data.get_next_batch("train", batch_size, num_steps, overlap=0.5, one_hot=True)
        x = np.array(x)
        y = np.argmax(y, axis=1)
        if activity * 1.0 in y:
            print("Found activity " + str(activity))
            act_idx = np.where(y == activity)
            layer_names = [l.name for l in model.layers]

            plot_raw_signals(x[act_idx][-1])
            if "att_hidden" in layer_names:
                v_model = Model(inputs=model.input, outputs=model.get_layer("att_hidden").output)
                att_hidden = np.squeeze(v_model.predict_on_batch(x)[act_idx][-1], -1)
                plot_att(att_hidden)
            if "att_input" in layer_names:
                v_model = Model(inputs=model.input, outputs=model.get_layer("att_input").output)
                att_input = v_model.predict_on_batch(x)[act_idx][-1]
                plot_att(att_input)

            return
    return




# Examples of training

# model_lstm = create_lstm_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model_lstm, pamap2_data)

model_att_hidden = create_attention_time_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model_att_hidden, pamap2_data, num_epochs=60)
visualize(model_att_hidden, pamap2_data, 3)

# model = create_attention_time_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model, pamap2_data)

# model_att_input = create_attention_input_rnn_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model_att_input, pamap2_data)
#
# visualize(model_att_input, pamap2_data, 3)



#model_att_input = create_attention_input_rnn_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes)
# train(model_att_input, pamap2_data)
#isualize(model_att_input, pamap2_data, 3)

#
# model_att_multihead_input = create_attention_input_multihead_model(batch_size, num_hidden_units, num_steps, num_features, num_classes, num_heads)
# train(model_att_multihead_input, pamap2_data)