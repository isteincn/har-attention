#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:34:59 2018

Attention layer added to the hidden output of LSTMs, attention weights is based on time step,

@author: haoxiang
"""
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.n_hidden = self.output_dim[-1]
        self.n_channel = input_shape[-1]

        # weight for past attention (*, channel, channel)
        self.weight_att = self.add_weight(name='weight_att',
                                          shape=(self.n_hidden, self.n_channel, self.n_channel),
                                          initializer='uniform',
                                          trainable=True)

        # weight for input (*, channel, channel)
        self.weight_input = self.add_weight(name='weight_input',
                                          shape=(self.n_hidden, self.n_channel, self.n_channel),
                                          initializer='uniform',
                                          trainable=True)

        # weight for output energy
        self.weight_e = self.add_weight(name='weight_energy',
                                        shape=(self.n_hidden, self.n_channel, self.n_channel),
                                        initializer='uniform',
                                        trainable=True)

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!
        return

    def batch_mult(self, weight, batch_x):
        weight = tf.reshape(weight, [-1, self.n_channel])
        output = tf.matmul(batch_x, tf.transpose(weight))
        output = tf.reshape(output, [-1, self.n_hidden, self.n_channel])
        return output

    def batch_att_mult(self, weight, batch_att):
        # split along hidden unit axis
        batch_att_list = tf.unstack(batch_att, axis=1)
        w_list = tf.unstack(weight, axis=0) # weight_e matrix for each hidden unit
        out_list = []

        for i in range(self.n_hidden): # calculate att for each hidden unit
            # att: (batch, channels)
            att = batch_att_list[i]
            # w: (channel, channel)
            w = w_list[i]
            out_list.append(tf.matmul(att, tf.transpose(w)))

        # combine att along hidden unit axis
        output = tf.stack(out_list, axis=1)
        return output
    
    def call(self, inputs, **kwargs):
        # initial attention
        rank = inputs.get_shape().ndims
        if rank == 2: # no batch
            # att = K.ones((self.n_)) * 1.0 / self.input_dim[-1]
            # att = K.expand_dims(att)
            #
            # input_list = tf.unstack(inputs)
            # output_list = []
            # for x in input_list:
            #     att = K.dot(self.weight_input, K.expand_dims(x)) + K.dot(self.weight_att, att)
            #     att = K.tanh(att)
            #     att = K.dot(self.weight_e, att)
            #     att = tf.reduce_sum(att, -1)
            #     output_list.append(K.softmax(att))
            # output = tf.stack(output_list, axis=0)
            # return output
            pass

        elif rank == 3: # train on batch
            input_list = tf.unstack(inputs, axis=1)
            # att: (batch, hidden, channel)
            att = K.ones([tf.shape(inputs)[0], self.n_hidden, self.n_channel]) / (1.0 * self.n_channel)
            output_list = [] # split among time step
            for x in input_list:
                # x: (batch, channels)
                att = self.batch_mult(self.weight_input, x) + self.batch_att_mult(self.weight_att, att)
                att = tf.tanh(att)
                att = self.batch_att_mult(self.weight_e, att)
                att = K.softmax(att)
                # att: batch * hidden * 3, x: batch * 3
                x = tf.expand_dims(x, -1)
                output_list.append(tf.reduce_sum(tf.matmul(att, x), axis=-1))
            output = tf.stack(output_list, axis=1)

            print(output)
            return output
        return

    def compute_output_shape(self, input_shape):
        return self.output_dim
