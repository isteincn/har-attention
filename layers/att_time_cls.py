#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:34:59 2018
@author: haoxiang

Attention layer attached to the hidden layer output
Input: (batch, time_step, hidden_unit)
Use the last hidden unit to generate attention of each time step, and only output weighted sum of hidden states

"""
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class Attention(Layer):

    def __init__(self, output_dim, continuous=False, **kwargs):
        self.output_dim = output_dim
        assert "batch_size" in kwargs
        self.batch_size = kwargs["batch_size"]
        self.continuous = continuous
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_hidden = input_shape[-1]
        self.n_timestep = input_shape[-2]

        # the attention weight based on last time step hidden state
        self.weight_energy = self.add_weight(name='weight_energy',
                                          shape=(self.n_hidden, self.n_timestep),
                                          initializer='uniform')

        self.bias_energy = self.add_weight(name='bias_eng',
                                        shape=[self.n_timestep],
                                        initializer='zero')

        # the attention weight based on last time step hidden state
        self.weight_att = self.add_weight(name='weight_att',
                                          shape=(self.n_timestep, self.n_timestep),
                                          initializer='uniform')

        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!
        return

    def call(self, inputs, **kwargs):
        # initial attention
        rank = inputs.get_shape().ndims
        assert rank == 3
        # batch of hidden states at last time step
        energy = inputs[:, -1]

        # projection to weight
        # energy: (batch, hidden), weight: (hidden, timestep)
        energy = K.dot(energy, self.weight_energy) + self.bias_energy
        energy = K.tanh(energy)

        # calculate attention
        att = K.dot(energy, self.weight_att)
        att = K.softmax(att)

        # calc continuous regularization
        if self.continuous:
            loss = K.sum(K.sum(K.abs(att[:,1:] - att[:,0:-1]), 0, True)) / self.n_timestep
            self.add_loss(loss, inputs)
        # weighted sum of hidden outputs
        # att: (batch, 1, timestep), input: (batch, time step, hidden)
        att = tf.expand_dims(att, -1)
        # output = K.batch_dot(att, inputs)
        # output = tf.squeeze(output, 1)
        return att

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.n_timestep, 1
