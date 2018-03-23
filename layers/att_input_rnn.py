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

# Input is last hidden state of RNN, and generate weights for time steps and sensors
# attention only on time, is a special case where sensor = 1
class Attention(Layer):
    def __init__(self, n_feature=12, n_sensor=1, continuous=False, **kwargs):

        self.n_sensor = n_sensor
        self.n_feature = n_feature
        self.continuous = continuous

        if n_feature % n_sensor != 0:
            raise ValueError("Num features must be multiple of num sensors")

        assert "batch_size" in kwargs
        self.batch_size = kwargs["batch_size"]
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.n_hidden = input_shape[-1]
        self.n_timestep = input_shape[-2]

        # the attention weight based on last time step hidden state
        self.weight_energy = self.add_weight(name='weight_energy',
                                             shape=(self.n_hidden, self.n_timestep * self.n_sensor),
                                             initializer='uniform')

        self.bias_energy = self.add_weight(name='bias_eng',
                                           shape=[self.n_timestep * self.n_sensor],
                                           initializer='zero')

        # the attention weight based on last time step hidden state
        self.weight_att = self.add_weight(name='weight_att',
                                          shape=(self.n_timestep * self.n_sensor, self.n_timestep * self.n_sensor),
                                          initializer='uniform')

        super(Attention, self).build(input_shape)
        return

    def call(self, inputs, **kwargs):
        rank = inputs.get_shape().ndims
        assert rank == 3

        energy = inputs[:, -1]
        # energy: (batch, hidden), weight: (hidden, timestep * sensors)
        energy = K.dot(energy, self.weight_energy) + self.bias_energy
        energy = K.tanh(energy)

        # calculate attention
        att = K.dot(energy, self.weight_att)
        att = K.softmax(att) * self.n_sensor

        # format output
        # att: (batch, timestep * sensor) -> (batch, timestep, num_features)
        att = K.reshape(att, (self.batch_size, self.n_timestep, self.n_sensor, 1))
        att = K.repeat_elements(att, self.n_feature / self.n_sensor, 3)
        att = K.reshape(att, (self.batch_size, self.n_timestep, self.n_feature))
        return att

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.n_timestep, self.n_feature
