import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys
import random
import re

class Model:
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def bulid(self, wordEmbeddingDimention, imageFeatureDimention, learning_rate=0.0001):	
        word_features = tf.placeholder(tf.float32, [None, wordEmbeddingDimention])
        visual_features = tf.placeholder(tf.float32, [None, imageFeatureDimention])
        W_left_w1 = Model.weight_variable([wordEmbeddingDimention, imageFeatureDimention])
        b_left_w1 = Model.bias_variable([imageFeatureDimention])
        self.left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
        self.loss_w = tf.reduce_mean(tf.square(self.left_w1 - visual_features))
        regularisers_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))
        self.loss_total = self.loss_w + 1e-3 * regularisers_w
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_w)
