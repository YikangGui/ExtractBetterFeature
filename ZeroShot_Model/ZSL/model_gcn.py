#-*- coding: UTF-8 -*- 
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS
# Q1:  l2_normalize 为什么维度要选择1
# Q2： imagefeature中是否存在负值


class GCN_dense(object):
    def __init__(self, placeholders, input_dim, flag_gcn=True):
        self.name = self.__class__.__name__.lower()
        self.flag_gcn = flag_gcn
        self.inputs = placeholders['features']
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.input_dim = input_dim
        self.placeholders = placeholders
        # self.input_dim = self.inputs.get_shape().as_list()[1]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])
        self.activations = []
        self.layers = []
        self.activations = []
        self.vars = {}
        self.loss = 0
        self.outputs = None
        self.opt_op = None
        self.grads = None
        self.build()

    def build(self):
        # build network layers
        self.addNetworkLayers()

        # bulid network structure
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self.loss_function()
        self.opt_op = self.optimizer.minimize(self.loss)
        # self.grads = tf.gradients(self.loss, self.inputs)[0]  # does not work on sparse vector

    def loss_function(self):
        # Weight decay loss
        self.loss_weight = 0
        if self.flag_gcn:
            for i in range(len(self.layers)):
                for var in self.layers[i].vars.values():
                    self.loss_weight += FLAGS.weight_decay * tf.nn.l2_loss(var)
        else:
            for i in range(len(self.layers)):
                self.loss_weight = FLAGS.weight_decay*(tf.nn.l2_loss(self.layers[i].W_left_w1) + tf.nn.l2_loss(self.layers[i].b_left_w1))

        self.loss_diff = mask_mse_loss(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
        # self.loss_diff = mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1), self.placeholders['labels_mask'])
        # self.loss_diff *= 10
        self.loss = (self.loss_weight + self.loss_diff)

    def addNetworkLayers(self):
        if self.flag_gcn:
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=self.output_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: tf.maximum(x, 0.01 * x),
                                                # act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                                bias=True,
                                                dropout=False))

        else:
            self.layers.append(NN_Norm(input_dim=self.input_dim, output_dim=self.output_dim))
