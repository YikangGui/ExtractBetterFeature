from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
import pickle as pkl
import numpy as np
from utils import *
from models import GCN_dense_mse
from wheel import MyFunction, Generator_gcn

config = MyFunction.gainConfig()
seed = config.getint('Model', 'seed')

np.random.seed(seed)
tf.set_random_seed(seed)
A_matrix_fp = config.get('Model', 'A_matrix_fp')
fc_vector_alltrain_fp = config.get('Model', 'fc_vector_alltrain_fp')
wordEmbedding_fp = config.get('Model', 'wordembEdding_fp')
model_key = config.get('Model', 'model_key')
model_name = config.get('Model', 'model_name')
use_trainval = config.getboolean('Model', 'use_trainval')
output_path = config.get('Model', 'output_path')
patience_len = config.getint('Model', 'patience_len')
patience_value = config.getfloat('Model', 'patience_value')
test_num = config.getint('Model', 'test_num')
ImageFeatureDimention = config.getint('Model', 'ImageFeatureDimention')
label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
train_sample_num = config.getint('Model', 'train_sample_num')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '../../data/glove_res50/', 'Dataset string.')
flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_string('save_path', '../output/', 'save dir')
flags.DEFINE_integer('epochs', 8000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 800, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1000, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 2024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 2024, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


#feat_suffix = 'allx_dense'

adj, features, y_train, train_mask = Generator_gcn.load_data_zero_shot(A_matrix_fp, wordEmbedding_fp, fc_vector_alltrain_fp, train_sample_num, test_num, ImageFeatureDimention, label2ClassName_fp)
features, div_mat = preprocess_features_dense2(features)

if FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

model = model_func(placeholders, input_dim=features.shape[1], logging=True)
sess = tf.Session(config = create_config_proto())
sess.run(tf.global_variables_initializer())

now_lr = FLAGS.learning_rate
loss_all = []
for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)
    loss_all.append(outs[1])
    if epoch % 20 == 0:
        #now_lr = MyFunction.process_lr(outs[3], loss_all, patience_len, patience_value)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))
    if epoch % 50 == 0:
        outs = sess.run(model.outputs, feed_dict=feed_dict)
        filename = output_path + '/' + model_name + '_' + model_key + '_' + str(epoch) + '_imageFeature_predict.txt'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        MyFunction.save2DimentionVector(filename, outs)
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)    

print("Optimization Finished!")
sess.close()
