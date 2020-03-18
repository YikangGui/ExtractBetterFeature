from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
import pickle as pkl
import numpy as np
#from utils import *
from wheel import *
from model_gcn import GCN_dense
from wheel import MyFunction, Generator_gcn, Specific, ComputeTestAcc
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--version', dest='version',
                        help='version',
                        default='0', type=str)
    parser.add_argument('--seed', dest='seed',
                        help='seed',
                        default=78, type=int)
    args = parser.parse_args()
    return args
args = parse_args()
seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

config = MyFunction.gainConfig()
A_matrix_fp = config.get('Model_GCN', 'A_matrix_fp')
fc_vector_alltrain_fp = config.get('Model_GCN', 'fc_vector_alltrain_fp')
wordEmbedding_fp = config.get('Model_GCN', 'wordembEdding_fp')
output_path = config.get('Model_GCN', 'output_path')
output_path_ImageFeature = config.get('Model_GCN', 'output_path_ImageFeature')
output_path_result = config.get('Model_GCN', 'output_path_result')
test_image_unlable_fp = config.get('Model_GCN', 'test_image_labled_fp')

model_key = config.get('Model_GCN', 'model_key')
model_name = config.get('Model_GCN', 'model_name')
use_trainval = config.getboolean('Model_GCN', 'use_trainval')
test_num = config.getint('Model_GCN', 'test_num')
ImageFeatureDimention = config.getint('Model_GCN', 'ImageFeatureDimention')
label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
train_sample_num = config.getint('Model_GCN', 'train_sample_num')
fc_vector_test_fp = config.get('Model_GCN', 'fc_vector_test_npy_fp')
test_image_labled_fp = config.get('Model_GCN', 'test_image_labled_fp')
train_label_fp = config.get('Model_GCN', 'train_label_fp')
train_imageName_Lable_fp = config.get('Model_GCN', 'train_imageName_Lable_fp')
train_feat_fp = config.get('Model_GCN', 'fc_vector_alltrain_npy_fp')
Flag_AllImage = config.getboolean('Model_GCN', 'Flag_AllImage')
Falg_saveImageFeature = config.getboolean('Model_GCN', 'Falg_saveImageFeature')
Flag_fewImage = False
Flag_updateMask = False

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


# adj is A materix; features is imput of GCN; 
# train_test_label_order is the order of every row in adj
# testIndex and trainIndex are 0/1 vectors
adj, features, train_test_label_order, TrainIndex, TestIndex = Generator_gcn.load_data_zero_shot(A_matrix_fp, wordEmbedding_fp, fc_vector_alltrain_fp, ImageFeatureDimention, label2ClassName_fp, train_label_fp)
print(features.shape)
# label_allFeat_map is the label of GCN
label_allFeat_map = Generator_gcn.load_label_allFeat_map(train_imageName_Lable_fp, train_feat_fp)
y_train = Generator_gcn.nextBatchImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention)
y_train_1 = Generator_gcn.nextBatchImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention)
y_train_2 = Generator_gcn.nextBatchImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention)

train_mask = Generator_gcn.sample_mask_sigmoid(TrainIndex, y_train.shape[0], y_train.shape[1])
print("y shape", y_train.shape)
fc_vector_test = np.load(fc_vector_test_fp)
try:
    fc_vector_test = np.load(fc_vector_test_fp)
    real_test_label = MyFunction.gainRealLable(test_image_labled_fp)
    test_lable_orderd = [e for inx, e in enumerate(train_test_label_order) if TestIndex[inx] == 1]
except:
    print("no real_test_label")


if FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=()),
    'dropout': tf.placeholder(tf.float32, shape=())
}

model = model_func(placeholders, input_dim=features.shape[1])
sess = tf.Session(config=create_config_proto())
sess.run(tf.global_variables_initializer())

now_lr = FLAGS.learning_rate
now_dropout = FLAGS.dropout

for epoch in range(FLAGS.epochs):
    t = time.time()
    # every iteration, we will tarin model using different image feats
    if Flag_fewImage:
        y_train = y_train_1 if epoch%2 == 0 else y_train_2
    if Flag_AllImage:
        y_train = Generator_gcn.meanImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention)
    if Flag_updateMask:
        train_mask = Generator_gcn.updatesample_mask_sigmoid(TrainIndex, y_train.shape[0], y_train.shape[1])
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})
    feed_dict.update({placeholders['dropout']: now_dropout})

    outs = sess.run([model.opt_op, model.loss, model.loss_diff, model.optimizer._lr], feed_dict=feed_dict)

    if epoch % 1000 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.8f}".format(float(outs[3])))

    if epoch % 1000 == 0:
        feed_dict.update({placeholders['dropout']: 0})
        outs = sess.run(model.outputs, feed_dict=feed_dict)

        if True:
            imageFeature_pre_test = np.array([e for inx, e in enumerate(outs) if TestIndex[inx] == 1])
            result, cur_epoch_acc = MyFunction.compute_accuracy(imageFeature_pre_test, fc_vector_test, test_lable_orderd, real_test_label)
            merge_result = ComputeTestAcc.mergeResult(result, test_image_labled_fp)
            result_name = output_path_result + 'result_' + str(epoch) + "_" + str(cur_epoch_acc) + '.txt'
            MyFunction.saveVector(result_name, merge_result)

        filename = output_path + '/' + model_name + '_' + model_key + '_' + str(epoch) + '_imageFeature_predict.txt'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        Test_image_feat_file = []
        for inx in range(len(test_lable_orderd)):
            cur_label_ = test_lable_orderd[inx]
            cur_imageFeat_string = " ".join([str(e) for e in imageFeature_pre_test[inx]])
            Test_image_feat_file.append([cur_label_ + ":" + cur_imageFeat_string])
        MyFunction.save2DimentionVector(filename, Test_image_feat_file)
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename + '\n')

if Falg_saveImageFeature:
    TestImageFeaturePredicted = []
    for i in range(len(imageFeature_pre_test)):
        cur_label_imageFeature = test_lable_orderd[i] + ':' + " ".join([str(e) for e in list(imageFeature_pre_test[i])])
        TestImageFeaturePredicted.append(cur_label_imageFeature)
    if not os.path.exists(output_path_ImageFeature):
        os.makedirs(output_path_ImageFeature)
    savefp = output_path_ImageFeature + '/' + 'TestImageFeaturePredicted_C_' + args.version + '.txt'
    MyFunction.saveVector(savefp, TestImageFeaturePredicted)

print("Optimization Finished!")
sess.close()
