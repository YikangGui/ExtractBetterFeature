from nn_model import Model
from wheel import Generator, MyFunction
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import os


config = MyFunction.gainConfig()

wordEmbeddingDimention = config.getint('NN_model', 'wordEmbeddingDimention')
imageFeatureDimention = config.getint('NN_model', 'imageFeatureDimention')
epoch = config.getint('NN_model', 'epoch')
learning_rate = config.getfloat('NN_model', 'learning_rate')
Flag_oneVsAllPicture = config.getboolean('NN_model', 'Flag_oneVsAllPicture')
output_path = config.get('NN_model', 'output_path')
ckpt_path = config.get('NN_model', 'ckpt_path')
label_wordEmbedding_fp = config.get('NN_model', 'label_wordEmbedding_fp')
fc_vector_alltrain_fp = config.get('NN_model', 'fc_vector_alltrain_fp')
fc_vector_alltrain_npy_fp = config.get('NN_model', 'fc_vector_alltrain_npy_fp')
batchSize = config.getint('NN_model', 'batchSize')
FC1_dimention = config.getint('NN_model', 'FC1_dimention')
test_lable_fp = config.get('NN_model', 'test_lable_fp')
Falg_txt = config.getboolean('NN_model', 'Falg_txt')
Flag_npy = config.getboolean('NN_model', 'Flag_npy')
train_imageName_Lable_fp = config.get('NN_model', 'train_imageName_Lable_fp')
fc_vector_test_fp = config.get('NN_model', 'fc_vector_test_npy_fp')
test_image_labled_fp = config.get('NN_model', 'test_image_labled_fp')
Flag_train = config.getboolean("NN_model", 'Flag_train')
Test_set = config.get('NN_model', 'Test_set')
mask_flag_fp = config.get('NN_model', 'mask_flag_fp')
use_mask = config.getboolean('NN_model', 'use_mask')

if Flag_oneVsAllPicture:
    train_features, train_lables = Generator.gainTrainData_onePicture(trainImageFeature_fp, trainExtractWordembedding_fp)
    test_features = Generator.gainTestData(testExtractWordembedding_fps)
elif Falg_txt:
    train_features, train_lables = Generator.gainTrainData_allPictures_txt(label_wordEmbedding_fp, fc_vector_alltrain_fp)
    test_features = Generator.gainTestData_allPictures(label_wordEmbedding_fp, test_lable_fp)
elif Flag_npy:
    if use_mask:
        train_features, train_lables = Generator.gainTrainData_allPictures_npy(label_wordEmbedding_fp, fc_vector_alltrain_npy_fp, train_imageName_Lable_fp, mask_flag_fp)
    else:
        train_features, train_lables = Generator.gainTrainData_allPictures_npy(label_wordEmbedding_fp, fc_vector_alltrain_npy_fp, train_imageName_Lable_fp)
    test_features = Generator.gainTestData_allPictures(label_wordEmbedding_fp, test_lable_fp)
print("x:", len(train_features), len(train_features[0]))
print("y:", len(train_lables), len(train_lables[0]))
print("loda data is ok")
test_lable_orderd = MyFunction.readVector(test_lable_fp)

try:
    fc_vector_test = np.load(fc_vector_test_fp)
    real_test_label = MyFunction.gainRealLable(test_image_labled_fp)
except:
    print("no real_test_label")

def cosine_tf(q, a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    print(pooled_len_1.shape)
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = 1 - tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    score = tf.reduce_mean(score)
    return score 

model = Model().bulid(wordEmbeddingDimention, imageFeatureDimention)

word_features = tf.placeholder(tf.float32, [None, wordEmbeddingDimention])
visual_features = tf.placeholder(tf.float32, [None, imageFeatureDimention])
W_left_w1 = Model.weight_variable([wordEmbeddingDimention, imageFeatureDimention])
b_left_w1 = Model.bias_variable([imageFeatureDimention])

W_left_w2 = Model.weight_variable([imageFeatureDimention, wordEmbeddingDimention])
b_left_w2 = Model.bias_variable([wordEmbeddingDimention])

left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
left_w2 = tf.nn.relu(tf.matmul(left_w1, W_left_w2) + b_left_w2)
# left_w1_norm = tf.nn.l2_normalize(left_w1, dim=1)
# visual_features_norm = tf.nn.l2_normalize(visual_features, dim=1)
# loss_w = tf.losses.cosine_distance(visual_features, left_w1, axis=1)
loss_w_cons = cosine_tf(visual_features, left_w1)
# loss_w = tf.losses.mean_squared_error(visual_features, left_w1)
loss_w = tf.reduce_mean(tf.square(left_w1 - visual_features))
# loss_w2 = tf.reduce_mean(tf.square(left_w2 - word_features))
regularisers_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))
loss_all = loss_w + 1e-3 * regularisers_w + loss_w_cons
# loss_all = loss_w + 0 * regularisers_w + loss_w2
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=3)
min_los = 1000
if Flag_train:
    for i in range(epoch):
        if Flag_oneVsAllPicture:
            _, cur_loss_all, cur_loss_real = sess.run([train_step, loss_all, loss_w], feed_dict={word_features: train_features, visual_features: train_lables})
        else:
            batchSize_train_features, batchSize_train_lables = Generator.gainNextBatchSizeSample(train_features, train_lables, batchSize)
            _ = sess.run([train_step], feed_dict={word_features: batchSize_train_features, visual_features: batchSize_train_lables})   
        if i % 1000 == 0:
            los, los_all = sess.run([loss_w, loss_all], feed_dict={word_features: batchSize_train_features, visual_features: batchSize_train_lables})
            print('loss: '+str(los) +'loss_all ' +str(los_all))
            if los < min_los:
                min_los = los
                saver.save(sess, ckpt_path+'/mnist.ckpt', global_step=i+1)
        if i % 1000 == 0:
            print("save at epoch", i)
            imageFeature_pre_test = sess.run(left_w1, feed_dict={word_features: test_features})
            try:
                # imageFeature_pre_test_norm = preprocessing.normalize(imageFeature_pre_test, norm='l2')
                # fc_vector_test_norm = preprocessing.normalize(fc_vector_test, norm='l2')
                result = MyFunction.compute_accuracy(imageFeature_pre_test, fc_vector_test, test_lable_orderd, real_test_label)
            except:
                pass
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            save_predict_imageFeature_fp = output_path + str(i) + '_' + Test_set + '_imageFeature_predict.txt'
            MyFunction.save2DimentionVector(save_predict_imageFeature_fp, imageFeature_pre_test)
else:
    model_file=tf.train.latest_checkpoint(ckpt_path)
    saver.restore(sess, model_file)
    imageFeature_pre_test = sess.run(left_w1, feed_dict={word_features: test_features})
    #MyFunction.compute_accuracy(imageFeature_pre_test, fc_vector_test, test_lable_orderd, real_test_label)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(output_path)
    save_predict_imageFeature_fp = output_path + str(100) + '_' + Test_set + '_imageFeature_predict.txt'
    MyFunction.save2DimentionVector(save_predict_imageFeature_fp, imageFeature_pre_test)
