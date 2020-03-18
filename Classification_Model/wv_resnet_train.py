from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from utils import Extractor, MyFunction
from wv_resnet_model import Resnet
import configparser as ConfigParser

confit_fp = './yang.conf'
config = ConfigParser.RawConfigParser()
config.read(confit_fp)

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 500
data_augmentation = True
dropout_rate_fc = 0
dropout_rate_cc = 0.6
n = 3
version = 1
pre_model_fp = './saved_models/cifar10_ResNet_20traindelete_model.043.0.544866.h5'
subtract_pixel_mean = True
model_key = "wv_resnet_model"
model_path = './train/%s/' % (model_key)

load_weights = False
final_pool = 4
Train = True

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet_%s' % (model_key)

train_txt_fp = config.get("FILE_PATH", 'train_imagepath_label')
train_pt = './train/train/'
class_wordembedings_txt_fp = config.get("FILE_PATH", 'class_wordembeddings_reduced_100')
label_list_fp = config.get('FILE_PATH', 'lable_list')
x_train, y_train, x_test, y_test = Extractor.readTrainDataVersion2(train_txt_fp, train_pt, class_wordembedings_txt_fp, label_list_fp)
print(y_train.shape)

input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.{val_acc:03f}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

model = Resnet().buildModel(version=version, input_shape=[64,64,3], depth=depth, final_pool=final_pool, dropout_rate_fc=dropout_rate_fc)
if load_weights:
    model.load_weights(pre_model_fp)
model.compile(loss='mean_absolute_error',
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()


# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_reducer = ReduceLROnPlateau(factor=0.5,
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer]
if Train:
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None)
            # fraction of images reserved for validation (strictly between 0 and 1)
            #validation_split=0.0)    

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)    

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0]//batch_size,
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=2, workers=4,
                            callbacks=callbacks,
                            validation_steps=x_test.shape[0]//batch_size)

model.save('./my_model.h5')
