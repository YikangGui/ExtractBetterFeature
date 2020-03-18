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
from resnet_model import Resnet
#from resnet_model_256 import Resnet
import configparser as ConfigParser

confit_fp = './yang.conf'
config = ConfigParser.ConfigParser()
config._interpolation = ConfigParser.BasicInterpolation()
config.read(confit_fp)

batch_size = config.getint("MODEL", "batch_size")
epochs = config.getint("MODEL", "epochs")
data_augmentation = config.getboolean("MODEL", "data_augmentation")
num_classes = config.getint("MODEL", "num_classes")
dropout_rate_fc = config.getfloat("MODEL", "dropout_rate_fc")
dropout_rate_cc = config.getfloat("MODEL", "dropout_rate_cc")
n = config.getint("MODEL", "n")
version = config.getint("MODEL", "version")
pre_model_fp = config.get("MODEL", "pre_model_fp")
Flag_save_model = config.getboolean("MODEL", 'Flag_save_model')
subtract_pixel_mean = config.getboolean("MODEL", "subtract_pixel_mean")
model_key = config.get("MODEL", "model_key")
model_path = config.get("MODEL", 'model_path')
load_weights = config.getboolean("MODEL", 'load_weights')
final_pool = config.getint("MODEL", "final_pool")
Train = config.getboolean("MODEL", 'Train')
model_save_fp = config.get("MODEL", 'model_save_fp')
train_label_20_fp = config.get('FILE_PATH', 'train_label_20')
train_lable_fp = config.get('MODEL', 'train_lable')
train_imageName_Lable_fp = config.get("MODEL", 'train_imageName_Lable_fp')
image_path = config.get("MODEL", 'image_path')

if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2


model_type = 'ResNet_%s' % (model_key)
save_dir = os.path.join(config.get('MODEL', 'data_pre_pt'), 'saved_models')

x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable_fp, train_imageName_Lable_fp, image_path)
num_classes = train_cate_num
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

model_name = 'cifar10_%s_model.{epoch:03d}.{val_acc:03f}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

model = Resnet().buildModel(version=version, input_shape=[x_train.shape[1],x_train.shape[2],3], depth=depth, num_classes=num_classes, final_pool=final_pool, dropout_rate_fc=dropout_rate_fc, Flag_save_model=Flag_save_model)
if load_weights and False:
    model.load_weights(pre_model_fp, by_name=True)
model.compile(loss='categorical_crossentropy',
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
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None)
        datagen.fit(x_train)    
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0]//batch_size,
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=2, workers=4,
                            callbacks=callbacks,
                            validation_steps=x_test.shape[0]//batch_size)
'''
train_result = model.evaluate(x_train, y_train, verbose=1)
print(train_result)
test_result = model.evaluate(x_test, y_test, verbose=1)
print(test_result)
'''
'''
y_pre = model.predict(x_test, verbose=1)
#MyFunction.analysisClass(y_pre, y_test, num_classes, train_lable_fp)
'''
if Flag_save_model:
    if os.path.exists(model_path) == False:
            os.mkdir(model_path)
    model.save(model_save_fp)
    print("The model has been saved")
