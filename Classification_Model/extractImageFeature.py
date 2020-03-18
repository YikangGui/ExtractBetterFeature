from __future__ import print_function
from keras import backend as K
from utils import Extractor, MyFunction, Analysis
from keras.models import load_model
import configparser as ConfigParser
import numpy as np
import os

confit_fp = './yang.conf'
config = ConfigParser.ConfigParser()
config._interpolation = ConfigParser.BasicInterpolation()
config.read(confit_fp)
mode_fp = config.get('EXTRACT_IMAGE_FEATURE', 'extract_imageFeature_mode')

model = load_model(mode_fp)
model.summary()

if config.getboolean('EXTRACT_IMAGE_FEATURE', 'extract_all_train'):
    print('getting all train fc vector...')
    model_key = config.get('EXTRACT_IMAGE_FEATURE', 'model_key')
    model_path = config.get("EXTRACT_IMAGE_FEATURE", 'model_path')
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
    save_fp = model_path + config.get('EXTRACT_IMAGE_FEATURE', 'fc_vector_alltrain')
    train_lable_fp = config.get('EXTRACT_IMAGE_FEATURE', 'train_lable_fp')
    train_imageName_Lable_fp = config.get("EXTRACT_IMAGE_FEATURE", 'train_imageName_Lable_fp')
    image_path_train = config.get("EXTRACT_IMAGE_FEATURE", 'image_path_train')
    X, Y = Extractor.gainVal(train_lable_fp, train_imageName_Lable_fp, image_path_train)
    print("total x:", X.shape[0])
    X = X.astype('float32') / 255
    x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable_fp, train_imageName_Lable_fp, image_path_train)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    X -= x_train_mean
    pre_y, pre_fc = model.predict(X, verbose=1)
    #print(MyFunction.computeAcc(Y, pre_y))
    MyFunction.saveFcLayer(pre_y, Y, pre_fc, save_fp, train_lable_fp)

if config.getboolean('EXTRACT_IMAGE_FEATURE', 'extract_val'):
    print("getting val fc vector...")
    model_key = config.get('MODEL', 'model_key')
    train_lable = config.get('MODEL', 'train_lable')
    val_lable_fp = config.get('FILE_PATH', 'val_lable')

    model_path = './data/%s/' % (model_key)
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
    save_fp = model_path + config.get('SAVE_FILE_PATH', 'fc_vector_val')
    X, Y = Extractor.gainVal(val_lable_fp)
    X = X.astype('float32') / 255
    x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    X -= x_train_mean
    pre_y, pre_fc = model.predict(X, verbose=1)
    MyFunction.saveValFcLayer(Y, pre_fc, save_fp, val_lable_fp)

if config.getboolean('EXTRACT_IMAGE_FEATURE', 'gainTestInageFeature'):
    #is for 40 labled test
    print("getting testImageFeature...")
    model_key = config.get('MODEL', 'model_key')
    train_lable = config.get('MODEL', 'train_lable')
    val_lable_fp = config.get('FILE_PATH', 'test_lable')

    model_path = './data/%s/' % (model_key)
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
    save_fp = config.get('FILE_PATH', 'test_image_feature_fp')
    X, Y = Extractor.gainVal(val_lable_fp, isForTest=True)
    X = X.astype('float32') / 255
    x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    X -= x_train_mean
    pre_y, pre_fc = model.predict(X, verbose=1)
    MyFunction.saveValFcLayer(Y, pre_fc, save_fp, val_lable_fp)

if config.getboolean('EXTRACT_IMAGE_FEATURE', 'extract_test'):
    print("extracting test fc vector and gaining final_result...")
    train_lable_fp = config.get('EXTRACT_IMAGE_FEATURE', 'train_lable_fp')
    train_imageName_Lable_fp = config.get("EXTRACT_IMAGE_FEATURE", 'train_imageName_Lable_fp')
    test_imageName_Lable_fp = config.get("EXTRACT_IMAGE_FEATURE", 'test_imageName_Lable_fp')
    image_path_test = config.get("EXTRACT_IMAGE_FEATURE", 'image_path_test')
    image_path_train = config.get("EXTRACT_IMAGE_FEATURE", 'image_path_train')
    model_key = config.get('EXTRACT_IMAGE_FEATURE', 'model_key')
    model_path = config.get("EXTRACT_IMAGE_FEATURE", 'model_path')
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
    save_test_fc_fp = model_path + config.get("EXTRACT_IMAGE_FEATURE", 'fc_vector_test')

    X = Extractor.gainRealTest(test_imageName_Lable_fp, image_path_test)
    X = X.astype('float32') / 255
    x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable_fp, train_imageName_Lable_fp, image_path_train)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    X -= x_train_mean
    pre_y, pre_fc = model.predict(X, verbose=1)
    MyFunction.saveTestFcLayer(pre_fc, save_test_fc_fp)
    print("is ok")

if config.getboolean('EXTRACT_IMAGE_FEATURE', 'AnalysisTestPictureFromCalssificationModel'):
    #is for 40 labled test
    print("getting testImageFeature...")
    model_key = config.get('MODEL', 'model_key')
    train_lable = config.get('MODEL', 'train_lable')
    val_lable_fp = config.get('FILE_PATH', 'test_lable')
    test_imagepath_lable_fp = config.get('FILE_PATH', 'test_imagepath_lable')
    model_path = './data/%s/' % (model_key)
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
    save_fp = config.get('FILE_PATH', 'test_image_feature_fp')
    X, Y = Extractor.gainVal(val_lable_fp, isForTest=True)
    X = X.astype('float32') / 255
    x_train, y_train, x_test, y_test, train_cate_num = Extractor.gainTrainAndTest(train_lable)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    X -= x_train_mean
    pre_y, pre_fc = model.predict(X, verbose=1)
    Analysis.analysisTestDataPredictionResult(pre_y, train_lable, test_imagepath_lable_fp)
    #MyFunction.saveValFcLayer(Y, pre_fc, save_fp, val_lable_fp)