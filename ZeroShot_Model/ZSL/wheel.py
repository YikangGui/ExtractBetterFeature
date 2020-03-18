#-*- coding: UTF-8 -*- 
import numpy as np
import configparser as ConfigParser
from progressbar import *
from sklearn.decomposition import PCA
import os,shutil
import pandas as pd
import networkx as nx
from random import random, randint
import operator
import math
from numpy import *
import os
import colorsys
import numpy as np
from sklearn.manifold import TSNE
from sklearn import svm
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.manifold import TSNE
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import itertools


class Generator_searchNN:
    @staticmethod
    def readFeatAndLabel(file_list, fc_vector_test_fp, test_image_labled_fp):
        for cur_root, _, cur_files in os.walk(file_list):
            root = cur_root
            files = cur_files

        allfeat_line = []
        for cur_file in files:
            cur_path = root + cur_file
            cur_lines = MyFunction.readVector(cur_path)
            allfeat_line += cur_lines

        labels = []
        feats = []
        for cur_line in allfeat_line:
            cur_label = cur_line.split(":")[0]
            cur_feat = [float(e) for e in cur_line.split(":")[1].split()]
            labels.append(cur_label)
            feats.append(cur_feat)

        labels_unique = set(labels)
        label_inx_map = {cur_label: inx for inx, cur_label in enumerate(labels_unique)}
        labels_inx = [label_inx_map[cur_label] for cur_label in labels]

        feat_test = np.load(fc_vector_test_fp)
        test_label = MyFunction.gainRealLable(test_image_labled_fp)
        test_label_inx = [label_inx_map[cur_label] for cur_label in test_label]
        return feats, labels_inx, feat_test, test_label_inx



class KNN:
    def __init__(self, Image_feature_fp_list, fc_vector_test_fp,\
                 test_image_labled_fp=None, test_feat_fp=None):
        self.Image_feature_fp_list = Image_feature_fp_list
        self.fc_vector_test_fp = fc_vector_test_fp
        self.test_image_labled_fp = test_image_labled_fp
        self.test_feat_fp = test_feat_fp
        if self.test_feat_fp:
            print("sss")
            self.label_allFeat_map = Generator_gcn.load_label_allFeat_map(test_image_labled_fp, test_feat_fp)

    def gainTrainData(self, Image_feature_fp_list):
        Y = []
        X = []
        for cur_fp in Image_feature_fp_list:
            print(cur_fp)
            image_features = MyFunction.readVector(cur_fp)
            for line in image_features:
                cur_label = line.split(":")[0]
                cur_fc_value = line.strip().split(":")[1]
                cur_fc_value = cur_fc_value.split(" ")
                cur_fc_value = [float(e) for e in cur_fc_value]
                Y.append(cur_label)
                X.append(cur_fc_value)
        return X, Y

    def gainTestX(self, fc_vector_test_fp):
        fc_vector_test = np.load(fc_vector_test_fp)
        return fc_vector_test

    def gainTestY(self, test_image_labled_fp):
        test_image_labeld = MyFunction.readVector(test_image_labled_fp)
        Y = []
        for line in test_image_labeld:
            Y.append(line.split('\t')[1])
        return Y

    def updateTrainX(self, train_X, train_Y):
        all_label = list(self.label_allFeat_map.keys())
        total_class = len(all_label)
        ran = randint(0, total_class)
        selected_label = all_label[ran]
        if selected_label == 'ZJL999':
            return train_X
        curLabel_allFeat = self.label_allFeat_map[selected_label]
        total_feat = len(curLabel_allFeat)
        ran = randint(0, total_feat)
        selected_feat = curLabel_allFeat[ran]
        index_list = []
        for inx, label in enumerate(train_Y):
            if selected_label == label:
                index_list.append(inx)
        total_same_label = len(index_list)
        selected_index = index_list[randint(0, total_same_label)]
        train_X[selected_index] = selected_feat
        return train_X


    def predictByKNN_updateVersion(self, n_neighbors):
        train_X, train_Y = self.gainTrainData(self.Image_feature_fp_list)
        test_X = self.gainTestX(self.fc_vector_test_fp)
        test_Y = self.gainTestY(self.test_image_labled_fp)
        p = 1.0 * len(train_X) / len(test_X)
        print("p", p)
        best_acc = -1
        for i in range(10):
            train_X_tem = self.updateTrainX(train_X, train_Y)
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
            knn.fit(train_X_tem, train_Y)
            test_Y_predict = knn.predict(test_X)
            try:
                cur_acc = metrics.accuracy_score(test_Y_predict, test_Y)/0.9865
                cur_acc_p = (cur_acc - p) / (1 - p)
                print("acc", cur_acc_p, "best_acc", best_acc, 'o_acc', cur_acc)
                if cur_acc_p > best_acc:
                    best_acc = cur_acc_p
                    train_X = train_X_tem
            except:
                print("can not compute acc")
        return test_Y_predict

    def predictBySVM(self):
        train_X, train_Y = self.gainTrainData(self.Image_feature_fp_list)
        label2cnnidx = {label_code: idx for idx, label_code in enumerate(list(set(train_Y)))}
        train_Y = [label2cnnidx[x] for x in train_Y]
        print(type(train_X[0][0]))
        print(type(train_Y[0]))
        print("train num", len(train_X))
        test_X = self.gainTestX(self.fc_vector_test_fp)
        test_Y = self.gainTestY(self.test_image_labled_fp)
        clf = svm.SVC(C=1000)
        label2cnnidx['ZJL999'] = 999
        test_Y = [label2cnnidx[x] for x in test_Y]
        #clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
        clf.fit(train_X, train_Y)
        test_Y_predict = clf.predict(test_X)
        try:
            cur_acc = metrics.accuracy_score(test_Y_predict, test_Y)
            print("acc", cur_acc)
        except:
            print("can not compute acc")

    @staticmethod
    def computeDistance(train_X, train_Y, test_Y_predict, test_Y, test_X):
        pos_count = 0
        neg_count = 0
        pos_list = []
        neg_list = []
        for inx in range(len(test_Y)):
            cur_test_X = test_X[inx]
            cur_predic_label = test_Y_predict[inx]
            cur_real_label = test_Y[inx]
            distance = []
            for cur_train_X in train_X:
                distance.append(MyFunction.cosine_distance(cur_test_X, cur_train_X))
                # distance.append(MyFunction.L2distance(cur_test_X, cur_train_X))
            distance = sorted(distance)
            '''
            if cur_real_label == cur_predic_label:
                print("+", distance[-1], distance[-2])
            else:
                print("-", distance[-1], distance[-2])
            '''
            # if distance[-1] - distance[-2] > 0.04:
            if True:
                if cur_real_label == cur_predic_label:
                    pos_count += 1
                    pos_list.append(distance[-1] - distance[-2])
                else:
                    neg_count += 1
                    neg_list.append(distance[-1] - distance[-2])

            '''
            if distance[0] < 0.05:
                if cur_real_label == cur_predic_label:
                    pos_count += 1
                else:
                    neg_count += 1
            '''
        print(pos_count, neg_count)
        print(sum(pos_list)/len(pos_list))
        print(sum(neg_list)/len(neg_list))


    def predictByKNN(self, n_neighbors, num_class=65, select_k=55, epoch=10):
        train_X, train_Y = self.gainTrainData(self.Image_feature_fp_list)
        print("train num", len(train_X))
        test_X = self.gainTestX(self.fc_vector_test_fp)
        test_Y = self.gainTestY(self.test_image_labled_fp)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        knn.fit(train_X, train_Y)
        final_result = knn.predict(test_X)
        cur_acc = metrics.accuracy_score(final_result, test_Y)
        print(cur_acc)
        '''
        test_Y_predict = []
        for i in range(epoch):
            select_index = np.random.choice(num_class, select_k, replace=False)
            train_X = np.array(train_X)
            train_Y = np.array(train_Y)
            knn.fit(train_X[select_index], train_Y[select_index])
            cur_test_Y_predict = knn.predict(test_X)
            print(cur_test_Y_predict)
            try:
                cur_acc = metrics.accuracy_score(cur_test_Y_predict, test_Y)
                print("acc", cur_acc)
            except:
                print("can not compute acc")
            test_Y_predict.append(cur_test_Y_predict)
        final_result = []
        for i in range(len(test_Y)):
            cur_result = np.array(test_Y_predict)[:, i]
            voted_label = pd.value_counts(cur_result).index[0]
            final_result.append(voted_label)
        cur_acc = metrics.accuracy_score(final_result, test_Y)
        print("acc", cur_acc)
        '''


        '''
        knn2 = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        knn2.fit(train_X[13:59], train_Y[13:59])
        test_Y_predict2 = knn2.predict(test_X)
        Analysis.analysisDifferenceBetweenFinalResult_vector(test_Y, test_Y_predict, test_Y_predict2)
        '''
        # KNN.computeDistance(train_X, train_Y, test_Y_predict, test_Y, test_X)
        return final_result

    def predictByKNN_2Times(self, n_neighbors, num_class=45):
        train_X, train_Y = self.gainTrainData(self.Image_feature_fp_list)
        test_X = self.gainTestX(self.fc_vector_test_fp)
        test_Y = self.gainTestY(self.test_image_labled_fp)
        epoch = int(len(train_X)/num_class)
        all_predict_result = pd.DataFrame()
        all_train_X = []
        for cur_epoch in range(epoch-1):
            if cur_epoch % 3 == 0:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
            elif cur_epoch % 3 == 1:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='l2')
            else:
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='l1')
            begin_index = cur_epoch * 45
            end_index = (cur_epoch + 1) * 45
            knn.fit(train_X[begin_index:end_index], train_Y[begin_index:end_index])
            test_Y_predict = knn.predict(test_X)
            all_predict_result[cur_epoch] = test_Y_predict
            label_imageFeature_map = dict(zip(train_Y[begin_index:end_index], train_X[begin_index:end_index]))
            all_train_X.append(label_imageFeature_map)
            try:
                cur_acc = metrics.accuracy_score(test_Y_predict, test_Y)/0.9865
                print("acc", cur_acc)
            except:
                print("can not compute acc")

        begin_index = (epoch-1) * 45
        end_index = (epoch) * 45
        knn2 = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        knn2.fit(train_X[begin_index:end_index], train_Y[begin_index:end_index])
        test_Y_predict2 = knn2.predict(test_X)
        # Analysis.analysisDifferenceBetweenFinalResult_vector(test_Y, test_Y_predict, test_Y_predict2)
        '''
        # compute by mode
        final_result_mode = list(all_predict_result.mode(axis=1)[0])
        cur_acc = metrics.accuracy_score(final_result_mode, test_Y)/0.9865
        print("acc mode", cur_acc)
        # compute by min distance(max samiliarity)
        final_result_min = []
        for i_apr in range(all_predict_result.shape[0]):
            cur_pre_result = list(all_predict_result.iloc[i_apr])
            similarity_list = []
            cur_feat = test_X[i_apr]
            for i_cpr in range(len(cur_pre_result)):
                cur_ImageFeature = all_train_X[i_cpr][cur_pre_result[i_cpr]]
                cur_samilirity = MyFunction.cosineSimilarity(cur_feat, cur_ImageFeature)
                similarity_list.append(cur_samilirity)
            index_sorted = np.argsort(similarity_list)
            selected_index = index_sorted[-1]
            final_result_min.append(cur_pre_result[selected_index])
        cur_acc = metrics.accuracy_score(final_result_min, test_Y)/0.9865
        print("acc max samiliarity", cur_acc)

        # Analysis.analysisDifferenceBetweenFinalResult_vector(test_Y, final_result_mode, final_result_min)
        '''
        # ACC_Threshold
        count_single_P2 = 0
        count_single_max = 0
        count_common = 0
        count_improve = 0
        for i in range(len(test_Y)):
            cur_real_label = test_Y[i]
            #print(cur_real_label, all_predict_result.iloc[i].value_counts())
            cur_pre_result = list(all_predict_result.iloc[i])
            if cur_real_label in cur_pre_result and cur_real_label == test_Y_predict2[i]:
                count_common += 1
            elif cur_real_label not in cur_pre_result and cur_real_label == test_Y_predict2[i]:
                count_single_P2 += 1
            elif cur_real_label in cur_pre_result and cur_real_label != test_Y_predict2[i]:
                count_single_max += 1
                if test_Y_predict2[i] not in cur_pre_result:
                    count_improve += 1
        print("max acc", count_single_max/len(test_Y) + count_common/len(test_Y))
        print("common acc", count_common/len(test_Y))
        print("sigle acc", count_single_P2/len(test_Y))
        print("single max", count_single_max/len(test_Y))
        print("improve", count_improve/len(test_Y))

        begin_index = (epoch-1) * 45
        end_index = (epoch) * 45
        knn2 = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        train_X_45 = np.array(train_X[begin_index:end_index])
        train_Y_45 = np.array(train_Y[begin_index:end_index])
        test_Y_predict_post = []
        for i in range(len(test_Y)):
            cur_condidate = list(set(all_predict_result.iloc[i]))
            cur_condidate_inx = [list(train_Y_45).index(e) for e in cur_condidate]
            knn2.fit(train_X_45[cur_condidate_inx], train_Y_45[cur_condidate_inx])
            cur_test_Y_predict = knn2.predict(test_X[i].reshape(1, -1))
            test_Y_predict_post.append(cur_test_Y_predict)
        cur_acc = metrics.accuracy_score(test_Y_predict_post, test_Y)/0.9865
        print("acc mode", cur_acc)
        return test_Y_predict


#class TestC_wordEmbedding
class AnalysisAccOfEveryClass:
    @staticmethod
    def plotAccOfEveryClass(acc_fp_list):
        for cur_fp in acc_fp_list:
            label_acc = MyFunction.readVector(cur_fp)
            label_acc_list = []
            for line in label_acc:
                label_acc_list.append([line.split(':')[0].strip(" ").strip("ZJL"), float(line.split(':')[1].strip(" "))])
            label_acc_list = np.array(sorted(label_acc_list, key=lambda y:y[0]))
            plt.scatter(label_acc_list[:, 0], label_acc_list[:, 1])
        plt.show()

    @staticmethod
    def devideClassByAccLevel(accOfEvervClass_fp, save_fp):
        accOfEvervClass = MyFunction.readVector(accOfEvervClass_fp)
        label_acc_map = {}
        for line in accOfEvervClass:
            cur_label = line.split(":")[0].strip(" ")
            cur_acc = float(line.split(":")[1].strip(" "))
            label_acc_map[cur_label] = cur_acc
        acclevel_label_map = {}
        for key in label_acc_map:
            cur_acc = label_acc_map[key]
            cur_level = int(int(cur_acc*100) / 10)
            if cur_level not in acclevel_label_map:
                acclevel_label_map[cur_level] = []
            acclevel_label_map[cur_level].append(key)

        acclevel_label_list = []
        for level in sorted(acclevel_label_map.keys()):
            cur_level = str(level) + ":" + " ".join([str(e) for e in acclevel_label_map[level]])
            acclevel_label_list.append(cur_level)

        MyFunction.saveVector(save_fp, acclevel_label_list)


class TestC_ImageFeature:
    @staticmethod
    def plotTSNE(feat_data_fp, final_result_fp):
        feat, label, n_samples, n_features = TestC_ImageFeature.get_data(\
            feat_data_fp, final_result_fp)
        feat, label = TestC_ImageFeature.filter_data(feat, label)
        print("ssss", len(label))
        X_embedded = TSNE(n_components=2, verbose=1).fit_transform(feat)
        fig, ax = plt.subplots(figsize=[20, 16])
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=label, ax=ax)
        plt.show(fig)

    @staticmethod
    def getLabel(final_result):
        label_ZJL = []
        for line in final_result:
            label_ZJL.append(line.split('\t')[1])
        label_ZJL = pd.Series(label_ZJL)
        label2cnnidx = {label_code: idx for idx, label_code in enumerate(label_ZJL.unique().tolist())}
        label_num = label_ZJL.apply(lambda x: label2cnnidx[x])
        return label_num, label2cnnidx

    @staticmethod
    def get_data(feat_data_fp, final_result_fp=None, k=20):
        test_feat = np.load(feat_data_fp)
        final_result = MyFunction.readVector(final_result_fp)
        label, _ = TestC_ImageFeature.getLabel(final_result)
        test_feat, label = TestC_ImageFeature.filter_data(test_feat, label, num_save=10)
        n_samples, n_features = test_feat.shape
        return test_feat, label, n_samples, n_features

    @staticmethod
    def filter_data(feat, label, num_save=100, num_class=20):
        label_count_map = {}
        filter_feat = []
        filter_label = []
        for (cur_feat, cur_label) in zip(feat, label):
            if len(label_count_map.keys()) >= num_class and cur_label not in label_count_map:
                continue
            if cur_label not in label_count_map:
                label_count_map[cur_label] = 0
            if label_count_map[cur_label] < num_save:
                label_count_map[cur_label] += 1
                filter_feat.append(cur_feat)
                filter_label.append(cur_label)
        return np.array(filter_feat), np.array(filter_label)

    @staticmethod
    def plot_embedding(data, label, title):
        plt.switch_backend('agg')
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        num_class = pd.Series(label).unique().size
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]-20, ):
            cur_color = 0.9
            '''
            if cur_color != label[0] / num_class:
                continue
            '''
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(cur_color),
                     fontdict={'weight': 'bold', 'size': 9})

        for i in range(data.shape[0]-20, data.shape[0]):
            cur_color = 0.5
            '''
            if cur_color != label[0] / num_class:
                continue
            '''
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(cur_color),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    @staticmethod
    def grainPredictImageFeatureDataBasedOnLabel(label, predict_imageFeature_fp, final_result_fp):
        final_result = MyFunction.readVector(final_result_fp)
        _, label2cnnidx = TestC_ImageFeature.getLabel(final_result)
        label = set(label)
        predict_imageFeature = MyFunction.readVector(predict_imageFeature_fp)
        label_predictImageFeature_map = {}
        for line in predict_imageFeature:
            if line.split(":")[0] not in label2cnnidx.keys():
                continue
            cur_label = label2cnnidx[line.split(":")[0]]
            cur_ImageFeature = [float(e) for e in line.split(":")[1].split()]
            label_predictImageFeature_map[cur_label] = cur_ImageFeature
        predictImageFeature = []
        predictImageFeature_label = []
        for cur_label in label:
            predictImageFeature_label.append(cur_label)
            predictImageFeature.append(label_predictImageFeature_map[cur_label])
        return np.array(predictImageFeature), np.array(predictImageFeature_label)

    @staticmethod
    def plotTestC(feat_data_fp, predict_imageFeature_fp, final_result_fp=None, fig_save_fp=None):
        data, label, n_samples, n_features = TestC_ImageFeature.get_data(feat_data_fp, final_result_fp)
        predictImageFeature, predictImageFeature_label = TestC_ImageFeature.grainPredictImageFeatureDataBasedOnLabel(label, predict_imageFeature_fp, final_result_fp)
        print(predictImageFeature.shape)
        label = np.r_[label, predictImageFeature_label]
        data = np.r_[data, predictImageFeature]
        print(label.shape)
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
        result = tsne.fit_transform(data)
        fig = TestC_ImageFeature.plot_embedding(result, label, 't-SNE embedding of the digits')
        if fig_save_fp:
            plt.savefig(fig_save_fp)
        plt.show(fig)


class PlantPostProcess:
    def __init__(self):
        self.color_rgb_map = {'red': [212, 175, 0], \
                         'green': [56, 123, 56], 'yellow': [212, 175, 0], \
                         'brown': [163, 132, 41], 'blue': [53, 55, 176], \
                         'wight': [256, 256, 256], 'black': [0, 0, 0]}

    def returnSpecificColorBasedOnRGB(r, g, b):
        flag_1 = 200
        flag_2 = 50
        flag_3 = 100
        if r > flag_1 and g < flag_2 and b < flag_2:
            return 'red'
        if r < flag_2 and g > flag_1 and b < flag_2:
            return 'green'
        if r > flag_1 and g > flag_1 and b < flag_2:
            return 'yellow'
        if r < flag_2 and g < flag_2 and b < flag_2:
            return 'black'
        if r > flag_1 and g > flag_1 and b > flag_1:
            return 'wight'
        return 'brown'

    def gainPredictColorBasedOnImageSampleMethod(self, image):
        num_sample = 100
        for i in range(num_sample):
            random_w = 20 * random.random(0, 1) + 20
            random_h = 20 * random.random(0, 1) + 20

    def gainPredictColorBasedOnDominantColor(self, dominant_color):
        predictColor = None
        min_distance = 9999999
        for cur_key in self.color_rgb_map:
            cur_color = self.color_rgb_map[cur_key]
            diff = np.array(cur_color) - np.array(dominant_color)
            cur_distance = sum(e**2 for e in diff)
            if cur_distance < min_distance:
                min_distance = cur_distance
                predictColor = cur_key
        return predictColor

    def get_dominant_color(self, image):
        image = image.convert('RGBA')
        print(image)
        max_score = -10000
        dominant_color = None
        for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
            if a == 0:
                continue
            saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
            y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
            y = (y - 16.0) / (235 - 16)
            if y > 0.9:
                continue
            score = (saturation + 0.1) * count
            if score > max_score:
                max_score = score
                dominant_color = [r, g, b]
        return dominant_color

    def getLabelBasedOnImageAndImageFeature(self, image, imageFeature=None):
        dominant_color_rgb = self.get_dominant_color(image)
        predict_color = self.gainPredictColorBasedOnDominantColor(dominant_color_rgb)
        return predict_color, dominant_color_rgb


class PrepareData:
    @staticmethod
    def constractWordEmbedding(className_fp=None, class_wordembeddings_fp=None, extractWordembedding_fp=None):
        extract_wordembedding = []
        class_wordEmbedding_map = MyFunction.gainWordEmbedding(class_wordembeddings_fp)
        className = MyFunction.readVector(className_fp)
        for cur_className in className:
            extract_wordembedding.append(class_wordEmbedding_map[cur_className])
        MyFunction.saveVector(extractWordembedding_fp, extract_wordembedding)


class Generator:
    @staticmethod
    def gainNextBatchSizeSample(x_, y_, batchSize):
        x_ = list(x_)
        y_ = list(y_)
        total_num = len(x_)
        batchSize_x = []
        batchSize_y = []
        for i in range(batchSize):
            random_index = randint(0, total_num-1)
            batchSize_x.append(x_[random_index])
            batchSize_y.append(y_[random_index])
        return batchSize_x, batchSize_y

    @staticmethod
    def gainTrainData_allPictures_txt(label_wordEmbedding_fp, fc_vector_alltrain_fp):
        print("collecting train data ...")
        fc_vector_alltrain = MyFunction.readVector(fc_vector_alltrain_fp)
        label_wordEmbedding = MyFunction.gainWordEmbedding(label_wordEmbedding_fp, 'float')
        x_ = []
        y_ = []
        for line in fc_vector_alltrain:
            cur_lable = line.split(":")[0]
            cur_y = [float(e) for e in line.split(":")[1].split()]
            cur_x = label_wordEmbedding[cur_lable]
            x_.append(cur_x)
            y_.append(cur_y)
        return x_, y_

    @staticmethod
    def gainTrainData_allPictures_npy(label_wordEmbedding_fp, fc_vector_alltrain_fp, train_imageName_Lable_fp, mask_flag_fp=None):
        print("collecting train data ...")
        train_imageName_Lable = MyFunction.readVector(train_imageName_Lable_fp)
        print("train image name lable num", len(train_imageName_Lable))
        fc_vector_alltrain = np.load(fc_vector_alltrain_fp)
        print("sss", fc_vector_alltrain.shape)
        print("shape fc vector", fc_vector_alltrain.shape)
        label_wordEmbedding = MyFunction.gainWordEmbedding(label_wordEmbedding_fp, 'float')
        if mask_flag_fp:
            right_flag = MyFunction.readVector(mask_flag_fp)
        x_ = []
        y_ = []
        for i in range(len(train_imageName_Lable)):
            cur_lable = train_imageName_Lable[i].strip('/r/t/n').split()[1]
            cur_y = [e for e in fc_vector_alltrain[i]]
            cur_x = label_wordEmbedding[cur_lable]
            if mask_flag_fp != None:
                if int(right_flag[i]) == 0:
                    continue
            x_.append(cur_x)
            y_.append(cur_y)
        return x_, y_

    @staticmethod
    def gainTestData_allPictures(label_wordEmbedding_fp, test_lable_fp):
        test_lable = MyFunction.readVector(test_lable_fp)
        label_wordEmbedding = MyFunction.gainWordEmbedding(label_wordEmbedding_fp, 'float')
        x_ = []
        for line in test_lable:
            x_.append(label_wordEmbedding[line.strip('\n\r\t')])
        return x_

    @staticmethod
    def gainTestData(extractWordembedding_fp):
        return MyFunction.load_WordEmbedding(extractWordembedding_fp)

    @staticmethod
    def gainTrainData_onePicture(trainImageFeature_fp, extractWordembedding_fp):
        lables = MyFunction.load_ImageFeature(trainImageFeature_fp)
        features = MyFunction.load_WordEmbedding(extractWordembedding_fp)
        return features, lables


class Gainer:
    @staticmethod
    def gain_wordEmbedding_map(wordEmbedding_fp, format='float'):
        wordEmbedding_string = MyFunction.readVector(wordEmbedding_fp)
        label_wordEmbedding_map = {}
        for line in wordEmbedding_string:
            cur_label = line.split()[0]
            cur_wordEmbedding_string = line.split()[1:]
            cur_wordEmbedding_float = [float(e) for e in cur_wordEmbedding_string]
            label_wordEmbedding_map[cur_label] = cur_wordEmbedding_float
        return label_wordEmbedding_map


class Analysis:
    @staticmethod
    def analysisTestCResultCNN(predict_resultCNN_fp, real_result_fp):
        predict_result = MyFunction.readVector(predict_resultCNN_fp)
        real_result = MyFunction.readVector(real_result_fp)
        real_result = [line.split("\t")[1] for line in real_result]
        realResult_predictResult_map = {}
        #print(real_result)
        for (cur_real_result, cur_pre_result) in zip(real_result, predict_result):
            if cur_real_result not in realResult_predictResult_map:
                realResult_predictResult_map[cur_real_result] = {}
            if cur_pre_result not in realResult_predictResult_map[cur_real_result]:
                realResult_predictResult_map[cur_real_result][cur_pre_result] = 0
            realResult_predictResult_map[cur_real_result][cur_pre_result] += 1
        for cur_real_label in realResult_predictResult_map:
            cur_real_label_map = realResult_predictResult_map[cur_real_label]
            print(cur_real_label, sum(list(cur_real_label_map.values())))
            cur_real_label_map = sorted(cur_real_label_map.items(), key=lambda d: d[1], reverse=True)
            print(cur_real_label_map)

    @staticmethod
    def analysisAccOfFirstAttribute(attribute_per_class_fp, real_result_fp, predict_result_fp, label2ClassName_fp):
        label_className_map = MyFunction.gainLabelToClassnale(label2ClassName_fp)
        real_result = MyFunction.readVector(real_result_fp)
        predict_result1 = MyFunction.readVector(predict_result_fp)
        label_firstClass_acc = {}
        attribute = ConstractA.getAttribute(label2ClassName_fp, attribute_per_class_fp)
        className_firstclass_map = ConstractA.returnFirstAttribute(attribute)

        for inx in range(len(real_result)):
            real_label = real_result[inx].split('\t')[1]
            if real_label == "ZJL999":
                continue
            predict_label = predict_result1[inx].split('\t')[1]
            real_className = label_className_map[real_label]
            predict_className = label_className_map[predict_label]

            if real_label not in label_firstClass_acc:
                label_firstClass_acc[real_label] = [0, 0]
            if className_firstclass_map[real_className] == className_firstclass_map[predict_className]:
                label_firstClass_acc[real_label][0] += 1
            label_firstClass_acc[real_label][1] += 1
        avg_acc = 0
        for cur_label in label_firstClass_acc:
            print(cur_label, label_className_map[cur_label], ":", label_firstClass_acc[cur_label][0]/label_firstClass_acc[cur_label][1])
            avg_acc += label_firstClass_acc[cur_label][0]/label_firstClass_acc[cur_label][1] / len(label_firstClass_acc)
        print(avg_acc)

        firstClass_acc_map = {}
        for cur_label in label_firstClass_acc:
            cur_className = label_className_map[cur_label]
            cur_firstClass = className_firstclass_map[cur_className]
            if cur_firstClass not in firstClass_acc_map:
                firstClass_acc_map[cur_firstClass] = []
            firstClass_acc_map[cur_firstClass].append(label_firstClass_acc[cur_label])

        for cur_firstClass in firstClass_acc_map:
            print(cur_firstClass, firstClass_acc_map[cur_firstClass])


    @staticmethod
    def analysisDifferenceBetweenFinalResult_vector(real_result, final_result1, final_result2):
        total_reuslt = len(real_result)
        common_right = 0
        single_result1_right = 0
        single_result2_right = 0
        for inx in range(len(real_result)):
            real_label = real_result[inx]
            result1 = final_result1[inx]
            result2 = final_result2[inx]
            if real_label == result1 and real_label == result2:
                common_right += 1
            elif real_label != result1 and real_label == result2:
                single_result2_right += 1
            elif real_label == result1 and real_label != result2:
                single_result1_right += 1
        print("common count", common_right)
        print("result1 single count", single_result1_right)
        print("result2 single count", single_result2_right)
        print("total acc", 1.0*(single_result1_right + single_result2_right + common_right)/total_reuslt)

    @staticmethod
    def analysisDifferenceBetweenFinalResult(real_result_fp, final_result1_fp, final_result2_fp):
        real_result = MyFunction.readVector(real_result_fp)
        final_result1 = MyFunction.readVector(final_result1_fp)
        final_result2 = MyFunction.readVector(final_result2_fp)
        total_reuslt = len(real_result)
        common_right = 0
        single_result1_right = 0
        single_result2_right = 0
        for inx in range(len(real_result)):
            real_label = real_result[inx].split('\t')[1]
            result1 = final_result1[inx].split('\t')[1]
            result2 = final_result2[inx].split('\t')[1]
            if real_label == result1 and real_label == result2:
                common_right += 1
            elif real_label != result1 and real_label == result2:
                single_result2_right += 1
            elif real_label == result1 and real_label != result2:
                single_result1_right += 1
        print("common count", common_right)
        print("result1 single count", single_result1_right)
        print("result2 single count", single_result2_right)
        print("total acc", 1.0*(single_result1_right + single_result2_right + common_right)/total_reuslt)

    @staticmethod
    def gainTopkRelateClassLabelForEveryTestClassByWordEmbedding(wordEmbedding_fp, label2ClassName_fp, test_label_fp, k=5):
        wordEmbedding = MyFunction.gainWordEmbedding(wordEmbedding_fp, 'float')
        train_test_label = np.array(list(wordEmbedding.keys()))
        label_className_map = MyFunction.gainLabelToClassnale(label2ClassName_fp)
        test_label = MyFunction.readVector(test_label_fp)
        test_className_list = [label_className_map[e] for e in test_label]

        #train search in all
        label_relateLable_map = {}
        for cur_key1 in wordEmbedding:
            all_avg_cons = []
            for cur_key2 in wordEmbedding:
                cur_cons = MyFunction.cosineSimilarity(wordEmbedding[cur_key1], wordEmbedding[cur_key2])
                all_avg_cons.append(cur_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_test_label[all_avg_cons_sorted_index[1:k+1]]
            label_relateLable_map[cur_key1] = top_k_relate_label

        # test search in train
        test_label_relateLable_map = {}
        for cur_key1 in wordEmbedding:
            if cur_key1 not in test_className_list:
                continue
            all_avg_cons = []
            for cur_key2 in wordEmbedding:
                if cur_key2 in test_className_list:
                    cur_cons = 0
                else:
                    cur_cons = MyFunction.cosineSimilarity(wordEmbedding[cur_key1], wordEmbedding[cur_key2])
                all_avg_cons.append(cur_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_test_label[all_avg_cons_sorted_index[0:k]]
            test_label_relateLable_map[cur_key1] = top_k_relate_label

        for cur_label in test_label_relateLable_map:
            label_relateLable_map[cur_label] = test_label_relateLable_map[cur_label]

        # here the label is className
        for cur_label in label_relateLable_map:
            if cur_label in test_className_list:
                print(cur_label, ":",
                      label_relateLable_map[cur_label][0],\
                      label_relateLable_map[cur_label][1],\
                      label_relateLable_map[cur_label][2],\
                      label_relateLable_map[cur_label][3],\
                      label_relateLable_map[cur_label][4])

    @staticmethod
    def compute_avgSamiliarty(test_feat, train_feat):
            all_cons = 0
            for cur_test_feat in test_feat:
                for cur_train_feat in train_feat:
                    all_cons += MyFunction.cosineSimilarity(cur_test_feat, cur_train_feat)
            return float(all_cons)/(len(test_feat) * len(train_feat))

    @staticmethod
    def gainTopkRelateClassLabelForEveryTestClassByImageFeat(train_feat_fp, test_feat_fp, train_image_labeled_fp, test_image_labeled_fp, label2ClassName_fp, k_compare=10, k=3):
        train_image_labeled = MyFunction.readVector(train_image_labeled_fp)
        test_image_labeled = MyFunction.readVector(test_image_labeled_fp)
        lable_className_map = MyFunction.gainLabelToClassnale(label2ClassName_fp)

        train_feat = np.load(train_feat_fp)
        test_feat = np.load(test_feat_fp)
        train_label_kFeat_map = {}
        for i in range(len(train_image_labeled)):
            cur_label = train_image_labeled[i].split('\t')[1]
            cur_imageFeat = train_feat[i]
            if cur_label not in train_label_kFeat_map:
                train_label_kFeat_map[cur_label] = []
            if len(train_label_kFeat_map[cur_label]) < k_compare:
                train_label_kFeat_map[cur_label].append(cur_imageFeat)

        test_label_kFeat_map = {}
        for i in range(len(test_image_labeled)):
            cur_label = test_image_labeled[i].split('\t')[1]
            if cur_label == 'ZJL999':
                continue
            cur_imageFeat = test_feat[i]
            if cur_label not in test_label_kFeat_map:
                test_label_kFeat_map[cur_label] = []
            if len(test_label_kFeat_map[cur_label]) < k_compare:
                test_label_kFeat_map[cur_label].append(cur_imageFeat)

        train_test_label_kFeat_map = MyFunction.merge_two_dicts(train_label_kFeat_map, test_label_kFeat_map)
        train_label = np.array(list(train_label_kFeat_map.keys()))
        test_label = np.array(list(test_label_kFeat_map.keys()))
        train_test_label = np.array(list(train_test_label_kFeat_map.keys()))

        # train search in train and test
        train_label_relateLable_map = {}
        for cur_key1 in train_test_label_kFeat_map:
            all_avg_cons = []
            for cur_key2 in train_test_label_kFeat_map:
                avg_cons = Analysis.compute_avgSamiliarty(train_test_label_kFeat_map[cur_key1], train_test_label_kFeat_map[cur_key2])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_test_label[all_avg_cons_sorted_index[1:k+1]]
            train_label_relateLable_map[cur_key1] = top_k_relate_label

        '''
        train search in test
        train_label_relateLable_map = {}
        for cur_key1 in train_test_label_kFeat_map:
            all_avg_cons = []
            for cur_key2 in test_label_kFeat_map:
                avg_cons = Analysis.compute_avgSamiliarty(train_test_label_kFeat_map[cur_key1], test_label_kFeat_map[cur_key2])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = test_label[all_avg_cons_sorted_index[0:k]]
            train_label_relateLable_map[cur_key1] = top_k_relate_label
            # print(cur_key1, top_k_relate_label)
        '''

        '''
        # train search in train
        train_label_relateLable_map = {}
        for cur_key1 in train_test_label_kFeat_map:
            all_avg_cons = []
            for cur_key2 in train_label_kFeat_map:
                if cur_key2 in ['ZJL386', 'ZJL479', 'ZJL369']:
                    avg_cons = 0
                else:
                    avg_cons = Analysis.compute_avgSamiliarty(train_test_label_kFeat_map[cur_key1], train_label_kFeat_map[cur_key2])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_label[all_avg_cons_sorted_index[1:k+1]]
            train_label_relateLable_map[cur_key1] = top_k_relate_label
            # print(cur_key1, top_k_relate_label)
        '''

        # test search in train
        test_label_relateLable_map = {}
        for cur_key_test in test_label_kFeat_map:
            all_avg_cons = []
            for cur_key_train in train_label_kFeat_map:
                if cur_key_train in ['ZJL386', 'ZJL479', 'ZJL369']:
                    avg_cons = 0
                else:
                    avg_cons = Analysis.compute_avgSamiliarty(test_label_kFeat_map[cur_key_test], train_label_kFeat_map[cur_key_train])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_label[all_avg_cons_sorted_index[0:k]]
            test_label_relateLable_map[cur_key_test] = top_k_relate_label

        # update train_label_relatedLable_map
        for cur_label in test_label_relateLable_map:
            train_label_relateLable_map[cur_label] = test_label_relateLable_map[cur_label]
        
        # add missing class
        train_label_relateLable_map['nematode'] = ['nematode', 'nematode', 'nematode']
        train_label_relateLable_map['lasagna'] = ['lasagna', 'lasagna', 'lasagna']
        train_label_relateLable_map['ravioli'] = ['ravioli', 'ravioli', 'ravioli']
        train_label_relateLable_map['crepe'] = ['crepe', 'crepe', 'crepe']
        train_label_relateLable_map['hotpot'] = ['hotpot', 'hotpot', 'hotpot']
        train_label_relateLable_map['waffle'] = ['waffle', 'waffle', 'waffle']

        for cur_label in train_label_relateLable_map:
            if cur_label in lable_className_map:
                # if cur_label not in test_label_relateLable_map:
                #    continue
                if 'ZJL999' not in train_label_relateLable_map[cur_label]:
                    print(lable_className_map[cur_label],":",\
                          lable_className_map[train_label_relateLable_map[cur_label][0]],\
                          lable_className_map[train_label_relateLable_map[cur_label][1]],\
                          lable_className_map[train_label_relateLable_map[cur_label][2]])
                else:
                    print(cur_label, train_label_relateLable_map[cur_label])
        print('nematode : nematode nematode nematode')
        print('lasagna : lasagna lasagna lasagna')
        print('ravioli : ravioli ravioli ravioli')
        print('crepe : crepe crepe crepe')
        print('hotpot : hotpot hotpot hotpot')
        print('waffle : waffle waffle waffle')

        '''
        print("begin")
        unique_label = set()
        for test_label in test_label_relateLable_map:
            #print(test_label_relateLable_map[test_label][0])
            unique_label.add(test_label_relateLable_map[test_label][0])
            unique_label.add(test_label_relateLable_map[test_label][1])

        print("unique")
        print(len(unique_label))
        for e in unique_label:
            print(e)
        #print(unique_label)
        '''
    @staticmethod
    def gainTopkRelateClassLabelForEveryTestClassByImageFeat_allTrain(train_feat_fp_list, test_feat_fp, train_image_labeled_fp_list, test_image_labeled_fp, label2ClassName_fp, k_compare=10, k=3):
        train_feat_all = []
        for train_feat_fp in train_feat_fp_list:
            train_feat = list(np.load(train_feat_fp))
            train_feat_all += train_feat
        train_feat_all = np.array(train_feat_all)
        test_feat = np.load(test_feat_fp)

        train_image_labeled_all = []
        for train_image_labeled_fp in train_image_labeled_fp_list:
            train_image_labeled = MyFunction.readVector(train_image_labeled_fp)
            train_image_labeled_all += train_image_labeled
        test_image_labeled = MyFunction.readVector(test_image_labeled_fp)

        lable_className_map = MyFunction.gainLabelToClassnale(label2ClassName_fp)

        test_label_kFeat_map = {}
        for i in range(len(test_image_labeled)):
            cur_label = test_image_labeled[i].split('\t')[1]
            if cur_label == 'ZJL999':
                continue
            cur_imageFeat = test_feat[i]
            if cur_label not in test_label_kFeat_map:
                test_label_kFeat_map[cur_label] = []
            if len(test_label_kFeat_map[cur_label]) < k_compare:
                test_label_kFeat_map[cur_label].append(cur_imageFeat)

        train_label_kFeat_map = {}
        for i in range(len(train_image_labeled_all)):
            cur_label = train_image_labeled_all[i].split('\t')[1]
            cur_imageFeat = train_feat_all[i]
            if cur_label not in train_label_kFeat_map:
                train_label_kFeat_map[cur_label] = []
            if len(train_label_kFeat_map[cur_label]) < k_compare:
                train_label_kFeat_map[cur_label].append(cur_imageFeat)
        
        train_test_label_kFeat_map = MyFunction.merge_two_dicts(train_label_kFeat_map, test_label_kFeat_map)
        train_label = np.array(list(train_label_kFeat_map.keys()))
        train_test_label = np.array(list(train_test_label_kFeat_map.keys()))
        
        '''
        train_label_relateLable_map = {}
        for cur_key1 in train_test_label_kFeat_map:
            print(cur_key1)
            all_avg_cons = []
            for cur_key2 in train_test_label_kFeat_map:
                avg_cons = Analysis.compute_avgSamiliarty(train_test_label_kFeat_map[cur_key1], train_test_label_kFeat_map[cur_key2])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_test_label[all_avg_cons_sorted_index[1:k+1]]
            train_label_relateLable_map[cur_key1] = top_k_relate_label
        '''
        test_label_relateLable_map = {}
        for cur_key_test in test_label_kFeat_map:
            print(cur_key_test)
            if cur_key_test == "ZJL999":
                continue
            all_avg_cons = []
            for cur_key_train in train_label_kFeat_map:
                avg_cons = Analysis.compute_avgSamiliarty(test_label_kFeat_map[cur_key_test], train_label_kFeat_map[cur_key_train])
                all_avg_cons.append(avg_cons)
            all_avg_cons_sorted_index = list(np.argsort(all_avg_cons))
            all_avg_cons_sorted_index = np.flip(all_avg_cons_sorted_index, axis=0)
            top_k_relate_label = train_label[all_avg_cons_sorted_index[0:k]]
            test_label_relateLable_map[cur_key_test] = top_k_relate_label

        for cur_label in test_label_relateLable_map:
            if cur_label in lable_className_map:
                if 'ZJL999' not in test_label_relateLable_map[cur_label]:
                    print(lable_className_map[cur_label],":",\
                          lable_className_map[test_label_relateLable_map[cur_label][0]],\
                          lable_className_map[test_label_relateLable_map[cur_label][1]],\
                          lable_className_map[test_label_relateLable_map[cur_label][2]])
                else:
                    print(cur_label, test_label_relateLable_map[cur_label])

        '''
        for cur_label in test_label_relateLable_map:
            train_label_relateLable_map[cur_label] = test_label_relateLable_map[cur_label]

        for cur_label in train_label_relateLable_map:
            if cur_label in lable_className_map:
                if 'ZJL999' not in train_label_relateLable_map[cur_label]:
                    print(lable_className_map[cur_label],":",\
                          lable_className_map[train_label_relateLable_map[cur_label][0]],\
                          lable_className_map[train_label_relateLable_map[cur_label][1]],\
                          lable_className_map[train_label_relateLable_map[cur_label][2]])
                else:
                    print(cur_label, train_label_relateLable_map[cur_label])
        '''

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.locator_params(nbins=100, axis='x')

        plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)    

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")    

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.tight_layout()
        plt.show()

    @staticmethod
    def gainCnnModelAcc(predict_result_fp, real_label_fp, right_flag_fp=None):
        predict_result = MyFunction.readVector(predict_result_fp)
        real_label = MyFunction.readVector(real_label_fp)
        predict_result = [e.split('\t')[1] for e in predict_result]
        real_label = [e.split('\t')[1] for e in real_label]
        class_names = list(set(real_label))
        cnf_matrix = confusion_matrix(real_label, predict_result, labels=class_names)
        cnf_matrix = np.around(cnf_matrix, decimals=2)
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        for cur_inx in range(len(class_names)):
            cur_label_all_rate = cnf_matrix[cur_inx]
            cur_label_all_rate_sorted_inx = list(np.argsort(cur_label_all_rate))
            cur_label_all_rate_sorted_inx = np.flip(cur_label_all_rate_sorted_inx, axis=0)
            top_4 = [cur_label_all_rate[e] for e in cur_label_all_rate_sorted_inx[0:4]]
            if cnf_matrix[cur_inx][cur_inx] < top_4[0] or True:
                print(class_names[cur_inx], cnf_matrix[cur_inx][cur_inx])
                #print([[class_names[e], cur_label_all_rate[e]] for e in cur_label_all_rate_sorted_inx[0:4]])

        #Analysis.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        #                               title='Normalized confusion matrix')
        '''
        right_flag = []
        label_count_map = {}
        acc_count = 0
        total_ZJL999 = 0
        for i in range(len(real_label)):
            cur_real_label = real_label[i].split('\t')[1]
            cur_pred_label = predict_result[i].split('\t')[1]
            if cur_real_label == 'ZJL999':
                total_ZJL999 += 1
            if cur_real_label not in label_count_map:
                label_count_map[cur_real_label] = [0,0]

            if cur_real_label == cur_pred_label:
                acc_count += 1
                right_flag.append(1)
                label_count_map[cur_real_label][1] += 1 
            else:
                right_flag.append(0)
            label_count_map[cur_real_label][0] += 1 
        for cur_label in label_count_map:
            print("%s : %.2f" %(cur_label, label_count_map[cur_label][1]/label_count_map[cur_label][0]))

        print("acc is ", acc_count/(len(predict_result)-total_ZJL999))
        if right_flag_fp:
            MyFunction.saveVector(right_flag_fp, right_flag)
        '''



    @staticmethod
    def K_SimiliarClass_WordEmbeddingDistance(wordEmbedding_fp, k):
        label_wordEmbedding_map = Gainer.gain_wordEmbedding_map(wordEmbedding_fp)
        label_KsmiliarLabel_map = {}
        for cur_label1 in label_wordEmbedding_map:
            label_distance = {}
            for cur_label2 in label_wordEmbedding_map:
                cur_distance = MyFunction.cosineSimilarity(label_wordEmbedding_map[cur_label1], label_wordEmbedding_map[cur_label2])
                label_distance[cur_label2] = cur_distance
            label_distance_sorted = sorted(label_distance.items(), key=lambda d:d[1], reverse = True)
            label_KsmiliarLabel_map[cur_label1] = label_distance_sorted[1:k+1]
            print(cur_label1, label_KsmiliarLabel_map[cur_label1])
        return label_KsmiliarLabel_map

    @staticmethod
    def analysisDatasetJDistance(train_J_feat_npy_fp):
        #计算某个image feature与其他imagefeature的距离
        train_J_feat = np.load(train_J_feat_npy_fp)
        main_feat = train_J_feat[1]
        for index, cur_feat in enumerate(train_J_feat):
            samiliarty = Analysis.cosineSimilarity(main_feat, cur_feat)
            print(index+1, samiliarty)

    @staticmethod
    def getClassNameAndAttributeByLabel(lable_fp, label2ClassName_fp, lable_attribute_fp):
        label_class_file = open(label2ClassName_fp, 'r')
        line = label_class_file.readline()
        label_className_map = {}
        className_label_map = {}
        while line:
            cur_label = line.strip().split()[0]
            cur_className = line.strip().split()[1]
            label_className_map[cur_label] = cur_className
            className_label_map[cur_className] = cur_label
            line = label_class_file.readline()

        attribute = MyFunction.getAttribute(label2ClassName_fp, lable_attribute_fp)
        first_class_map = MyFunction.returnFirstAttribute(attribute)
        firstAttribute_num_map = {'animal': 0, 'transportation': 0, 'clothes': 0, 'plant': 0, 'tableware': 0, 'device': 0, 'food': 0, 'None': 0}
        firstAttribute_className_map = {'animal': [], 'transportation': [], 'clothes': [], 'plant': [], 'tableware': [], 'device': [], 'food': [], 'None': []}
        label_list_file = open(lable_fp, 'r')
        lines = label_list_file.readlines()
        label_list = [line.strip() for line in lines]
        for cur_label in label_list:
            cur_className = label_className_map[cur_label]
            cur_category = first_class_map[cur_className]
            firstAttribute_num_map[cur_category] += 1
            firstAttribute_className_map[cur_category] += [cur_className]
        print(firstAttribute_num_map)
        print(firstAttribute_className_map)

    @staticmethod
    def split_finalResult(label2ClassName_fp, final_result_fp, B_test_pt, split_fp, process_num, className_Num_fp):
        label2ClassName = MyFunction.gainLabelToClassnale(label2ClassName_fp)
        className_imageNameSet = {}
        className_Num_list = []
        with open(final_result_fp, 'r+') as f:
            for line in f.readlines():
                linelists = line.strip().split()
                img = linelists[0]
                cls_code = linelists[1]
                cls_name = label2ClassName[cls_code]
                if cls_name not in className_imageNameSet.keys():
                    className_imageNameSet[cls_name] = [img]
                else:
                    className_imageNameSet[cls_name].append(img)

        for cur_key in className_imageNameSet:
            className_Num_list.append(cur_key + ':' + str(len(className_imageNameSet[cur_key])))
        MyFunction.saveVector(className_Num_fp, className_Num_list)
        for k in className_imageNameSet.keys():
            if not os.path.exists(split_fp+k):
                os.mkdir(split_fp+k)
            for img in className_imageNameSet[k]:
                srcfile = B_test_pt+img
                dstfile_full = split_fp+k+'\\'+img
                shutil.copyfile(srcfile, dstfile_full)

    @staticmethod
    def analysisTestImageFeatureandTrainImageFeatuerRelationship(image_feature_trian_fp, image_feature_test_fp, lable2Classname_fp, best_print=False):
        image_feature_test = MyFunction.gainImageFeature(image_feature_test_fp)
        image_feature_train = MyFunction.gainImageFeature(image_feature_trian_fp)
        print(len(image_feature_train))
        lable_classname_map = MyFunction.gainLabelToClassnale(lable2Classname_fp)
        if best_print:
            for cur_lable_in_test in image_feature_test:
                best_dis = 99999
                best_lable = 'None'
                for cur_lable_in_train in image_feature_train:
                    cur_dis = MyFunction.L2distance(image_feature_train[cur_lable_in_train], image_feature_test[cur_lable_in_test])
                    if cur_dis < best_dis:
                        best_dis = cur_dis
                        best_lable = cur_lable_in_train
                print(lable_classname_map[cur_lable_in_test], lable_classname_map[best_lable])
        else:
            for cur_lable_in_test in image_feature_test:
                count = 0
                dis_lable_list = []
                for cur_lable_in_train in image_feature_train:
                    cur_lable_in_train_dele_ = cur_lable_in_train.split("_")[0]
                    cur_dis = MyFunction.cosineSimilarity(image_feature_train[cur_lable_in_train], image_feature_test[cur_lable_in_test])
                    dis_lable_list.append([cur_dis, lable_classname_map[cur_lable_in_train_dele_]]) 
                print(lable_classname_map[cur_lable_in_test])
                print(sorted(dis_lable_list, key=lambda student: student[0], reverse=True))

class FeatAnalysis:
    @staticmethod
    def returnLabelFeatMap(feat_fp, real_result_fp):
        real_result = MyFunction.readVector(real_result_fp)
        feat = np.load(feat_fp)
        label_feat_map = {}
        for inx in range(len(real_result)):
            cur_feat = feat[inx]
            cur_label = real_result[inx].split('\t')[1]
            if cur_label not in label_feat_map:
                label_feat_map[cur_label] = []
            label_feat_map[cur_label].append(cur_feat)
        return label_feat_map

    @staticmethod
    def computeInterAndExterDistanceOFImageFeat(feat_fp, real_result_fp, kSample=50):
        label_feat_map = FeatAnalysis.returnLabelFeatMap(feat_fp, real_result_fp)
        inner_dis = []
        for cur_label_1 in label_feat_map:
            cur_label_allFeat = np.array(label_feat_map[cur_label_1])
            feat_num = len(cur_label_allFeat)
            random_list1 = [randint(0, feat_num) for _ in range(kSample)]
            random_list2 = [randint(0, feat_num) for _ in range(kSample)]
            cur_dis = Analysis.compute_avgSamiliarty(cur_label_allFeat[random_list1],\
                                                     cur_label_allFeat[random_list2])
            inner_dis.append(cur_dis)
        print("inner distance", sum(inner_dis)/len(inner_dis))

        outter_dis = []
        num_class = len(list(label_feat_map.keys()))
        for cur_label_1 in label_feat_map:
            cur_label_allFeat = np.array(label_feat_map[cur_label_1])
            feat_num_1 = len(cur_label_allFeat)
            random_list1 = [randint(0, feat_num_1) for _ in range(kSample)]
            outer_class_inx = randint(0, num_class)
            outer_label = list(label_feat_map.keys())[outer_class_inx]
            outer_label_allFeat = np.array(label_feat_map[outer_label])
            feat_num_2 = len(outer_label_allFeat)
            random_list2 = [randint(0, feat_num_2) for _ in range(kSample)]
            cur_dis = Analysis.compute_avgSamiliarty(cur_label_allFeat[random_list1],\
                                                     outer_label_allFeat[random_list2])
            outter_dis.append(cur_dis)
        print("outter distance", sum(outter_dis)/len(outter_dis))

    @staticmethod
    def returnStatisticInformation(feat_fp, real_result_fp):
        label_feat_map = FeatAnalysis.returnLabelFeatMap(feat_fp, real_result_fp)
        mean_all = []
        std_all = []
        for cur_label in label_feat_map:
            cur_label_allFeat = np.array(label_feat_map[cur_label])
            mean = cur_label_allFeat.mean()
            std = cur_label_allFeat.std()
            mean_all.append(mean)
            std_all.append(std)
        print(mean_all[0], mean_all[1], mean_all[2])
        print(std_all[0], std_all[1], std_all[2])
        print("mean", np.mean(mean_all))
        print("std", np.mean(std_all))


class Specific:
    @staticmethod
    def changeImageFeatureToFeatFile(file_list, label_fp, feat_fp):
        for cur_root, _, cur_files in os.walk(file_list):
            root = cur_root
            files = cur_files

        allfeat_line = []
        for cur_file in files:
            cur_path = root + cur_file
            cur_lines = MyFunction.readVector(cur_path)
            allfeat_line += cur_lines

        labels = []
        feats = []
        for cur_line in allfeat_line:
            cur_label = cur_line.split(":")[0]
            cur_feat = [float(e) for e in cur_line.split(":")[1].split()]
            labels.append('s\t' + cur_label)
            feats.append(cur_feat)
        feats = np.array(feats)
        np.save(file_list + feat_fp, feats)
        MyFunction.saveVector(file_list + label_fp, labels)


    @staticmethod
    def gainImageFeatureDirectlyFromImageFeat(test_feat_fp, test_image_labled_fp, save_fp=None):
        test_feat = np.load(test_feat_fp)
        test_image_labled = MyFunction.readVector(test_image_labled_fp)
        label_feature_map = {}
        label_feature_list = []
        for inx in range(len(test_image_labled)):
            cur_label = test_image_labled[inx].split("\t")[1]
            if cur_label not in label_feature_map:
                label_feature_map[cur_label] = []
            label_feature_map[cur_label].append(test_feat[inx])

        for key in label_feature_map.keys():
            total_sample = len(label_feature_map[key])
            select_inx = np.random.randint(total_sample, size=1)
            selected_feat = label_feature_map[key][select_inx[0]]
            selected_feat_string = " ".join([str(e) for e in selected_feat])
            label_feature_list.append(key + ":" + selected_feat_string)
        MyFunction.saveVector(save_fp, label_feature_list)

    @staticmethod
    def findDifference(real_result_fp):
        Analysis.analysisDifferenceBetweenFinalResult_vector()

    @staticmethod
    def compareTwoResultAcc(acc_fp1, acc_fp2):
        acc_fp1 = MyFunction.readVector(acc_fp1)
        acc_fp2 = MyFunction.readVector(acc_fp2)
        acc1_map = {}
        for line in acc_fp1:
            cur_label = line.split()[0]
            acc = line.split()[1]
            acc1_map[cur_label] = acc

        acc2_map = {}
        for line in acc_fp2:
            cur_label = line.split()[0]
            acc = line.split()[1]
            acc2_map[cur_label] = acc

        for cur_label in acc1_map.keys():
            acc1 = acc1_map[cur_label]
            acc2 = acc2_map[cur_label]
            if float(acc1) - 0.05 > float(acc2):
                print(cur_label, acc1, acc2)



    @staticmethod
    def anaLysisRelatedClass(related_fp):
        related_fp = MyFunction.readVector(related_fp)
        center_label = []
        first_class_label = []
        second_class_label = []
        for line in related_fp:
            cur_center_label = line.split(":")[0].strip(" ")
            cur_first_class_label = line.split(":")[1].split(" ")[1]
            cur_second_class_label = line.split(":")[1].split(" ")[2]
            center_label.append(cur_center_label)
            first_class_label.append(cur_first_class_label)
            second_class_label.append(cur_second_class_label)
        label_DataFrame = pd.DataFrame({'0': center_label, '1': first_class_label, '2': second_class_label})
        # label_DataFrame[1] = cur_center_label
        print(label_DataFrame['1'].value_counts())
        print(label_DataFrame['2'].value_counts())

    @staticmethod
    def mergeRelatedClass(related_fp1, related_fp2, label2ClassName_fp, test_label_fp, save_fp):
        test_label = MyFunction.readVector(test_label_fp)
        label2ClassName = MyFunction.gainLabelToClassnale(label2ClassName_fp)
        related_fp1 = MyFunction.readVector(related_fp1)
        related_fp2 = MyFunction.readVector(related_fp2)
        related_fp1.sort()
        related_fp2.sort()
        print(related_fp2[0])
        testClassName_list = [label2ClassName[e] for e in test_label]
        merge_related = []
        for (line1, line2) in zip(related_fp1, related_fp2):
            print(line1)
            print(line2)
            cur_className = line1.split(':')[0].strip(" ")
            if cur_className in testClassName_list:
                merge_related.append(line1)
            else:
                merge_related.append(line2)
        MyFunction.saveVector(save_fp, merge_related)

    @staticmethod
    def findDifferenceBetweenRelatedClass(fp1, fp2, test_label_fp, label2ClassName):
        class_relatedClass1 = MyFunction.readVector(fp1)
        class_relatedClass2 = MyFunction.readVector(fp2)
        test_label = MyFunction.readVector(test_label_fp)
        test_className = []
        label2ClassName = MyFunction.gainLabelToClassnale(label2ClassName)

        class_relatedClass1_map = {}
        for line in class_relatedClass1:
            cur_className = line.split(":")[0].strip(" ")
            related_className = line.split(":")[1].split(" ")
            class_relatedClass1_map[cur_className] = related_className

        class_relatedClass2_map = {}
        for line in class_relatedClass2:
            cur_className = line.split(":")[0].strip(" ")
            related_className = line.split(":")[1].split(" ")
            class_relatedClass2_map[cur_className] = related_className

        for e in test_label:
            test_className.append(label2ClassName[e])

        for key in class_relatedClass1_map.keys():
            if key in test_className:
                className1 = class_relatedClass1_map[key][1]
                className2 = class_relatedClass2_map[key][1]
                print(key, className1, className2)

    @staticmethod
    def testHighDimentionVectorHasAddAttribute(feat_fp, real_result_fp):
        real_result = MyFunction.readVector(real_result_fp)
        feat = np.load(feat_fp)
        label_feat_map = {}
        for inx in range(len(real_result)):
            cur_feat = feat[inx]
            cur_label = real_result[inx].split('\t')[1]
            if cur_label not in label_feat_map:
                label_feat_map[cur_label] = []
            label_feat_map[cur_label].append(cur_feat)

        label_avgFeat_map = {}
        for cur_label in label_feat_map:
            cur_allFeat = label_feat_map[cur_label]
            label_avgFeat_map[cur_label] = np.sum(cur_allFeat, axis=0)/len(cur_allFeat)

        for cur_label_1 in label_feat_map:
            cur_label_1_avgFeat = label_avgFeat_map[cur_label_1]
            cur_select_feat = label_feat_map[cur_label_1][0:10]
            for cur_label_2 in label_feat_map:
                dis1 = Analysis.compute_avgSamiliarty([cur_label_1_avgFeat], label_feat_map[cur_label_2][0:20])
                dis2 = Analysis.compute_avgSamiliarty(cur_select_feat, label_feat_map[cur_label_2][0:20])
                print(dis1, dis2)
            break

    @staticmethod
    def gainAllRelatedLabelWithTest(train_test_Label_topK_relateTrainLabel_fp, train_label_fp, label2ClassName_fp):
        train_label = MyFunction.readVector(train_label_fp)
        train_test_Label_topK_relateTrainLabel = MyFunction.readVector(train_test_Label_topK_relateTrainLabel_fp)
        className_label_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        all_related_label_set = set()
        for line in train_test_Label_topK_relateTrainLabel:
            # print(line)
            cur_label = className_label_map[line.split(":")[0].strip(" ")]
            related_label = [className_label_map[e] for e in line.split(":")[1].split()[:1]]
            # print(cur_label)
            if cur_label not in train_label:
                all_related_label_set |= set(related_label)
            else:
                continue
                for cur_related_label in related_label:
                    if cur_related_label not in train_label:
                        all_related_label_set |= set(cur_label)
        for e in all_related_label_set:
            print(e)
        print(len(all_related_label_set))
        return all_related_label_set

    @staticmethod
    def changAttributeIntoWordEmbeddingFormat(attribute_fp, wordEmbedding_fp):
        attribute = MyFunction.readVector(attribute_fp)
        new_result = []
        for line in attribute:
            cur_label = line.split('\t')[0]
            cur_attribute = line.split('\t')[1:]
            cur_new_result = cur_label + ' ' + " ".join(cur_attribute)
            new_result.append(cur_new_result)
        MyFunction.saveVector(wordEmbedding_fp, new_result)

    @staticmethod
    def mergeFeatNPY(train_feat_A_fp, train_feat_B_fp, train_feat_C_fp, merge_fp):
        print("Reading feat ...")
        train_feat_A = np.load(train_feat_A_fp)
        print("shape file 1", train_feat_A.shape)
        train_feat_B = np.load(train_feat_B_fp)
        print("shape file 2", train_feat_B.shape)
        train_feat_C = np.load(train_feat_C_fp)
        print("shape file 3", train_feat_C.shape)

        train_feat = np.concatenate([train_feat_A, train_feat_B, train_feat_C], 0)
        print("shape file merge", train_feat.shape)
        np.save(merge_fp, train_feat)

    @staticmethod
    def gainImageMaskBasedOnLabel(imageName_lable_fp, label_fp, mask_fp):
        label = MyFunction.readVector(label_fp)
        imageName_lable = MyFunction.readVector(imageName_lable_fp)
        mask = []
        for line in imageName_lable:
            cur_label = line.split('\t')[1]
            if cur_label in label:
                mask.append(1)
            else:
                mask.append(0)
        MyFunction.saveVector(mask_fp, mask)

    @staticmethod
    def gainTrainLableAndTestLable(lable_className_fp, train_imageName_lable_fp, train_lable_1_fp, test_lable_1_fp):
        train_imageName_lable = MyFunction.readVector(train_imageName_lable_fp)
        lable_className_map = MyFunction.getLabelClassMap(lable_className_fp)
        train_lable = []
        test_lable = []
        for line in train_imageName_lable:
            cur_lable = line.split("\t")[1]
            if cur_lable not in train_lable:
                train_lable.append(cur_lable)
        for key in lable_className_map:
            if key not in train_lable:
                test_lable.append(key)
        MyFunction.saveVector(train_lable_1_fp, train_lable)
        MyFunction.saveVector(test_lable_1_fp, test_lable)

    @staticmethod
    def mergeLable(lable_fp_list, merge_fp=None):
        unique_label = set()
        for file_fp in lable_fp_list:
            label_1 = MyFunction.readVector(file_fp)
            unique_label |= set(label_1)
        if merge_fp:
            MyFunction.saveVector(merge_fp, unique_label)
        else:
            return unique_label

    @staticmethod
    def extractTestImageFeatureFromTrainAndTestPredictImageFeature(A_matriex_C_fp, train_test_imageFeature_predict_fp, label2ClassName_fp, testLabel_fp, TestImageFeaturePredicted_fp):
        A_matrix = MyFunction.readVector(A_matriex_C_fp)
        testLabel = MyFunction.readVector(testLabel_fp)
        train_test_imageFeature_predict = MyFunction.readVector(train_test_imageFeature_predict_fp)
        className2Label = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        label_imageFeature_Test = []
        for i, cur_line in enumerate(A_matrix):
            cur_className = cur_line.split()[0]
            cur_lable = className2Label[cur_className]
            cur_image_feature = train_test_imageFeature_predict[i]
            if cur_lable in testLabel:
                label_imageFeature_Test.append(cur_lable + ':' + cur_image_feature)
        MyFunction.saveVector(TestImageFeaturePredicted_fp, label_imageFeature_Test)


    @staticmethod
    def extractTrainImageFeatureFromNpy(feat_fp, imageName_Lable_fp, lable_feat_save_fp, k):
        imageName_Lable = MyFunction.readVector(imageName_Lable_fp)
        feat = np.load(feat_fp)
        lable_feat = []
        lable_count = {}
        for i in range(len(imageName_Lable)):
            cur_label = imageName_Lable[i].split("\t")[1]
            if cur_label == 'ZJL999':
                continue
            if cur_label not in lable_count:
                lable_count[cur_label] = 0
            if lable_count[cur_label] < k:
                lable_count[cur_label] += 1
                cur_train_feat_string = " ".join([str(e) for e in feat[i]])
                lable_feat.append(cur_label + ":" + cur_train_feat_string)
        MyFunction.saveVector(lable_feat_save_fp, lable_feat)

    @staticmethod
    def changeNPYtoTXTForTestSet(npy_fp, txt_fp):
        fc_vector_test = np.load(npy_fp)
        fc_vector_test_txt = []
        count = 0
        for line in fc_vector_test:
            count += 1
            if count % 1000 == 0:
                print(count)
            cur_line = [e for e in line]
            fc_vector_test_txt.append(cur_line)
        MyFunction.saveVector(txt_fp, fc_vector_test_txt)

    @staticmethod
    def mergeFinalResultBetweenFirstVersionAndPlantVersion(final_result_1version_fp, final_result_plant_version_fp, Plant_list, final_result_merge_fp):
        final_result_1version = MyFunction.readVector(final_result_1version_fp)
        final_result_plant_version = MyFunction.readVector(final_result_plant_version_fp)
        merge_final_result = []
        for i in range(len(final_result_1version)):
            line = final_result_1version[i]
            cur_lable = line.split()[1]
            if cur_lable in Plant_list:
                merge_final_result.append(final_result_plant_version[i])
            else:
                merge_final_result.append(final_result_1version[i])
        MyFunction.saveVector(final_result_merge_fp, merge_final_result)
        

    @staticmethod
    def gainLabelWordEmbeddingFromClassnameWordEmbedding(class_wordembedding_fp, label2ClassName_fp, save_fp):
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        class_wordEmbedding = MyFunction.readVector(class_wordembedding_fp)
        label_wordEmbedding_list = []
        for line in class_wordEmbedding:
            print(line)
            cur_className = line.split()[0]
            cur_wordEmbedding = " ".join(line.split()[1:])
            label_wordEmbedding_list.append(className_lable_map[cur_className] + ' ' + cur_wordEmbedding)
        MyFunction.saveVector(save_fp, label_wordEmbedding_list)

    @staticmethod
    def gainClassnameWordEmbeddingFromLabelWordEmbedding(label_wordembedding_fp, label2ClassName_fp, save_fp):
        lable_className_map = MyFunction.gainLabelToClassnale(label2ClassName_fp)
        label_wordembedding = MyFunction.readVector(label_wordembedding_fp)
        className_wordEmbedding_list = []
        for line in label_wordembedding:
            print(line)
            cur_label = line.split()[0]
            cur_wordEmbedding = " ".join(line.split()[1:])
            className_wordEmbedding_list.append(lable_className_map[cur_label] + ' ' + cur_wordEmbedding)
        MyFunction.saveVector(save_fp, className_wordEmbedding_list)

    @staticmethod
    def constractAllOneAMatrix(NumClass=155, A_fp=None, label_className_fp=None):
        label_className = MyFunction.readVector(label_className_fp)
        A = []
        for i in range(NumClass):
            cur_string = '1'
            for j in range(NumClass-1):
                cur_string += ' 1'
            cur_className = label_className[i].split('\t')[1]
            A.append(cur_className + " " + cur_string)
        MyFunction.saveVector(A_fp, A)
    
    @staticmethod
    def constractIdentityAMatrix(NumClass=155, A_fp=None, label_className_fp=None):
        label_className = MyFunction.readVector(label_className_fp)
        A = []
        for i in range(NumClass):
            cur_string = ''
            for j in range(NumClass):
                if i == j:
                    cur_string += ' 1'
                else:
                    cur_string += ' 0'
            cur_className = label_className[i].split('\t')[1]
            A.append(cur_className + cur_string)
        MyFunction.saveVector(A_fp, A)
    

    @staticmethod
    def mergeClassNameAndWordEmbedding(class_wordembedding_500_fp, label2ClassName_fp, save_fp):
        label2ClassName = MyFunction.readVector(label2ClassName_fp)
        #class_wordembedding_500 = MyFunction.readVector(class_wordembedding_500_fp)
        class_wordembedding_500 = np.load(class_wordembedding_500_fp)

        result = []
        for i in range(len(label2ClassName)):
            print(label2ClassName[i])
            result.append(label2ClassName[i].split("\t")[0] + ' ' + " ".join([str(e) for e in class_wordembedding_500[i].tolist()]) )
        MyFunction.saveVector(save_fp, result)

    @staticmethod
    def gainLableBasedOnClassName(className_fp, label2ClassName_fp, save_lable_fp):
        label2ClassName = MyFunction.readVector(label2ClassName_fp)
        className = MyFunction.readVector(className_fp)
        className_lable_map = {}
        for line in label2ClassName:
            className_lable_map[line.split()[1]] = line.split()[0]
        result_lable = []
        for cur_className in className:
            result_lable.append(className_lable_map[cur_className])
        MyFunction.saveVector(save_lable_fp, result_lable)
    
    @staticmethod
    def gainTestImageFeatureFromPredictResult(test_lable_fp, predict_result_fp, saveTestImageFeature_fp):
        test_lable = MyFunction.readVector(test_lable_fp)
        predict_ImageFeatureresult = MyFunction.readVector(predict_result_fp)
        test_predict_imageFeature = []
        for i in range(len(test_lable)):
            cur_lable = test_lable[i]
            cur_ImageFeature = predict_ImageFeatureresult[i]
            cur_result = cur_lable + ':' + cur_ImageFeature
            test_predict_imageFeature.append(cur_result)
        MyFunction.saveVector(saveTestImageFeature_fp, test_predict_imageFeature)
        print("### train and test ImageFeature has been saved!")
    '''
    @staticmethod
    def gainTestImageFeatureFromPredictResult(number_train, predict_ImageFeatureresult_fp, trainAndTestLable_fp, saveTrainImageFeature_fp, saveTestImageFeature_fp):
        trainAndTestLable = MyFunction.readVector(trainAndTestLable_fp)
        predict_ImageFeatureresult = MyFunction.readVector(predict_ImageFeatureresult_fp)
        train_predict_imageFeature = []
        test_predict_imageFeature = []
        for i in range(len(trainAndTestLable)):
            cur_lable = trainAndTestLable[i]
            cur_ImageFeature = predict_ImageFeatureresult[i]
            cur_result = cur_lable + ':' + cur_ImageFeature
            if i < number_train:
                train_predict_imageFeature.append(cur_result)
            else:
                test_predict_imageFeature.append(cur_result)
        MyFunction.saveVector(saveTrainImageFeature_fp, train_predict_imageFeature)
        MyFunction.saveVector(saveTestImageFeature_fp, test_predict_imageFeature)
        print("### train and test ImageFeature has been saved!")
        '''


class MyFunction:
    @staticmethod
    def merge_two_dicts(x, y):

        z = x.copy()
        z.update(y)
        return z

    @staticmethod
    def getLabelClassMap(label2ClassName_fp):
        label2ClassName_file = open(label2ClassName_fp, 'r')
        line = label2ClassName_file.readline()
        className_map = {}
        while line:
            cur_label = line.strip().split()[0]
            cur_className = line.strip().split()[1]
            className_map[cur_label] = cur_className
            line = label2ClassName_file.readline()
        return className_map

    @staticmethod
    def gainRealLable(test_image_labled_fp):
        test_image_labled = MyFunction.readVector(test_image_labled_fp)
        real_test_label = []
        for line in test_image_labled:
            real_test_label.append(line.strip('/n/t/r').split()[1])
        return real_test_label

    def compute_accuracy(predict_visual_feat, fc_vector_test, test_lable_orderd, real_test_label):
        '''
        predict_visual_feat_new = []
        test_lable_orderd_new = []
        for inx in range(len(test_lable_orderd)):
            if test_lable_orderd[inx] not in ['ZJL302', 'ZJL496', 'ZJL497', 'ZJL498', 'ZJL500', 'ZJL499']:
                predict_visual_feat_new.append(predict_visual_feat[inx])
                test_lable_orderd_new.append(test_lable_orderd[inx])
        test_lable_orderd = np.array(test_lable_orderd_new)
        predict_visual_feat = np.array(predict_visual_feat_new)
        '''

        pre_lable = []
        progress = ProgressBar()
        for i in progress(range(len(fc_vector_test))):
            outputLabel = MyFunction.kNNClassify_cos(fc_vector_test[i], predict_visual_feat, test_lable_orderd, 1)
            pre_lable.append(outputLabel)
        total_num = 0
        acc_num = 0

        for i in range(len(real_test_label)):
            cur_real_lable = real_test_label[i]
            cur_pre_result = pre_lable[i]
            if cur_real_lable not in test_lable_orderd:
                continue
            total_num += 1
            if cur_real_lable == cur_pre_result:
                acc_num += 1
        acc = 1.0*acc_num/total_num
        print("acc is ", acc)
        return pre_lable, acc

    @staticmethod
    def classificationForTest(cur_test_feat, predict_visual_weight, test_lable_orderd):
        cur_test_feat = np.append(cur_test_feat, [1])
        result_value = np.matmul(predict_visual_weight, cur_test_feat)
        sortedDistIndices = argsort(result_value)
        predict_label = test_lable_orderd[sortedDistIndices[-1]]
        return predict_label

    def compute_accuracy_weight(predict_visual_weight, fc_vector_test, test_lable_orderd, real_test_label):
        '''
        predict_visual_feat_new = []
        test_lable_orderd_new = []
        for inx in range(len(test_lable_orderd)):
            if test_lable_orderd[inx] not in ['ZJL302', 'ZJL496', 'ZJL497', 'ZJL498', 'ZJL500', 'ZJL499']:
                predict_visual_feat_new.append(predict_visual_feat[inx])
                test_lable_orderd_new.append(test_lable_orderd[inx])
        test_lable_orderd = np.array(test_lable_orderd_new)
        predict_visual_feat = np.array(predict_visual_feat_new)
        '''

        pre_lable = []
        progress = ProgressBar()
        for i in progress(range(len(fc_vector_test))):
            outputLabel = MyFunction.classificationForTest(fc_vector_test[i], predict_visual_weight, test_lable_orderd)
            pre_lable.append(outputLabel)
        total_num = 0
        acc_num = 0

        for i in range(len(real_test_label)):
            cur_real_lable = real_test_label[i]
            cur_pre_result = pre_lable[i]
            if cur_real_lable not in test_lable_orderd:
                continue
            total_num += 1
            if cur_real_lable == cur_pre_result:
                acc_num += 1
        acc = 1.0*acc_num/total_num
        print("acc is ", acc)
        return pre_lable, acc

    @staticmethod
    def get_attribute_list():
        head = list(['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'food',  'black', 'white', 'blue', 'brown',
            'orange', 'red', 'green', 'yellow', 'has_feathers', 'has_four_legs', 'has_two_legs', 'has_two_arms', 'for_entertainment',
            'for_business', 'for_communication', 'for_family', 'for_office use', 'for_personal', 'gorgeous', 'simple', 'elegant', 'cute',
            'pure', 'naive'])
        return head 

    @staticmethod
    def returnFirstAttribute(attribute):
        index_attribute = attribute.index
        class_list = ['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'food', 'None']
        className_firstclass_map = {}
        for i in range(attribute.shape[0]):
            cur_index = index_attribute[i]
            cur_class = attribute.loc[cur_index][0:7]
            cur_result = 6
            for j in range(len(cur_class)):
                e_cur_class = cur_class[j]
                if int(e_cur_class) == 1:
                    cur_result = j
                    break
            className_firstclass_map[cur_index] = class_list[j]
        return className_firstclass_map

    @staticmethod
    def getAttribute(label2ClassName_fp, lable_attribute_fp):
        label2ClassName_file = open(label2ClassName_fp, 'r')
        line = label2ClassName_file.readline()
        className_map = {}
        while line:
            cur_label = line.strip().split()[0]
            cur_className = line.strip().split()[1]
            className_map[cur_label] = cur_className
            line = label2ClassName_file.readline()

        label_list = MyFunction.get_attribute_list()
        attribute_list = []
        file_attribute = open(lable_attribute_fp)
        line = file_attribute.readline()
        className = []
        while line:
            cur_label = line.strip().split()[0]
            cur_className = className_map[cur_label]
            className.append(cur_className)
            cur_result = line.strip().split()[1:]
            attribute_list.append(cur_result)
            line = file_attribute.readline()
        attribute_pd = pd.DataFrame(data=attribute_list, index=className, columns=label_list)

        return attribute_pd

    @staticmethod
    def filterImageFeatureByLabel(allImageFeature_fp, newImageFeature_fp, newLable_fp):
        allImageFeature = MyFunction.readVector(allImageFeature_fp)
        newLable_classname = MyFunction.readVector(newLable_fp)
        newLable = []
        newImageFeature = []
        for e in newLable_classname:
            cur_lable = e.split()[0]
            newLable.append(cur_lable)

        for line in allImageFeature:
            cur_lable = line.split(':')[0]
            if cur_lable in newLable:
                newImageFeature.append(line)
        MyFunction.saveVector(newImageFeature_fp, newImageFeature)

    @staticmethod
    def gainLabelToClassnale(fp):
        lable_classname = MyFunction.readVector(fp)
        lable_className_map = {}
        for line in lable_classname:
            cur_lable = line.split()[0]
            cur_className = line.split()[1]
            lable_className_map[cur_lable] = cur_className
        return lable_className_map

    @staticmethod
    def kNNClassify_msr(newInput, dataSet, labels, k):  
        numSamples = dataSet.shape[0]
        diff = tile(newInput, (numSamples, 1)) - dataSet
        squaredDiff = diff ** 2
        squaredDist = sum(squaredDiff, axis=1)
        distance = squaredDist ** 0.5
        sortedDistIndices = argsort(distance)
        voteLabel = labels[sortedDistIndices[0]]
        return voteLabel

    @staticmethod
    def kNNClassify_cos(newInput, dataSet, labels, k):
        global distance
        distance = [0] * dataSet.shape[0]
        for i in range(dataSet.shape[0]):
            distance[i] = MyFunction.cosine_distance(newInput, dataSet[i])
            # distance[i] = MyFunction.L2distance(newInput, dataSet[i])
        sortedDistIndices = argsort(distance)
        voteLabel = labels[sortedDistIndices[0]]
        return voteLabel

    @staticmethod
    def cosine_distance(v1, v2):
        v1_sq = np.inner(v1, v1)
        v2_sq = np.inner(v2, v2)
        dis = 1 - np.inner(v1, v2) / math.sqrt(v1_sq * v2_sq)
        return dis

    @staticmethod
    def gainClassNameToLabel(fp):
        lable_classname = MyFunction.readVector(fp)
        className_label_map = {}
        for line in lable_classname:
            cur_lable = line.split()[0]
            cur_className = line.split()[1]
            className_label_map[cur_className] = cur_lable
        return className_label_map

    @staticmethod
    def gainImageFeature(image_feature_fp):
        image_feature = MyFunction.readVector(image_feature_fp)
        print(len(image_feature))
        lable_imageFeature_map = {}
        lable_num_map = {}
        for line in image_feature:
            cur_lable = line.split(":")[0]
            cur_image_feature = [float(e) for e in line.split(":")[1].split()]
            if cur_lable not in lable_imageFeature_map:
                lable_imageFeature_map[cur_lable] = cur_image_feature
                lable_num_map[cur_lable] = 1
            else:
                lable_num_map[cur_lable] += 1
                lable_imageFeature_map[cur_lable+"_"+str(lable_num_map[cur_lable])] = cur_image_feature
        return lable_imageFeature_map

    @staticmethod
    def L2distance(lis1, lis2):
        lis1 = np.array(lis1)
        lis2 = np.array(lis2)
        diff = lis1 - lis2
        return sum(diff**2)/len(diff)

    @staticmethod
    def constractImageFeature(ZJL_calssName_fp, alltrain_image_feature_fp, trainAndTestClassName_fp, saveImageFeature_fp):
        ZJL_calssName = MyFunction.readVector(ZJL_calssName_fp)
        alltrain_image_feature = MyFunction.readVector(alltrain_image_feature_fp)
        trainAndTestClassName = MyFunction.readVector(trainAndTestClassName_fp)
        calssName_ZJL_map = {}
        extractImageFeature = []
        ZJL_imageFeature_map = {}
        count_saved_imageFeature = 0
        for line in ZJL_calssName:
            cur_ZJL = line.split()[0]
            cur_className = line.split()[1]
            calssName_ZJL_map[cur_className] = cur_ZJL
        for line in alltrain_image_feature:
            ZJL_imageFeature_map[line.split(':')[0]] = ' '.join(line.split(":")[1:])
        for cur_className in trainAndTestClassName:
            cur_ZJL = calssName_ZJL_map[cur_className]
            if cur_ZJL in ZJL_imageFeature_map:
                count_saved_imageFeature += 1
                extractImageFeature.append(ZJL_imageFeature_map[cur_ZJL])
        MyFunction.saveVector(saveImageFeature_fp, extractImageFeature)
        print("total imageFeature", count_saved_imageFeature)

    @staticmethod
    def gainWordEmbedding(wordEmbedding_fp=None, return_type='string'):
        word_embeding_file = open(wordEmbedding_fp, 'r')
        line = word_embeding_file.readline()
        word_vector_map = {}
        while line:
            cur_word = line.strip().split()[0]
            if return_type == 'float':
                cur_embedding = [float(e) for e in line.strip().split()[1:]]
            elif return_type == 'string':
                cur_embedding = " ".join(line.strip().split()[1:])
            else:
                print("the flag type is wrrong")
            word_vector_map[cur_word] = cur_embedding
            line = word_embeding_file.readline()
        return word_vector_map

    @staticmethod
    def cosineSimilarity(A, B):
        t1 = np.inner(A, B)
        t2 = np.inner(A, A)
        t3 = np.inner(B, B)
        if t2 == 0 or t3 == 0:
            return 2
        #return t1
        return t1 / math.sqrt(t2 * t3)
    '''
    @staticmethod
    def cosineSimilarity(A, B):
        t1 = MyFunction.inner(A, B)
        t2 = MyFunction.inner(A, A)
        t3 = MyFunction.inner(B, B)
        if t2 == 0 or t3 == 0:
            return 2
        #return t1
        return t1 / (pow(t2, 0.5) * pow(t3, 0.5))
    '''
    @staticmethod
    def inner(A, B):
        count, i = 0, 0
        n = len(A)
        while i < n:
            count += (A[i] * B[i])
            i += 1
        return count

    @staticmethod
    def computeOffLineAcc(image_label_fp, pre_result_fp, test_lable_fp):
        total_num = 0
        acc_num = 0
        test_lable = MyFunction.readVector(test_lable_fp)
        image_labled = MyFunction.readVector(image_label_fp)
        pre_result = MyFunction.readVector(pre_result_fp)
        for i in range(len(image_labled)):
            cur_image_labled_line = image_labled[i]
            cur_pre_result_line = pre_result[i]
            if len(cur_image_labled_line.split('\t')) == 2:
                cur_real_lable = cur_image_labled_line.split('\t')[1]
                cur_pre_result = cur_pre_result_line.split('\t')[1]
                if cur_real_lable not in test_lable:
                    continue
                total_num += 1
                if cur_real_lable == cur_pre_result:
                    acc_num += 1
        print("off line acc is:", 1.0*acc_num/total_num)

    @staticmethod
    def process_lr(now_lr, loss_all, patience_len, patience_value):
        if len(loss_all) <= patience_len:
            return now_lr
        else:
            if loss_all[len(loss_all)-patience_len] - loss_all[-1] < patience_value:
                return now_lr/2.0
            else:
                return now_lr

    @staticmethod
    def gainConfig():
        confit_fp = './yang.conf'
        config = ConfigParser.ConfigParser()
        config._interpolation = ConfigParser.BasicInterpolation()
        config.read(confit_fp)
        return config

    @staticmethod
    def readFcLayer(pre_fc_fp):
        pre_fc_string = MyFunction.readVector(pre_fc_fp)
        pre_fc_double = []
        for line in pre_fc_string:
            pre_fc_double.append([float(e) for e in line.split()])
        return pre_fc_double

    @staticmethod
    def load_ImageFeature(trainImageFeature_fp):
        trainImageFeature_string = MyFunction.readVector(trainImageFeature_fp)
        trainImageFeature_double = []
        for line in trainImageFeature_string:
            cur_line_float = [float(e) for e in line.split()]
            trainImageFeature_double.append(cur_line_float)
        return np.array(trainImageFeature_double)

    @staticmethod
    def load_WordEmbedding(extractWordembedding_fp):
        #this is extractWordEmbedding, which means that there is not classname or label in every line
        extractWordembedding_string = MyFunction.readVector(extractWordembedding_fp)
        extractWordembedding_double = []
        for line in extractWordembedding_string:
            cur_line_float = [float(e) for e in line.split()]
            extractWordembedding_double.append(cur_line_float)
        return np.array(extractWordembedding_double)

    @staticmethod
    def save2DimentionVector(save_fp, vector2Dimention):
        save_result = []
        for line in vector2Dimention:
            new_line = " ".join([str(e) for e in line])
            save_result.append(new_line)
        MyFunction.saveVector(save_fp, save_result)

    @staticmethod
    def readVector(fp):
        f = open(fp,'r')
        result = []
        line = f.readline()
        while line:
            result.append(line.strip())
            line = f.readline()
        return result

    @staticmethod
    def saveVector(save_fp, vector):
        file = open(save_fp, 'w')
        for value in vector:
            file.write(str(value) + "\n")
        file.close()


class ComputeTestAcc:
    @staticmethod
    def gainPredictValue_L2(pre_fc_value, lable_imageFeatures_map):
        cur_smallest_dis_value = 100000000.0
        cur_best_label = 'None'
        for cur_label in lable_imageFeatures_map:
            cur_fc_value = lable_imageFeatures_map[cur_label]
            cur_dis_value = MyFunction.L2distance(cur_fc_value, pre_fc_value)
            if cur_dis_value < cur_smallest_dis_value:
                cur_best_label = cur_label
                cur_smallest_dis_value = cur_dis_value
        return cur_best_label

    @staticmethod
    def gainPredictValue_cosine(pre_fc_value, lable_imageFeatures_map):
        cur_best_same_value = 0.00
        cur_best_label = 'None'
        for cur_label in lable_imageFeatures_map:
            cur_fc_value = lable_imageFeatures_map[cur_label]
            cur_same_value = MyFunction.cosineSimilarity(cur_fc_value, pre_fc_value)
            if cur_same_value > cur_best_same_value:
                cur_best_label = cur_label
                cur_best_same_value = cur_same_value
        return cur_best_label

    @staticmethod
    def mergeResult(lables, image_fp):
        image_name = MyFunction.readVector(image_fp)
        imageName_lable = []
        for i in range(len(image_name)):
            cur_image_name = image_name[i].split('\t')[0]
            cur_label = lables[i]
            imageName_lable.append(cur_image_name + "\t" + cur_label)
        return imageName_lable

    @staticmethod
    def gainTestResult(pre_fc, Image_feature_fp, Mode='cos'):
        print("gain test result...")
        result = []
        progress = ProgressBar()
        image_features = MyFunction.readVector(Image_feature_fp)
        lable_imageFeatures_map = {}
        for line in image_features:
            cur_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            cur_fc_value = cur_fc_value.split(" ")
            cur_fc_value = [float(e) for e in cur_fc_value]
            lable_imageFeatures_map[cur_label] = cur_fc_value

        print("tital image", len(pre_fc))
        for i in progress(range(len(pre_fc))):
            cur_pre = pre_fc[i]
            if i < 7000000:
                if Mode == 'cos':
                    cur_result = ComputeTestAcc.gainPredictValue_cosine(cur_pre, lable_imageFeatures_map)
                else:
                    cur_result = ComputeTestAcc.gainPredictValue_L2(cur_pre, lable_imageFeatures_map)
            else:
                cur_result = 'ZJL000'
            result.append(cur_result)
        return result

    @staticmethod
    def basedOnTestImageFeatureToPredictTestResult(fc_vector_test_fp, test_image_feature_fp, test_image_unlable_fp, final_result_fp, test_image_labled_fp, Flag_npy, test_lable_fp):
        # image_label_fp=config.get("FILE_PATH", 'image-labled'), pre_result_fp=config.get("FILE_PATH", 'final_result')
        if Flag_npy:
            fc_vector_test = np.load(fc_vector_test_fp)
        else:
            fc_vector_test = MyFunction.readFcLayer(fc_vector_test_fp)
        result = ComputeTestAcc.gainTestResult(fc_vector_test, test_image_feature_fp)
        merge_result = ComputeTestAcc.mergeResult(result, test_image_unlable_fp)
        MyFunction.saveVector(final_result_fp, merge_result)
        try:
            MyFunction.computeOffLineAcc(image_label_fp=test_image_labled_fp, pre_result_fp=final_result_fp, test_lable_fp=test_lable_fp)
        except:
            print("can not compute acc")


class ConstractA():
    @staticmethod
    def constractAbyRelatedClass(related_class_fp, k=2, isSymmetrical=False):
        related_class = MyFunction.readVector(related_class_fp)
        adj = {}
        for line in related_class:
            cur_className = line.split(":")[0].strip(' ')
            related_className = line.split(':')[1].split()
            adj[cur_className] = []
            for i in range(k):
                adj[cur_className].append(related_className[i])

        AMatrix = ConstractA.transfromAintoMatrix(adj, isSymmetrical)
        return AMatrix


    @staticmethod
    def transfromAintoMatrix(cur_adj, isSymmetrical):
        keys = list(cur_adj.keys())
        #print(keys)
        num_keys = len(keys)
        A_matrix = np.array([[0]*num_keys] * num_keys)
        key_index_map = dict(zip(keys, range(len(keys))))
        for cur_key in cur_adj:
            cur_key_index = key_index_map[cur_key]
            A_matrix[cur_key_index][cur_key_index] = 1
            for cur_relate in cur_adj[cur_key]:
                cur_relate_index = key_index_map[cur_relate]
                A_matrix[cur_key_index][cur_relate_index] = 1
                if isSymmetrical:
                    A_matrix[cur_relate_index][cur_key_index] = 1
        result = []
        for i in range(len(A_matrix)):
            line = A_matrix[i]
            cur_lable = keys[i]
            cur_line_string = cur_lable + " " + ' '.join([str(e) for e in line])
            result.append(cur_line_string)
        return result

    @staticmethod
    def transfromAintoMatrix_differentImportance(cur_adj):
        keys = list(cur_adj.keys())
        #print(keys)
        num_keys = len(keys)
        A_matrix = np.array([[0]*num_keys] * num_keys)
        key_index_map = dict(zip(keys, range(len(keys))))
        for cur_key in cur_adj:
            total_relaed_class_num = len(cur_adj[cur_key])
            cur_key_index = key_index_map[cur_key]
            A_matrix[cur_key_index][cur_key_index] = 1 + total_relaed_class_num
            for cur_relate in cur_adj[cur_key]:
                cur_relate_index = key_index_map[cur_relate]
                A_matrix[cur_key_index][cur_relate_index] = total_relaed_class_num
                total_relaed_class_num -= 1
        result = []
        for i in range(len(A_matrix)):
            line = A_matrix[i]
            cur_lable = keys[i]
            cur_line_string = cur_lable + " " + ' '.join([str(e) for e in line])
            result.append(cur_line_string)
        return result

    @staticmethod
    def get_attribute_list(version='C'):
        if version == 'AB':
            head = list(['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'black', 'white', 'blue', 'brown',
                'orange', 'red', 'green', 'yellow', 'has_feathers', 'has_four_legs', 'has_two_legs', 'has_two_arms', 'for_entertainment',
                'for_business', 'for_communication', 'for_family', 'for_office use', 'for_personal', 'gorgeous', 'simple', 'elegant', 'cute',
                'pure', 'naive'])
        elif version == 'C':
            head = list(['animal', 'transportation', 'clothes', 'plant', 'kitchenware', 'device', 'food', 'musical_instrument', 'building', \
                'furniture','black', 'white', 'red', 'orange', 'yellow', 'green', 'blue', 'violet',
                'has_feathers', 'scales', 'shells', 'has_four_legs', 'has_two_legs', 'has_two_arms', 'has_wings',
                'for_entertainment', 'for_business', 'for_communication', 'for_family', 'for_officeuse', 'for_personal', 
                'globular', 'star-shaped', 'cubic', 'cylindrical', 'schistose', 'flat',
                'metal', 'plastic', 'galss', 'wood', 'cloth',
                'eaten', 'dive', 'fly', 'float', 'driven','lights', 'sound', 'photograph'])

        return head 

    @staticmethod
    def gainAbasedOnWordEmedding(word_embeding_fp, label2ClassName_fp, all_label_fp, min_distance=0.4):
        all_label = MyFunction.readVector(all_label_fp)
        label2ClassName = MyFunction.readVector(label2ClassName_fp)
        className_lable_map = {}
        for cur_line in label2ClassName:
            className_lable_map[cur_line.strip().split()[1]] = cur_line.strip().split()[0]

        word_embeding_file = open(word_embeding_fp, 'r')
        line = word_embeding_file.readline()
        word_vector_map = {}
        while line:
            cur_word = line.strip().split()[0]
            cur_lable = className_lable_map[cur_word]
            if cur_lable not in all_label:
                line = word_embeding_file.readline()
                continue
            cur_embedding = [float(e) for e in line.strip().split()[1:]]
            word_vector_map[cur_word] = cur_embedding
            line = word_embeding_file.readline()

        adj = {}
        for cur_key in word_vector_map:
            test_vector = word_vector_map[cur_key]
            adj[cur_key] = []
            for key in word_vector_map:
                if cur_key == key:
                    continue
                cur_vector = word_vector_map[key]
                distance = MyFunction.cosineSimilarity(cur_vector,test_vector)
                if distance > min_distance:
                    adj[cur_key] += [key]
        print("tital class num", len(word_vector_map))
        return adj

    @staticmethod
    def getAttribute(label2ClassName_fp, attribute_per_class_fp):
        label2ClassName_file = open(label2ClassName_fp, 'r')
        line = label2ClassName_file.readline()
        className_map = {}
        while line:
            cur_label = line.strip().split()[0]
            cur_className = line.strip().split()[1]
            className_map[cur_label] = cur_className
            line = label2ClassName_file.readline()

        label_list = ConstractA.get_attribute_list()
        attribute_list = []
        file_attribute = open(attribute_per_class_fp)
        line = file_attribute.readline()
        className = []
        while line:
            cur_label = line.strip().split()[0]
            cur_className = className_map[cur_label]
            className.append(cur_className)
            cur_result = line.strip().split()[1:]
            attribute_list.append(cur_result)
            line = file_attribute.readline()
        attribute_pd = pd.DataFrame(data=attribute_list, index=className, columns=label_list)

        return attribute_pd
    
    @staticmethod
    def returnFirstAttribute(attribute, version='C'):
        index_attribute = attribute.index
        if version == 'AB':
            class_list = ['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'None']
        elif version == "C":
            class_list = ['animal', 'transportation', 'clothes', 'plant', 'kitchenware', 'device', 'food', 'musical_instrument', 'building', 'furniture', 'None']
        className_firstclass_map = {}
        for i in range(attribute.shape[0]):
            cur_index = index_attribute[i]
            if version == "AB":
                cur_class = attribute.loc[cur_index][0:6]
                cur_result = 6
            elif version == "C":
                cur_class = attribute.loc[cur_index][0:10]
                cur_result = 10
            for j in range(len(cur_class)):
                e_cur_class = cur_class[j]
                if int(float(e_cur_class)) == 1:
                    cur_result = j
                    break
            className_firstclass_map[cur_index] = class_list[cur_result]
        return className_firstclass_map

    @staticmethod
    def constractA(word_embeding_fp, all_label_fp, min_distance, attribute_per_class_fp):
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        cur_adj = ConstractA.gainAbasedOnWordEmedding(word_embeding_fp, label2ClassName_fp, all_label_fp, min_distance)
        attribute = ConstractA.getAttribute(label2ClassName_fp, attribute_per_class_fp)
        first_class_map = ConstractA.returnFirstAttribute(attribute)
        
        #提出掉不同大类之间的链接
        for cur_key in cur_adj:
            relate_key = cur_adj[cur_key]
            new_relate_key = []
            for cur_relate_key in relate_key:
                if first_class_map[cur_key] not in ['animal', 'transportation', 'clothes', 'plant', 'kitchenware', 'device', 'food', 'musical_instrument', 'building', 'furniture']:
                    break
                if  first_class_map[cur_relate_key] == first_class_map[cur_key]:
                    new_relate_key.append(cur_relate_key)
            cur_adj[cur_key] = new_relate_key
        
        #cope animals
        for cur_key1 in cur_adj:
            relate_class_by_wordembeding = cur_adj[cur_key1]
            if first_class_map[cur_key1] != 'animal':
                continue
            first_key_attribute = list(attribute.loc[cur_key1][18:25])
            if first_key_attribute == ['0', '0', '0', '0', '0', '0', '0']:
                #print("999", cur_key1)
                continue
            new_relate_key = []
            for cur_key2 in cur_adj:
                if first_class_map[cur_key2] != 'animal':
                    continue
                if cur_key1 == cur_key2:
                    continue
                second_key_attribute = list(attribute.loc[cur_key2][18:25])
                if first_key_attribute == second_key_attribute:
                #and cur_key2 in relate_class_by_wordembeding:
                    new_relate_key.append(cur_key2)
            cur_adj[cur_key1] = new_relate_key

        #plant
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'plant':
                continue
            new_relate_key = []
            for cur_key2 in cur_adj:
                if first_class_map[cur_key2] != 'plant':
                    continue
                if cur_key1 == cur_key2:
                    continue
                new_relate_key.append(cur_key2)
            cur_adj[cur_key1] = new_relate_key

        #device
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'device':
                continue
            new_relate_key = []
            for cur_key2 in cur_adj:
                if first_class_map[cur_key2] != 'device':
                    continue
                if cur_key1 == cur_key2:
                    continue
                new_relate_key.append(cur_key2)
            cur_adj[cur_key1] = new_relate_key

        #musical
        key_relateKey_map = {'violin': ['cello'], 'cello': ['violin'], 'flute': ['bassoon'], 'bassoon': ['flute'],\
                            'harp': ['marimba'], 'marimba': ['harp']}
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'musical_instrument':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key

        #food
        related_key = [['artichoke', 'broccoli'], ['bagel', 'burrito', 'crepe', 'carbonara', 'waffle', 'hotdog'], ['hip', 'strawberry'], ['trifle', 'cup', 'hotpot'], ['dough', 'lasagna', 'ravioli']]
        key_relateKey_map = {}
        for group in related_key:
            for cur_key in group:
                key_relateKey_map[cur_key] = list(set(group) - set([cur_key]))
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'food':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key

        #buliding
        related_key = [['planetarium', 'vault', 'dome'], ['library', 'bookshop'], ['toyshop', 'stage']]
        key_relateKey_map = {}
        for group in related_key:
            for cur_key in group:
                key_relateKey_map[cur_key] = list(set(group) - set([cur_key]))
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'building':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key

        #furniture
        related_key = [['tub', 'washbasin'], ['cradle', 'crib']]
        key_relateKey_map = {}
        for group in related_key:
            for cur_key in group:
                key_relateKey_map[cur_key] = list(set(group) - set([cur_key]))
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'furniture':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key
          
        #kitchenware
        related_key = [['strainer', 'thimble', 'cup', 'pitcher'], ['corkscrew', 'spatula', 'cleaver']]
        key_relateKey_map = {}
        for group in related_key:
            for cur_key in group:
                key_relateKey_map[cur_key] = list(set(group) - set([cur_key]))
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'kitchenware':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key

        #transportation
        related_key = [['harvester', 'thresher', 'plow'], ['unicycle', 'snowmobile', 'dogsled'], ['yawl', 'trimaran', 'liner'], ['minibus', 'ambulance', 'minivan', 'pickup', 'canoe', 'grille']]
        key_relateKey_map = {}
        for group in related_key:
            for cur_key in group:
                key_relateKey_map[cur_key] = list(set(group) - set([cur_key]))
        for cur_key1 in cur_adj:
            if first_class_map[cur_key1] != 'transportation':
                continue
            if cur_key1 not in key_relateKey_map:
                new_relate_key = []
            else:
                new_relate_key = key_relateKey_map[cur_key1]
            cur_adj[cur_key1] = new_relate_key
        return cur_adj


class Generator_gcn:
    @staticmethod
    def sample_mask_sigmoid(idx, h, w):
        """Create mask."""
        mask = np.zeros((h, w))
        for i_, e in enumerate(idx):
            if e == 1:
                mask[i_] = np.ones((1, w))
        return np.array(mask)

    def updatesample_mask_sigmoid(idx, h, w, rate=0.5):
        """Create mask."""
        mask = np.zeros((h, w))
        for i_, e in enumerate(idx):
            if e == 1:
                rand_ = np.random.random_sample()
                if rand_ > rate:
                    mask[i_] = np.ones((1, w))
        return np.array(mask)

    @staticmethod
    def load_dictionary_A(A_matrix_fp):
        A_matrix_string = MyFunction.readVector(A_matrix_fp)
        A_matrix = []
        className_order = []
        index_relateindex_map = {}
        for line in A_matrix_string:
            className_order.append(line.split()[0])
            cur_line_string = line.split()[1:]
            cur_line_int = [float(e) for e in cur_line_string]
            A_matrix.append(cur_line_int)

        for i in range(len(A_matrix)):
            index_relateindex_map[i] = []
            #print(A_matrix[i])
            for j in range(len(A_matrix)):
                if A_matrix[i][j] == 1:
                    index_relateindex_map[i].append(j)
        return index_relateindex_map, className_order

    @staticmethod
    def load_ImageFeature(trainImageFeature_fp, test_num, ImageFeatureDimention, label2ClassName_fp, className_order):
        trainImageFeature_string = MyFunction.readVector(trainImageFeature_fp)
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        trainImageFeature_double = []
        lable_imageFeature_map = {}

        for line in trainImageFeature_string:
            cur_lable = line.split(":")[0]
            cur_image_feature = [float(e) for e in line.split(":")[1].split()]
            lable_imageFeature_map[cur_lable] = cur_image_feature
        #print(lable_imageFeature_map)
        for cur_className in className_order:
            if className_lable_map[cur_className] in lable_imageFeature_map:
                trainImageFeature_double.append(lable_imageFeature_map[className_lable_map[cur_className]])
            else:
                new_ImageFeature = list(np.array([0]*ImageFeatureDimention))
                trainImageFeature_double.append(new_ImageFeature)
        print('train image Feature 0', len(trainImageFeature_double[0]))
        return np.array(trainImageFeature_double)

    @staticmethod
    def load_WordEmbedding(Wordembedding_fp, className_order):
        Wordembedding_fp_list = []
        pre = Wordembedding_fp.split('&')[0]
        end = Wordembedding_fp.split('&')[-1]
        for i in range(1, len(Wordembedding_fp.split('&'))-1):
            cur_fp = pre + Wordembedding_fp.split('&')[i] + end
            Wordembedding_fp_list.append(cur_fp)

        className_Wordembedding_map = []
        for i in range(len(Wordembedding_fp_list)):
            className_Wordembedding_map.append(MyFunction.gainWordEmbedding(Wordembedding_fp_list[i], 'float'))

        extractWordembedding_double = []
        for inx, cur_className in enumerate(className_order):
                word_embedding = []
                for cur_i in range(len(Wordembedding_fp_list)):
                    word_embedding += className_Wordembedding_map[cur_i][cur_className]
                extractWordembedding_double.append(word_embedding)
        return np.array(extractWordembedding_double)

    @staticmethod
    def gainTrainAndTestIndex(className_order, label2ClassName_fp, train_lable_fp):
        TrainIndex = []
        TestIndex = []
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        train_label = MyFunction.readVector(train_lable_fp)
        for cur_className in className_order:
            if className_lable_map[cur_className] in train_label:
                TrainIndex.append(1)
                TestIndex.append(0)
            else:
                TrainIndex.append(0)
                TestIndex.append(1)
        return TrainIndex, TestIndex

    @staticmethod
    def load_data_zero_shot(A_matrix_fp, Wordembedding_fp, trainImageFeature_fp, ImageFeatureDimention, label2ClassName_fp, train_lable_fp):
        graph, className_order = Generator_gcn.load_dictionary_A(A_matrix_fp)
        TrainIndex, TestIndex = Generator_gcn.gainTrainAndTestIndex(className_order, label2ClassName_fp, train_lable_fp)
        allx = Generator_gcn.load_WordEmbedding(Wordembedding_fp, className_order)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        train_test_label_order = [className_lable_map[e] for e in className_order]
        features = allx

        return adj, features, train_test_label_order, TrainIndex, TestIndex
    @staticmethod
    def load_Weight(weight_fp, classOrder_fp, className_order, label2ClassName_fp):
        classOrder = MyFunction.readVector(classOrder_fp)
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        weight = np.load(weight_fp)
        label_weight_map = {}
        for (cur_weight, cur_label) in zip(weight, classOrder):
            cur_label = cur_label.split(":")[1].strip("' ").strip(",")[:-1]
            label_weight_map[cur_label] = cur_weight

        Y = []
        for cur_className in className_order:
            cur_label = className_lable_map[cur_className]
            if cur_label in label_weight_map.keys():
                Y.append(label_weight_map[cur_label])
            else:
                print(cur_label)
                Y.append([0] * weight.shape[1])
        return np.array(Y)
        


    @staticmethod
    def load_data_zero_shot_weight(A_matrix_fp, Wordembedding_fp,
                                   trainImageFeature_fp,
                                   ImageFeatureDimention,
                                   label2ClassName_fp,
                                   train_lable_fp,
                                   weight_fp,
                                   classOrder_fp):
        graph, className_order = Generator_gcn.load_dictionary_A(A_matrix_fp)
        TrainIndex, TestIndex = Generator_gcn.gainTrainAndTestIndex(className_order, label2ClassName_fp, train_lable_fp)
        allx = Generator_gcn.load_WordEmbedding(Wordembedding_fp, className_order)
        Y = Generator_gcn.load_Weight(weight_fp, classOrder_fp, className_order, label2ClassName_fp)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        className_lable_map = MyFunction.gainClassNameToLabel(label2ClassName_fp)
        train_test_label_order = [className_lable_map[e] for e in className_order]
        features = allx

        return adj, Y, features, train_test_label_order, TrainIndex, TestIndex
    
    @staticmethod
    def load_label_allFeat_map(train_imageName_Lable_fp, train_feat_fp):
        train_feat_fp_list = []
        pre = train_feat_fp.split('&')[0]
        end = train_feat_fp.split('&')[-1]
        for i in range(1, len(train_feat_fp.split('&'))-1):
            cur_fp = pre + train_feat_fp.split('&')[i] + end
            train_feat_fp_list.append(cur_fp)
        print(train_feat_fp_list)
        train_image_labeled_fp_list = []
        pre = train_imageName_Lable_fp.split('&')[0]
        end = train_imageName_Lable_fp.split('&')[-1]
        for i in range(1, len(train_imageName_Lable_fp.split('&'))-1):
            cur_fp = pre + train_imageName_Lable_fp.split('&')[i] + end
            train_image_labeled_fp_list.append(cur_fp)
        print(train_image_labeled_fp_list)
        train_feat_all = []
        for train_feat_fp in train_feat_fp_list:
            train_feat = list(np.load(train_feat_fp))
            train_feat_all += train_feat
        train_feat_all = np.array(train_feat_all)

        train_image_labeled_all = []
        for train_image_labeled_fp in train_image_labeled_fp_list:
            train_image_labeled = MyFunction.readVector(train_image_labeled_fp)
            train_image_labeled_all += train_image_labeled

        label_allFeat_map = {}
        for i in range(len(train_image_labeled_all)):
            cur_lable = train_image_labeled_all[i].split("\t")[1]
            if cur_lable not in label_allFeat_map:
                label_allFeat_map[cur_lable] = []
            cur_feat_double = train_feat_all[i]
            label_allFeat_map[cur_lable].append(cur_feat_double)
        return label_allFeat_map

    @staticmethod
    def nextBatchImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention):
        trainImageFeature_double = []
        for cur_label in train_test_label_order:
            if cur_label in label_allFeat_map.keys():
                num_feat = len(label_allFeat_map[cur_label])
                selected_index = randint(0, num_feat)
                new_ImageFeature = label_allFeat_map[cur_label][selected_index]
            else:
                new_ImageFeature = list(np.array([0]*ImageFeatureDimention))
            trainImageFeature_double.append(new_ImageFeature)
        #print("sssssss", selected_index)
        return np.array(trainImageFeature_double)

    @staticmethod
    def meanImageFeature(label_allFeat_map, train_test_label_order, ImageFeatureDimention):
        trainImageFeature_double = []
        for cur_label in train_test_label_order:
            if cur_label in label_allFeat_map.keys():
                cur_all_feat = np.array(label_allFeat_map[cur_label])
                new_ImageFeature = cur_all_feat.mean(axis=0)
            else:
                new_ImageFeature = list(np.array([0]*ImageFeatureDimention))
            trainImageFeature_double.append(new_ImageFeature)
        #print("sssssss", selected_index)
        return np.array(trainImageFeature_double)

    @staticmethod
    def extractRandomTrainImageFeatureFromNpy(train_feat, train_imageName_Lable):
        lable_feat = []
        unic_lable = set()
        for i in range(len(train_imageName_Lable)):
            cur_lable = train_imageName_Lable[i].split("\t")[1]
            if cur_lable not in unic_lable:
                unic_lable.add(cur_lable)
                cur_train_feat_string = " ".join([str(e) for e in train_feat[i]])
                lable_feat.append(cur_lable + ":" + cur_train_feat_string)


class PostProcess():
    @staticmethod
    def readVector(fp):
        f = open(fp,'r')
        result = []
        line = f.readline()
        while line:
            result.append(line.strip())
            line = f.readline()
        return result

    @staticmethod
    def saveVector(save_fp, vector):
        file = open(save_fp, 'w')
        for value in vector:
            file.write(str(value) + "\n")
        file.close()

    @staticmethod
    def getNewFileNameBasedOnInitialFileFilePath(path):
        cur_path = os.path.split(path)[0]
        initial_file_name_and_extendname = os.path.split(path)[1]
        initial_file_name, extend_name = os.path.splitext(initial_file_name_and_extendname)
        new_result_path = cur_path + '/' + initial_file_name + '_Wrong' + extend_name
        return new_result_path

    @staticmethod
    def setWrongAnswerRandomly(path=None, rate=0.3, rand=11, wrong_label='ZJL999'):
        initial_result = PostProcess.readVector(path)
        new_result = []
        random.seed(rand)
        rand_flag = random.randint(1, 100)
        count = 0
        for index, cur_line in enumerate(initial_result):
            cur_label = cur_line.split()[1]
            cur_imageName = cur_line.split()[0]
            random.seed(rand_flag + index)
            cur_random = random.random()
            if cur_random < 0.3:
                count += 1
                cur_label = wrong_label
            new_result.append(cur_imageName + '\t' + cur_label)
        print("%.2f %% data was set to wrong answer"%(count/len(initial_result)))
        new_result_path = PostProcess.getNewFileNameBasedOnInitialFileFilePath(path)
        PostProcess.saveVector(new_result_path, new_result)


config = MyFunction.gainConfig()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def create_config_proto():
    """Reset tf default config proto"""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    # config.operation_timeout_in_ms=8000
    config.log_device_placement = False
    return config

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict