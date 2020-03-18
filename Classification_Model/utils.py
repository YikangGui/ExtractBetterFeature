import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.utils import to_categorical
import configparser as ConfigParser
from progressbar import *
from sklearn.decomposition import PCA
import random 
import copy

confit_fp = './yang.conf'
config = ConfigParser.ConfigParser()
config._interpolation = ConfigParser.BasicInterpolation()
config.read(confit_fp)

Flag_resize = config.getboolean('MODEL', 'Flag_resize')
image_size = config.getint("MODEL", 'image_size')

class Analysis:
    @staticmethod
    def analysisTestDataPredictionResult(pre_y, lable_list_fp, test_imagepath_lable_fp):
        lableClassName_map = MyFunction.getLabelClassMap()
        f_train = open(lable_list_fp)
        temp = f_train.readlines()
        train_dict = dict(zip(range(230), [x.strip('\n\r') for x in temp]))
        realLable_related_lable = {}
        test_imagepath_lable = MyFunction.readVector(test_imagepath_lable_fp)
        lable_sorted_list = []
        for line in test_imagepath_lable:
            lable_sorted_list.append(line.split()[1])
        count = 0
        for cur_pre_result in pre_y:
            cur_pre_result = list(cur_pre_result)
            cur_pre_result_sorted = sorted(cur_pre_result, reverse=True)
            first_score_lable = train_dict[cur_pre_result.index(cur_pre_result_sorted[0])]
            second_score_lable = train_dict[cur_pre_result.index(cur_pre_result_sorted[1])]
            third_score_lable = train_dict[cur_pre_result.index(cur_pre_result_sorted[2])]
            real_lable = lable_sorted_list[count]
            realLable_related_lable[lableClassName_map[real_lable]] = [lableClassName_map[first_score_lable], lableClassName_map[second_score_lable], lableClassName_map[third_score_lable]]
            print(lableClassName_map[real_lable], realLable_related_lable[lableClassName_map[real_lable]])
            count += 1

    @staticmethod
    def testResultAnalysis(image_label_fp=config.get("FILE_PATH", 'image-labled'), pre_result_fp=config.get("FILE_PATH", 'final_result')):
        total_num = 0
        acc_num = 0
        image_labled = MyFunction.readVector(image_label_fp)
        pre_result = MyFunction.readVector(pre_result_fp)
        class_acc_map = {}
        firstAttribute_acc_map = {}
        attribute_pd = MyFunction.getAttribute()
        lableClassName_map = MyFunction.getLabelClassMap()

        for i in range(len(image_labled)):
            cur_image_labled_line = image_labled[i]
            cur_pre_result_line = pre_result[i]
            if len(cur_image_labled_line.split('\t')) == 2:
                cur_real_lable = cur_image_labled_line.split('\t')[1]
                cur_pre_result = cur_pre_result_line.split('\t')[1]
                total_num += 1
                if cur_real_lable == cur_pre_result:
                    acc_num += 1
                #gain class_acc
                if cur_real_lable not in class_acc_map:
                    class_acc_map[cur_real_lable] = [0,0]
                if cur_real_lable == cur_pre_result:
                    class_acc_map[cur_real_lable][0] += 1
                class_acc_map[cur_real_lable][1] += 1

                #gain first attribute acc
                if cur_real_lable not in firstAttribute_acc_map:
                    firstAttribute_acc_map[cur_real_lable] = [0, 0]
                cur_real_className = lableClassName_map[cur_real_lable]
                cur_pre_className = lableClassName_map[cur_pre_result]
                cur_real_firstAttribute = list(attribute_pd.loc[cur_real_className][0:6])
                cur_pre_firstAttribute = list(attribute_pd.loc[cur_pre_className][0:6]) 
                if cur_real_firstAttribute == cur_pre_firstAttribute:
                    firstAttribute_acc_map[cur_real_lable][0] += 1
                firstAttribute_acc_map[cur_real_lable][1] += 1


        print("off line acc is:", acc_num/total_num)
        print("class_acc is：")
        for e in class_acc_map:
            print(e, lableClassName_map[e], ":", class_acc_map[e][0]/class_acc_map[e][1], firstAttribute_acc_map[e][0]/firstAttribute_acc_map[e][1])

    @staticmethod
    def gainTestResult(pre_fc, Image_feature_fp):
        print("gain test result...")
        result = []
        progress = ProgressBar()
        for i in progress(range(len(pre_fc))):
            cur_pre = pre_fc[i]
            cur_result = Analysis.gainPredictValue(cur_pre, Image_feature_fp)
            result.append(cur_result)
        return result

    @staticmethod
    def gainPredictValue(pre_fc_value, Image_feature_fp):
        image_features = MyFunction.readVector(Image_feature_fp)
        cur_best_same_value = 0.00
        cur_best_label = 'None'
        for line in image_features:
            cur_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            cur_fc_value = cur_fc_value.split(" ")
            cur_fc_value = [float(e) for e in cur_fc_value]
            cur_same_value = Analysis.cosineSimilarity(cur_fc_value, pre_fc_value)
            if cur_same_value > cur_best_same_value:
                cur_best_label = cur_label
                cur_best_same_value = cur_same_value
        return cur_best_label

    @staticmethod
    def gainAccOfVal(fcLayer_fp, ValIMageFeature_fp):
        print("gain acc of val")
        fc_vector = MyFunction.readVector(fcLayer_fp)
        total_sample = len(fc_vector)
        count_postive = 0
        count = 0
        for line in fc_vector:
            count += 1
            if count%1000 == 0:
                print(count)
            cur_real_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            cur_fc_value = cur_fc_value.split(" ")
            cur_fc_value = [float(e) for e in cur_fc_value]
            cur_pre_label = Analysis.gainPredictValue(cur_fc_value, ValIMageFeature_fp)
            #print(cur_pre_label, cur_real_label)
            if cur_real_label == cur_pre_label:
                count_postive += 1
        print(count_postive/total_sample)

    @staticmethod
    def cosineSimilarity(A, B):
        t1 = Analysis.inner(A, B)
        t2 = Analysis.inner(A, A)
        t3 = Analysis.inner(B, B)
        if t2 == 0 or t3 == 0:
            return 2
        #return t1
        return t1 / (pow(t2, 0.5) * pow(t3, 0.5))

    @staticmethod
    def inner(A, B):
        count, i = 0, 0
        n = len(A)
        while i < n:
            count += (A[i] * B[i])
            i += 1
        return count

    @staticmethod
    def analysisFcLayer(fcLayer_fp):
        fc_vector = MyFunction.readVector(fcLayer_fp)
        label_fc_map = {}
        for line in fc_vector:
            cur_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            cur_fc_value = cur_fc_value.split(" ")
            cur_fc_value = [float(e) for e in cur_fc_value]
            if cur_label not in label_fc_map:
                label_fc_map[cur_label] = []
            label_fc_map[cur_label].append(cur_fc_value)

        for center_label in label_fc_map.keys():
            negative_count = 0
            center_fc = label_fc_map[center_label][1]
            distance = 0
            for e in label_fc_map[center_label]:
                distance += Analysis.cosineSimilarity(e, center_fc)
            inner_distance = distance/len(label_fc_map[center_label])

            for cur_label in label_fc_map.keys():
                distance = 0
                for e in label_fc_map[cur_label]:
                    #print(cur_label, Analysis.cosineSimilarity(e, center_fc))
                    distance += Analysis.cosineSimilarity(e, center_fc)
                outter_distance = distance/len(label_fc_map[cur_label])
                if outter_distance > inner_distance:
                    negative_count += 1
            print(center_label, negative_count)


class MyFunction:
    @staticmethod
    def L2distance(lis1, lis2):
        lis1 = np.array(lis1)
        lis2 = np.array(lis2)
        diff = lis1 - lis2
        return sum(diff*diff)/len(diff)

    @staticmethod
    def readFcLayer(pre_fc_fp):
        pre_fc_string = MyFunction.readVector(pre_fc_fp)
        pre_fc_double = []
        for line in pre_fc_string:
            pre_fc_double.append([float(e) for e in line.split()])
        return pre_fc_double

    @staticmethod
    def basedOnTrainImageFeatureToPredictTrainResult(pre_fc, Image_feature_fp):
        pre_fc = MyFunction.readVector(pre_fc)
        acc_count = 0
        all_count = 5000
        progress = ProgressBar()
        for cur_i in progress(range(all_count)):
            line = pre_fc[cur_i]
            cur_lable = line.split(":")[0]
            cur_pre_fc_string = line.split(":")[1].split()
            cur_pre_fc_float = [float(e) for e in cur_pre_fc_string]
            cur_result = Analysis.gainPredictValue(cur_pre_fc_float, Image_feature_fp)
            if cur_lable == cur_result:
                acc_count += 1
        print("train acc is ", acc_count/all_count)

    @staticmethod
    def basedOnTestImageFeatureToPredictTestResult(pre_fc_fp, Image_feature_fp, image_fp, save_final_resut_fp):
        pre_fc = MyFunction.readFcLayer(pre_fc_fp)
        result = Analysis.gainTestResult(pre_fc, Image_feature_fp)
        merge_result = MyFunction.mergeResult(result, image_fp)
        MyFunction.saveVector(save_final_resut_fp, merge_result)
        MyFunction.computeOffLineAcc(pre_result_fp=save_final_resut_fp)

    @staticmethod
    def get_attribute_list():
        head = list(['animal', 'transportation', 'clothes', 'plant', 'tableware', 'device', 'black', 'white', 'blue', 'brown',
            'orange', 'red', 'green', 'yellow', 'has_feathers', 'has_four_legs', 'has_two_legs', 'has_two_arms', 'for_entertainment',
            'for_business', 'for_communication', 'for_family', 'for_office use', 'for_personal', 'gorgeous', 'simple', 'elegant', 'cute',
            'pure', 'naive'])
        return head 

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
    def getAttribute():
        attribute_per_class_fp = config.get("FILE_PATH", 'attributes_per_class')
        label2ClassName_fp = config.get("FILE_PATH", 'lable_list')
        className_map = MyFunction.getLabelClassMap(label2ClassName_fp)
        label_list = MyFunction.get_attribute_list()
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
    def PCAforWordEmbeeding(class_wordembedings_txt_fp, save_fp):
        class_wordembedings = open(class_wordembedings_txt_fp, 'r')
        class_wordembedings = class_wordembedings.readlines()
        all_word_embeeding_vectors = []
        for line_wordembedding in class_wordembedings:
            all_word_embeeding_vectors.append(list(map(float,line_wordembedding.split()[1:])))
        pca = PCA(n_components=100)
        all_word_embeeding_vectors = np.array(all_word_embeeding_vectors)
        X_r = pca.fit(all_word_embeeding_vectors)
        all_word_embeeding_vectors_reduced = X_r.transform(all_word_embeeding_vectors)
        print(np.sum(X_r.explained_variance_ratio_))
        print(all_word_embeeding_vectors_reduced.shape)
        result = []
        for e1, e2 in zip(list(all_word_embeeding_vectors_reduced), class_wordembedings):
            cur_result = e2.split()[0] + " " + " ".join([str(e) for e in list(e1)])
            result.append(cur_result)

        MyFunction.saveVector(save_fp, result)

    @staticmethod
    def computeOffLineAcc(image_label_fp=config.get("FILE_PATH", 'image-labled'), pre_result_fp=config.get("FILE_PATH", 'final_result')):
        total_num = 0
        acc_num = 0
        image_labled = MyFunction.readVector(image_label_fp)
        pre_result = MyFunction.readVector(pre_result_fp)
        for i in range(len(image_labled)):
            cur_image_labled_line = image_labled[i]
            cur_pre_result_line = pre_result[i]
            if len(cur_image_labled_line.split('\t')) == 2:
                cur_real_lable = cur_image_labled_line.split('\t')[1]
                cur_pre_result = cur_pre_result_line.split('\t')[1]
                total_num += 1
                if cur_real_lable == cur_pre_result:
                    acc_num += 1
        print("off line acc is:", acc_num/total_num)

    @staticmethod
    def supplyResult(image_fp=config.get("FILE_PATH",'image-labled')):
        image = MyFunction.readVector(image_fp)
        first_value = -1
        classlable_linenum_map = {}
        new_image = []
        for i in range(len(image)):
            cur_line = image[i]
            if len(cur_line.split('\t')) == 2:
                if first_value == -1:
                    cur_label = cur_line.split('\t')[1]
                    first_value = 1
                else:
                    new_image.append(image[i].split('\t')[0] + '\t' + cur_label)
                    first_value = -1
                    continue

            if first_value == 1:
                new_image.append(image[i].split('\t')[0] + '\t' + cur_label)
            else:
                new_image.append(image[i].split('\t')[0])
        MyFunction.saveVector(image_fp, new_image)

    @staticmethod
    def mergeResult(lables, image_fp):
        image_name = MyFunction.readVector(image_fp)
        imageName_lable = []
        for i in range(len(image_name)):
            cur_image_name = image_name[i]
            cur_label = lables[i]
            imageName_lable.append(cur_image_name + "\t" + cur_label)
        return imageName_lable

    @staticmethod
    def computeAcc(Y, pre_y):
        Y = Y.argmax(axis=1)
        pre_y = pre_y.argmax(axis=1)
        count = 0
        for i in range(len(pre_y)):
            if pre_y[i] == Y[i]:
                count += 1
        return count/len(pre_y)

    @staticmethod
    def saveImageFeatureByBestStratigy(fcLayer_fp, save_fp_ImageFeature, save_fp_acc, n_patience=30):
        fc_vector = MyFunction.readVector(fcLayer_fp)
        lable_allFcOfSameLable_map = {}
        best_acc = []
        lable_bestImageFeature_list = []
        for line in fc_vector:
            cur_lable = line.split(':')[0]
            cur_FC = [float(e) for e in line.split(':')[1].split()]
            if cur_lable not in lable_allFcOfSameLable_map:
                lable_allFcOfSameLable_map[cur_lable] = [cur_FC]
            else:
                lable_allFcOfSameLable_map[cur_lable].append(cur_FC)

        lable_bestImageFeature_map = {}
        for cur_lable in lable_allFcOfSameLable_map:
            lable_bestImageFeature_map[cur_lable] = lable_allFcOfSameLable_map[cur_lable][0]
            #print(cur_lable, lable_bestImageFeature_map[cur_lable])
        flag_first = True
        for cur_lable in lable_allFcOfSameLable_map:
            if flag_first:
                flag_first = False
                first_lable = cur_lable
            cur_best_acc = -2
            all_FCOfcurLable_num = len(lable_allFcOfSameLable_map[cur_lable])
            for i in range(n_patience):
                select_index = random.randint(0, all_FCOfcurLable_num-1)
                lable_bestImageFeature_map_tem = copy.deepcopy(lable_bestImageFeature_map)
                lable_bestImageFeature_map_tem[cur_lable] = lable_allFcOfSameLable_map[cur_lable][select_index]
                cur_accOfCurLable = MyFunction.computerAccOfSpecificCurLable(lable_bestImageFeature_map_tem, lable_allFcOfSameLable_map[cur_lable], cur_lable)
                if cur_accOfCurLable > cur_best_acc:
                    #lable_bestImageFeature_map = lable_bestImageFeature_map_tem
                    lable_bestImageFeature_map[cur_lable] = copy.deepcopy(lable_allFcOfSameLable_map[cur_lable][select_index])
                    cur_best_acc = cur_accOfCurLable
            if cur_best_acc == -2:
                print("I am -2", all_FCOfcurLable_num)
            first_acc_change = MyFunction.computerAccOfSpecificCurLable(lable_bestImageFeature_map, lable_allFcOfSameLable_map[first_lable], first_lable)
            print("first acc", first_acc_change)
            print(cur_lable, cur_best_acc)
            best_acc.append(cur_lable + ":" + str(cur_best_acc))

        for cur_lable in lable_bestImageFeature_map:
            print(lable_bestImageFeature_map[cur_lable])
            print(type(lable_bestImageFeature_map[cur_lable]))
            lable_bestImageFeature_list.append(cur_lable + ':' + " ".join([str(e) for e in lable_bestImageFeature_map[cur_lable]]))
        MyFunction.saveVector(save_fp_ImageFeature, lable_bestImageFeature_list)
        MyFunction.saveVector(save_fp_acc, best_acc)

    @staticmethod
    def computerAccOfSpecificCurLable(lable_bestImageFeature_map, fc_curLable, lable_real):
        count_acc = 0
        count_all = 0
        for cur_fc in fc_curLable:
            cur_best_distance = 99999
            cur_lableOfbestResult = 'None'
            for cur_lable in lable_bestImageFeature_map:
                cur_distance = MyFunction.L2distance(lable_bestImageFeature_map[cur_lable], cur_fc)
                if cur_distance < cur_best_distance:
                    cur_lableOfbestResult = cur_lable
                    cur_best_distance = cur_distance
            if cur_lableOfbestResult == lable_real:
                count_acc += 1
            count_all += 1
        return count_acc/count_all

    @staticmethod
    def saveImageFeature(fcLayer_fp, save_fp):
        fc_vector = MyFunction.readVector(fcLayer_fp)
        label_fc_map = {}
        result_ImageFeature = []
        numImageFeatureToSave_perClass = 10
        for line in fc_vector:
            cur_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            if cur_label not in label_fc_map:
                label_fc_map[cur_label] = 1
                result_ImageFeature.append(str(cur_label) + ":" + cur_fc_value)
            elif label_fc_map[cur_label] < numImageFeatureToSave_perClass:
                label_fc_map[cur_label] += 1
                result_ImageFeature.append(str(cur_label) + ":" + cur_fc_value)
        MyFunction.saveVector(save_fp, result_ImageFeature)
    
    @staticmethod
    def saveImageFeature_avg(fcLayer_fp, save_fp, avg_num):
        fc_vector = MyFunction.readVector(fcLayer_fp)
        label_fc_map = {}
        avg_one_Image_feature = []
        avg_num_ImageFeature = []
        numImageFeatureToSave_perClass = avg_num
        for line in fc_vector:
            cur_label = line.split(":")[0]
            cur_fc_value = line.strip().split(":")[1]
            if cur_label not in label_fc_map:
                label_fc_map[cur_label] = 1
                #label_fc_map[cur_label].append(cur_fc_value)
                avg_num_ImageFeature.append(str(cur_label) + ":" + cur_fc_value)
            elif label_fc_map[cur_label] < numImageFeatureToSave_perClass:
                label_fc_map[cur_label] += 1
                avg_num_ImageFeature.append(str(cur_label) + ":" + cur_fc_value)
        image_feature_dimention = len(avg_num_ImageFeature[0].split(':')[1].split())
        new_imageFeature_float = np.array([0.0] * image_feature_dimention)
        count = 0
        for cur_imageFeature in avg_num_ImageFeature:
            count += 1
            new_label = cur_imageFeature.split(':')[0]
            new_imageFeature_float += np.array([float(e) for e in cur_imageFeature.split(':')[1].split()])
            if count % avg_num == 0:
                new_imageFeature_float /= avg_num
                new_imageFeature_string = new_label + ":" + " ".join([str(e) for e in new_imageFeature_float])
                avg_one_Image_feature.append(new_imageFeature_string)
        print("total image feature", len(avg_one_Image_feature))
        MyFunction.saveVector(save_fp, avg_one_Image_feature)

    @staticmethod
    def saveTestFcLayer(pre_fc, save_fp):
        string_fc = []
        for cur_fc in pre_fc:
            string_fc.append(" ".join([str(e) for e in cur_fc]))
        MyFunction.saveVector(save_fp, string_fc)

    @staticmethod
    def saveValFcLayer(y_test, pre_fc, save_fp, lable_list_fp):
        f_train = open(lable_list_fp)
        temp = f_train.readlines()
        train_dict = dict(zip(range(230), [x.strip('\n\r') for x in temp]))
        real_result = y_test.argmax(axis=1)
        print(real_result)
        fc_result = []
        print(len(real_result))
        for i in range(len(real_result)):
            cur_real = real_result[i]
            cur_label_ZHL = train_dict[cur_real]
            cur_fc_value = [str(e) for e in pre_fc[i]]
            cur_fc_result = str(cur_label_ZHL) + ":" + ' '.join(cur_fc_value)
            fc_result.append(cur_fc_result)
        MyFunction.saveVector(save_fp, fc_result)

    @staticmethod
    def saveFcLayer(pre_y, y_test, pre_fc, save_fp, lable_list_fp):
        f_train = open(lable_list_fp)
        temp = f_train.readlines()
        train_dict = dict(zip(range(230), [x.strip('\n\r') for x in temp]))
        predict_result = pre_y.argmax(axis=1) 
        real_result = y_test.argmax(axis=1)
        fc_result = []
        all_result = []
        acc_result = []
        for i in range(len(predict_result)):
            cur_pre = predict_result[i]
            cur_real = real_result[i]
            if cur_real not in all_result:
                all_result.append(cur_real)
            if cur_pre == cur_real:
                if cur_real not in acc_result:
                    acc_result.append(cur_real)
                cur_label_ZHL = train_dict[cur_pre]
                cur_fc_value = [str(e) for e in pre_fc[i]]
                cur_fc_result = str(cur_label_ZHL) + ":" + ' '.join(cur_fc_value)
                fc_result.append(cur_fc_result)
        MyFunction.saveVector(save_fp, fc_result)
        print("%f image was save", len(fc_result)/len(predict_result))

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
    def saveVector(save_fp,vector):
        file = open(save_fp, 'w')
        for value in vector:
            file.write(str(value) + "\n")
        file.close()

    @staticmethod
    def analysisClassForIVR2(predict_result, real_result, train_dict):
        num_classes = len(train_dict.keys())
        num_samples = len(list(predict_result))
        class_acc_map = {}
        class_num_map = {}
        for i in range(num_classes):
            class_acc_map[i] = 0
        for e in real_result:
            if e not in class_num_map:
                class_num_map[e] = 1
            else:
                class_num_map[e] += 1
        count = 0
        for cur_pre, cur_real in zip(predict_result, real_result):
            if cur_pre == cur_real:
                class_acc_map[cur_pre] += 1/class_num_map[cur_pre]
                count += 1
        print(count/num_samples)
        acc_recompute = 0
        for e in class_acc_map:
            acc_recompute += class_num_map[e] * class_acc_map[e]
        print("recompute acc is ", acc_recompute/num_samples)

        print("acc < 0.2")
        for e in class_acc_map:
            if class_acc_map[e] < 0.2:
                print(train_dict[e])


    @staticmethod
    def analysisClass(pre_y, y_test, num_classes, train_lable_fp='data/train190_lable.txt'):
        f_train = open(train_lable_fp)
        temp = f_train.readlines()
        train_dict = dict(zip(range(230), [x.strip('\n\r') for x in temp]))

        pre_y = np.array(pre_y)
        predict_result = pre_y.argmax(axis=1) 
        real_result = y_test.argmax(axis=1)
        class_acc_map = {}
        class_num_map = {}
        for i in range(num_classes):
            class_acc_map[i] = 0
        for e in real_result:
            if e not in class_num_map:
                class_num_map[e] = 1
            else:
                class_num_map[e] += 1
        count = 0
        for cur_pre, cur_real in zip(predict_result, real_result):
            if cur_pre == cur_real:
                class_acc_map[cur_pre] += 1/class_num_map[cur_pre]
                count += 1
        print(count/pre_y.shape[0])
        acc_recompute = 0
        for e in class_acc_map:
            acc_recompute += class_num_map[e] * class_acc_map[e]
        print("recompute acc is ", acc_recompute/len(pre_y))
        
        print("acc > 0.3")
        for e in class_acc_map:
            if class_acc_map[e] >= 0.1:
                print(train_dict[e])

        print("acc > 0.5")
        for e in class_acc_map:
            if class_acc_map[e] >= 0.20:
                print(train_dict[e])

        print("acc > 0.7")
        for e in class_acc_map:
            if class_acc_map[e] >= 0.05:
                print(train_dict[e])


class Extractor:
    @staticmethod
    def readTrainDataVersion2(train_txt_fp, train_pt, class_wordembedings_txt_fp, label_list_fp):
        data_train = open(train_txt_fp, 'r')
        data_train = data_train.readlines()# !!!!!!!!!!!!!!!!!!!!!！！！！！！！！！！！！！！！！！！这里要注意修改

        class_wordembedings = open(class_wordembedings_txt_fp,'r')
        class_wordembedings = class_wordembedings.readlines()
        word_to_vector = {}
        for line_wordembedding in class_wordembedings:
            word_to_vector[line_wordembedding.split()[0]] = list(map(float,line_wordembedding.split()[1:]))

        labelString2LabelWord = open(label_list_fp)
        labelString2LabelWord_map = {}
        for line_labelWord in labelString2LabelWord:
            labelString2LabelWord_map[line_labelWord.split()[0]] = line_labelWord.split()[1]

        length = len(data_train)
        train_x = np.zeros((length, image_size, image_size, 3))
        train_y = np.zeros((length, len(class_wordembedings[0].split())-1))

        for i in range(length):
            image_name = data_train[i].split()[0]
            labelString = data_train[i].split()[1]
            if Flag_resize:
                img = image.load_img(train_pt + image_name, target_size=[128, 128])
            else:
                img = image.load_img(train_pt + image_name)
            train_x[i] = image.img_to_array(img)
            cur_lableWord = labelString2LabelWord_map[labelString]
            train_y[i] = word_to_vector[cur_lableWord]
        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1,random_state=91)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def gainTrainAndTest(train_label_fp, train_imageName_Lable_fp, image_path, hight=image_size, width=image_size):
        print("read data...")
        f_train = open(train_label_fp)
        train_imageName_Lable = open(train_imageName_Lable_fp)
        temp = f_train.readlines()
        train_dict = dict(zip([x.strip('\n\r') for x in temp],range(230)))
        train_cate_num = len(train_dict.keys())
        train_imageName_Lable = train_imageName_Lable.readlines()
        length = len(train_imageName_Lable)
        data_x = np.zeros((length, hight, width, 3))
        data_y = np.zeros((length))
        num = 0
        result_all_inx = []
        for i in range(length):
            m,n = train_imageName_Lable[i].split()
            if Flag_resize:
                img = image.load_img(image_path + m, target_size=[128, 128])
            else:
                img = image.load_img(image_path + m)
            if n in train_dict.keys():
                data_x[num] = image.img_to_array(img)
                idx = train_dict[n]
                data_y[num] = int(idx)
                if int(idx) not in result_all_inx:
                    result_all_inx.append(int(idx))
                num = num + 1
        data_y = to_categorical(data_y, train_cate_num)

        X_train, X_test, y_train, y_test = train_test_split(data_x[:num], data_y[:num], test_size=0.1 ,random_state=91)
        return X_train, y_train, X_test, y_test, train_cate_num

    @staticmethod
    def gainVal(label_fp, train_imageName_Lable_fp, image_path, hight=image_size, width=image_size):
        f_train = open(label_fp)
        train_imageName_Lable = open(train_imageName_Lable_fp)
        temp = f_train.readlines()
        test_dict = dict(zip([x.strip('\n\r') for x in temp],range(230)))
        train_cate_num = len(test_dict.keys())
        print('total categories:'+str(train_cate_num))
        # get train samples
        train_imageName_Lable = train_imageName_Lable.readlines()
        length = len(train_imageName_Lable)
        # initial 
        data_x = np.zeros((length, hight, width, 3))
        data_y = np.zeros((length))
        num = 0
        # set values
        result_all_inx = []
        for i in range(length):
            m,n = train_imageName_Lable[i].split()
            if Flag_resize:
                img = image.load_img(image_path + m, target_size=[128, 128])
            else:
                img = image.load_img(image_path + m)
            if n in test_dict.keys():
                data_x[num] = image.img_to_array(img)
                idx = test_dict[n]
                data_y[num] = int(idx)
                if int(idx) not in result_all_inx:
                    result_all_inx.append(int(idx))
                num = num + 1
        data_y = to_categorical(data_y[:num], train_cate_num)
        return data_x[:num], data_y

    @staticmethod
    def gainRealTest(test_imageName_Lable_fp, image_path, hight=image_size, width=image_size):
        test_imageName_Lable = open(test_imageName_Lable_fp)
        test_imageName_Lable = test_imageName_Lable.readlines()
        length = len(test_imageName_Lable)
        data_x = np.zeros((length, hight, width, 3))
        for i in range(length):
            m = test_imageName_Lable[i].strip()
            if Flag_resize:
                img = image.load_img(image_path + m, target_size=[128, 128])
            else:
                img = image.load_img(image_path + m)
            data_x[i] = image.img_to_array(img)
        print(data_x.shape)
        return data_x


class Specific:

    @staticmethod
    def makeUpTestImageFeature(trainImageFeature_fp, testImageFeature_fp, makeUPLableList):
        trainImageFeature = MyFunction.readVector(trainImageFeature_fp)
        testImageFeature = MyFunction.readVector(testImageFeature_fp)
        trainLable_ImageFeature_map = {}
        for line in trainImageFeature:
            cur_lable = line.split(':')[0]
            cur_ImageFeature = line.split(':')[1]
            trainLable_ImageFeature_map[cur_lable] = cur_ImageFeature
        for cur_lable in makeUPLableList:
            new_ImageFeature = cur_lable + ':' + trainLable_ImageFeature_map[cur_lable]
            testImageFeature.append(new_ImageFeature)
        MyFunction.saveVector(testImageFeature_fp, testImageFeature)


