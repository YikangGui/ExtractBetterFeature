from wheel import ComputeTestAcc, Specific, MyFunction, PrepareData, Analysis, Generator, ConstractA, \
    PlantPostProcess, PostProcess, TestC_ImageFeature, AnalysisAccOfEveryClass,\
    KNN, FeatAnalysis, Generator_gcn
import numpy as np
import os
from PIL import Image
Flags = [430]
def run(Flag):
    Model = 'PredictResult_gcn'
    if True:

        unlabel = MyFunction.readVector('../../data/list_tc/imageName_lable/test_imageName_unlable_D.txt')
        file = []
        for line in unlabel:
            file.append(line + '\t' + 'ZJL349')
        MyFunction.saveVector('../../data/list_tc/imageName_lable/test_D_labeled.txt', file)

    if Flag == 49:
        feat_fp = '../../data/searchLabel/ImageFeature_A2/'
        save_feat_fp = 'feat.npy'
        save_label_fp = 'label.txt'
        Specific.changeImageFeatureToFeatFile(feat_fp, save_label_fp, save_feat_fp)

    if Flag == 48:
        test_feat_fp = '../../data/list_tc/feat/CNN16/test_D_feat.npy'
        test_image_labled_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_D.txt'
        save_fp = './realImageFeature_0.txt'
        Specific.gainImageFeatureDirectlyFromImageFeat(test_feat_fp, test_image_labled_fp, save_fp)

    if Flag == 47:
        acc_fp1 = './acc43.txt'
        acc_fp2 = './acc47.txt'
        Specific.compareTwoResultAcc(acc_fp1, acc_fp2)

    if Flag == 46:
        related_fp = './1.txt'
        Specific.anaLysisRelatedClass(related_fp)

    if Flag == 45:
        related_fp1 = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byWordEmbedding_D_1.txt'
        related_fp2 = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byImageFeat_D_3.txt'
        label2ClassName_fp = '../../data/list_tc/common_file/label_className.txt'
        test_label_fp = '../../data/list_tc/label/test_lable_1_D.txt'
        save_fp = './1.txt'
        Specific.mergeRelatedClass(related_fp1, related_fp2, label2ClassName_fp, test_label_fp, save_fp)

    if Flag == 44:
        fp1 = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byImageFeat_D_2.txt'
        fp2 = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byWordEmbedding_D_1.txt'
        test_label_fp = '../../data/list_tc/label/test_lable_1_D.txt'
        label2ClassName = '../../data/list_tc/common_file/label_className.txt'
        Specific.findDifferenceBetweenRelatedClass(fp1, fp2, test_label_fp, label2ClassName)

    if Flag == 43:
        # feat_fp = '../../data/list_tc/feat/CNN16/train_D_feat.npy'
        # real_result_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_D.txt'
        feat_fp = '../../data/searchLabel/ImageFeature_A2/feat.npy'
        real_result_fp = '../../data/searchLabel/ImageFeature_A2/label.txt'
        FeatAnalysis.returnStatisticInformation(feat_fp, real_result_fp)

    if Flag == 42:
        # feat_fp = '../../data/list_tc/feat/CNN12/train_C_feat.npy'
        # real_result_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_C.txt'
        feat_fp = '../../data/searchLabel/ImageFeature_A2/feat.npy'
        real_result_fp = '../../data/searchLabel/ImageFeature_A2/label.txt'
        FeatAnalysis.computeInterAndExterDistanceOFImageFeat(feat_fp, real_result_fp)

    if Flag == 41:
        feat_fp = '../../data/list_tc/feat/CNN1/train_C_feat.npy'
        real_result_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_C.txt'
        Specific.testHighDimentionVectorHasAddAttribute(feat_fp, real_result_fp)

    if Flag == 40:
        predict_resultCNN_fp = './test_C_predict.txt'
        real_result_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_C.txt'
        print("sss")
        Analysis.analysisTestCResultCNN(predict_resultCNN_fp, real_result_fp)

    if Flag == 39:
        train_test_Label_topK_relateTrainLabel_fp = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byImageFeat_CNN1_2.txt'
        train_label_fp = '../../data/list_tc/label/train_lable_1_ABC.txt'
        label2ClassName_fp = '../../data/list_tc/common_file/label_className.txt'
        lassThan20Label_list = ['../../data/list_tc/label/accLeccThan20Label_unfilter.txt']
        related_classLabel = Specific.gainAllRelatedLabelWithTest(train_test_Label_topK_relateTrainLabel_fp, train_label_fp, label2ClassName_fp)
        '''
        acc_lessThan20Label = Specific.mergeLable(lassThan20Label_list)
        for label in acc_lessThan20Label:
            if label not in related_classLabel:
                print(label)
        '''

    if Flag == 38:
        attribute_pd = '../../data/list_tc/common_file/attributes_per_class_C.txt'
        real_result_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_C.txt'
        predict_result_fp = '../../data/list_tc/final_Result/ZSL_A8/final_result_C.txt'
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        Analysis.analysisAccOfFirstAttribute(attribute_pd, real_result_fp, predict_result_fp, label2ClassName_fp)

    if Flag == 37:
        real_result_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_C.txt'
        final_result1_fp = '../../data/list_tc/final_Result/ZSL_A2/final_result_C.txt'
        final_result2_fp = '../../data/list_tc/final_Result/ZSL_A8/final_result_C.txt'
        Analysis.analysisDifferenceBetweenFinalResult(real_result_fp, final_result1_fp, final_result2_fp)

    if Flag == 36:
        acc_fp1 = '../../data/list_tc/final_Result/ZSL_A9/accOfEveryClass.txt'
        acc_fp2 = '../../data/list_tc/final_Result/ZSL_A8/accOfEveryClass.txt'
        acc_fp_list = [acc_fp1, acc_fp2]
        AnalysisAccOfEveryClass.plotAccOfEveryClass(acc_fp_list)

    if Flag == 35:
        fp_test1 = '../../data/KNN/TestImageFeaturePredicted/D2/realImageFeature_'
        fp1 = '../../data/KNN/TestImageFeaturePredicted/D1/gcn_A2_19000_imageFeature_predict.txt'
        fp2 = '../../data/KNN/TestImageFeaturePredicted/D1/gcn_A2_16000_imageFeature_predict.txt'
        Image_feature_fp_list = []
        for i in range(1, 3):
            Image_feature_fp_list.append(fp_test1+str(i)+'.txt')

        #Image_feature_fp_list.append(fp2)
        #Image_feature_fp_list.append(fp1)
        print(Image_feature_fp_list)

        fc_vector_test_fp = '../../data/list_tc/feat/CNN16/test_D_feat.npy'
        test_image_labled_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_D.txt'
        test_image_unlable_fp = '../../data/list_tc/imageName_lable/test_imageName_unlable_D.txt'
        final_result_fp = './final_result_C.txt'
        test_feat_fp = '../../data/list_tc/feat/CNN16/test_D_feat.npy'
        result = KNN(Image_feature_fp_list, fc_vector_test_fp,\
                     test_image_labled_fp, test_feat_fp).predictByKNN(n_neighbors=1)
        # merge_result = ComputeTestAcc.mergeResult(result, test_image_unlable_fp)
        # MyFunction.saveVector(final_result_fp, merge_result)

    if Flag == 34:
        attribute_fp = '../../data/list_tc/wordEmbedding/attributes_per_class.txt'
        wordEmbedding_fp = '../../data/list_tc/wordEmbedding/className_wordEmbedding_ATT1.txt'
        Specific.changAttributeIntoWordEmbeddingFormat(attribute_fp, wordEmbedding_fp)

    if Flag == 33:
        accOfEvervClass_fp = '../../data/list_tc/final_Result/ZSL_A2/accOfEveryClass.txt'
        save_fp = '../../data/list_tc/final_Result/ZSL_A2/dividedLabel_ByAcc.txt'
        AnalysisAccOfEveryClass.devideClassByAccLevel(accOfEvervClass_fp, save_fp)

    if Flag == 32:
        wordEmbedding_fp = '../../data/list_tc/wordEmbedding/className_wordEmbedding_SH.txt'
        label2ClassName_fp = '../../data/list_tc/common_file/label_className.txt'
        test_label_fp = '../../data/list_tc/label/test_lable_1_D.txt'
        Analysis.gainTopkRelateClassLabelForEveryTestClassByWordEmbedding(wordEmbedding_fp,  label2ClassName_fp, test_label_fp)

    if Flag == 31:
        save_pt = '../../data/list_tc/A_matriex/A_matriex_D_666.txt'
        related_class_fp = '../../Analysis/txt/train_test_Label_topK_relateTrainLabel_byImageFeat_D_10.txt'
        # related_class_fp = './best_related.txt'
        result = ConstractA.constractAbyRelatedClass(related_class_fp, k=2, isSymmetrical=False)
        MyFunction.saveVector(save_pt, result)

    if Flag == 30:
        train_feat_fp = '../../data/list_tc/feat/CNN21/train_D_feat.npy'
        test_feat_fp = '../../data/list_tc/feat/CNN21/test_D_feat.npy'
        train_image_labeled_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_D.txt'
        test_image_labeled_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_D.txt'
        label2ClassName_fp = '../../data/list_tc/common_file/label_className.txt'
        '''
        train_feat_fp = ['../../data/list_tc/feat/CNN16/train_A_feat.npy', \
                         '../../data/list_tc/feat/CNN16/train_B_feat.npy',\
                         '../../data/list_tc/feat/CNN16/train_C_feat.npy',\
                         '../../data/list_tc/feat/CNN16/train_D_feat.npy']
        train_image_labeled_fp = ['../../data/list_tc/imageName_lable/train_imageName_Lable_A.txt',\
                                  '../../data/list_tc/imageName_lable/train_imageName_Lable_B.txt',\
                                  '../../data/list_tc/imageName_lable/train_imageName_Lable_C.txt',\
                                  '../../data/list_tc/imageName_lable/train_imageName_Lable_D.txt']
        '''
        Analysis.gainTopkRelateClassLabelForEveryTestClassByImageFeat(
            train_feat_fp, test_feat_fp, train_image_labeled_fp,
            test_image_labeled_fp, label2ClassName_fp, k_compare=10)

    if Flag == 29:
        train_feat_A_fp = '../../data/list_tc/feat/CNN1/train_A_feat.npy'
        train_feat_B_fp = '../../data/list_tc/feat/CNN1/train_B_feat.npy'
        train_feat_C_fp = '../../data/list_tc/feat/CNN1/train_C_feat.npy'
        merge_fp = '../../data/list_tc/feat/CNN1/train_ABC_feat.npy'
        Specific.mergeFeatNPY(train_feat_A_fp, train_feat_B_fp, train_feat_C_fp, merge_fp)

    if Flag == 26:
        # 为nn创建mask 
        label_fp= '../../data/list_tc/label/train_lable_2_C.txt'
        imageName_lable_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_ABC.txt'
        mask_fp = '../../data/list_tc/image_mask/mask_ABC_2.txt'
        Specific.gainImageMaskBasedOnLabel(imageName_lable_fp, label_fp, mask_fp)

    if Flag == 25:
        # predict_result_fp = '../../data/list_tc/gcn_model_key/A2/result/result_18000_0.472333000997009.txt'
        real_label_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_D.txt'
        predict_result_fp = '../../data/list_tc/final_Result/ZSL_A24/result_8000_0.46809571286141577.txt'
        right_flag_fp = None
        #predict_result_fp = '../../data/list_tc/CNN_predict/CNN1/train_C_predict.txt'
        #real_label_fp = '../../data/list_tc/imageName_lable/train_imageName_Lable_C.txt'
        #right_flag_fp = '../../data/list_tc/CNN_predict/CNN1/train_C_flag.txt'
        Analysis.gainCnnModelAcc(predict_result_fp, real_label_fp, right_flag_fp)

    if Flag == 24:
        wordEmbedding_fp = '../../data/list_tc/wordEmbedding/className_wordEmbedding_500_C.txt'
        Analysis.K_SimiliarClass_WordEmbeddingDistance(wordEmbedding_fp, 3)

    if Flag == 23:
        pre_fp = '../../data/list_tc/'
        lable_className_fp = pre_fp + 'common_file/label_className.txt'
        train_imageName_lable_fp = pre_fp + 'imageName_lable/train_imageName_Lable_C.txt'
        train_lable_1_fp = 'label/train_lable_1_C.txt'
        test_lable_1_fp = '/test_lable_1_C.txt'
        Specific.gainTrainLableAndTestLable(lable_className_fp, train_imageName_lable_fp, train_lable_1_fp, test_lable_1_fp)

    if Flag == 22:
        lable_1_fp = '../../data/list_tc/label/train_lable_1_A.txt'
        lable_2_fp = '../../data/list_tc/label/train_lable_1_B.txt'
        lable_3_fp = '../../data/list_tc/label/train_lable_1_C.txt'
        merge_fp = './train_lable_1_ABC.txt'
        Specific.mergeLable(lable_1_fp, lable_2_fp,lable_3_fp, merge_fp)

    if Flag == 21:
        train_J_feat_npy_fp = '../data/classficaData/J1_IVR2_1/train_J_feat.npy'
        Analysis.analysisDatasetJDistance(train_J_feat_npy_fp)

    if Flag == 20:
        feat_path = '../../data/list_tc/feat/CNN16/train_D_feat.npy'
        final_result = '../../data/list_tc/imageName_lable/train_imageName_Lable_D.txt'
        #final_result = '../../data/list_tc/final_Result/ZSL_A3/final_result.txt'
        save_image_fp = './test_D.jpg'
        predict_imageFeature_fp = '../../data/list_tc/gcn_model_key/A2/output/gcn_A2_14000_imageFeature_predict.txt'
        TestC_ImageFeature.plotTestC(feat_path, predict_imageFeature_fp, final_result, save_image_fp)

    if Flag == 19:
        path = "./final_result_C.txt"
        PostProcess.setWrongAnswerRandomly(path)

    if Flag == 18:
        # 从GCN预测的结果中，提取出test的image feature。 提供A（因为A中提供了顺序） 以及 imageFeature_predict train + test
        A_matriex_C_fp = config.get(Model, 'A_matrix_fp')
        train_test_imageFeature_predict_fp = config.get(Model, 'train_test_imageFeature_predict_fp')
        TestImageFeaturePredicted_fp = config.get(Model, 'TestImageFeaturePredicted_fp')
        test_label_fp = config.get(Model, 'test_label_fp')
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        Specific.extractTestImageFeatureFromTrainAndTestPredictImageFeature(A_matriex_C_fp, train_test_imageFeature_predict_fp, label2ClassName_fp, test_label_fp, TestImageFeaturePredicted_fp)

    if Flag == 17:
        # 从npy的train feat中为每个类提取出一个imagefeature 并保存成txt文件。格式为lable:imageFeature
        #     需要提供train_imageName_Lable_C。txt 为了得到每个类的label
        feat_fp = '../../data/list_tc/feat/CNN1/test_C_feat.npy'
        imageName_Lable_fp = '../../data/list_tc/imageName_lable/test_imageName_Lable_C.txt'
        lable_feat_save_fp = '../../data/KNN/TestImageFeaturePredicted/T7/TestImageFeaturePredicted_C.txt'
        Specific.extractTrainImageFeatureFromNpy(feat_fp, imageName_Lable_fp,
                                                 lable_feat_save_fp, 100)

    if Flag == 16:
        npy_fp = list_tc_pre + 'nn_model_key/A5/train_B_feat.npy'
        txt_fp = list_tc_pre + 'nn_model_key/A5/fc_vector_test_A.txt'
        Specific.changeNPYtoTXTForTestSet(npy_fp, txt_fp)

    if Flag == 15:
        final_result_1version_fp = list_tc_pre + 'final_Result/3/final_result.txt'
        final_result_plant_version_fp = list_tc_pre + 'final_Result/4/final_result.txt'
        Plant_list = ['ZJL253']
        final_result_merge_fp = list_tc_pre + 'final_Result/4/final_result_merge.txt'
        Specific.mergeFinalResultBetweenFirstVersionAndPlantVersion(final_result_1version_fp, final_result_plant_version_fp, Plant_list, final_result_merge_fp)

    if Flag == 14:
        lable_fp = list_tc_pre + 'train_lable_A.txt'
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        lable_attribute_fp = config.get('FileName', 'lable_attribute_fp')
        Analysis.getClassNameAndAttributeByLabel(lable_fp, label2ClassName_fp, lable_attribute_fp)

    if Flag == 13:
        allTestImageFeature_fp = config.get(Model, 'TestImageFeaturePredicted_fp')
        newTestImageFeature_fp = config.get(Model, 'newTestImageFeaturePredicted_fp')
        newTestLable_fp = config.get(Model, 'newTestLable40_fp')
        MyFunction.filterTestImageFeature(allTestImageFeature_fp, newTestImageFeature_fp, newTestLable_fp)

    if Flag == 12:
        list_tc_pre = '../../data/'
        word_embeding_fp = config.get('ConstractA', 'word_embeding_fp')
        all_label_fp = config.get('ConstractA', 'all_label_fp')
        min_distance = 0.3
        attribute_per_class_fp = config.get('ConstractA', 'attribute_per_class_fp')
        adj = ConstractA.constractA(word_embeding_fp, all_label_fp, min_distance, attribute_per_class_fp)
        for cur_key in adj:
            print(cur_key, adj[cur_key])
        A = ConstractA.transfromAintoMatrix(adj)
        MyFunction.saveVector(list_tc_pre+'A_matriex.txt', A)

    if Flag == 11:
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        final_result_fp = config.get('Analysis', 'final_result_fp')
        test_pt = config.get('Analysis', 'test_pt')
        split_fp = config.get('Analysis', 'split_fp')
        if not os.path.exists(split_fp):
            os.mkdir(split_fp)
        className_Num_fp = config.get('Analysis', 'className_Num_fp')
        process_num = 2000
        Analysis.split_finalResult(label2ClassName_fp, final_result_fp, test_pt, split_fp, process_num, className_Num_fp)

    if Flag == 9:
        '''
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        class_wordembedding_fp = '../../data/list_tc/wordEmbedding/className_wordEmbedding_500.txt'
        save_fp = '../../data/list_tc/wordEmbedding/label_wordembedding_500.txt'
        Specific.gainLabelWordEmbeddingFromClassnameWordEmbedding(class_wordembedding_fp, label2ClassName_fp, save_fp)
        '''
        label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        label_wordembedding_fp = '../../data/list_tc/wordEmbedding/label_wordEmbedding_ATT_D.txt'
        save_fp = '../../data/list_tc/wordEmbedding/className_wordEmbedding_ATT_D.txt'
        Specific.gainClassnameWordEmbeddingFromLabelWordEmbedding(label_wordembedding_fp, label2ClassName_fp, save_fp)
        

    if Flag == 8:
        #image_feature_trian_fp = list_tc_pre + 'alltrain_image_feature_selectedbest.txt'
        image_feature_trian_fp = list_tc_pre + 'alltrain_image_feature_10.txt'
        image_feature_test_fp = list_tc_pre + 'test_image_feature.txt'
        lable2Classname_fp = list_tc_pre + 'label2ClassName.txt'
        Analysis.analysisTestImageFeatureandTrainImageFeatuerRelationship(image_feature_trian_fp, image_feature_test_fp, lable2Classname_fp)

    if Flag == 7:
        ZJL_calssName_fp = list_tc_pre + 'label2ClassName.txt'
        alltrain_image_feature_fp = list_tc_pre + 'alltrain_image_feature_selectedbest.txt'
        trainAndTestClassName_fp = list_tc_pre + 'trainAndTestClassName.txt'
        saveImageFeature_fp = list_tc_pre + 'trainImageFeature.txt'
        MyFunction.constractImageFeature(ZJL_calssName_fp, alltrain_image_feature_fp, trainAndTestClassName_fp, saveImageFeature_fp)

    if Flag == 6:
        NumClass = 205
        A_fp = '../../data/list_tc/A_matriex/A_matriex_C_3.txt'
        label_className_fp = '../../data/list_tc/common_file/label_className_C.txt'
        #Specific.constractAllOneAMatrix(NumClass=NumClass, A_fp=A_fp, label_className_fp=label_className_fp)
        Specific.constractIdentityAMatrix(NumClass=NumClass, A_fp=A_fp, label_className_fp=label_className_fp)

    if Flag == 5:
        className_fp = list_tc_pre + 'trainAndTestClassName.txt'
        class_wordembeddings_fp = list_tc_pre + 'class_wordembedding_500.txt'
        save_extractWordembedding_fp = list_tc_pre + 'extractWordembedding.txt'
        PrepareData.constractWordEmbedding(className_fp=className_fp, class_wordembeddings_fp=class_wordembeddings_fp, extractWordembedding_fp=save_extractWordembedding_fp)

    if Flag == 4:
        #class_wordembedding_500_fp = config.get('FileName', 'class_wordembedding_500_fp')
        # label2ClassName_fp = config.get('FileName', 'label2ClassName_fp')
        class_wordembedding_500_fp = './refined_w2v_all.npy'
        label2ClassName_fp = './label_list.txt'
        save_fp = 'sae.txt'
        Specific.mergeClassNameAndWordEmbedding(class_wordembedding_500_fp, label2ClassName_fp, save_fp)    

    if Flag == 3:
        Flag_npy = config.getboolean(Model, 'Flag_npy')
        if Flag_npy:
            fc_vector_test_fp = config.get(Model, 'fc_vector_test_npy_fp')
        else:
            fc_vector_test_fp = config.get(Model, 'fc_vector_test_fp')
        TestImageFeaturePredicted_fp = config.get(Model, 'TestImageFeaturePredicted_fp')
        test_image_unlable_fp = config.get(Model, 'test_image_unlable_fp')
        test_image_labled_fp = config.get(Model, 'test_image_labled_fp')
        final_result_fp = config.get(Model, 'final_result_fp')
        test_lable_fp = config.get(Model, 'test_label_fp')
        Flag_npy = config.getboolean(Model, 'Flag_npy')
        ComputeTestAcc.basedOnTestImageFeatureToPredictTestResult(fc_vector_test_fp, TestImageFeaturePredicted_fp, test_image_unlable_fp, final_result_fp, test_image_labled_fp, Flag_npy, test_lable_fp)    

    if Flag == 2:
        className_fp = list_tc_pre + 'trainAndTestClassName.txt'
        label2ClassName_fp = list_tc_pre + 'label2ClassName.txt'
        save_lable_fp = list_tc_pre + 'trainAndTestLable.txt'
        Specific.gainLableBasedOnClassName(className_fp, label2ClassName_fp, save_lable_fp)    

    if Flag == 1:
        # model_key nn0 
        '''
        trainAndTestLable_fp = config.get(Model, 'trainAndTestLable_fp')
        predict_result_fp = config.get(Model, 'predict_result_fp')
        TrainImageFeaturePredicted_fp = config.get(Model, 'TrainImageFeaturePredicted_fp')
        TestImageFeaturePredicted_fp = config.get(Model, 'TestImageFeaturePredicted_fp')
        num_trainClass = config.getint(Model, 'num_trainClass')
        Specific.gainTestImageFeatureFromPredictResult(num_trainClass, predict_result_fp, trainAndTestLable_fp, TrainImageFeaturePredicted_fp, TestImageFeaturePredicted_fp)
        '''
        test_lable_fp = config.get(Model, 'test_label_fp')
        predict_result_fp = config.get(Model, 'predict_result_fp')
        TestImageFeaturePredicted_fp = config.get(Model, 'TestImageFeaturePredicted_fp')
        Specific.gainTestImageFeatureFromPredictResult(test_lable_fp, predict_result_fp, TestImageFeaturePredicted_fp)


# 2 gainLableBasedOnClassName 根据className 返回 相应的Label
# 10 gainTrainData_allPictures
# 13 constract 从全部的imagefeature中提取出部分imagefeature 提供需要的class 的label
# 14 getClassNameAndAttributeByLabel


# --- Prediction
# 35 KNN
# 18 extractTestImageFeatureFromTrainAndTestPredictImageFeature
# 1 gainTestImageFeatureFromPredictResult
# 3 basedOnTestImageFeatureToPredictTestResult

# label
# 22 mergeLable 将多个文件的label合并到一起
# 23 gainTrainLableAndTestLable 
# 26 gainImageMaskBasedOnLabel
# 39 gainAllRelatedLabelWithTest

# --- A
# 31 constractAbyRelatedClass 根据相关的K个类构造A
# 12 constractA 将字典形式的A转化成最终形式的01矩阵A
# 6 constractAllOneAMatrix 制造全1的A
# 45 mergeRelatedClass 将两个版本的 relate class 融合

# --- wordEmbedding and ImageFeat
# 4 mergeClassNameAndWordEmbedding 将classname 与 无classname的word embdeeding进行融合
# 9 gainLabelWordEmbeddingFromClassnameWordEmbedding
# 34 changAttributeIntoWordEmbeddingFormat 将att中的tab换成空格
# 20 plotTestC 为wordembedding
# 24 K_SimiliarClass_WordEmbeddingDistance 找到相似的类根据WordEmbedding
# 30 gainTopkRelateClassLabelForEveryTestClassByImageFeat 找到相似的类根据imageFeat
# 32 gainTopkRelateClassLabelForEveryTestClassByWordEmbedding 找到相似的类根据WordEmbedding
# 17 extractTrainImageFeatureFromNpy 从npy的train feat中为每个类提取出一个imagefeature 并保存成txt文件。格式为lable:imageFeature 为GCN提供输入
# 48 gainImageFeatureDirectlyFromImageFeat

# --- AnaLysis
# 21 analysisDatasetJDistance 计算某个image feature与其他imagefeature的距离
# 8  analysisTestImageFeatureandTrainImageFeatuerRelationship
# 11 split_finalResult 将不同的图片分到不同的文件夹中
# 33 devideClassByAccLevel
# 46 anaLysisRelatedClass 分析related class

# --- FeatAnalysis
# 43 returnStatisticInformation
# 42 computeInterAndExterDistanceOFImageFeat

# --- Specific
# 16 changeNPYtoTXTForTestSet 抛弃
# 5 constractWordEmbedding 抛弃 为gcn构造wordembedding
# 7 constractImageFeature 抛弃 为GCN构造train的Imagefeature
# 15 mergeFinalResultBetweenFirstVersionAndPlantVersion 合并最终结果 给出第一版本的lable 将其全部替换成最后一个版本的label
# 38 analysisAccOfFirstAttribute 分析第一大类的acc
# 41 testHighDimentionVectorHasAddAttribute 验证高纬度是具有可加性的

# --- CNN
# 29 mergeFeatNPY
# 25 gainCnnModelAcc 根据CNN的feat 得到哪个是预测正确的 你哪个是预测错误的

# --- plot
# 20 plotTSNE


list_tc_pre = '../../data/list_tc/'
config = MyFunction.gainConfig()
for cur_Flag in Flags:
    run(cur_Flag)

'''



'''