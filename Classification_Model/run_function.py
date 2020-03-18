from utils import Extractor, MyFunction, Analysis, Specific
import configparser as ConfigParser


confit_fp = './yang.conf'
config = ConfigParser.ConfigParser()
config._interpolation = ConfigParser.BasicInterpolation()
config.read(confit_fp)

# Flag 
# 1 gainAccOfTest
# 2 supplyimage
# 3 computeOffLineAcc
# 4 gain PCAforWordEmbeeding
# 5 testResultAnalysis
# 6 saveImageFeature
# 7 makeUpTestImageFeature
# 8 basedOnTestImageFeatureToPredictTestResult
Flag = 9

if Flag == 8:
    pre_fp = './data/20traindelete_model/'
    model_key = 'gcn0'
    Flag_test = False
    if Flag_test:
        pre_fc = pre_fp + 'fc_vector_test.txt'
        Image_feature_fp = pre_fp + model_key + 'TestImageFeaturePredicted.txt'
        image_fp = config.get("FILE_PATH", 'image')
        save_final_resut_fp = pre_fp + config.get('SAVE_FILE_PATH', 'final_result')
        MyFunction.basedOnTestImageFeatureToPredictTestResult(pre_fc, Image_feature_fp, image_fp, save_final_resut_fp)
    else:
        #Image_feature_fp = pre_fp + 'alltrain_image_feature.txt'
        pre_fc = pre_fp + 'fc_vector_alltrain.txt'
        Image_feature_fp = pre_fp + 'alltrain_image_feature_selectedrandom.txt'
        #Image_feature_fp = pre_fp + 'alltrain_image_feature.txt'
        MyFunction.basedOnTrainImageFeatureToPredictTrainResult(pre_fc, Image_feature_fp)


if Flag == 7:
    trainImageFeature_fp = './train/gcn0/fc_vector_test.txtTrainImageFeaturePredicted.txt'
    testImageFeature_fp = './train/gcn0/TestImageFeaturePredicted.txt'
    makeUPLableList = ['ZJL184']
    Specific.makeUpTestImageFeature(trainImageFeature_fp, testImageFeature_fp, makeUPLableList)

if Flag == 6:
    fcLayer_fp = './data//20traindelete_model/fc_vector_alltrain.txt'
    save_fp_ImageFeature = './data//20traindelete_model/alltrain_image_feature_10.txt'
    save_fp_acc = './data//20traindelete_model/alltrain_image_feature_acc.txt'
    #MyFunction.saveImageFeatureByBestStratigy(fcLayer_fp, save_fp_ImageFeature, save_fp_acc)
    MyFunction.saveImageFeature(fcLayer_fp, save_fp_ImageFeature)

if Flag == 5:
    Analysis.testResultAnalysis()

if Flag == 4:
    class_wordembedings_txt_fp = config.get("FILE_PATH", 'class_wordembeddings')
    class_wordembedings_txt_reduced_fp = config.get("SAVE_FILE_PATH", 'class_wordembedings_txt_reduced')
    MyFunction.PCAforWordEmbeeding(class_wordembedings_txt_fp, class_wordembedings_txt_reduced_fp)

if Flag == 3:
    MyFunction.computeOffLineAcc()

if Flag == 2:
    MyFunction.supplyResult()

if Flag == 1:
    fc_vector_val_fp = config.get('FILE_PATH', 'fc_vector_val_fp')
    ValIMageFeature_save_fp = config.get('SAVE_FILE_PATH', 'ValImageFeature')
    MyFunction.saveImageFeature(fc_vector_val_fp, ValIMageFeature_save_fp)
    Analysis.gainAccOfVal(fc_vector_val_fp, ValIMageFeature_save_fp)