from utils_wei import load_data_for_training_cnn, train_cnn, \
    load_data_for_feature_extract, extract_features, load_data_for_zsl, \
    load_data_for_feature_extract_ZJJ, predict_result, extract_scores,\
    load_data_for_training_cnn_ABCD, train_cnn_centerLoss
from irv2 import InceptionResNetV2
from rn import AttributeNetwork, RelationNetwork, train_rn
import torch
# from se_resnet import se_resnet50
from center_loss_mseNorm import CenterLoss
# from torchsummary import summary

def train_cnn_ivr():
    n_feature = 1024
    drop_rate = 0.2
    lowAccLabel_fp = '../data/list_tc/label/accLeccThan20Label_filter.txt'

    loaders, cnnidx2label = load_data_for_training_cnn_ABCD(
                            batch_size=20 * 1, 
                            lowAccLabel_fp=lowAccLabel_fp)
    n_classes = len(list(cnnidx2label.keys()))
    print("total class is ", n_classes)

    model = InceptionResNetV2(num_classes=n_classes, num_feature=n_feature, drop_rate=drop_rate)
    DEVICE = torch.device('cuda:0')
    gpu_ids = [0]

    train_cnn(model, loaders['train_cnn'],
              loaders['val_cnn'], cnnidx2label, DEVICE,
              multi_gpu=None, repick=True, n_classes=n_classes)


def train_cnn_ivr_centerloss():
    n_classes = 365
    lowAccLabel_fp = '../data/list_tc/label/accLeccThan2Label_filter.txt'

    loaders, cnnidx2label = load_data_for_training_cnn_ABCD(
                            batch_size=32 * 1, 
                            lowAccLabel_fp=lowAccLabel_fp)

    model = InceptionResNetV2(num_classes=n_classes, num_feature=1024, drop_rate=0.2)
    # model = se_resnet50(num_classes=365)
    criterion_cent = CenterLoss(num_classes=365, feat_dim=1024, use_gpu=True)
    DEVICE = torch.device('cuda:0')
    gpu_ids = [0]

    train_cnn_centerLoss(model, criterion_cent, loaders['train_cnn'],
              loaders['val_cnn'], cnnidx2label, DEVICE,
              multi_gpu=gpu_ids, repick=True, n_classes=n_classes)

def result_predict():
    model = InceptionResNetV2(num_classes=365, num_feature=1024)
    DEVICE_ID = 0
    device = torch.device(f'cuda:{DEVICE_ID}')
    model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_1/pth/irv2_9_0.678389.pth'))
    model = model.to(device)
    loaders, cnnidx2label = load_data_for_feature_extract()
    # predict_result(model, loaders['train_A'], device, '../data/train_A_predict.txt', cnnidx2label)
    # predict_result(model, loaders['train_B'], device, '../data/train_B_predict.txt', cnnidx2label)
    predict_result(model, loaders['test_C'], device, './test_C_predict.txt', cnnidx2label)


def feature_extract():

    model = InceptionResNetV2(num_classes=365, num_feature=1024)
    DEVICE_ID = 0
    device = torch.device(f'cuda:{DEVICE_ID}')
    model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_13/pth/irv2_25_0.688679.pth'))
    model = model.to(device)
    '''
    
    model = InceptionResNetV2(num_classes=365, num_feature=1024)
    gpu_ids = [0]
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[0]}')
    model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_13/pth/irv2_11_0.819228.pth'))
    model = model.module.to(device)
    '''
    loaders, cnnidx2label = load_data_for_feature_extract()
    print("load data is  ok")
    '''
    extract_features(model, loaders['train_A'], device, '../data/train_A_feat.npy')
    extract_features(model, loaders['train_B'], device, '../data/train_B_feat.npy')
    extract_features(model, loaders['test_A'], device, '../data/test_A_feat.npy')
    extract_features(model, loaders['test_B'], device, '../data/test_B_feat.npy')
    extract_features(model, loaders['train_C'], device, '../data/train_C_feat.npy')
    extract_features(model, loaders['test_C'], device, '../data/test_C_feat.npy')
    '''
    extract_features(model, loaders['train_D'], device, '../data/train_D_feat.npy')
    # extract_features(model, loaders['test_D'], device, '../data/test_D_feat.npy')


def score_extract():
    train_lable_fp = '../data/list_tc/label/train_lable_1_C.txt'
    model = InceptionResNetV2(num_classes=365, num_feature=1024)
    # model = se_resnet50(num_classes=365)
    DEVICE_ID = 0
    device = torch.device(f'cuda:{DEVICE_ID}')
    model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_1/pth/irv2_9_0.678389.pth'))
    model = model.to(device)
    loaders, cnnidx2label = load_data_for_feature_extract()
    extract_scores(model, loaders['test_C'], device, './tem.txt', cnnidx2label, train_lable_fp)

def feature_extract_ZJJ():
    model = InceptionResNetV2(num_classes=365, num_feature=1024)
    DEVICE_ID = 0
    model = torch.nn.DataParallel(model, device_ids=[DEVICE_ID])
    device = torch.device(f'cuda:{DEVICE_ID}')
    model.load_state_dict(torch.load('../data/tem/irv2.pth'))
    model = model.module.to(device)

    loaders = load_data_for_feature_extract_ZJJ()
    extract_features(model, loaders['train_A'], device, '../data/train_J_feat.npy')

def train_zsl():
    loaders, word_embeddings, val_idx_range, test_idx_range = load_data_for_zsl(batch_size=128)
    DEVICE = torch.device('cuda:1')
    attribute_network = AttributeNetwork(500, 1024, 1536)
    relation_network = RelationNetwork(1536 * 2, 1024)
    train_rn(attribute_network, relation_network, loaders['train'],loaders['val'], word_embeddings, val_idx_range, DEVICE)


if __name__ == '__main__':
    train_cnn_ivr()
    # train_cnn_ivr_centerloss()
    # result_predict()
    # score_extract()
    # feature_extract()
    # feature_extract_ZJJ()
    # train_zsl()
