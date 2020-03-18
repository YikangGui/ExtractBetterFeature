from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import PIL
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
# from torchsummary import summary


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        file_name = self.df.iloc[item]['file_name']

        path = f'../data/{file_name}'
        image = self.transformer(Image.open(path).convert('RGB'))

        sample = {'image': image}

        # if 'word_emb_idx' in self.df.columns:
        #     sample['word_emb_idx'] = self.df.iloc[item]['word_emb_idx']

        if 'label_cnn' in self.df.columns:
            sample['label_cnn'] = self.df.iloc[item]['label_cnn']

        return sample


class FeatureSet(Dataset):
    def __init__(self, df, features):
        self.df = df
        self.features = features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        feature = self.features[item]
        idx = self.df.iloc[item]['idx']

        return {
            'feature': feature,
            'idx': idx
        }


# def load_data(batch_size=16):
#     label_list = pd.read_csv('../data/DatasetB/label_list.txt', delimiter='\t', header=None)
#     label2weidx = {label_list.iloc[i][0]: i for i in range(len(label_list))}
#     word_embedding = np.loadtxt('./external_data/class_wordembeddings_flickr.txt')
#     test_label_list = pd.read_csv('./external_data/B_test_label_list.txt', delimiter=' ', header=None)
#
#     train_A = pd.read_csv('../data/DatasetA/train.txt', delimiter='\t', header=None)
#     train_B = pd.read_csv('../data/DatasetB/train.txt', delimiter='\t', header=None)
#     test = pd.read_csv('../data/DatasetB/image.txt', delimiter='\t', header=None)
#     val = pd.read_csv('./external_data/A_test_labelled.txt', delimiter='\t', header=None)
#
#     val['file_name'] = 'DatasetA/test/' + val[0]
#     test['file_name'] = 'DatasetB/test/' + test[0]
#     train_A['file_name'] = 'DatasetA/train/' + train_A[0]
#     train_B['file_name'] = 'DatasetB/train/' + train_B[0]
#
#     train = pd.concat([train_A, train_B], 0)
#     train['word_emb_idx'] = train[1].apply(lambda x: label2weidx[x])
#     val['word_emb_idx'] = val[1].apply(lambda x: label2weidx[x])
#
#     label2cnnidx = {label_code: idx for idx, label_code in enumerate(train[1].unique().tolist())}
#     # idx2label = {label2idx[label]: label for label in label2idx}
#     train['label_cnn'] = train[1].apply(lambda x: label2cnnidx[x])
#
#     train_cnn, val_cnn = train_test_split(train, stratify=train['label_cnn'].values, train_size=0.9, test_size=0.1)
#
#     transformer_da = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(299, (0.7, 1), interpolation=PIL.Image.LANCZOS),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5]),
#     ])
#
#     transformer = transforms.Compose([
#         transforms.Resize(299, interpolation=PIL.Image.LANCZOS),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5]),
#     ])
#
#
#
#     datasets = {
#         'train': ImageSet(train, transformer),
#         'val': ImageSet(val, transformer),
#         'test': ImageSet(test, transformer),
#         'train_cnn': ImageSet(train_cnn, transformer_da),
#         'val_cnn':   ImageSet(val_cnn, transformer)
#     }
#
#     dataloaders = {
#         ds: DataLoader(datasets[ds],
#                        batch_size=batch_size,
#                        num_workers=8,
#                        shuffle=False if ds == 'test' else True) for ds in datasets
#     }
#
#     return dataloaders


def load_data_for_zsl(batch_size=128):
    label_list = pd.read_csv('../data/DatasetB/label_list.txt', delimiter='\t', header=None)
    label2idx = {label_list.iloc[i][0]: i for i in range(len(label_list))}
    word_embedding = np.loadtxt('./external_data/class_wordembeddings_flickr.txt')
    test_label_list = pd.read_csv('./external_data/B_test_label_list.txt', delimiter=' ', header=None)

    train_A = pd.read_csv('../data/DatasetA/train.txt', delimiter='\t', header=None)
    train_B = pd.read_csv('../data/DatasetB/train.txt', delimiter='\t', header=None)
    test_B = pd.read_csv('../data/DatasetB/image.txt', delimiter='\t', header=None)
    val = pd.read_csv('./external_data/A_test_labelled.txt', delimiter='\t', header=None)
    val_unseen_idx = ~val[1].isin(train_A[1])
    val = val[val_unseen_idx]

    train_A_feat = np.load('../data/train_A_feat.npy')
    train_B_feat = np.load('../data/train_B_feat.npy')
    test_A_feat = np.load('../data/test_A_feat.npy')
    test_B_feat = np.load('../data/test_B_feat.npy')

    train = pd.concat([train_A, train_B], 0)
    train['idx'] = train[1].apply(lambda x: label2idx[x])
    val['idx'] = val[1].apply(lambda x: label2idx[x])
    test = test_B

    val_idx_range = val['idx'].unique().tolist()
    test_idx_range = test_label_list[0].apply(lambda x: label2idx).tolist()

    train_feat = np.concatenate([train_A_feat, train_B_feat], 0)
    val_feat = test_A_feat[val_unseen_idx]
    test_feat = test_B_feat

    datasets = {
        'train': FeatureSet(train, train_feat),
        'val': FeatureSet(val, val_feat),
        'test': FeatureSet(test, test_feat),

    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False if ds == 'test' else True) for ds in datasets
    }

    return dataloaders, word_embedding, val_idx_range, test_idx_range

def deleteTrainABdata(train_cnn, val_cnn, trainAB_num):
    return train_cnn[(train_cnn['id'] >= trainAB_num)], val_cnn[(val_cnn['id'] >= trainAB_num)]

def deleteTraindata_byLabel(train_cnn, val_cnn, lowAccLabel_fp):
    lowAccLabel = readVector(lowAccLabel_fp)
    train_cnn['Flag_train'] = [e not in lowAccLabel for e in train_cnn[1]]
    val_cnn['Flag_val'] = [e not in lowAccLabel for e in val_cnn[1]]
    return train_cnn[(train_cnn['Flag_train'] == True)], val_cnn[(val_cnn['Flag_val'] == True)]


def load_data_for_training_cnn(batch_size=16, image_size=299, lowAccLabel_fp=None):
    # train_A = pd.read_csv('../data/DatasetA/train.txt', delimiter='\t', header=None)
    # train_B = pd.read_csv('../data/DatasetB/train.txt', delimiter='\t', header=None)
    train_C = pd.read_csv('../data/DatasetC/train.txt', delimiter='\t', header=None)
    train_D = pd.read_csv('../data/DatasetD/train.txt', delimiter='\t', header=None)
    # trainA_num = train_A.shape[0]
    # trainB_num = train_B.shape[0]
    trainC_num = train_C.shape[0]
    trainD_num = train_D.shape[0]
    # train_A['file_name'] = 'DatasetA/train/' + train_A[0]
    # train_B['file_name'] = 'DatasetB/train/' + train_B[0]
    train_C['file_name'] = 'DatasetC/train/' + train_C[0]
    train_D['file_name'] = 'DatasetD/train/' + train_D[0]

    # train = pd.concat([train_A, train_B, train_C, train_D], 0)
    train = pd.concat([train_C, train_D], 0)
    label2cnnidx = {label_code: idx for idx, label_code in enumerate(train[1].unique().tolist())}
    cnnidx2label = {idx: label_code for idx, label_code in enumerate(train[1].unique().tolist())}
    # train['id'] = range(0, trainA_num+trainB_num+trainC_num+trainD_num)
    total_image_num = trainC_num+trainD_num
    train['id'] = range(0, total_image_num)

    train['label_cnn'] = train[1].apply(lambda x: label2cnnidx[x])
    print("total train num is:", total_image_num)
    train_cnn, val_cnn = train_test_split(train, stratify=train['label_cnn'].values, train_size=0.9, test_size=0.1, random_state=100)
    # train_cnn, val_cnn = deleteTrainABdata(train_cnn, val_cnn, trainA_num+trainB_num)
    # train_cnn, val_cnn = deleteTraindata_byLabel(train_cnn, val_cnn, lowAccLabel_fp)
    transformer_da = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, (0.7, 1), interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    transformer = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    datasets = {
        'train_cnn': ImageSet(train_cnn, transformer_da),
        'val_cnn':   ImageSet(val_cnn, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets
    }

    return dataloaders, cnnidx2label

def load_data_for_training_cnn_ABCD(batch_size=16, image_size=299, lowAccLabel_fp=None):
    train_A = pd.read_csv('../data/DatasetA/train.txt', delimiter='\t', header=None)
    train_B = pd.read_csv('../data/DatasetB/train.txt', delimiter='\t', header=None)
    train_C = pd.read_csv('../data/DatasetC/train.txt', delimiter='\t', header=None)
    train_D = pd.read_csv('../data/DatasetD/train.txt', delimiter='\t', header=None)
    trainA_num = train_A.shape[0]
    trainB_num = train_B.shape[0]
    trainC_num = train_C.shape[0]
    trainD_num = train_D.shape[0]
    train_A['file_name'] = 'DatasetA/train/' + train_A[0]
    train_B['file_name'] = 'DatasetB/train/' + train_B[0]
    train_C['file_name'] = 'DatasetC/train/' + train_C[0]
    train_D['file_name'] = 'DatasetD/train/' + train_D[0]

    train = pd.concat([train_A, train_B, train_C, train_D], 0)
    label2cnnidx = {label_code: idx for idx, label_code in enumerate(train[1].unique().tolist())}
    cnnidx2label = {idx: label_code for idx, label_code in enumerate(train[1].unique().tolist())}

    total_image_num = trainA_num+trainB_num+trainC_num+trainD_num
    train['id'] = range(0, total_image_num)

    train['label_cnn'] = train[1].apply(lambda x: label2cnnidx[x])
    print("total train num is:", total_image_num)
    train_cnn, val_cnn = train_test_split(train, stratify=train['label_cnn'].values, train_size=0.9, test_size=0.1, random_state=100)
    # train_cnn, val_cnn = deleteTrainABdata(train_cnn, val_cnn, trainA_num+trainB_num)
    # train_cnn, val_cnn = deleteTraindata_byLabel(train_cnn, val_cnn, lowAccLabel_fp)
    transformer_da = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, (0.7, 1), interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    transformer = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    datasets = {
        'train_cnn': ImageSet(train_cnn, transformer_da),
        'val_cnn':   ImageSet(val_cnn, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets
    }

    return dataloaders, cnnidx2label

def load_data_for_feature_extract(batch_size=128, image_size=299):
    train_A = pd.read_csv('../data/DatasetA/train.txt', delimiter='\t', header=None)
    train_B = pd.read_csv('../data/DatasetB/train.txt', delimiter='\t', header=None)
    train_C = pd.read_csv('../data/DatasetC/train.txt', delimiter='\t', header=None)
    train_D = pd.read_csv('../data/DatasetD/train.txt', delimiter='\t', header=None)
    test_A = pd.read_csv('../data/DatasetA/image.txt', delimiter='\t', header=None)
    test_B = pd.read_csv('../data/DatasetB/image.txt', delimiter='\t', header=None)
    test_C = pd.read_csv('../data/DatasetC/image.txt', delimiter='\t', header=None)
    test_D = pd.read_csv('../data/DatasetD/image.txt', delimiter='\t', header=None)

    train = pd.concat([train_A, train_B, train_C, train_D], 0)
    cnnidx2label = {idx: label_code for idx, label_code in enumerate(train[1].unique().tolist())}

    test_A['file_name'] = 'DatasetA/test/' + test_A[0]
    test_B['file_name'] = 'DatasetB/test/' + test_B[0]
    test_C['file_name'] = 'DatasetC/test/' + test_C[0]
    test_D['file_name'] = 'DatasetD/test/' + test_D[0]
    train_A['file_name'] = 'DatasetA/train/' + train_A[0]
    train_B['file_name'] = 'DatasetB/train/' + train_B[0]
    train_C['file_name'] = 'DatasetC/train/' + train_C[0]
    train_D['file_name'] = 'DatasetD/train/' + train_D[0]

    transformer = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    datasets = {
        'train_A': ImageSet(train_A, transformer),
        'train_B': ImageSet(train_B, transformer),
        'train_C': ImageSet(train_C, transformer),
        'train_D': ImageSet(train_D, transformer),
        'test_A': ImageSet(test_A, transformer),
        'test_B': ImageSet(test_B, transformer),
        'test_C': ImageSet(test_C, transformer),
        'test_D': ImageSet(test_D, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets
    }

    return dataloaders, cnnidx2label

def load_data_for_feature_extract_ZJJ(batch_size=2):
    train_A = pd.read_csv('../data/DatasetJ/ImageName_Label.txt', delimiter='\t', header=None)
    train_A['file_name'] = 'DatasetJ/train/' + train_A[0]

    transformer = transforms.Compose([
        transforms.Resize(299, interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    datasets = {
        'train_A': ImageSet(train_A, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets
    }

    return dataloaders

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.5, 0.5, 0.5]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def permute(x):
    pivot = x.size(0) // 2
    return torch.cat([x[pivot:], x[:pivot]], 0)


def mixup(x, y, n_class, epsilon=0.1, alpha=1.):

    y = torch.zeros([len(y), n_class], device=y.device).scatter_(1, y.view(-1, 1), 1)
    y = (1 - epsilon) * y + epsilon / n_class

    lam = np.random.beta(alpha, alpha)

    mix_x = x * lam + permute(x) * (1 - lam)
    mix_y = y * lam + permute(y) * (1 - lam)

    x = torch.cat([x, mix_x], 0)
    y = torch.cat([y, mix_y], 0)

    return x, y


def cross_entropy(logits, y):
    log_probs = F.log_softmax(logits, 1)
    return (-y * log_probs).sum(1).mean(0)


def train_cnn(model, tr_loader, va_loader,\
              cnnidx2label, device, n_ep=100, n_classes=None,\
              multi_gpu=None, repick=True, input_size=299):
    if multi_gpu is not None:
        model = torch.nn.DataParallel(model, device_ids=multi_gpu)
        device = torch.device(f'cuda:{multi_gpu[0]}')

    if repick:
        print("load model")
        model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_13/pth/irv2_3_0.424006.pth'))

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)

    model.to(device)
    # summary(model, (3, input_size, input_size))

    random_erasing = RandomErasing()

    best_acc = 0.00

    for i_ep in range(n_ep):
        model.train()
        count = 0
        for _, sample in tqdm(enumerate(tr_loader), total=len(tr_loader)):
            image = sample['image'].to(device)
            label = sample['label_cnn'].to(device)
            image, label = mixup(image, label, n_classes)
            for i in range(len(image)):
                image[i] = random_erasing(image[i])

            optimizer.zero_grad()
            features, logits = model(image)
            loss = cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            if count %1000 == 0:
                print(loss)
            count += 1

        model.eval()
        with torch.no_grad():
            losses = []
            preds = []
            ys = []
            counter = 0
            for _, sample in tqdm(enumerate(tr_loader), total=len(tr_loader)):
                image = sample['image'].to(device)
                label = sample['label_cnn'].to(device)

                _, logits = model(image)
                loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
                losses += [loss]
                ys += [label.detach().cpu().numpy()]
                preds += [(logits.max(1)[1].detach().cpu().numpy())]
                counter += 1
                if counter == 30:
                    break

            preds = np.concatenate(preds, 0).reshape(-1)
            ys = np.concatenate(ys, 0).reshape(-1)
            # analysisClassForIVR2(preds, ys, cnnidx2label)
            acc = accuracy_score(ys, preds)

            loss = np.concatenate(losses).reshape(-1).mean()
            scheduler.step(loss)
            print("train acc is ")
            print(f'{i_ep}: {acc:.5%} | {loss:.5f}')

        model.eval()
        with torch.no_grad():
            losses = []
            preds = []
            ys = []
            for _, sample in tqdm(enumerate(va_loader), total=len(va_loader)):
                image = sample['image'].to(device)
                label = sample['label_cnn'].to(device)

                _, logits = model(image)
                loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
                losses += [loss]
                ys += [label.detach().cpu().numpy()]
                preds += [(logits.max(1)[1].detach().cpu().numpy())]

            preds = np.concatenate(preds, 0).reshape(-1)
            ys = np.concatenate(ys, 0).reshape(-1)
            # analysisClassForIVR2(preds, ys, cnnidx2label)
            acc = accuracy_score(ys, preds)

            loss = np.concatenate(losses).reshape(-1).mean()
            scheduler.step(loss)
            print(f'{i_ep}: {acc:.5%} | {loss:.5f}')
            print("best_acc", best_acc, "acc", acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.cpu().state_dict(), '../data/classficaData/B3_IVR2_13/pth/irv2_%d_%f.pth' %(i_ep, best_acc))
                model.to(device)

def train_cnn_centerLoss(model, criterion_cent, tr_loader, va_loader,\
              cnnidx2label, device, n_ep=100, n_classes=365,\
              multi_gpu=None, repick=True, input_size=299, 
              lr_cent=0.1, weight_cent=1):
    if multi_gpu is not None:
        model = torch.nn.DataParallel(model, device_ids=multi_gpu)
        device = torch.device(f'cuda:{multi_gpu[0]}')

    if repick:
        print("load model")
        model.load_state_dict(torch.load('../data/classficaData/B3_IVR2_13//irv2_1024.pth'))
        # criterion_cent.load_state_dict(torch.load('../data/classficaData/B3_IVR2_12/pth/irv2_9_0.671674_center.pth'))

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0001)
    optimizer_centloss = torch.optim.Adam(criterion_cent.parameters(), lr=lr_cent)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)

    model.to(device)

    random_erasing = RandomErasing()

    best_acc = 0.00
    # best_model_wts = copy.deepcopy(model.cpu().state_dict())

    for i_ep in range(n_ep):
        model.train()
        count_iter = 0
        for _, sample in tqdm(enumerate(tr_loader), total=len(tr_loader)):
            image = sample['image'].to(device)
            label = sample['label_cnn'].to(device)
            # image, label, label_index = mixup(image, label, n_classes)
            for i in range(len(image)):
                image[i] = random_erasing(image[i])

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()
            features, logits = model(image)
            # loss_xent = cross_entropy(logits, label)
            loss_xent = F.cross_entropy(logits, label)
            loss_cent = criterion_cent(features, label)
            loss_cent *= weight_cent
            if count_iter % 1000 == 0:
                print(loss_xent, loss_cent)
            loss = loss_xent + loss_cent
            loss.backward()
            optimizer.step()
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / weight_cent)
            optimizer_centloss.step()
            count_iter += 1

        model.eval()
        with torch.no_grad():
            losses = []
            preds = []
            ys = []
            counter = 0
            for _, sample in tqdm(enumerate(tr_loader), total=len(tr_loader)):
                image = sample['image'].to(device)
                label = sample['label_cnn'].to(device)

                _, logits = model(image)
                loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
                losses += [loss]
                ys += [label.detach().cpu().numpy()]
                preds += [(logits.max(1)[1].detach().cpu().numpy())]
                counter += 1
                if counter == 30:
                    break

            preds = np.concatenate(preds, 0).reshape(-1)
            ys = np.concatenate(ys, 0).reshape(-1)
            # analysisClassForIVR2(preds, ys, cnnidx2label)
            acc = accuracy_score(ys, preds)

            loss = np.concatenate(losses).reshape(-1).mean()
            scheduler.step(loss)
            print(f'{i_ep}: {acc:.5%} | {loss:.5f}')

        model.eval()
        with torch.no_grad():
            losses = []
            preds = []
            ys = []
            for _, sample in tqdm(enumerate(va_loader), total=len(va_loader)):
                image = sample['image'].to(device)
                label = sample['label_cnn'].to(device)

                _, logits = model(image)
                loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
                losses += [loss]
                ys += [label.detach().cpu().numpy()]
                preds += [(logits.max(1)[1].detach().cpu().numpy())]

            preds = np.concatenate(preds, 0).reshape(-1)
            ys = np.concatenate(ys, 0).reshape(-1)
            # analysisClassForIVR2(preds, ys, cnnidx2label)
            acc = accuracy_score(ys, preds)

            loss = np.concatenate(losses).reshape(-1).mean()
            scheduler.step(loss)
            print(f'{i_ep}: {acc:.5%} | {loss:.5f}')
            print("best_acc", best_acc, "acc", acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.cpu().state_dict(), '../data/classficaData/B3_IVR2_14/pth/irv2_%d_%f.pth' %(i_ep, best_acc))
                torch.save(criterion_cent.cpu().state_dict(), '../data/classficaData/B3_IVR2_14/pth/irv2_%d_%f_center.pth' %(i_ep, best_acc))
                model.to(device)
                criterion_cent.to(device)

def extract_features_ZJJ(model, lodar, devive, save_path):
    # Myfunction.analysisClassForIVR2()
    pass

def extract_scores(model, loader, device, path, cnnidx2label,  train_lable_fp):
    scores = []
    related_class = []
    model.eval()
    train_label = readVector(train_lable_fp)
    with torch.no_grad():
        for _, sample in tqdm(enumerate(loader), total=len(loader)):
            image = sample['image'].to(device)

            f = model.get_scores(image).detach().cpu().numpy()
            scores += [f]
            break
    scores = np.concatenate(scores, 0)
    for cur_score in scores:
        cur_related_class = []
        cur_score_sorted_index = list(np.argsort(cur_score))
        cur_score_sorted_index = np.flip(cur_score_sorted_index, axis=0)
        for e_inx in cur_score_sorted_index:
            if cnnidx2label[e_inx] in train_label:
                cur_related_class.append(cnnidx2label[e_inx])
                cur_related_class.append(cur_score[e_inx])
        related_class.append(cur_related_class)
    saveVector(path, related_class)
    # np.save(path, features)

def extract_features(model, loader, device, path):
    features = []
    model.eval()
    with torch.no_grad():
        for _, sample in tqdm(enumerate(loader), total=len(loader)):
            image = sample['image'].to(device)

            f = model.get_features(image).detach().cpu().numpy()
            features += [f]

    features = np.concatenate(features, 0)
    np.save(path, features)

def predict_result(model, loader, device, path, cnnidx2label):
    preds = []
    label_result = []
    model.eval()
    with torch.no_grad():
        for _, sample in tqdm(enumerate(loader), total=len(loader)):
            image = sample['image'].to(device)
            logits = model(image)
            preds += [(logits.max(1)[1].detach().cpu().numpy())]
        preds = np.concatenate(preds, 0).reshape(-1)
    label_result = [cnnidx2label[e] for e in preds]
    saveVector(path, label_result)


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
    acc_recompute = 0
    for e in class_num_map:
        acc_recompute += class_num_map[e] * class_acc_map[e]
    # print("recompute acc is ", acc_recompute/num_samples)

    print("acc < 0.2")
    for e in class_acc_map:
        if class_acc_map[e] < 0.3 and class_acc_map[e] > 0.001:
            print(train_dict[e])

def saveVector(save_fp,vector):
    file = open(save_fp, 'w')
    for value in vector:
        file.write(str(value) + "\n")
    file.close()

def readVector(fp):
    f = open(fp,'r')
    result = []
    line = f.readline()
    while line:
        result.append(line.strip())
        line = f.readline()
    return result