import os
import torchreid
from cuhk03 import model
from cuhk03.data import *
from reid.utils import feature_operate as FO
import scipy.io
import numpy as np


class union_find:
    def __init__(self, length):
        self.length = length
        self.ids = np.arange(length)

    def union(self, i, j):
        id_i = self.ids[i]
        id_j = self.ids[j]
        for i in range(self.length):
            if self.ids[i] == id_i:
                self.ids[i] = id_j

    def get_set(self):
        keys = []
        result = {}
        for i in range(self.length):
            value = self.ids[i]
            if value in keys:
                result[value].append(i)
            else:
                keys.append(value)
                result[value] = [i]
        return result


def get_features():
    mat_name = os.path.join("./", 'feature_result.mat')
    online_mat_name = "/content/drive/My Drive/Colab/ReID works/CVPR fintuning/mat/market_feature.mat"

    if os.path.exists(mat_name):
        result = scipy.io.loadmat(mat_name)
        train_feature = result['train_feature']
        return train_feature

    if os.path.exists(online_mat_name):
        with open(online_mat_name, "rb") as f, open(mat_name, 'wb') as fw:
            fw.write(f.read())
        result = scipy.io.loadmat(mat_name)
        train_feature = result['train_feature']
        return train_feature

    datamanager = torchreid.data.ImageDataManager(
        root='./',
        sources='cuhk03',
        targets='cuhk03',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    )

    info = datamanager.train_loader
    data_info = info.dataset.train + info.dataset.query + info.dataset.gallery

    net = model.get_model()
    data_loader = get_loader(data_info)

    train_feature = FO.extract_cnn_feature(net, loader=data_loader, vis=False, is_normlize=False)
    train_id, train_camera = data_loader.dataset.original_id, data_loader.dataset.cameras

    result = {'train_feature': train_feature.numpy()}
    scipy.io.savemat(mat_name, result)

    with open(mat_name, "rb") as f, open(online_mat_name, 'wb') as fw:
        fw.write(f.read())

    train_feature = result['train_feature']
    return train_feature


def get_similarity(tf):
    """
    :param tf: 图片的特征
    :return:
    """
    feature = torch.tensor(tf).cuda()

    score = torch.mm(feature, feature.t()).detach().cpu().numpy()
    indexs = np.argsort(-score, axis=1)
    return indexs


def connect_with_mutual(indexs):
    u = union_find(indexs.shape[0])
    for i in range(indexs.shape[0]):
        for k in indexs[i][:13]:
            if i in indexs[k][:13]:
                u.union(i, k)
        if i % 100 == 0:
            print(i)
    return u.get_set()


train_feature = get_features()
indexs = get_similarity(train_feature)
connected = connect_with_mutual(indexs)
print(len(connected.keys()))
