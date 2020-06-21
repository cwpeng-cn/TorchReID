import os
import torchreid
from cuhk03 import model
from cuhk03.data import *
from reid.utils import feature_operate as FO
import scipy.io
import numpy as np
import pandas as pd
from glob import glob
from os import path as osp
import csv


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

    data_loader = get_loader(data_info)
    train_id, train_camera, train_path = data_loader.dataset.original_id, data_loader.dataset.cameras, data_loader.dataset.paths

    if os.path.exists(mat_name):
        result = scipy.io.loadmat(mat_name)
        train_feature = result['train_feature']
        return train_feature, train_id, train_camera, train_path

    if os.path.exists(online_mat_name):
        with open(online_mat_name, "rb") as f, open(mat_name, 'wb') as fw:
            fw.write(f.read())
        result = scipy.io.loadmat(mat_name)
        train_feature = result['train_feature']
        return train_feature, train_id, train_camera, train_path

    net = model.get_model()
    train_feature = FO.extract_cnn_feature(net, loader=data_loader, vis=False, is_normlize=False)
    train_id, train_camera, train_path = data_loader.dataset.original_id, data_loader.dataset.cameras, data_loader.dataset.paths

    result = {'train_feature': train_feature.numpy()}
    scipy.io.savemat(mat_name, result)

    with open(mat_name, "rb") as f, open(online_mat_name, 'wb') as fw:
        fw.write(f.read())

    train_feature = result['train_feature']
    return train_feature, train_id, train_camera, train_path


def get_similarity(tf):
    """
    :param tf: 图片的特征
    :return:
    """
    feature = torch.tensor(tf).cuda()

    score = torch.mm(feature, feature.t()).detach().cpu().numpy()
    indexs = np.argsort(-score, axis=1)
    return indexs


def connect_with_mutual(indexs, num=13):
    u = union_find(indexs.shape[0])
    for i in range(indexs.shape[0]):
        for k in indexs[i][:num]:
            if i in indexs[k][:num]:
                u.union(i, k)
        if i % 100 == 0:
            print(i)
    return u.get_set()


num = 20
train_feature, train_id, train_camera, train_path = get_features()
indexs = get_similarity(train_feature)
connected = connect_with_mutual(indexs, num)

reliable_keys = []
for key in connected.keys():
    if len(connected[key]) >= 3 and len(connected[key]) <= 32:
        reliable_keys.append(key)

for key in reliable_keys:
    print(key, ":", np.array(train_id)[connected[key]])

# csv 写入
csv_name = 'result_stn_cuhk03_{}_3_32.csv'.format(num)
out = open(csv_name, 'a', newline='')
# 设定写入模式
csv_write = csv.writer(out, dialect='excel')
for i, key in enumerate(reliable_keys):
    csv_write.writerow(np.array(train_path)[connected[key]])
print("write over")

with open(csv_name, "rb") as f, open("/content/drive/My Drive/Colab/ReID works/CVPR fintuning/excel/" + csv_name,
                                     'wb') as fw:
    fw.write(f.read())
