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
from sklearn.cluster import SpectralClustering


def get_features():
    mat_name = os.path.join("./", 'feature_result.mat')
    online_mat_name = "/content/drive/My Drive/Colab/ReID works/CVPR fintuning/mat/cuhk03_feature.mat"

    datamanager = torchreid.data.ImageDataManager(
        root='./',
        sources='cuhk03',
        targets='cuhk03',
        height=256,
        width=128,
        cuhk03_labeled=True,
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


print("提取权重......")
train_feature, train_id, train_camera, train_path = get_features()
print('权重提取完成')

model = SpectralClustering(n_clusters=400, n_neighbors=8)
model.fit_predict(train_feature)

# csv 写入
csv_name = 'cuhk03_spectralclustering_400.csv'
out = open(csv_name, 'a', newline='')
# 设定写入模式
csv_write = csv.writer(out, dialect='excel')
for label in range(400):
    csv_write.writerow(np.array(train_path)[np.where(model.labels_ == label)])
print("write over")
out.close()

with open(csv_name, "rb") as f, open("/content/drive/My Drive/Colab/ReID works/CVPR fintuning/excel/cuhk03/" + csv_name,
                                     'wb') as fw:
    fw.write(f.read())
