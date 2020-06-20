import os
import torchreid
from cuhk03 import model
from cuhk03.data import *
from reid.utils import feature_operate as FO
import scipy.io


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


train_feature = get_features()
print(train_feature)
print(len(train_feature))
