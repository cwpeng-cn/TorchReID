import os
import torchreid
from cuhk03 import model
from cuhk03.data import *
from reid.utils import feature_operate as FO

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

model = model.get_model()
data_loader = get_loader(data_info)

print(len(data_loader.dataset))
