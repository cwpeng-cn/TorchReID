import os
import torchreid
from . import model


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

data_loader = datamanager.train_loader
dataset = data_loader.dataset.train + data_loader.dataset.query + data_loader.dataset.gallery

print(len(dataset))


print(model.get_model())