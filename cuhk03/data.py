from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import os.path as osp
from glob import glob
import re
from torchvision import transforms
from torch.utils.data import DataLoader


class Dataset(data.Dataset):
    def __init__(self, data_info):
        self.data_info = data_info
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.original_id = []
        self.cameras = []
        self.paths = []
        self.ret = []
        self.preprocess()

    def preprocess(self, relabel=True):
        all_pids = {}
        for info in self.data_info:
            fpath, pid, cam = info[0], info[1], info[2]
            self.original_id.append(pid)
            self.cameras.append(cam)
            self.paths.append(fpath)
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            self.ret.append((fpath, pid, cam))

    def __getitem__(self, index):
        image = Image.open(self.ret[index][0])
        id_ = self.ret[index][1]
        cam = self.ret[index][2]
        return self.transform(image), id_, cam

    def __len__(self):
        return len(self.ret)


def get_dataset(data_info):
    return Dataset(data_info)


def get_loader(data_info):
    dataset = get_dataset(data_info)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
