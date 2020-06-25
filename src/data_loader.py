#Code is partly adapted from https://github.com/kevinzakka/recurrent-visual-attention

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from pathlib import Path
import skimage.io as io
from PIL import Image
import json
import time
import logging
import torchfile as thf

logger = logging.getLogger(__name__)


class mnist_data(Dataset):
    """
    Cluttered MNIST Dataset
    """

    def __init__(self, data_dir, idx, transforms=None, frac=1.0):
        """
        Params
        ----------------
        :param data_dir     : (string) directory containing dataset
        :param idx          : (array_like) list of indices to use
        :param transforms   : (array_like) contains transformations applied to data
        :param frac         : (float) fraction of labels to use for semi-supervised training
        """

        self.data_path = Path(data_dir)

        thdata = thf.load(self.data_path)
        self.data = torch.from_numpy(thdata[b"data"][idx]).float()
        self.labels = torch.from_numpy(thdata[b"labels"][idx])

        self.frac = frac
        if frac:
            num_labeled = int(self.frac * len(idx))
            self.labeled = self.data[:num_labeled]
            self.unlabeled = self.data[num_labeled:]
        else:
            self.unlabeled = self.data
        self.n = 0

        self.transforms = transforms

    def __len__(self):
        """
        Returns
        ----------------
        :returns l: (int) length of dataset
        """
        if self.frac < 1:
            return len(self.unlabeled)
        else:
            return len(self.labeled)

    def __getitem__(self, idx):
        """
        Returns item at index idx
        Params
        ----------------
        :param idx : (int) index of item to be returned
        Returns
        ----------------
        :returns item: (array_like) 2D image Tensor of dimension (H,W)
        :returns ann : (array_like) annotation of image as one hot vector
        """
        if not self.frac:
            return self.unlabeled[idx], [], []
        if self.frac < 1:
            out = self.unlabeled[idx], self.labeled[self.n], self.labels[self.n]
            self.n = (self.n + 1) % len(self.labeled)
        else:
            out = [], self.labeled[idx], self.labels[idx]
        return out


def get_data_loader(
    data_dir,
    batch_size,
    random_seed,
    num=1000,
    num_val=1000,
    train=True,
    num_workers=0,
    pin_memory=False,
    frac=1.0,
    data_size=100000,
):
    """
    Params
    ----------------
    :param data_dir     : (string) directory containing dataset.
    :param batch_size   : (int) samples per batch.
    :param random_seed  : (int) random seed.
    :param num          : (int) number of training samples to use
    :param num_val      : (int) number of validation samples to use
    :param data_size    : (int) size of the full data set (train + test)
    :param num_workers  : (int) number of subprocesses to use when loading the dataset.
    :param pin_memory   : (bool) whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    :param frac         : (float) fraction of labels to use for semi-supervised training
    Returns
    ----------------
    :returns train_loader: (DataLoader) training set iterator.
    :returns valid_loader: (DataLoader) validation set iterator.
    """

    if num_workers > batch_size:
        num_workers = 0

    if train:

        train_idx = torch.arange(0, num)
        val_idx = torch.arange(num, num + num_val)

        traindata = mnist_data(
            data_dir, train_idx, frac=frac
        )
        valdata = mnist_data(
            data_dir, val_idx
        )

        train_sampler = RandomSampler(traindata)
        valid_sampler = RandomSampler(valdata)

        train_loader = DataLoader(
            traindata,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valdata,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return (train_loader, valid_loader)

    else:
        test_idx = torch.arange(data_size - num, data_size)
        testdata = mnist_data(
            data_dir, test_idx
        )
        test_loader = DataLoader(
            testdata,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return test_loader
