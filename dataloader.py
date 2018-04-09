import os
import torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from dataset import Vim2Dataset
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image


class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        # self.batch_table = {4:16, 8:16, 16:16, 32:8, 64:8, 128:8, 256:6, 512:1, 1024:1} # change this according to available gpu memory.
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4

    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))

        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))
        # self.dataset = ImageFolder(
        #             root=self.root,
        #             transform=transforms.Compose(   [
        #                                             transforms.Scale(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
        #                                             transforms.ToTensor(),
        #                                             ]))
        self.dataset = Vim2Dataset(  self.root,
                                subject_id=1,
                                is_val=False,
                                stimuli_transform = transforms.Compose(
                                    [
                                        # transforms.Scale(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
                                        transforms.ToTensor()
                                    ]
                                ),
                                voxel_response_transform=None,
                                # add_noise=[-0.002, 0.002],
                                pre_resize_stimuli=(self.imsize,self.imsize),
                                add_noise=False,
                                debug=False,
                                verbose=False)

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)


    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]
