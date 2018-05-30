#!/usr/bin/env python

"""
    omniglot.py
"""

from __future__ import print_function

from PIL import Image
import torch.utils.data as data
from torchvision.datasets.utils import list_files

import os
import torch
import numpy as np
from glob import glob
from collections import defaultdict
from torch.utils.data import TensorDataset

class Omniglot(data.Dataset):
    def __init__(self, root='data/omniglot'):
        self._imgs = []
        self._classes = []
        characters = glob(os.path.join(root, '*/*'))
        for idx, character in enumerate(characters):
            self._classes.append(idx)
            for image in glob(os.path.join(character, '*.png')):
                self._imgs.append((image, idx))
    
    def __len__(self):
        return len(self._imgs)
        
    def get(self, index, rotation=0):
        path, label = self._imgs[index]
        
        img = Image.open(path).resize((28, 28))
        if rotation != 0:
            img = img.rotate(rotation)
        img = np.array(img).astype('float32')
        img = torch.FloatTensor(img).unsqueeze(0)
        
        return img, label


class OmniglotTaskWrapper:
    def __init__(self, dataset, classes, rotation=False):
        self.dataset = dataset
        self.rotation = rotation
        
        self.lookup = defaultdict(list)
        for idx,(f,lab) in enumerate(dataset._imgs):
            if lab in classes:
                self.lookup[lab].append(idx)
        
        self.classes = list(self.lookup.keys())
    
    def sample_task(self, num_classes, num_shots, test_shots=0, replace=False):
        """
            Note: rotation is handled slightly differently than in OPENAI version
            In this setup, we'll never sample the same character w/ different
            rotations simultaneously.  However, since the number of characters
            is large, that probably doesn't happen very much in the OPENAI implementation
            anyway.
        """
        classes = np.random.choice(self.classes, num_classes, replace=replace)
        
        task, labs = [], []
        for i, lab in enumerate(classes):
            rotation = np.random.choice([0, 90, 180, 270]) if self.rotation else 0
            for idx in np.random.choice(self.lookup[lab], num_shots, replace=replace):
                task.append(self.dataset.get(idx, rotation=rotation))
                labs.append(i)
        
        imgs = list(zip(*task))[0]
        imgs = torch.cat(imgs).unsqueeze(1)
        
        labs = torch.LongTensor(labs)
        
        if test_shots > 0:
            sel = torch.arange(labs.shape[0]) % num_shots < test_shots
            train_imgs, train_labs = imgs[~sel], labs[~sel]
            test_imgs, test_labs = imgs[sel], labs[sel]
            return TensorDataset(train_imgs, train_labs), TensorDataset(test_imgs, test_labs)
        else:
            return TensorDataset(imgs, labs)
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    @staticmethod
    def make_gen(task, batch_size, num_batches):
        task = list(task)
        X, y = zip(*task)
        X = torch.cat(X).unsqueeze(1)
        y = torch.LongTensor(y)
        
        batch_count = 0
        sel = []
        while True:
            for idx in np.random.permutation(X.shape[0]):
                sel.append(idx)
                if len(sel) == batch_size:
                    sel = torch.LongTensor(sel)
                    yield X[sel], y[sel]
                    
                    sel = []
                    batch_count += 1
                    if batch_count == num_batches:
                        return
    
    @property
    def num_classes(self):
        return len(self.classes)