#!/usr/bin/env python

"""
    main.py
"""

import sys
import json
import argparse
import numpy as np
from time import time
from copy import deepcopy
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms

from basenet import BaseNet
from basenet.helpers import set_seeds, to_numpy

from omniglot import Omniglot, OmniglotTaskWrapper

# --
# Helpers

def accuracy(pred, act):
    return float((to_numpy(pred).argmax(axis=-1) == to_numpy(act)).mean())

def copy_model(model):
    # !! ugly hack to get around device serialization error
    del model.device
    new_model = deepcopy(model)
    model.device = cuda
    new_model.device = cuda
    return model, new_model

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--train-shots', type=int, default=10)
    parser.add_argument('--inner-batch-size', type=int, default=10)
    parser.add_argument('--inner-iters', type=int, default=5)
    parser.add_argument('--eval-shots', type=int, default=1)
    parser.add_argument('--eval-inner-batch-size', type=int, default=5)
    parser.add_argument('--eval-inner-iters', type=int, default=50)
    parser.add_argument('--meta-batch-size', type=int, default=5)
    parser.add_argument('--meta-iters', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--meta-step-size', type=float, default=1.0)
    parser.add_argument('--meta-step-size-final', type=float, default=0.0)
    parser.add_argument('--eval-interval', type=int, default=10)
    
    parser.add_argument('--num-train-classes', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

# --
# Model definition

class OmniglotModel(BaseNet):
    def __init__(self, num_classes, in_channels=1, hidden_channels=64):
        super().__init__(loss_fn=F.cross_entropy)
        
        self.conv = []
        for _ in range(4):
            self.conv += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=[2, 2], padding=1),
                nn.BatchNorm2d(hidden_channels, track_running_stats=False),
                nn.ReLU()
            ]
            in_channels = hidden_channels
        
        self.conv = nn.Sequential(*self.conv)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
    def stash_weights(self):
        self._old_state = deepcopy(self.state_dict())
    
    def unstash_weights(self):
        self.load_state_dict(deepcopy(self._old_state))
    
    def meta_step(self, new_states, meta_step_size):
        assert self._old_state is not None, "OmniglotModel._old_state is None"
        
        tmp = {}
        for k in self._old_state.keys():
            old_var = self._old_state[k]
            new_var = torch.mean(torch.stack([s[k] for s in new_states]), dim=0)
            tmp[k] = (1 - meta_step_size) * old_var + meta_step_size * new_var.clone()
        
        self.load_state_dict(deepcopy(tmp))
        del self._old_state

# --
# Run

args = parse_args()
set_seeds(args.seed)

# --
# IO

dataset = Omniglot(root='./data/omniglot')

train_classes, test_classes = train_test_split(dataset._classes, train_size=args.num_train_classes)

train_taskset = OmniglotTaskWrapper(dataset, classes=train_classes, rotation=False)
test_taskset  = OmniglotTaskWrapper(dataset, classes=test_classes, rotation=False)

cuda = torch.device('cuda')
model = OmniglotModel(num_classes=args.num_classes).to(cuda)

model.init_optimizer(
    opt=torch.optim.Adam,
    params=model.parameters(),
    lr=args.lr,
    betas=[0.0, 0.999],
)

t = time()
for meta_iter in range(args.meta_iters):
    
    # --
    # Training loop
    
    model.stash_weights()
    new_states = []
    for _ in range(args.meta_batch_size):
        # Reptile steps
        task_gen = OmniglotTaskWrapper.make_gen(
            train_taskset.sample_task(args.num_classes, args.train_shots),
            batch_size=args.inner_batch_size,
            num_batches=args.inner_iters
        )
        
        for batch_idx, (data, target) in enumerate(task_gen):
            output, _ = model.train_batch(data, target)
        
        new_states.append(deepcopy(model.state_dict()))
        model.unstash_weights()
    
    # Update weights
    frac_done = meta_iter / args.meta_iters
    cur_meta_step_size = frac_done * args.meta_step_size_final + (1 - frac_done) * args.meta_step_size
    model.meta_step(new_states, cur_meta_step_size)
    
    # --
    # Eval loop
    
    if not meta_iter % args.eval_interval:
        res = OrderedDict([
            ("meta_iter", meta_iter),
            ("elapsed_time", time() - t),
        ])
        
        for taskname, taskset in [("train", train_taskset), ("test", test_taskset)]:
            # Copy model
            model, eval_model = copy_model(model)
            
            # Reptile steps
            task, test_task = taskset.sample_task(args.num_classes, args.eval_shots + 1, test_shots=1)
            task_gen = OmniglotTaskWrapper.make_gen(
                task,
                batch_size=args.eval_inner_batch_size,
                num_batches=args.eval_inner_iters
            )
            
            for batch_idx, (data, target) in enumerate(task_gen):
                output, _ = eval_model.train_batch(data, target)
            
            # Evaluate on holdout (transductive)
            X_test = torch.cat([t[0] for t in test_task]).unsqueeze(1)
            y_test = torch.LongTensor([t[1] for t in test_task])
            preds, _ = eval_model.eval_batch(X_test, y_test)
            res.update(OrderedDict([
                (("%s_acc" % taskname), accuracy(preds, y_test)),
            ]))
            
            # Reset model
            del eval_model
        
        print(json.dumps(res))
        sys.stdout.flush()