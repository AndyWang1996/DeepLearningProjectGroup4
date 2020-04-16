from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

'''
Part of format and full model from pytorch examples repo: 
https://github.com/pytorch/examples/blob/master/mnist/main.py
'''


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 'F', 'FC1', 'FC2', 'FC3']
        modulelist = []
        channel = 1

        for layer in layers:
            if layer == 'M':  # 2*2; 1/4 pooling (100*100 => 50*50 => 25*25 => 12*12 => 6*6)
                modulelist.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif layer == 'F':  # FLATTEN
                modulelist.append(nn.Flatten())
            elif layer == 'FC1':  # Full connection 1 in = 6*6*512 -> 1024 out
                modulelist.append(nn.Linear(6 * 6 * 512, 1024))
                modulelist.append(nn.ReLU(inplace=True))
            elif layer == 'FC2':  # Full connection 2 in = 1024 -> 256 out
                modulelist.append(nn.Linear(1024, 256))
                modulelist.append(nn.ReLU(inplace=True))
            elif layer == 'FC3':  # Full connection 3 in = 256 -> 2 out
                modulelist.append(nn.Linear(256, 2))
            else:  # Conv2d inout = in_channel; output = out_channel
                modulelist.append(nn.Conv2d(in_channels=channel, out_channels=layer, kernel_size=3, padding=1))
                modulelist.append(nn.ReLU(inplace=True))
                channel = layer
        self.vgg = nn.ModuleList(modulelist)

    def forward(self, x):
        x = x
        i = 1
        for module in self.vgg:
            if i == 30:
                x = module(x)
                x = F.softmax(x, -1)
            else:
                x = module(x)
                i = i + 1

        output = x
        return output


def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)


def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
    # Need this line for things like dropout etc.  
    model.eval()
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.clone()
            label = label.to(device)
            output = model(data)
            output = output.to(device)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label))
    loss = torch.mean(torch.stack(losses))
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average='binary')
    precision = precision_score(targets, preds, average='binary')
    f1 = f1_score(targets, preds, average='binary')
    fpr, tpr, thresholds = roc_curve(targets, preds, pos_label=1)
    au = auc(fpr, tpr)

    return loss, acc, recall, precision, f1, fpr, tpr, thresholds, au


def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())

    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc
