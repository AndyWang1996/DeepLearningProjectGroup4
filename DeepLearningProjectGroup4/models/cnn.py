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
'''Part of format and full model from pytorch examples repo: https://github.com/pytorch/examples/blob/master/mnist/main.py'''
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        nn = self.conv1(x)
        nn = F.relu(nn)
        nn = self.conv2(nn)
        nn = F.max_pool2d(nn, 2)
        nn = self.dropout1(nn)
        nn = torch.flatten(nn, 1)
        nn = self.fc1(nn)
        nn = F.relu(nn)
        nn = self.dropout2(nn)
        nn = self.fc2(nn)
        output = F.softmax(nn, -1)
        return output

def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            output = model(data)
            output = output.cuda(device)
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
    model = model.cuda(device)
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            output = output.cuda(device)
            label = label.cuda(device)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            # print(type(cost(output, label)))
            # print(type(cost(output, label)))
            losses.append(float(cost(output, label)))
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc

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
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc

