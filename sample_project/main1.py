from __future__ import print_function
import os
import time
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import models
from utils import Datasets
from utils.params import Params
from utils.plotting import plot_training


warnings.filterwarnings("ignore")

    
def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "--write_data",
        required = False,
        default=False,
                help="Set to true to write_data."
        )
    args = parser.parse_args()
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    # Check if a GPU is available and use it if so. 
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net()
    # Send the model to the chosen device. 
    # To use multiple GPUs
    # model = nn.DataParallel(model)
    model.to(device)
    # Grap your training and validation functions for your network.
    train = model_module.train
    val = model_module.val
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    # Write data if specified in command line arguments.
    path = "images/DeepLearningProjectGroup4/sample_project/models/"

    if args.write_data:

        filelist = os.listdir(path)  #包含了path下所有文件的文件名
        for file in filelist:
            data = data.append(pd.read_csv(path + file))


        # data = pd.read_csv('fashionmnist/train_full_image_data.csv')



        test_data = pd.read_csv('fashionmnist/test_full_image_data.csv')
        val_split = round(data.shape[0]*0.2)
        data = shuffle(data)
        train_data = data.iloc[val_split:]
        val_data = data.iloc[:val_split]
        train_data.to_csv(os.path.join(params.data_dir, "train.csv"), index=False)
        val_data.to_csv(os.path.join(params.data_dir, "val.csv"), index=False)
        test_data.to_csv(os.path.join(params.data_dir, "test.csv"), index=False)

    # This is useful if you have multiple custom datasets defined. 
    Dataset = getattr(Datasets, params.dataset_class)
    train_data = Dataset(params.data_dir,"train.csv", flatten=params.flatten)
    val_data = Dataset(params.data_dir,"val.csv", flatten=params.flatten)
    train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    if not os.path.exists("figs"): os.makedirs("figs")

    val_accs = []
    val_losses = []
    val_rec = []
    val_pre = []
    val_f1 = []
    val_auc = []
    
    train_losses = []
    train_accs = []
    train_rec = []
    train_pre = []
    train_f1 = []
    train_auc = []
    
    for epoch in range(1, params.num_epochs + 1):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train(model, device, train_loader, optimizer)
        # Evaluate on both the training and validation set. 
        train_loss, train_acc, train_recall, train_precision, train_f1s,train_fprs, train_tprs, train_threshold, train_aucs = val(model, device, train_loader)
        val_loss, val_acc, val_recall, val_precision, val_f1s,val_fprs, val_tprs, val_threshold, val_aucs= val(model, device, val_loader)
        # Collect some data for logging purposes. 
        train_losses.append(float(train_loss))
        train_accs.append(train_acc)
        train_rec.append(train_recall)
        train_pre.append(train_precision)
        train_f1.append(train_f1s)
        train_auc.append(train_aucs)
        
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)
        val_rec.append(val_recall)
        val_pre.append(val_precision)
        val_f1.append(val_f1s)
        val_auc.append(val_aucs)

        
        

        print('\ntrain Loss: {:.3f}    train acc: {:.3f}    train recall: {:.3f}    train precision: {:.3f}    train f1: {:.3f}    train_auc: {:.3f}'.format(train_loss, train_acc, train_recall, train_precision,train_f1s,train_aucs))


        print('val Loss: {:.3f}    val acc: {:.3f}    val recall: {:.3f}    val precision: {:.3f}    val f1: {:.3f}    val_auc: {:.3f}\n'.format(val_loss, val_acc, val_recall, val_precision, val_f1s, val_aucs))


        # Here is a simply plot for monitoring training. 
        # Clear plot each epoch 
        fig = plot_training(train_losses, train_accs, val_losses, val_accs)
        
        fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name)))
        
        # Save model every few epochs (or even more often if you have the disk space).
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(params.checkpoint_dir,"checkpoint_{}_epoch_{}".format(args.model_name,epoch)))
    # Some log information to help you keep track of your model information. 
    logs ={
        "model": args.model_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "train_recall": train_recall,
        "train_precision": train_precision,
        "train_f1": train_f1,
        "train_auc": train_auc,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_recall":val_recall,
        "val_precision":val_precision,
        "val_f1":val_f1,
        "val_auc": val_auc,
        "best_val_epoch": int(np.argmax(val_accs)+1),
        "model": args.model_name,
        "lr": params.lr,
        "batch_size":params.batch_size
    }

    with open(os.path.join(params.log_dir,"{}_{}.json".format(args.model_name,  start_time)), 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    main()
