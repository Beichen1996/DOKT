'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
#import visdom
from tqdm import tqdm

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
from config import *
from data.sampler import SubsetSequentialSampler
from sys import argv

##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
]) 

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_train = CIFAR100('../data', train=True, download=True, transform=train_transform)
cifar10_test  = CIFAR100('../data', train=False, download=True, transform=test_transform)



iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss            = m_backbone_loss 

        loss.backward()
        optimizers['backbone'].step()
    return m_backbone_loss

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    cors = np.array([0]*100)
    tot = np.array([0]*100)
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(len(labels)):
                tot[labels[i]] +=1
                if(labels[i] == preds[i]):
                    cors[labels[i]] +=1
    
    return 100 * correct / total, cors / tot
#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    for epoch in range(num_epochs):
        
        losss = train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)
        schedulers['backbone'].step() 

        if epoch % 20 == 4:
            acc,_ = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}  one of losss: {:.3f}'.format(acc, best_acc, losss))
    print('>> Finished.')
    return best_acc



##
# Main
if __name__ == '__main__':
    vis = None
    plot_data = { }
    EPOCH = 360#330
    MILESTONES = [200, 300]
    #MILESTONES = [160]
    BATCH = 32
    version = int(argv[1])
    for trial in range(0,10):
        name =  './cifar100lt/' + str(version)+'/100lt_' + str(217*(trial+1)) + '.npy'
        print (name)
        indices = np.load( name).tolist()
        labeled_set = indices
        all_indices = set(np.arange(NUM_TRAIN))
        indices = list(range(NUM_TRAIN))
        unlabeled_set = np.setdiff1d(indices, labeled_set).tolist()
        
        train_loader = DataLoader(cifar10_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        resnet18    = resnet.ResNet18(num_classes=100).cuda()
        models      = {'backbone': resnet18}
        torch.backends.cudnn.benchmark = True

        print(  len(list(set(labeled_set)))  , len(list(set(unlabeled_set)))  )

        criterion      = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                momentum=MOMENTUM, weight_decay=WDECAY)
        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

        optimizers = {'backbone': optim_backbone}
        schedulers = {'backbone': sched_backbone}

        # Training and test
        acc = train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
        print('Label set size {}: Test acc {}'.format(len(labeled_set), acc))
