#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.vgg import VGG
from models.lenet import LeNet
import models.alexnet as alexnet
import models.googlenet as googlenet
import attacks
import numpy as np
import pandas as pd
from collections import OrderedDict
import os


# In[ ]:


i = 0


# In[ ]:


use_cuda = torch.cuda.is_available()


# In[ ]:


def load_cifar():
	print('==> Preparing data..')
	transform_train = transforms.Compose([
    	transforms.RandomCrop(32, padding=4),
    	transforms.RandomHorizontalFlip(),
    	transforms.ToTensor(),
    	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	transform_test = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader, testloader


# In[ ]:


def test(model, criterion, testloader, attacker):
    
    correct, correct_adv, total = 0.0, 0.0, 0.0

    for data in testloader:
        inputs, labels = data
        inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
        labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

        y_hat = model(inputs)
        loss = criterion(y_hat, labels)
        loss.backward()

        predicted = torch.max(y_hat.data, 1)[1]
        correct += predicted.eq(labels.data).sum().item()

        adv_inputs, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model)
        correct_adv += num_unperturbed.item()

        total += labels.size(0)

       

    return correct/total, correct_adv/total


# In[ ]:


def test_epsilon(model, criterion, testloader, attacker, model_name, att_name):
    epsilons = [0.0,0.2,0.4,0.6,0.8,1.0]
    resultsDF = pd.DataFrame(columns=('Model','Attacker','Epsilon','Test_acc','Test_att_acc'))
    name = model_name
    global i
    for epsilon in epsilons:
        correct, correct_adv, total = 0.0, 0.0, 0.0
        for data in testloader:
            inputs, labels = data
            inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
            labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)
            
            y_hat = model(inputs)
            loss = criterion(y_hat, labels)
            loss.backward()

            predicted = torch.max(y_hat.data, 1)[1]
            correct += predicted.eq(labels.data).sum().item()

            _, adv_labels, num_unperturbed = attacker.attack(inputs,labels, model, epsilon)
            adv_inputs  = attacker.perturb(inputs,epsilon=epsilon)
            correct_adv += num_unperturbed.item()

            total += labels.size(0)

        fake = adv_inputs
        samples_name = 'images/'+name+str(epsilon) +'_samples.png'
        vutils.save_image(fake.data, samples_name, normalize = True)
        print('Test Acc Acc: %.4f | Test Attacked Acc: %.4f | Epsilon: %.2f'  % (100.*correct/total, 100.*correct_adv/total,epsilon))
        resultsDF.loc[i] = [model_name,att_name,epsilon,correct/total,correct_adv/total]
        i = i + 1
    resultsDF.to_csv('DCGAN_attack_results.csv',mode='a',header=(not os.path.exists('DCGAN_attack_results.csv')))


    return correct/total, correct_adv/total


# In[ ]:


weights = {
    'lenet':
    ['target/lenet.pth','attacker/lenet_attacker.pth'],
    'VGG16': ['target/VGG16.pth','attacker/VGG16_attacker.pth'],
    'googlenet':
    ['target/googlenet.pth','attacker/googlenet_attacker.pth']
    
    }


# In[ ]:


criterion = nn.CrossEntropyLoss()
for m in weights.keys():
    for n in weights.keys():
        if (m==n): # white box attack
            target_architectures = {
                                'lenet': LeNet,
                                'VGG16': VGG,    
                                'googlenet': googlenet.GoogLeNet
                            }
            _,testloader = load_cifar()
            model = target_architectures[n]()
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model.load_state_dict(torch.load(weights[n][0]))
            attacker = attacks.DCGAN(train_adv=False)
            attacker.load(weights[m][1])
            print("Classifier (target): " +  n + ", Generator (attacker): " + m)
            test_acc, test_adv_acc = test_epsilon(model, criterion, testloader, attacker,n,m)

            
            


# In[ ]:




