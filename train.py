#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.vgg import VGG
from models.lenet import LeNet
import models.googlenet as googlenet
import attacks
import numpy as np
import os


# In[ ]:


use_cuda = torch.cuda.is_available()
i = 0 # Epsilon counter for logging


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


def train(model, optimizer, criterion, trainloader, architecture, attacker=None, num_epochs=25, freq=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        total, correct, correct_adv, total_adv  = 0.0, 0.0, 0.0, 1.0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = Variable((inputs.cuda() if use_cuda else inputs), requires_grad=True)
            labels = Variable((labels.cuda() if use_cuda else labels), requires_grad=False)

            y_hat = model(inputs)
            loss = criterion(y_hat, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = y_hat.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum().item()

			# print statistics
            running_loss += loss.item()

            if attacker :
                adv_inputs, adv_labels, num_unperturbed = attacker.attack(inputs, labels, model, optimizer)
                correct_adv += num_unperturbed.item()
                total_adv += labels.size(0)

            if (i+1) % freq == 0:
                print ('[%s: %d, %5d] loss: %.4f' % (architecture,epoch + 1, i + 1, running_loss / 2), correct/total, correct_adv/total_adv)
                running_loss = 0.0


    return correct/total, correct_adv/total_adv


# In[ ]:


def prep(model):
	if model and use_cuda:
		model.cuda()
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True
	return model


# In[ ]:


trainloader, testloader = load_cifar()
criterion = nn.CrossEntropyLoss()
do_train = True
# use the models that you want to train. If you have a powerful enough GPU, you can fit models like GoogLeNet and DenseNet121
# model, model name , epochs
architectures = [
		(LeNet, 'lenet', 1),
        (VGG, 'VGG16', 50),
		(googlenet.GoogLeNet, 'googlenet', 200),	
	]

for init_func, name, epochs in architectures:
	for tr_adv in [False,True]:
		print (name, tr_adv)
		model = prep(init_func())
		attacker = attacks.DCGAN(train_adv=tr_adv)
		#attacker = None

		optimizer = optim.Adam(model.parameters(), lr=1e-4)
		if do_train:
			train_acc, train_adv_acc = train(model, optimizer,criterion, trainloader, name, attacker, num_epochs=epochs)
			suffix = '_AT' if tr_adv else ''
			if attacker :
				attacker.save('attacker/{0}{1}_attacker.pth'.format(name, suffix))
			torch.save(model.state_dict(),'target/{0}{1}.pth'.format(name, suffix))
		


# In[ ]:




