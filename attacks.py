import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys

class FGSM(object):
	def __init__(self, epsilon=0.25):
		self.epsilon = epsilon

	def attack(self, inputs, labels, model, *args):
		
		adv_inputs = inputs.data + self.epsilon * torch.sign(inputs.grad.data)
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
		adv_inputs = Variable(adv_inputs, requires_grad=False)

		predictions = torch.max(model(adv_inputs).data, 1)[1].cpu().numpy()
		num_unperturbed = (predictions == labels.data.cpu().numpy()).sum()
		adv_inputs = [ adv_inputs[i] for i in range(inputs.size(0)) ]

		return adv_inputs, predictions, num_unperturbed




class DCGAN(object):
	def __init__(self, num_channels=3, ngf=100, cg=0.01, learning_rate=1e-4, train_adv=False):
		"""
		Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
		create adversarial attacks.

		- num_channels is the number of channels in the input
		- ngf is size of the conv layers
		- cg is the normalization constant for perturbation (higher means encourage smaller perturbation)
		- learning_rate is learning rate for generator optimizer
		- train_adv is whether the model being attacked should be trained adversarially
		"""
		self.generator = nn.Sequential(
			# input is (nc) x 32 x 32
			nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout2d(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout2d(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 1, 1, 0, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 3 x 32 x 32
			nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=False),
			nn.Tanh()
		)

		self.cuda = torch.cuda.is_available()

		if self.cuda:
			self.generator.cuda()
			self.generator = torch.nn.DataParallel(self.generator, device_ids=range(torch.cuda.device_count()))
			cudnn.benchmark = True

		self.criterion = nn.CrossEntropyLoss()
		self.cg = cg
		self.optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
		self.train_adv = train_adv

	def attack(self, inputs, labels, model, model_optimizer=None, epsilon=1.0, *args):
		perturbation = self.generator(Variable(inputs.data))
		adv_inputs = inputs + epsilon*perturbation
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)

		predictions = model(adv_inputs)
		# exponent value (p) in the norm needs to be 4 or higher! IMPORTANT!
		loss = torch.exp(-1 * self.criterion(predictions, labels)) + self.cg * (torch.norm(perturbation, 4))
                #print (torch.norm(perturbation, 2) ** 1).data[0]

		# optimizer step for the generator
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		# optimizer step for the discriminator (if training adversarially)
		if self.train_adv and model_optimizer:
			discriminator_loss = self.criterion(predictions, labels)
			model_optimizer.zero_grad()
			discriminator_loss.backward()
			model_optimizer.step()

		# print perturbation.data.mean(), inputs.data.mean()
		# print loss.data[0], torch.norm(perturbation, 2).data[0], torch.norm(inputs, 2).data[0]

		# prep the predictions and inputs to be returned
		predictions = torch.max(predictions.data, 1)[1].cpu().numpy()
		num_unperturbed = (predictions == labels.data.cpu().numpy()).sum()
		adv_inputs = [ adv_inputs[i] for i in range(inputs.size(0)) ]

		return adv_inputs, predictions, num_unperturbed

	def perturb(self, inputs, epsilon=1.0):
		perturbation = self.generator(Variable(inputs.data))
		adv_inputs = inputs + epsilon*perturbation
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
		return adv_inputs

	def save(self, fn):
		torch.save(self.generator.state_dict(), fn)

	def load(self, fn):
		self.generator.load_state_dict(torch.load(fn))
