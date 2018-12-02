import numpy as np
import os
import torch
import tqdm
from utils import makeInp

class Trainer(object):
	"""Manage training process with given configurations."""
	def __init__(self, config, savePath):
		super(Trainer, self).__init__()
		print('trainer...')
		self.lr = config['lr']
		self.savePath = savePath
		# os.makedirs(self.savePath, exist_ok=True)

	def adjust_learning_rate(self, optimizer, epoch):
		"""Sets the learning rate to the initial LR decayed by 2 every 50 epochs
		
		Args:
			optimizer: An instance of torch.optim.Optimizer.
				optimizer.param_group Specifies what Tensors should be optimized on.
			epoch: Integer, the currect training epoch
		"""
		lr = self.lr * (0.5 ** (epoch // 50))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def devLoss(self, ld, net, crit):
		"""Calculate the loss on validate set.

		Args:
			ld: A Dataloader containing the validate set.
			net: To model to evaluate
			crit: The module that calculate loss based on output sequences with multiple methods
		Returns:
			devLoss: The average loss on dev set.
		"""
		net.eval()
		ld = iter(ld)
		numIters = len(ld)
		devLoss = np.zeros(numIters)
		with torch.set_grad_enabled(False):
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar:
				inputs = makeInp(next(ld))
				outputs = net(inputs)
				loss = crit(outputs,inputs)
				devLoss[itr] = loss
				qdar.set_postfix(loss=str(np.round(loss.cpu().detach().numpy(),3)))
		devLoss = devLoss.mean()
		print('Average loss on dev set: '+str(devLoss))
		return devLoss

	def saveNet(self,net,isBest=False):
		"""Save the state_dict of model.
		See https://pytorch.org/tutorials/beginner/saving_loading_models.html

		Args:
			net: The model to save.
			isBest: Whether the model to be saved is the best model so far.
				If so, the model will be saved to bestmodel.pth.tar, otherwise checkpoint.pth.tar
		"""
		fileName = 'bestmodel.pth.tar' if isBest else 'checkpoint.pth.tar' 
		filePath = os.path.join(self.savePath, fileName)
		# create directory recursively. No OSError is raised if the target directory already exists
		os.makedirs(self.savePath, exist_ok=True) 
		torch.save({'state_dict': net.state_dict()},filePath)
		if isBest:
			print('>>> Saving best model...')
		else:
			print('Saving model...')
		
	def train(self, loader, net, crit, evaluator, config):
		"""Train the model. 
		
		The model will be saved after each epoch and the best one will be saved seperately.
		Args:
			loader: A LoadHandler instance, contains the loaded dataset.
			net: The main Seq2seq model to use.
			crit: The module that calculate loss based on output sequences with multiple methods
			evaluator: An Evaluator instance encapsulted the calculation of BLEU, 
				sentiment classify accuracy and language model loss.
			config: Specified configurations of the trainer.
		"""
		print('start to train...')
		
		# Specify which parameters to optimize.
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.lr)
		# train
		minLoss = float('inf')
		epoch = config['opt'].epoch
		while True:
			print('epoch: '+str(epoch))
			# Switch the model into train mode. Some layers such as BatchNorm and Dropout will be affected.
			net.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			ld = iter(loader.ldTrain) # Construct an iterator on the DataLoader instance.
			numIters = len(ld) # total number of samples
			qdar = tqdm.tqdm(range(numIters),
									total= numIters, # manually specify the expected iterations
									ascii=True) # use 1-9# to replace unicode(smooth block)
			for itr in qdar: 
				inputs = makeInp(next(ld)) # Move tensors onto GPU
				# TODO: training logic
				# Context manager disabling gradient computation locally to reduce memory use.
				with torch.set_grad_enabled(True):
					outputs = net(inputs, teacher_forcing_ratio=max((1-epoch/10),0))
					loss = crit(outputs,inputs)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				qdar.set_postfix(loss=str(np.round(loss.cpu().detach().numpy(),3)))

			# save model
			self.saveNet(net)
			# loss on dev	
			devLoss = self.devLoss(loader.ldDev,net,crit)
			# eval on dev
			# BLEU, Acc = evaluator.evaluate(loader.ldDevEval, net)
			# save best model
			if devLoss < minLoss:
				minLoss = devLoss
				self.saveNet(net,isBest=True)
			epoch += 1
