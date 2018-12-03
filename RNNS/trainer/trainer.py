import numpy as np
import os
import torch
import tqdm
from utils import makeInp

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, config, savePath):
		super(Trainer, self).__init__()
		print('trainer...')
		self.lr = config['lr']
		self.savePath = savePath
		os.makedirs(self.savePath, exist_ok=True)

	def adjust_learning_rate(self, optimizer, epoch):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr = self.lr * (0.5 ** (epoch // 10))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def devLoss(self, ld, net, crit):
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
		fileName = 'bestmodel.pth.tar' if isBest else 'checkpoint.pth.tar' 
		filePath = os.path.join(self.savePath, fileName)
		os.makedirs(self.savePath, exist_ok=True)
		torch.save({'state_dict': net.state_dict()},filePath)
		if isBest:
			print('>>> Saving best model...')
		else:
			print('Saving model...')
		
	def train(self, loader, net, crit, evaluator, config):
		print('start to train...')
		
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.lr)
		# train
		minLoss = float('inf')
		epoch = config['opt'].epoch
		while True:
			print('epoch: '+str(epoch))
			net.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			ld = iter(loader.ldTrain)
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar: 
				inputs = makeInp(next(ld))
				with torch.set_grad_enabled(True):
					outputs = net(inputs, teacher_forcing_ratio=max((1-epoch/10),0))
					# import pdb;pdb.set_trace()
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

class LangTrainer(object):
	"""docstring for Trainer"""
	def __init__(self, config, savePath):
		super(LangTrainer, self).__init__()
		print('LM trainer...')
		self.lr = config['lm_lr']
		self.savePath = savePath
		os.makedirs(self.savePath, exist_ok=True)
		self.celoss = torch.nn.CrossEntropyLoss()

	def adjust_learning_rate(self, optimizer, epoch):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr = self.lr * (0.5 ** (epoch // 10))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def devLoss(self, ld, net):
		def getLabel(x):
			return torch.cat((x[:,1:],torch.zeros_like(x)[:,0:1]),1)
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
				labels = getLabel(inputs['sentence'])
				outputs = net(inputs)
				loss = self.celoss(outputs.view(-1,outputs.shape[2]),labels.view(-1))
				devLoss[itr] = loss
				qdar.set_postfix(loss=str(np.round(loss.cpu().detach().numpy(),3)))
		devLoss = devLoss.mean()
		print('Average loss on dev set: '+str(devLoss))
		return devLoss

	def saveNet(self,net,isBest=False,isStyle=0):
		# fileName = 'lm_bestmodel.pth.tar' if isBest else 'lm_checkpoint.pth.tar' 
		fileName = 'lm_bestmodel' if isBest else 'lm_checkpoint' 
		fileName += '_neg' if isStyle==0 else '_pos'
		fileName += '.pth.tar'

		filePath = os.path.join(self.savePath, fileName)
		os.makedirs(self.savePath, exist_ok=True)
		torch.save({'state_dict': net.state_dict()},filePath)
		if isBest:
			print('>>> Saving best model...')
		else:
			print('Saving model...')

	def checkPath(self,net,isBest=True,isStyle=0):
		# fileName = 'lm_bestmodel.pth.tar' if isBest else 'lm_checkpoint.pth.tar' 
		fileName = 'lm_bestmodel' if isBest else 'lm_checkpoint' 
		fileName += '_neg' if isStyle==0 else '_pos'
		fileName += '.pth.tar'

		filePath = os.path.join(self.savePath, fileName)
		print(filePath)
	def train(self, loader, net, config,isStyle=0):
		print('start to train language model...')
		def getLabel(x):
			# Shift a tensor and return
			# import pdb;pdb.set_trace()
			# n = x.numpy()
			# n_shift = np.roll(n,-1,axis=1) # move in the sequence_len dim
			# n_shift[:,-1] = 0
			# return torch.tensor(n_shift)
			
			return torch.cat((x[:,1:],torch.zeros_like(x)[:,0:1]),1)

			# return torch.from_numpy(n_shift)
		# def getLoss(pred,labels):
		# 	loss = torch.nn.CrossEntropyLoss()
		# 	x = pred.view(-1,pred.shape[2])
		# 	y = pred.view(-1)
		# 	# pdb.set_trace()
		# 	return loss(x,y)
		self.checkPath(net,isStyle=1)
		self.checkPath(net,isStyle=0)
		# import pdb;pdb.set_trace()
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.lr)
		# train
		minLoss = float('inf')
		epoch = config['opt'].epoch

		while epoch<10:
			print('epoch: '+str(epoch))
			net.train()
			self.adjust_learning_rate(self.optimizer, epoch)
			if isStyle == 0:
				ld = iter(loader.ldTrain_neg)
				print("Training negtive model")
			else:
				ld = iter(loader.ldTrain_pos)
				print("Training positive model.")
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar: 
				inputs = makeInp(next(ld))
				# import pdb;pdb.set_trace()
				with torch.set_grad_enabled(True):
					# import pdb;pdb.set_trace()
					labels = getLabel(inputs['sentence'])
					# labels.detach_()
					# import pdb;pdb.set_trace()
					outputs = net(inputs['sentence'],inputs['st_inp_lengths'])
					# import pdb;pdb.set_trace()
					loss = self.celoss(outputs.view(-1,outputs.shape[2]),labels.view(-1))
					import pdb;pdb.set_trace()
					loss = loss/torch.sum(inputs['st_inp_lengths']).type(loss.dtype)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				qdar.set_postfix(loss=str(np.round(loss.cpu().detach().numpy(),3)))

			# save model
			self.saveNet(net,isStyle=isStyle)
			# loss on dev	
			# TODO
			if isStyle == 0:
				devLoss = self.devLoss(loader.ldDev_neg,net)
			else:
				devLoss = self.devLoss(loader.ldDev_pos,net)
			# eval on dev
			# BLEU, Acc = evaluator.evaluate(loader.ldDevEval, net)
			# save best model
			if devLoss < minLoss:
				minLoss = devLoss
				self.saveNet(net,isStyle=isStyle,isBest=True)
			epoch += 1



