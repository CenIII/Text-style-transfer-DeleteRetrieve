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
		lr = self.lr * (0.5 ** (epoch // 30))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def devLoss(self, ld, net, crit):
		net.eval()
		ld = iter(ld)
		devLoss = np.zeros(len(ld))
		with torch.set_grad_enabled(False):
			numIters = len(ld)
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
		
	def train(self, loader, net, crit, evaluator):
		print('start to train...')
		
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.lr)
		# train
		minLoss = float('inf')
		epoch = 0.
		while True:
			net.train()
			# self.adjust_learning_rate(self.optimizer, epoch)
			ld = iter(loader.ldTrain)
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar: #range(len(ld)):
				inputs = makeInp(next(ld))
				with torch.set_grad_enabled(True):
					outputs = net(inputs, teacher_forcing_ratio=max((0.2-epoch/2),0))
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
