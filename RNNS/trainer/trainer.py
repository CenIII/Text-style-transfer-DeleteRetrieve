import numpy as np
import os
import torch
import tqdm

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, config, savePath):
		super(Trainer, self).__init__()
		print('trainer...')
		self.lr = config['lr']
		self.savePath = savePath
		os.makedirs(self.savePath, exist_ok=True)

	def devLoss(self, ld, net, crit):
		ld = iter(ld)
		devLoss = np.zeros(len(ld))
		with torch.set_grad_enabled(False):
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar:
				inputs = next(ld)
				outputs = net(inputs)
				loss = crit(outputs,inputs)
				devLoss[itr] = loss
				qdar.set_postfix(loss=str(np.round(loss.detach().numpy(),2)))
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
		while True:
			ld = iter(loader.ldTrain)
			numIters = 10#len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar: #range(len(ld)):
				inputs = next(ld)
				# print(">>>>>>>>inputs: "+str(inputs))
				with torch.set_grad_enabled(True):
					outputs = net(inputs)
					loss = crit(outputs,inputs)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				qdar.set_postfix(loss=str(np.round(loss.detach().numpy(),2)))

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
