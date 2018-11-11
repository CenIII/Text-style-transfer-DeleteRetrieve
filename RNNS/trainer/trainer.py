import numpy as np
import os
import torch

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, config, savePath):
		super(Trainer, self).__init__()
		self.lr = config['lr']
		self.savePath = savePath #os.path.join(config['exp_path'], config['opt'].exp)
		os.makedirs(self.savePath, exist_ok=True)

	def devLoss(self, ld, net, crit):
		ld = iter(ld)
		devLoss = np.zeros(len(ld))
		with torch.set_grad_enabled(False):
			for itr in range(len(ld)):
				inputs = next(ld)
				outputs = net(inputs)
				loss = crit(outputs)
				devLoss[itr] = loss
		devLoss = devLoss.mean()
		print('Average loss on dev set: '+str(devLoss))
		return devLoss

	def saveNet(self,net,isBest=False):
		fileName = 'bestmodel.pth.tar' if isBest else 'checkpoint.pth.tar' 
		filePath = os.path.join(self.savePath, fileName)
		os.makedirs(self.savePath, exist_ok=True)
		torch.save({'state_dict': net.state_dict()})
		if isBest:
			print('>>> Saving best model...')
		else:
			print('Saving model...')
		
	def train(self, loader, net, crit, evaluator):
		self.optimizer = torch.optim.Adam(net.parameters(), self.lr)
		log = None
		dispText = {'loss1':0,'loss2':1} #TODO: ...
		# train
		maxAcc = 0
		while True:
			ld = iter(loader.ldTrain)
			for itr in range(len(ld)):
				inputs = next(ld)
				print(">>>>>>>>inputs: "+str(inputs))
				with torch.set_grad_enabled(True):
					outputs = net(inputs)
					loss = crit(outputs,inputs)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			# save model
			self.saveNet(net)
			# loss on dev	
			self.devLoss(loader.ldDev,net,crit)
			# eval on dev
			BLEU, Acc = evaluator.evaluate(loader.ldDevEval, net)

			# save best model
			if Acc>maxAcc:
				maxAcc = Acc

				self.saveNet(net,isBest=True)
