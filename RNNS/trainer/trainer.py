
class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, arg):
		super(Trainer, self).__init__()
		self.arg = arg
		

	def computeDevLoss(self):
		while True:
			dev_loss += ...
		
	def train(self, loader, net, crit, evaluator):

		# train
		for epoch in range():
			for itr in range():

				loss.backward()
			# loss on dev	
			self.computeDevLoss(loader.ldDev,net,crit)

			# eval on dev
			BLEU, Acc = evaluator.evaluate(loader.ldDevEval, net)

			# save best model

