# import Measure


class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self):
		super(Evaluator, self).__init__()
		print('evaluator...')
		self.bleu = BLEU()
		self.classifier = Classifier()

	def predictLine(self):
		pass

	def dumpOuts(self):
		pass

	def predict(self, loader, net):
		pass

	def evaluateMetrics(self, preds):
		pass

		# evaluate
	def evaluate(self, loader, net):
		pass