# import Measure


class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self):
		super(Evaluator, self).__init__()
		print('evaluator...')
		self.bleu = BLEU()
		self.classifier = Classifier()
		self.ind2wordDict = None

	def ind2word(self,sequence):

		pass

	def predictLine(self, ld, net, line):
		# get marker

		# retrieve

		# word2ind

		# predict

		# ind2word

		pass

	def dumpOuts(self):
		# each pred take 3 lines
		# pred #
		# sentence: ...
		# brkSentence, marker: ...
		# pred: ...

		pass

	def predict(self, ld, net):
		ld = iter(ld)
		predList = [] #([brkSent],[marker],[pred])
		with torch.set_grad_enabled(False):
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar:
				inputs = next(ld)
				outputs = net(inputs)

				brkSent = inputs['brk_sentence']
				marker = inputs['marker']
				sentence = inputs['sentence']
				pred = outputs[2]['sequence']

				predList.append((sentence,brkSent,marker,pred))
		predList_w = self.ind2word(predList)
		self.dumpOuts(predList_w)
		return predList

	def evaluateMetrics(self, preds):
		pass

		# evaluate
	def evaluate(self, ld, net):
		predList = self.predict(ld, net)
		BLEU, Acc = self.evaluateMetrics(predList)
		return BLEU, Acc