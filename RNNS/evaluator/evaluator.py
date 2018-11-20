# import Measure
import os
import pickle
import torch
import tqdm
from utils import makeInp, seq_collate

class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self,config,expPath):
		super(Evaluator, self).__init__()
		print('evaluator...')
		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.ind2wordDict = self._buildInd2Word(self.wordDict)
		self.savePath = expPath
		os.makedirs(self.savePath, exist_ok=True)

	def _buildInd2Word(self,wordDict):
		vocabs = sorted(self.wordDict.items(), key=lambda x: x[1])
		vocabs = [vocabs[i][0] for i in range(len(vocabs))]
		return vocabs

	def ind2word(self,sequence):
		if not isinstance(sequence,torch.Tensor) or (sequence.dim()>0):
			return [self.ind2word(sequence[i]) for i in range(len(sequence))]
		else:
			return self.ind2wordDict[sequence]

	def predictLine(self, ld, net, line, style):
		net.eval()
		batch = ld.dataset.loadLine(line, style)
		inp = seq_collate([batch])
		# predict
		out = net(inp)
		# ind2word
		pred = out[2]['sequence'][:out[2]['length'][0]]
		pred = self.ind2word(pred)
		pred = [pred[i][0][0] for i in range(len(pred))]
		if '<unk>' in pred:
			pred.remove('<unk>')
		if '<m_end>' in pred:
			pred.remove('<m_end>')
		if '@@END@@' in pred:
			pred.remove('@@END@@')
		return ' '.join(pred)

	def dumpOuts(self, predList):
		# each pred take 3 lines
		# pred #
		# sentence: ...
		# brkSentence, marker: ...
		# pred: ...
		with open(os.path.join(self.savePath,'preds.outs'),'w') as f:
			cnt = 0
			for ent in predList:
				f.write('# '+str(cnt)+'\n')
				sent = 'sentence:'+' '.join(ent[0][0])+'\n'
				f.write(sent)
				brk = 'brk_sentence:'+' '.join(ent[1][0])+'\n'
				f.write(brk)
				mk = 'marker:'+' '.join(ent[2][0])+'\n'
				f.write(mk)
				pred = [ent[3][i][0][0] for i in range(len(ent[3]))]
				pred = 'pred: '+' '.join(pred)+'\n'
				f.write(pred)
				cnt += 1

	def predict(self, ld, net):
		net.eval()
		ld = iter(ld.ldDevEval)
		predList = [] #([brkSent],[marker],[pred])
		with torch.set_grad_enabled(False):
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar:
				inputs = makeInp(next(ld))
				outputs = net(inputs)

				brkSent = inputs['brk_sentence']
				marker = inputs['marker']
				sentence = inputs['sentence']
				pred = outputs[2]['sequence'][:outputs[2]['length'][0]]

				predList.append([sentence,brkSent,marker,pred])
		predList_w = self.ind2word(predList)
		self.dumpOuts(predList_w)
		return predList

	def evaluateMetrics(self, preds):
		pass

		# evaluate
	# def evaluate(self, ld, net):
	# 	predList = self.predict(ld, net)
	# 	BLEU, Acc = self.evaluateMetrics(predList)
	# 	return BLEU, Acc