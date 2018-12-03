# import Measure
import os
import pickle
import torch
import tqdm
from utils import makeInp, seq_collate
from .metrics import Metrics
from model import Classifier

class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self,config,expPath, config_all):
		super(Evaluator, self).__init__()
		print('evaluator...')
		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.ind2wordDict = self._buildInd2Word(self.wordDict)
		self.savePath = expPath
		os.makedirs(self.savePath, exist_ok=True)

		config_all["model"]["bidirectional"] = 0
		classifier_net = Classifier(**config_all["model"])
		
		self.metrics = Metrics(config_all["metric"]["classifier_weight_path"], config_all["metric"]["ref_file"], classifier_net, config_all["model"]["wordDict"])
		self.mode = config_all['opt'].mode

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
				# mk = 'marker:'+' '.join(ent[2][0])+'\n'
				# f.write(mk)
				pred = [ent[2][i][0][0] for i in range(len(ent[2]))]
				pred = 'pred: '+' '.join(pred)+'\n'
				f.write(pred)
				cnt += 1

	def predict(self, ld, net):
		net.eval()
		ld = ld.ldDevEval if self.mode=='val' else ld.ldTestEval
		ld = iter(ld)
		predList = [] #([brkSent],[marker],[pred])
		styleList = []
		with torch.set_grad_enabled(False):
			numIters = len(ld)
			qdar = tqdm.tqdm(range(numIters),
									total= numIters,
									ascii=True)
			for itr in qdar:
				inputs = makeInp(next(ld))
				outputs = net(inputs) # Result1, Result2
				outputs = outputs[0]
				brkSent = inputs['brk_sentence']
				# marker = inputs['marker']
				sentence = inputs['sentence']
				style = inputs['style']
				pred = outputs[2]['sequence'][:outputs[2]['length'][0]]

				predList.append([sentence,brkSent,pred])
				styleList.append(style)
		predList_w = self.ind2word(predList)		
		self.dumpOuts(predList_w)
		predList_w = self.constructSentence(predList_w)
		return predList_w, styleList
	
	def constructSentence(self, predList_w):
		results = []
		tags = ['<unk>', '<m_end>','@@START@@', '@@END@@']
		for sentence in predList_w:
			result_sentence = []
			sentence = sum(sum(sentence[2],[]),[])
			for word in sentence:
				if word not in tags:
					result_sentence.append(word)
				# elif (word == '<unk>'):
				# 	result_sentence += sentence[2][idx]
				# 	idx += 1
			results.append(result_sentence)
		return results

	def evaluateMetrics(self, preds):
		if self.mode == 'val':
			bleu = -1
		else:
			bleu = self.metrics.bleuMetrics(preds)
		acc = self.metrics.classifierMetrics(preds)
		return bleu, acc

	def evaluate(self, ld, net):
		predList_w, styleList = self.predict(ld, net)
		preds = {"positive":[],"negative":[]}
		for i in range(len(predList_w)):
			if styleList[i] == 1:
				key = "positive"
			else:
				key = "negative"
			preds[key].append(predList_w[i])
		bleu,acc = self.evaluateMetrics(preds)
		return bleu, acc
		

		# evaluate
	# def evaluate(self, ld, net):
	# 	predList = self.predict(ld, net)
	# 	BLEU, Acc = self.evaluateMetrics(predList)
	# 	return BLEU, Acc