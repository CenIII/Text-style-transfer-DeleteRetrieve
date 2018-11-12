import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate
import numpy as np 

class YelpDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = -1
	OppStyle = {POS:NEG,NEG:POS}
	def __init__(self, config, datafile): #, wordDictFile): #, labeled=True, needLabel=True):
		super(YelpDataset, self).__init__()
		print('- dataset: '+datafile)
		# self.data = {self.POS:[], self.NEG:[]}
		self.data = self.readData(datafile)
		with open(config['posStyleDict'], "rb") as fp:   #Pickling
			self.pos_style_dict =  pickle.load(fp)
		with open(config['negStyleDict'], "rb") as fp:   #Pickling
			self.neg_style_dict = pickle.load(fp)
		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.sos_id = self.wordDict['@@START@@']
		self.eos_id = self.wordDict['@@END@@']
		self.isTrans = config['isTrans']

	def isValidSentence(self,sentence):
		if(sentence == [] or 
			sentence == 'Positive' or 
			sentence == 'Negative'):
			return False
		return True

	def readData(self,datafile):
		data = [] #{self.POS:[], self.NEG:[]}
		# proc .0 file (negative)
		def subread(postfix,style):
			with open(datafile+postfix,'r') as f:
				line = f.readline()
				while line:
					sentence = line.split(' ')[:-1]
					if self.isValidSentence(sentence):
						data.append((style, sentence))
					line = f.readline()
		subread('.0',self.NEG)
		subread('.1',self.POS)
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.loadOne(idx)

	def extractMarker(self, sentence, style):
		maxc = -float('inf')
		words = sentence
		cnt = 0
		if style == self.POS:
			style_count = self.pos_style_dict
		elif style == self.NEG:
			style_count = self.neg_style_dict
		for n in range(1, 5):
			for l in range(0, len(words)-n+1):
				tmp = ' '.join(words[l:l+n])
				if style_count.get(tmp, 0) > maxc:
					maxc = style_count.get(tmp, 0)
					cur = (tmp, l ,l+n)
					cnt += 1
		if cnt==0:
			print(sentence)
		marker = cur[0].split(' ')
		marker = self.applyNoise(marker, style_count)
		return words[:cur[1]]+['<unk>']+words[cur[2]:], marker

	def retrieveTargetMarker(self, brkSentence, targetStyle):
		# an API wrapper
		# whether we need deleted marker for this task or not is debatable
		pass
	# 	return targetMarker

	def applyNoise(self, marker, style_count):
		if len(marker) <= 1:
			return marker
		if np.random.uniform(0,1,1)<0.1:
			# print(marker)
			minScore = float('inf')
			cur = None
			# for loop marker
			for i in range(len(marker)):
				sc = style_count.get(marker[i], 0)
				if  sc < minScore:
					minScore = sc
					cur = i
			del marker[i]	
			# print(marker)
		return marker

	def word2index(self, sList):
		resList = []
		for sentence in sList:
			indArr = []
			# indArr.append(self.sos_id)
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
			indArr.append(self.eos_id)
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList

	def loadOne(self,idx):
		style, sentence = self.data[idx]
		# print('style: '+str(style)+' sentence:'+str(sentence))
		brkSentence, marker = self.extractMarker(sentence, style=style)
		if self.isTrans:
			marker = self.retrieveTargetMarker(brkSentence, targetStyle=self.OppStyle[style])
		# print('brkSentence: '+str(brkSentence)+' marker: '+str(marker))
		brkSentence, marker, sentence = self.word2index([brkSentence, marker, sentence])
		# targetMarker = self.retrieveTargetMarker(brkSentence, targetStyle=OppStyle[style])
		return (brkSentence, marker, sentence) #targetMarker


class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()
		print('loader handler...')	
		trainData = YelpDataset(config,config['trainFile'])
		self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		devData = YelpDataset(config,config['devFile'])
		self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)