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
	def __init__(self, datafile, wordDictFile): #, labeled=True, needLabel=True):
		super(YelpDataset, self).__init__()
		# self.data = {self.POS:[], self.NEG:[]}
		self.data = self.readData(datafile)
		with open('../AuxData/pos_style_count', "rb") as fp:   #Pickling
			self.pos_style_dict =  pickle.load(fp)
		with open('../AuxData/neg_style_count', "rb") as fp:   #Pickling
			self.neg_style_dict = pickle.load(fp)
		with open(wordDictFile,"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.sos_id = self.wordDict['@@START@@']
		self.eos_id = self.wordDict['@@END@@']

	def readData(self,datafile):
		data = [] #{self.POS:[], self.NEG:[]}
		# proc .0 file (negative)
		def subread(postfix,style):
			with open(datafile+postfix,'r') as f:
				line = f.readline()
				while line:
					data.append((style, line.split(' ')[:-1]))
					line = f.readline()
		subread('.0',self.NEG)
		subread('.1',self.POS)
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.loadOne(idx)

	def extractMarker(self, sentence, style):
		maxc = 0
		words = sentence
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
		marker = cur[0].split(' ')
		return words[:cur[1]]+['<unk>']+words[cur[2]:], marker
		# maxc = 0
		# words = sentence
		# if style == self.POS:
		# 	style_count = self.pos_style_dict
		# elif style == self.NEG:
		# 	style_count = self.neg_style_dict
		# for n in range(1, 5):
		# 	for l in range(0, len(words)-n+1):
		# 		tmp = words[l:l+n]
		# 		if style_count.get(tmp, 0) > maxc:
		# 			maxc = style_count.get(tmp, 0)
		# 			cur = (tmp, l ,l+n)
		# return words[:cur[1]] + ["<unk>"] + words[cur[2]:], cur[0]

	# def retrieveTargetMarker(self, brkSentence, targetStyle=self.NEG):
	# 	# an API wrapper
	# 	# whether we need deleted marker for this task or not is debatable

	# 	return targetMarker

	def applyNoise(self, marker):
		n_marker = []

		return n_marker

	def word2index(self,sList):
		resList = []
		for sentence in sList:
			indArr = []
			indArr.append(self.sos_id)
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
		print("brk: "+str(brkSentence))
		print("marker: "+str(marker))

		# print('brkSentence: '+str(brkSentence)+' marker: '+str(marker))
		brkSentence, marker, sentence = self.word2index([brkSentence, marker, sentence])
		# targetMarker = self.retrieveTargetMarker(brkSentence, targetStyle=OppStyle[style])
		return (brkSentence, marker, sentence) #targetMarker


class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()	
		self.ldTrain = DataLoader(YelpDataset(config['trainFile'],config['wordDict']),batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		self.ldDev = DataLoader(YelpDataset(config['devFile'],config['wordDict']),batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(YelpDataset(config['devFile'],config['wordDict']),batch_size=1, shuffle=False)