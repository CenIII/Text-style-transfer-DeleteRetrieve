import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate, StyleMarker
import numpy as np 

class YelpDataset(Dataset):
	"""Custom dataset inherit torch.utils.data.Dataset. 
	
	See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for more information.

	Attributes:
		data: A list of (style, sentence) loaded from the dataset.
		wordDict: A dict in form of {word: word_index}.
		pos_style_dict, neg_style_dict: A dict in form of {marker: score}
		sos_id: Int,The word index for auxilary token '@@START@@'.
		eos_id: Int, Index for '@@END@@'.
		isTrans: Int, whether to transfer style.
	"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = 0
	OppStyle = {POS:NEG,NEG:POS}
	def __init__(self, config, datafile,forceNoNoise=False,hasStyle=None): #, wordDictFile): #, labeled=True, needLabel=True):
		"""
		Args:
			config: Configuration of loader.
			datafile: Path to the training set.
		"""
		super(YelpDataset, self).__init__()
		print('- dataset: '+datafile)
		# self.data = {self.POS:[], self.NEG:[]}
		self.data = self.readData(datafile,hasStyle=None)
		with open(config['posStyleDict'], "rb") as fp:   #Pickling
			self.pos_style_dict =  pickle.load(fp)
		with open(config['negStyleDict'], "rb") as fp:   #Pickling
			self.neg_style_dict = pickle.load(fp)
		with open(config['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp)
		self.sos_id = self.wordDict['@@START@@']
		self.eos_id = self.wordDict['@@END@@']
		self.isTrans = config['isTrans']
		self.useNoise = 0 if forceNoNoise else config['useNoise']


		self.sm = StyleMarker(config['selfatt'],self.wordDict)
		# self.sm.get_att(['the', 'service', 'was', 'really', 'good', 'too'])
		# self.sm.mark(['i', 'had', 'the', 'baja', 'burro', '...', 'it', 'was', 'heaven'])
		pass

	def isValidSentence(self,sentence):
		if(sentence == [] or 
			sentence == 'Positive' or 
			sentence == 'Negative'):
			return False
		return True

	def readData(self,datafile,hasStyle=None):
		"""Read data file and parse it into a list of (style, sentence).
		Args:
			datafile: the path to dataset.
		Returns:
			data: A list of (style, sentence) tuple, in which style is an interger and sentence is a list of tokens.
		"""
		data = [] #{self.POS:[], self.NEG:[]}
		# proc .0 file (negative)
		def subread(postfix,style):
			with open(datafile+postfix,'r') as f:
				line = f.readline()
				# i = 0
				while line:
					sentence = line.split(' ')[:-1]
					if self.isValidSentence(sentence):
						data.append((style, sentence))
					line = f.readline()
					# i += 1
		if hasStyle:
			subread('.'+str(hasStyle),int(hasStyle))
		else:
			subread('.0',self.NEG)
			subread('.1',self.POS)
		return data

	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return len(self.data)

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		style, sentence = self.data[idx]
		return self.loadLine(sentence, style)

	def applyNoise(self, ptList,sentLen):
		curList = sum([[l,r] for (l,r) in ptList],[])
		newList = []
		for (l,r) in ptList:
			if np.random.uniform(0,1,1)<0.1:
				if np.random.uniform(0,1,1)<0.5:
					newl = max([0,l-1])
					if newl in curList:
						newl = l
					newList.append((newl,r))
				else:
					newr = min([sentLen,r+1])
					if newr in curList:
						newr = r
					newList.append((l,newr))
			else:
				newList.append((l,r))
		return newList

	def extractMarker(self, sentence):
		"""Find the marker with the most sentiment."""
		ptList = self.sm.mark(sentence)
		if self.useNoise:
			ptList = self.applyNoise(ptList,len(sentence))
		brkSent = []
		marker = []
		fullSent = []
		pt = 0
		for (l,r) in ptList:
			brkSent += sentence[pt:l]+['<unk>']+['<m_end>']
			marker.append(sentence[l:r])
			fullSent += sentence[pt:l]+['<unk>']+marker[-1]+['<m_end>']
			pt = r
		brkSent += sentence[pt:]
		fullSent += sentence[pt:]
		# print(str(brkSent)+'>>><<<<'+str(marker)+'<<<>>>>'+str(fullSent))
		return brkSent, marker, fullSent

	def loadLine(self, sentence, style):
		"""Given a sentence and its style, perform the Delete stage of our algorithm.

		Returns:
			brkSentence: The word-index of marker-excluded sentence with auxilary tokens <unk> and <m_end> to indicate the position of marker.
			style: Target style. The behavior may vary depending on OptStyle.
			sentence: The word-indexed original sencentence with auxilary tokens.
			marker: The word-indexed marker
		"""
		# print(sentence)
		brkSentence, marker, sentence = self.extractMarker(sentence)

		tmp = self.word2index([brkSentence] + marker)
		brkSentence = tmp[0]
		marker = tmp[1:]
		sentence = self.word2index([sentence],sos=True)[0]
		
		if self.isTrans:
			style = self.OppStyle[style]
		return (brkSentence, [style], sentence, marker) #targetMarker

	def word2index(self, sList, sos=False):
		"""For each sentence in a list of sentences, convert its tokens into word index including auxilary tokens.
		Args:
			sList: A list of sentences.
			sos: A boolean indicating whether to add <sos> token at the start of sentences.
		"""
		resList = []
		for sentence in sList:
			indArr = []
			if sos:
				indArr.append(self.sos_id)
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
			indArr.append(self.eos_id) 
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList
		


class LoaderHandler(object):
	"""Load dataset to be used later as required in the config.
	
	Attributes:
		ldDev: An instance of Pytorch Dataloader. batch_size depends on config. collate_fn overloaded in utils.py
		ldDevEval: The same as ldDev, except the batch_size is always 1.
		ldTrain: It is created only in training mode. Data is shuffled.
	"""
	def __init__(self, config):
		""" Init instance. Only one instance should be created.
		Args:
			config: An instance of ConfigParser, contains information parsed from both command line and config file.
		"""
		super(LoaderHandler, self).__init__()
		print('loader handler...')	
		mode = config['opt'].mode
		config = config['loader']
		if mode == 'test':
			testData = YelpDataset(config,config['testFile'],forceNoNoise=True)
			self.ldTestEval = DataLoader(testData,batch_size=1, shuffle=False, collate_fn=seq_collate)
			return
		if mode == 'train':
			trainData = YelpDataset(config,config['trainFile'])
			self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		if mode == "pretrain":
			trainData_pos = YelpDataset(config,config['trainFile'],hasStyle=1)
			self.ldTrain_pos = DataLoader(trainData_pos,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
			trainData_neg = YelpDataset(config,config['trainFile'],hasStyle=0)
			self.ldTrain_neg = DataLoader(trainData_neg,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
			devData_pos = YelpDataset(config,config['devFile'],forceNoNoise=True)
			self.ldDev_pos = DataLoader(devData_pos,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
			devData_neg = YelpDataset(config,config['devFile'],forceNoNoise=True)
			self.ldDev_neg = DataLoader(devData_neg,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		# elif mode == 'val':
		devData = YelpDataset(config,config['devFile'],forceNoNoise=True)
		self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)
		# else:
		