import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate
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
	def __init__(self, config, datafile): #, wordDictFile): #, labeled=True, needLabel=True):
		"""
		Args:
			config: Configuration of loader.
			datafile: Path to the training set.
		"""
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
						data.append((style, sentence)) # style: int; sentence: list of string
					line = f.readline()
					# i += 1
		# The samples are stored seperately by their sentiment.
		subread('.0',self.NEG)
		subread('.1',self.POS)
		return data

	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return len(self.data)

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		style, sentence = self.data[idx]
		# print('style: '+str(style)+' sentence:'+str(sentence))
		return self.loadLine(sentence, style)

	def extractMarker(self, sentence, style):
		"""Find the marker with the most sentiment."""
		# TODO: Build a better extractor. 
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
		# marker = self.applyNoise(marker, style_count)
		return words[:cur[1]]+['<unk>']+['<m_end>']+words[cur[2]:], marker, words[:cur[1]]+['<unk>']+marker+['<m_end>']+words[cur[2]:]

	def retrieveTargetMarker(self, brkSentence, targetStyle):
		# an API wrapper
		# whether we need deleted marker for this task or not is debatable
		pass
	# 	return targetMarker

	def loadLine(self, sentence, style):
		"""Given a sentence and its style, perform the Delete stage of our algorithm.

		Returns:
			brkSentence: The word-index of marker-excluded sentence with auxilary tokens <unk> and <m_end> to indicate the position of marker.
			style: Target style. The behavior may vary depending on OptStyle.
			sentence: The word-indexed original sencentence with auxilary tokens.
			marker: The word-indexed marker
		"""
		# sentence = sentence.split(' ')
		brkSentence, marker, sentence = self.extractMarker(sentence, style=style)
		# print(sentence)
		# if self.isTrans:
		# 	marker = self.retrieveTargetMarker(brkSentence, targetStyle=self.OppStyle[style])
		# print('brkSentence: '+str(brkSentence)+' marker: '+str(marker))
		brkSentence, marker = self.word2index([brkSentence, marker])
		sentence = self.word2index([sentence],sos=True)[0]
		# targetMarker = self.retrieveTargetMarker(brkSentence, targetStyle=OppStyle[style])
		if self.isTrans:
			style = self.OppStyle[style]
		return (brkSentence, [style], sentence, marker) #targetMarker

	def applyNoise(self, marker, style_count):
		"""Add noise to the marker. Not used."""
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
		super(LoaderHandler, self).__init__() # Init with object.__init__()
		print('loader handler...')	
		mode = config['opt'].mode # spcified mode from command line
		config = config['loader'] # we only care about information of loader here.
		if mode == 'train':
			trainData = YelpDataset(config,config['trainFile'])
			# collate_fn specifies how exactly the samples need to be batched. override by utils.seq_collate()
			self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		devData = YelpDataset(config,config['devFile'])
		self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
		self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)