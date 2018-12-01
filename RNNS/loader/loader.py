import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate
import numpy as np 

class YelpDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = 0
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
				# i = 0
				while line:
					sentence = line.split(' ')[:-1]
					if self.isValidSentence(sentence):
						data.append((style, sentence))
					line = f.readline()
					# i += 1
		subread('.0',self.NEG)
		subread('.1',self.POS)
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		style, sentence = self.data[idx]
		# print('style: '+str(style)+' sentence:'+str(sentence))
		return self.loadLine(sentence, style)

	def extractMarker(self, sentence, style):
		maxc = np.array([-float('inf'),-float('inf')])
		words = sentence
		cnt = 0
		mk = [None, None]
		cur = 0

		def rg():
			g = []
			for m in mk:
				if m is not None:
					g += list(range(m[1],m[2]))
			return g

		if style == self.POS:
			style_count = self.pos_style_dict
		elif style == self.NEG:
			style_count = self.neg_style_dict
		for n in range(1, 3):
			for l in range(0, len(words)-n+1):
				tmp = ' '.join(words[l:l+n])
				score = style_count.get(tmp, 0)
				if score > min(maxc):
					g = rg()
					if l in g or l+n-1 in g:
						continue
					# print(score)
					maxc[np.argmin(maxc)] = score
					mk[cur] = (tmp, l ,l+n, score)
					cur = (cur+1)%2
					cnt += 1
		if cnt==0:
			print(sentence)

		if None not in mk:
			ind = 0 if mk[0][3]<mk[1][3] else 1
			if len(words)==2 or mk[ind][3]<10:
				mk[ind] = None

		brkSent = []
		pt = 0
		marker = []
		sentence = []
		if None not in mk:
			if mk[0][1] > mk[1][1]:
				tmp = mk[0]
				mk[0] = mk[1]
				mk[1] = tmp

			if mk[0][2] == mk[1][1]:
				mk[0] = (mk[0][0]+' '+mk[1][0],mk[0][1],mk[1][2])
				mk[1] = None

		for m in mk:
			cnt = 0
			if m is not None:
				brkSent += words[pt:m[1]]+['<unk>']+['<m_end>']
				marker.append(m[0].split(' '))
				sentence += words[pt:m[1]]+['<unk>']+marker[-1]+['<m_end>']
				pt = m[2]
				cnt += 1
			if cnt==1:
				break
		brkSent += words[pt:]
		sentence += words[pt:]
		# print(str(brkSent)+'>>><<<<'+str(marker)+'<<<>>>>'+str(sentence))

		# marker = mk[0].split(' ')
		# marker = self.applyNoise(marker, style_count)
		return brkSent, marker,sentence #words[:mk[1]]+['<unk>']+['<m_end>']+words[mk[2]:], marker, words[:mk[1]]+['<unk>']+marker+['<m_end>']+words[mk[2]:]

	def retrieveTargetMarker(self, brkSentence, targetStyle):
		# an API wrapper
		# whether we need deleted marker for this task or not is debatable
		pass
	# 	return targetMarker

	def loadLine(self, sentence, style):
		# sentence = sentence.split(' ')
		brkSentence, marker, sentence = self.extractMarker(sentence, style=style)
		# TODO: assume marker has multi markers and is a list of list
		# print(sentence)
		# if self.isTrans:
		# 	marker = self.retrieveTargetMarker(brkSentence, targetStyle=self.OppStyle[style])
		# print('brkSentence: '+str(brkSentence)+' marker: '+str(marker))
		tmp = self.word2index([brkSentence] + marker)
		brkSentence = tmp[0]
		marker = tmp[1:]
		sentence = self.word2index([sentence],sos=True)[0]
		# targetMarker = self.retrieveTargetMarker(brkSentence, targetStyle=OppStyle[style])
		if self.isTrans:
			style = self.OppStyle[style]
		return (brkSentence, [style], sentence, marker) #targetMarker

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

	def word2index(self, sList, sos=False):
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
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()
		print('loader handler...')	
		mode = config['opt'].mode
		config = config['loader']
		if mode == 'train':
			trainData = YelpDataset(config,config['trainFile'])
			self.ldTrain = DataLoader(trainData,batch_size=config['batchSize'], shuffle=True, num_workers=2, collate_fn=seq_collate)
		elif mode == 'val':
			devData = YelpDataset(config,config['devFile'])
			self.ldDev = DataLoader(devData,batch_size=config['batchSize'], shuffle=False, num_workers=2, collate_fn=seq_collate)
			self.ldDevEval = DataLoader(devData,batch_size=1, shuffle=False, collate_fn=seq_collate)
		else:
			testData = YelpDataset(config,config['testFile'])
			self.ldTestEval = DataLoader(testData,batch_size=1, shuffle=False, collate_fn=seq_collate)
