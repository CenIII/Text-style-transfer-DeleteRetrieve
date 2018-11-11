import torch
from torch.utils.data import Dataset, DataLoader


class YelpDataset(Dataset):
	"""docstring for Dataset"""
	# dataset behave differently when requesting label or unlabel data
	POS = 1
	NEG = -1
	OppStyle = {POS:NEG,NEG:POS}
	def __init__(self, datafile): #, labeled=True, needLabel=True):
		super(YelpDataset, self).__init__()
		# self.data = {self.POS:[], self.NEG:[]}
		self.data = self.readData(datafile)
		with open('../AuxData/pos_style_count', "rb") as fp:   #Pickling
			self.pos_style_dict =  pickle.load(fp)
		with open('../AuxData/neg_style_count', "rb") as fp:   #Pickling
			self.neg_style_dict = pickle.load(fp)

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

	def extractMarker(self, sentence, style, gamma = 2):
		res = []
		result = []
		words = sentence.split()
		if style == self.POS:
			style_count = self.pos_style_dict
		elif style == self.NEG:
			style_count = self.neg_style_dict
		for n in range(1, 5):
			state = False
			for l in range(0, len(words)-n+1):
				tmp = ' '.join(words[l:l+n])
				if style_count[tmp] > gamma and (all(i in res for i in subset(words[l:l+n])) or all(i not in res for i in subset(words[l:l+n])) or n == 1):
					# print('word:'+tmp+'score:'+str(style_count[tmp]))
					res.append(words[l:l+n])
					result.append(tmp)
					for j in subset(words[l:l+n]):
						temp = ' '.join(j)
						if temp in result:
							result.remove(temp)
					state = True
			if not state:
				break

		return sentence, result

	# def extractMarker(self, sentence, style=self.POS):
	# 	# an API wrapper

	# 	return brkSentence, marker

	# def retrieveTargetMarker(self, brkSentence, targetStyle=self.NEG):
	# 	# an API wrapper
	# 	# whether we need deleted marker for this task or not is debatable

	# 	return targetMarker

	def loadOne(self,idx):
		style, sentence = self.data[idx]
		brkSentence, marker = self.extractMarker(sentence, style=style)
		# targetMarker = self.retrieveTargetMarker(brkSentence, targetStyle=OppStyle[style])
		return brkSentence, marker #targetMarker


class LoaderHandler(object):
	"""docstring for LoaderHandler"""
	def __init__(self, config):
		super(LoaderHandler, self).__init__()	
		self.ldTrain = DataLoader(YelpDataset(config['trainfile']),batch_size=self.batchsize, shuffle=True, num_workers=2, pin_memory=False)
		self.ldDev = DataLoader(YelpDataset(config['devfile']),batch_size=self.batchsize, shuffle=False, num_workers=2, pin_memory=False)
		self.ldDevEval = DataLoader(YelpDataset(config['devfile']),batch_size=1, shuffle=False, pin_memory=False)