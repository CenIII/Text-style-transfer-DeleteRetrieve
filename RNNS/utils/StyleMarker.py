from model import StructuredSelfAttention
import torch
import numpy as np

class StyleMarker(object):
	"""docstring for StyleMarker"""
	def __init__(self,checkpoint,wordDict):
		super(StyleMarker, self).__init__()
		self.wordDict = wordDict
		# self.ind2word = {v:k for k,v in self.wordDict.items()}
		self.model = None #StructuredSelfAttention()
		self.reloadModel(checkpoint)
		# if torch.cuda.is_available():
		# 	self.model = self.model.cuda()
		with open('./utils/stopwords','r') as f:
			lines = f.readlines()
			self.stopwords = [sw[:-1] for sw in lines]

	def reloadModel(self,checkpoint):
		self.model = StructuredSelfAttention(batch_size=1,
											 lstm_hid_dim=100,
											 d_a = 100,
											 r=2,
											 vocab_size=len(self.wordDict),
											 max_len=25,
											 type=0,
											 n_classes=1,
											 use_pretrained_embeddings=False,
											 embeddings=None)
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
		# model.load_state_dict(self.checkpoint['state_dict'])
		model_dict = self.model.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {}
		for k, v in checkpoint.items():
			if(k in model_dict):
				pretrained_dict[k] = v
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		# 3. load the new state dict
		self.model.load_state_dict(model_dict)
		return self.model
		
	def word2index(self, sList):
		resList = []
		for sentence in sList:
			indArr = []
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList

	def visualize(self,att,text):
		pass

	def get_att(self, text):
		# word dict
		seq = torch.tensor(self.word2index([text])[0])
		# if torch.cuda.is_available():
		# 	seq = seq.cuda()
		# pass forward
		with torch.set_grad_enabled(False):
			_, att = self.model(seq.detach())
		# get att
		return att

	def mark(self,text,hoplen=1,thr=0.5):
		atts = self.get_att(text)[0]
		# ret point list
		seqlen = len(atts[0])
		idx_l = list(range(seqlen))
		ptList = []

		mkList = []
		for att in atts:
			tgts = sorted(zip(att,idx_l))[-hoplen:]
			for v,idx in tgts:
				if v > thr and text[idx] not in self.stopwords:
					mkList.append(idx)

		if mkList == []:
			mkList.append(sorted(zip(atts[0],idx_l))[-1][1])
			# print('no find')

		mkList = sorted(mkList)
		p = mkList[0]
		if len(mkList)==1:
			return [(p,p+1)]
		else:
			t = p
			for c in mkList[1:]:
				if c==t+1:
					t = c
					continue
				else:
					ptList.append((p,t+1))
					p = c
					t = p
			ptList.append((p,t+1))
			return ptList
		
