import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import subset, seq_collate, StyleMarker
import numpy as np 
import os 

class YelpDataset(Dataset):

	def __init__(self):
		super(YelpDataset, self).__init__()
		self.dataPath = '../Data/longyelp/'
		self.data,self.label,self.lengths = self.readData()
		self.force_max_len = 200

	def readData(self):
		with open(os.path.join(self.dataPath,'train.pkl'),'rb') as f:
			datadict = pickle.load(f)
		data = datadict['data']
		label = datadict['label']
		lengths = datadict['lengths']
		return data,label,lengths

	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return len(self.data)

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		sentence = self.data[idx]
		label = self.label[idx]
		length = self.lengths[idx]
		return sentence, label, length

	def collate_fn(self, batch):
		batchSize = len(batch)
		maxLen = 0
		lengths = np.zeros(batchSize, dtype=np.int)
		labels = np.zeros(batchSize, dtype=np.int)
		for i in range(batchSize):
			labels[i] = batch[i][1]
			lengths[i] = min(self.force_max_len, batch[i][2])
		maxLen = min(self.force_max_len, np.max(lengths))
		sents = np.zeros([batchSize,maxLen], dtype=np.int)
		for i in range(batchSize):
			sents[i,:lengths[i]] = batch[i][0][:lengths[i]]

		inds = np.argsort(-lengths)
		sents = torch.LongTensor(sents[inds])
		labels = torch.LongTensor(labels[inds])
		lengths = torch.LongTensor(lengths[inds])
		nonz = (lengths<=0).nonzero()
		if len(nonz)>0:
			ending = nonz[0][0]
			sents = sents[:ending]
			labels = labels[:ending]
			lengths = lengths[:ending]
		return sents, labels, lengths

