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
		lengths = np.zeros(batchSize)
		labels = np.zeros(batchSize)
		for i in range(batchSize):
			labels[i] = batch[i][1]
			lengths[i] = batch[i][2]
		maxLen = np.max(lengths)
		sents = np.zeros([batchSize,maxLen])
		for i in range(batchSize):
			sents[i] = batch[i][0]
		return torch.LongTensor(sents), torch.LongTensor(labels), torch.LongTensor(lengths)

