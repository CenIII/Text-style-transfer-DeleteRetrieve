import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseLangRNN import baseLangRNN
import pickle
import numpy as np

class languageModel(nn.Module):
	def __init__(self, embedding=None, wordDict=None, hidden_size=300, style_size=100, input_dropout_p=0, max_len=100, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru', decode_function=F.log_softmax):
		super(languageModel, self).__init__()
		print('Language model...')
		if embedding==None:
			print('no embedding given. please try again')
			exit(0)
		embedding = torch.FloatTensor(np.load(embedding))
		vocab_size = len(embedding)
		with open(wordDict,"rb") as fp:
			self.wordDict = pickle.load(fp)
		sos_id = self.wordDict['@@START@@']
		eos_id = self.wordDict['@@END@@']
		unk_id = self.wordDict['<unk>']
		m_end_id = self.wordDict['<m_end>']
		self.lm_model = baseLangRNN(vocab_size, max_len, hidden_size, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=True,
				embedding=embedding, update_embedding=False)

	def flatten_parameters(self):
		self.encoder.rnn.flatten_parameters()
		self.decoder.rnn.flatten_parameters()

	def forward(self, sentence, sen_length):
		result, _ = self.lm_model(sentence, sen_length) # torch.Size([batch_size,seq_len,vocab_size])
		return result

		