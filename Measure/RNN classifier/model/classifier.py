import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import math

class Classifier(nn.Module):
	""" Standard sequence-to-sequence architecture with configurable encoder
	and decoder.

	Args:
		encoder (EncoderRNN): object of EncoderRNN
		decoder (DecoderRNN): object of DecoderRNN
		decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

	Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
		- **input_variable** (list, option): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs. This information is forwarded to the encoder.
		- **input_lengths** (list of int, optional): A list that contains the lengths of sequences
			in the mini-batch, it must be provided when using variable length RNN (default: `None`)
		- **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
		  each sequence is a list of token IDs. This information is forwarded to the decoder.
		- **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
		  is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
		  teacher forcing would be used (default is 0)

	Outputs: decoder_outputs, decoder_hidden, ret_dict
		- **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
		  outputs of the decoder.
		- **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
		  state of the decoder.
		- **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
		  representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
		  predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
		  sequences, where each list is of attention weights }.

	"""

	def __init__(self, embedding=None, wordDict=None, hidden_size=300, input_dropout_p=0, max_len=100, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru', decode_function=F.log_softmax,num_classes=2):
		super(Classifier, self).__init__()
		print('net...')
		if embedding==None:
			print('no embedding given. please try again')
			exit(0)
		self.embedding = torch.FloatTensor(np.load(embedding))
		vocab_size = len(embedding)

		with open(wordDict,"rb") as fp:
			self.wordDict = pickle.load(fp)
		sos_id = self.wordDict['@@START@@']
		eos_id = self.wordDict['@@END@@']
		self.encoder = Encoder(vocab_size,300, hidden_size, n_layers, dropout_p,bidirectional, rnn_cell, self.embedding, False, True)
		if(not bidirectional):
			attention_dim = hidden_size
		else:
			attention_dim = 2 * hidden_size
			
		self.attention = Attention(attention_dim, attention_dim, attention_dim)
		self.decoder = nn.Linear(attention_dim, num_classes)
		self.decode_function = decode_function

	def flatten_parameters(self):
		self.encoder.rnn.flatten_parameters()
		self.decoder.rattentionnn.flatten_parameters()

	def forward(self, inputs):
		outputs, hidden = self.encoder(inputs['sentence'],inputs['st_inp_lengths'])
		if isinstance(hidden, tuple): # LSTM
			hidden = hidden[1] # take the cell state

		if self.encoder.bidirectional: # need to concat the last 2 hidden layers
			hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
		else:
			hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

		energy, linear_combination = self.attention(hidden, outputs, outputs) 
		logits = F.softmax(self.decoder(linear_combination))
		return logits, energy



		
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
                bidirectional=True, rnn_type='GRU', embedding = None, update_embedding = True, variable_lengths=False):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.variable_lengths = variable_lengths
#        assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type.upper()) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
                            dropout=dropout, bidirectional=bidirectional)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dropout = nn.Dropout(p=dropout)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding

    def forward(self, input_var, input_lengths=None):
        inds = np.argsort(-input_lengths)
        input_var = input_var[inds]
        input_lengths = input_lengths[inds]
        rev_inds = np.argsort(inds)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[rev_inds]
        hidden = hidden[:,rev_inds]

        return output, hidden


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        # values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

class Criterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self, config):
		super(Criterion, self).__init__()
		print('crit...')
		self.celoss = nn.CrossEntropyLoss()

	def LanguageModelLoss(self):
		pass

	def ReconstructLoss(self):
		pass

	def forward(self, outputs, inputs):
		labels = inputs['label']
		output = outputs[0]
		batchSize = len(labels)
		loss = 0
		for i in range(batchSize):
			loss += self.celoss(output[i].unsqueeze(0),labels[i])
			# loss += - torch.sum(torch.gather(wordLogPs,1,gtWdIndices.unsqueeze(1)))/float(lengths[i]-1)
		loss = loss/batchSize
		return loss
		
