import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
import pickle
import numpy as np
from .languageModel import languageModel
from utils import utils
class Seq2seq(nn.Module):
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

	def __init__(self, embedding=None, wordDict=None, hidden_size=300, style_size=100, input_dropout_p=0, max_len=100, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='gru', decode_function=F.log_softmax):
		super(Seq2seq, self).__init__()
		print('net...')
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
		self.encoder = EncoderRNN(vocab_size, max_len, hidden_size, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=True,
				embedding=embedding, update_embedding=False)
		self.style_emb = nn.Embedding(2,style_size)
		self.decoder = DecoderRNN(vocab_size, max_len, int((hidden_size+style_size)*(bidirectional+1)), sos_id, eos_id, unk_id, m_end_id, n_layers=n_layers, rnn_cell=rnn_cell, bidirectional=bidirectional, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, use_attention=False, embedding=embedding, update_embedding=False)
		self.decode_function = decode_function

	def flatten_parameters(self):
		self.encoder.rnn.flatten_parameters()
		self.decoder.rnn.flatten_parameters()

	def forward(self, inputs, target_variable=None,
				teacher_forcing_ratio=0):
		tf_ratio = teacher_forcing_ratio if self.training else 0
		encoder_outputs, encoder_hidden = self.encoder(inputs['brk_sentence'], inputs['bs_inp_lengths'])

		style_embedding = self.style_emb(inputs['style'])
		target_style_embedding = self.style_emb(1-inputs['style'])
		result = self.decoder(inputs=[inputs['sentence'],inputs['brk_sentence'],inputs['mk_inp_lengths']],#target_variable,
							  style_embd=style_embedding,
							  encoder_hidden=encoder_hidden, #encoder_hidden0,
							  encoder_outputs=encoder_outputs,
							  function=self.decode_function,
							  teacher_forcing_ratio=tf_ratio,
							  outputs_maxlen=max(inputs['st_inp_lengths']))
		result2 = self.decoder(inputs=[inputs['sentence'],inputs['brk_sentence'],inputs['mk_inp_lengths']],#target_variable,
						style_embd=target_style_embedding,
						encoder_hidden=encoder_hidden, #encoder_hidden0,
						encoder_outputs=encoder_outputs,
						function=self.decode_function,
						teacher_forcing_ratio=tf_ratio,
						outputs_maxlen=max(inputs['st_inp_lengths']))
		# import pdb;pdb.set_trace()
		return result,result2



class Criterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self, config):
		super(Criterion, self).__init__()
		print('crit...')
		self.celoss = nn.CrossEntropyLoss()
		self.config = config

		with open(config['crit']['wordDict'],"rb") as fp:
			self.wordDict = pickle.load(fp)

		if config['crit']['use_lang_model']==1:
			self.lm_pos = languageModel(**config['lang_model'])
			self.lm_neg = languageModel(**config['lang_model'])	
			
	def load_crit(self):
		config = self.config
		if config['crit']['use_lang_model']==1:
			print("Loading language models.")
			self.lm_pos = utils.reloadLM(self.lm_pos,config,style=1)
			self.lm_neg = utils.reloadLM(self.lm_neg,config,style=0)
		else:
			print('Not using language model.')


	def LanguageModelLoss(self,sentence_input,length,style):
		# print(sentence_input.shape[1],length)
		# assert sentence.shape[1]==length
		sentence = sentence_input[:,:length]
		sid = self.wordDict['@@START@@']

		if torch.cuda.is_available():
			labels = torch.cat([sentence,torch.zeros(1).view(1,1).type(torch.int64).cuda()],dim=1)
			sentence = torch.cat([torch.tensor(sid).view(1,1).cuda(),sentence],dim=1) # add <sos>
		else: 
			labels = torch.cat([sentence,torch.zeros(1).view(1,1).type(torch.int64)],dim=1)
			sentence = torch.cat([torch.tensor(sid).view(1,1),sentence],dim=1) # add <sos>
		
		
		length = length+1
		# import pdb;pdb.set_trace()
		if style == 1:
			outputs = self.lm_pos(sentence.view(1,-1),torch.tensor(length).view(-1))
		else:
			outputs = self.lm_neg(sentence.view(1,-1),torch.tensor(length).view(-1))
		
		
		loss = self.celoss(outputs.view(-1,outputs.shape[2]),labels.view(-1))
		loss = loss/(length-1)
		return loss

	def ReconstructLoss(self):
		pass

	def forward(self, outputs, inputs):
		labels = inputs['sentence']
		lengths = inputs['st_inp_lengths'] # batch_size
		styles = inputs['style'] # (batch_size, 1)

		decoder_outputs = outputs[0][0] # Modified. calulate reconstruction error.
		decoder_outputs = torch.cat([torch.tensor(k).unsqueeze(1) for k in decoder_outputs],1) #[batch, seqlength, vocabsize]
		
		if self.config['crit']['use_lang_model']==1:
			# TODO: use cat & split
			transfer_decoder_outputs = outputs[1][2] # size(batch_size, vocab_size)
			transfer_sentence = transfer_decoder_outputs['sequence']
			t_cat = torch.cat(transfer_sentence,dim=1)
			transfer_sentence = torch.split(t_cat,1,dim=0)

			transfer_length = transfer_decoder_outputs['length']

		# transfer_sentence = torch.cat(transfer_decoder_outputs['sequence'], dim=1)
		# transfer_length = torch.tensor(transfer_decoder_outputs['length'])+1
		# sos_pad = torch.ones(transfer_sentence.shape[0],1,dtype=torch.int64)*self.wordDict['@@START@@']
		# transfer_sentence = torch.cat(sos_pad,transfer_sentence)

		batchSize = len(labels)
		loss = 0
		for i in range(batchSize):
			wordLogPs = decoder_outputs[i][:lengths[i]-1]
			gtWdIndices = labels[i][1:lengths[i]]
			if self.config['crit']['use_lang_model']==1:
				# import pdb;pdb.set_trace()
				loss += self.LanguageModelLoss(transfer_sentence[i],transfer_length[i],styles[i])
			loss += self.celoss(wordLogPs, gtWdIndices)
			# loss += - torch.sum(torch.gather(wordLogPs,1,gtWdIndices.unsqueeze(1)))/float(lengths[i]-1)
		loss = loss/batchSize
		return loss
			


		