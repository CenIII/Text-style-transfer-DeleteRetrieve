import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
import pickle
import numpy as np
import os
from functools import reduce
from torch.nn.utils import weight_norm

if torch.cuda.is_available():
	import torch.cuda as device
else:
	import torch as device

class Seq2att(nn.Module):
	def __init__(self, hidden_size=2048, style_size=100, input_dropout_p=0, max_len=200, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='lstm', decode_function=F.sigmoid):
		super(Seq2att, self).__init__()

		self.dataPath = '../Data/longyelp/'
		with open(os.path.join(self.dataPath,'vocabs.pkl'),'rb') as f:
			vocabs = pickle.load(f)

		self.wordDict = vocabs['word_dict']
		embedding = vocabs['word_embs']
		embedding = torch.FloatTensor(embedding)
		vocab_size = len(embedding)

		self.encoder = EncoderRNN(vocab_size, max_len, hidden_size, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=True,
				embedding=embedding, update_embedding=False)
		

		self.linear_first = nn.Linear(2048,1024,bias=False)
		# self.linear_first.bias.data.fill_(0)

		self.decoder = DecoderRNN(1, 30, 1024, n_layers=n_layers, 
									rnn_cell=rnn_cell, bidirectional=bidirectional, use_attention=True) #int((hidden_size+style_size)*(bidirectional+1))

	def getLeftOver(self, attns, out_lens, encoder_outputs):
		attns = torch.cat(attns,dim=1) # (batchSize, steps, numHiddens)  [16, 30, 463]
		batchSize, _, numHiddens = attns.shape
		left_over = [] # should be 16x463
		def do_mult(x1, x2): return x1 * x2
		for i in range(batchSize):
			if out_lens[i]==0:
				left_over.append(encoder_outputs[i])
				continue
			effAtts = 1. - attns[i,:out_lens[i]]
			leftAtt = reduce(do_mult, effAtts)
			left_over.append(encoder_outputs[i]*(leftAtt.unsqueeze(1)))
		left_over = torch.stack(left_over, dim=0)#.detach()  # todo: is it right to do detach?
		return left_over

	def forward(self, sents, labels, lengths, advclss):
		'''
		inputs: sents, labels, lengths

		outputs: score array, out lengths, att matrix, enc out left over
		'''
		encoder_outputs, encoder_hidden = self.encoder(sents, lengths)
		
		eh_ky_1 = F.tanh(self.linear_first(encoder_hidden[0]))       
		eh_ky_2 = F.tanh(self.linear_first(encoder_hidden[1])) 
		eh_ky = (eh_ky_1,eh_ky_2)
		# todo: eo_ky normalization?
		eo_ky =  F.normalize(F.tanh(self.linear_first(encoder_outputs)),dim=2)

		eh_ky_detach = (eh_ky[0].detach(),eh_ky[1].detach())
		eo_ky_detach = eo_ky.detach()
		eo_detach = encoder_outputs.detach()
		# todo: generated weight normalization?
		ret_dict = self.decoder(inputs=None,
								encoder_hidden=eh_ky_detach,
								encoder_outputs_key=eo_ky_detach,
								encoder_outputs=eo_detach,
								advclss=advclss,
								att_lengths=lengths,
								labels=labels)
		hiddens = {}
		hiddens['enc_outputs'] = encoder_outputs
		hiddens['enc_outputs_key'] = eo_ky
		hiddens['left_over_key'] = self.getLeftOver(ret_dict['attention_score'], 
												ret_dict['length'], 
												eo_ky)

		return hiddens, ret_dict


 
class AdvClassifier(nn.Module):
	"""
	The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
	and without pruning. Slight modifications have been done for speedup
	"""
   
	def __init__(self,batch_size,lstm_hid_dim,d_a,r,n_classes = 1):
		"""
		Initializes parameters suggested in paper
 
		Args:
			batch_size  : {int} batch_size used for training
			lstm_hid_dim: {int} hidden dimension for lstm
			d_a         : {int} hidden dimension for the dense layer
			r           : {int} attention-hops or attention heads
			n_classes   : {int} number of classes
		"""
		super(AdvClassifier,self).__init__()

		# self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
		# self.linear_first.bias.data.fill_(0)
		self.linear_second = weight_norm(nn.Linear(d_a,r,bias=False), name='weight')
		self.linear_second.weight_g.data.fill_(1.)
		# self.linear_second.bias.data.fill_(0)
		self.n_classes = n_classes
		self.linear_final = nn.Linear(lstm_hid_dim,self.n_classes)
		self.batch_size = batch_size       
		self.lstm_hid_dim = lstm_hid_dim
		self.r = r
	   
		
	def softmax(self,input, axis=1):
		"""
		Softmax applied to axis=n
 
		Args:
		   input: {Tensor,Variable} input on which softmax is to be applied
		   axis : {int} axis on which softmax is to be applied
 
		Returns:
			softmaxed tensors
	   
		"""
		input_size = input.size()
		trans_input = input.transpose(axis, len(input_size)-1)
		trans_size = trans_input.size()
		input_2d = trans_input.contiguous().view(-1, trans_size[-1])
		soft_max_2d = F.softmax(input_2d,dim=1)
		soft_max_nd = soft_max_2d.view(*trans_size)
		return soft_max_nd.transpose(axis, len(input_size)-1)
		
	def forward(self,keys=None,hiddens=None,sent_emb=None):   # enc_outs: left overs
		if sent_emb is None:
			assert(keys is not None)
			assert(hiddens is not None)
			# x = F.tanh(self.linear_first(hiddens))       
			x = self.linear_second(keys)       
			x = self.softmax(x,1)       
			attention = x.transpose(1,2)
			sent_emb = attention@hiddens
			avg_sent_emb = torch.sum(sent_emb,1)/self.r
		else:
			attention = None
			avg_sent_emb = sent_emb
		output = F.sigmoid(self.linear_final(avg_sent_emb))
		return output,attention


class DecCriterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self):
		super(DecCriterion, self).__init__()

	def l2_matrix_norm(self,m):
		"""
		Frobenius norm calculation
 
		Args:
		   m: {Variable} ||AAT - I||
 
		Returns:
			regularized value
	   
		"""
		return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(device.DoubleTensor)

	def reg_loss(self,attn):
		if len(attn)>0:
			hops = len(attn)
			mat = attn.mm(attn.transpose(0,1))-torch.eye(hops).type(device.FloatTensor)
			return self.l2_matrix_norm(mat)
		else:
			return 0.

	
	def forward(self, seq2att_outs, labels):
		labels = labels.type(device.FloatTensor)
		attns = torch.cat(seq2att_outs['attention_score'],1) #(16,30,204)
		out_lens = device.LongTensor(seq2att_outs['length'])
		scores = torch.cat(seq2att_outs['score'],1) #(16,30)
		
		# part 1: supervise on seq2att_outs, make every step (except for the last step) output match to labels
		batch_size, steps = scores.shape
		max_len = attns.shape[2]
		mask = torch.zeros_like(scores)
		for i in range(batch_size):
			mask[i,:out_lens[i]] = 1.
		labels_rep = labels.unsqueeze(1).repeat(1,steps)
		loss1 = -torch.sum((labels_rep*torch.log(scores+1e-18)+(1-labels_rep)*torch.log(1-scores+1e-18))*mask)/batch_size

		loss_reg = 0.
		for i in range(batch_size):
			loss_reg += self.reg_loss(attns[i,:out_lens[i]])

		return loss1, loss_reg #, loss2, loss3


class AdvCriterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self):
		super(AdvCriterion, self).__init__()

	def forward(self, advclss_outs, labels):
		labels = labels.type(device.FloatTensor)
		orig_outs, left_outs = advclss_outs
		batch_size = len(orig_outs)
		loss2_1 = -torch.sum((labels*torch.log(orig_outs+1e-18)+(1-labels)*torch.log(1-orig_outs+1e-18)))
		loss2_2 = -torch.sum((labels*torch.log(left_outs+1e-18)+(1-labels)*torch.log(1-left_outs+1e-18)))
		loss2 = (loss2_1+loss2_2)/(2*batch_size)
		return loss2










			

	   