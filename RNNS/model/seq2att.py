import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
import pickle
import numpy as np
import os
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
		

		self.linear_first = torch.nn.Linear(2048,1024)
		self.linear_first.bias.data.fill_(0)

		self.decoder = DecoderRNN(1, 30, 1024, n_layers=n_layers, 
									rnn_cell=rnn_cell, bidirectional=bidirectional, use_attention=True) #int((hidden_size+style_size)*(bidirectional+1))

	def getLeftOver(self, attns, out_lens, encoder_outputs):
		attns = torch.cat(attns,dim=1) # (batchSize, steps, numHiddens)  [16, 30, 463]
		batchSize, _, numHiddens = attns.shape
		left_over = [] # should be 16x463
		def do_mult(x1, x2): return x1 * x2
		for i in range(batchSize):
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
		eo_ky =  F.tanh(self.linear_first(encoder_outputs))

		# todo: eo_l normalization?
		

		eh_ky_detach = (eh_ky[0].detach(),eh_ky[2].detach())
		eo_ky_detach = eo_ky.detach()
		eo_detach = eo.detach()
		# todo: generated weight normalization?
		decoder_outs = self.decoder(inputs=None,
								encoder_hidden=eh_ky_detach,
								encoder_outputs_key=eo_ky_detach,
								encoder_outputs=eo_detach,
								advclss=advclss,
								labels=labels)
		hiddens = {}
		hiddens['enc_outputs'] = encoder_outputs
		hiddens['enc_outputs_key'] = eo_ky
		hiddens['left_over_key'] = self.getLeftOver(ret_dict['attention_score'], 
												ret_dict['length'], 
												eo_ky)

		return hiddens, decoder_outs


 
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
		self.linear_second = torch.nn.Linear(d_a,r)
		self.linear_second.bias.data.fill_(0)
		self.n_classes = n_classes
		self.linear_final = torch.nn.Linear(lstm_hid_dim,self.n_classes)
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
			avg_sent_emb = sent_emb
		output = F.sigmoid(self.linear_final(avg_sent_emb))
		return output,attention


class DecCriterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self):
		super(DecCriterion, self).__init__()
	
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
			mask[i,:(out_lens[i]-1)] = 1.
		labels_rep = labels.unsqueeze(1).repeat(1,steps)
		loss1 = -torch.sum((labels_rep*torch.log(scores+1e-18)+(1-labels_rep)*torch.log(1-scores+1e-18))*mask)/batch_size

		# def checkFirmPreds(scores,margin=0.05):
		# 	tmp = (torch.zeros(len(scores))-1.).type(device.FloatTensor)
		# 	tmp1 = 2*(scores>(0.5+margin)).type(device.FloatTensor).squeeze()
		# 	tmp2 = (scores<(0.5-margin)).type(device.FloatTensor).squeeze()
		# 	tmp = tmp+tmp1+tmp2
		# 	return tmp

		# # part 2: if advclss_outs agree with labels (>0.52), sup on last step of seq2att, also sup on the last att of seq2att
		# adv_scores, adv_attn = advclss_outs  # (16,1), (16,1,204)
		# adv_labels = checkFirmPreds(adv_scores)
		# agree_mask = adv_labels.data.eq(labels.data)

		# 	# find index of 1 in agree mask
		# active_indices = (agree_mask != 0).nonzero().squeeze()
		# 	# gather target scores, attns
		# ac_scores = scores[active_indices].view(-1,steps)
		# ac_attns = attns[active_indices].view(-1,steps,max_len)
		# ac_out_lens = out_lens[active_indices].view(-1)

		# assert(len(ac_out_lens)==len(active_indices))
		# loss2 = 0.
		# if len(ac_out_lens)>=1:
		# 	ac_scores = ac_scores.gather(1,(ac_out_lens-1).unsqueeze(1))
		# 	ac_attns = ac_attns.gather(1,(ac_out_lens-1).unsqueeze(1).unsqueeze(2).repeat(1,1,max_len))
		# 	# calc loss
		# 	ac_adv_scores = labels[active_indices] #adv_scores[active_indices]
		# 	ac_adv_attn = adv_attn[active_indices]
		# 	loss2_1 = -torch.sum((ac_adv_scores*torch.log(ac_scores+1e-18)+(1 - ac_adv_scores)*torch.log(1-ac_scores+1e-18)))/len(active_indices)
		# 	loss2_2 = torch.sum((ac_adv_attn - ac_attns)**2)/len(active_indices)
		# 	loss2 = loss2_1+loss2_2

		# part 3: don't forget to train advclss
		# loss3 = -torch.sum((labels*torch.log(adv_scores+1e-18)+(1-labels)*torch.log(1-adv_scores+1e-18)))/batch_size

		return loss1#, loss2, loss3


class AdvCriterion(nn.Module):
	"""docstring for Criterion"""
	def __init__(self):
		super(AdvCriterion, self).__init__()

	def forward(self, advclss_outs, labels):
		orig_outs, left_outs = advclss_outs
		batch_size = len(orig_outs)
		loss2_1 = -torch.sum((labels*torch.log(orig_outs+1e-18)+(1-labels)*torch.log(1-orig_outs+1e-18)))/batch_size
		loss2_2 = -torch.sum((labels*torch.log(left_outs+1e-18)+(1-labels)*torch.log(1-left_outs+1e-18)))/batch_size
		loss2 = loss2_1+loss2_2
		return loss2










			

	   