import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
import pickle
import numpy as np

class Seq2att(nn.Module):
	def __init__(self, hidden_size=2048, style_size=100, input_dropout_p=0, max_len=200, dropout_p=0, n_layers=1, bidirectional=False, rnn_cell='lstm', decode_function=F.sigmoid):
		super(Seq2seq, self).__init__()

		self.dataPath = '../Data/longyelp/'
		with open(os.path.join(self.dataPath,'vocabs.pkl'),'rb') as f:
			vocabs = pickle.load(f)

		self.wordDict = vocabs['word_dict']
		embedding = vocabs['word_embs']
		embedding = torch.FloatTensor(embedding)
		vocab_size = len(embedding)

		self.encoder = EncoderRNN(vocab_size, max_len, hidden_size, 
				input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers, bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=True,
				embedding=embedding, update_embedding=True)
		self.style_emb = nn.Embedding(2,style_size)
		self.decoder = DecoderRNN(1, 20, int((hidden_size+style_size)*(bidirectional+1)), n_layers=n_layers, 
									rnn_cell=rnn_cell, bidirectional=bidirectional, use_attention=True)

	def forward(self, inputs):
		encoder_outputs, encoder_hidden = self.encoder(inputs['brk_sentence'], inputs['bs_inp_lengths'])
		style_embedding = self.style_emb(inputs['style'])
		result = self.decoder(inputs=None,
							  style_embd=style_embedding,
							  encoder_hidden=encoder_hidden,
							  encoder_outputs=encoder_outputs)
	
		return result


 
class AdvClassifier(torch.nn.Module):
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

        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
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
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
        
    def forward(self,enc_outs):   # enc_outs: left overs
        x = F.tanh(self.linear_first(enc_outs))       
        x = self.linear_second(x)       
        x = self.softmax(x,1)       
        attention = x.transpose(1,2)       
        sentence_embeddings = attention@enc_outs       
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
        output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
        return output,attention


class Criterion(object):
	"""docstring for Criterion"""
	def __init__(self):
		super(Criterion, self).__init__()
	
	def forward(self, seq2att_outs, advclss_outs, labels):
		pass
		

       