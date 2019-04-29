import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils
 
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
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
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
       
	   
	# #Regularization
 #    def l2_matrix_norm(self,m):
 #        """
 #        Frobenius norm calculation
 
 #        Args:
 #           m: {Variable} ||AAT - I||
 
 #        Returns:
 #            regularized value
 
       
 #        """
 #        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(torch.cuda.DoubleTensor)