import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_SCORE = 'score'
    KEY_LENGTH = 'length'
    # KEY_SEQUENCE = 'sequence'
    # KEY_LEFT_OVER = 'left_over'


    def __init__(self, output_size, max_len, hidden_size,
            n_layers=1, rnn_cell='gru', bidirectional=False, use_attention=False):
        super(DecoderRNN, self).__init__(output_size, max_len, hidden_size, 0, 0, n_layers, rnn_cell)
        self.MARGIN = 0.05
        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(300, hidden_size, n_layers, batch_first=True, dropout=0)

        self.output_size = output_size
        self.max_length = max_len
        self.use_attention = use_attention

        if use_attention:
            self.attention = Attention()

        # self.out = nn.Linear(2048, self.output_size)


    def forward_step(self, input_var, hidden, encoder_outputs_key, encoder_outputs, function, att_lengths=None, advclss=None):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = torch.zeros([batch_size, 1, 300]) 
        if torch.cuda.is_available():
            embedded = embedded.cuda()

        queries, hidden = self.rnn(embedded, hidden)
        # query normalization
        queries = F.normalize(queries,dim=2)

        attn = None
        if self.use_attention:
            # todo: use part of encoder_outputs for att
            self.attention.set_mask(att_lengths, encoder_outputs_key.shape[1])
            sent_emb, attn = self.attention(queries, encoder_outputs_key, encoder_outputs)

        predicted_score = advclss(sent_emb=sent_emb.contiguous().view(-1, 2048))[0].view(batch_size, output_size, -1) #function()
        return predicted_score, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, encoder_outputs_key=None, advclss=None, att_lengths=None, labels=None,
                    function=F.sigmoid):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # todo: cat to both hidden and cell state
        # style_embd = style_embd.unsqueeze(0)
        # encoder_hidden = (torch.cat((encoder_hidden[0],style_embd),2), torch.cat((encoder_hidden[1],style_embd),2))

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function)
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        lengths = np.array([max_length] * batch_size)

        def getEOS(left_value, labels):
            tmp1 = labels & left_value.data.le(0.5+self.MARGIN).type(device.LongTensor).squeeze()
            tmp2 = (1-labels) & left_value.data.ge(0.5-self.MARGIN).type(device.LongTensor).squeeze()
            return (tmp1+tmp2).unsqueeze(1)
            
        def decode(step, step_output, step_attn, left_value):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            eos_batches = getEOS(left_value, labels)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(decoder_outputs)
            return

        def updateEncOutputsKeys(encoder_outputs, step_attn):  # (batchsize,1,hiddensize) (batchsize,1,hiddensize)
            encoder_outputs = encoder_outputs*(1-step_attn.transpose(1,2)) # todo: is it correct?
            return encoder_outputs

        # Manual unrolling is used to support random teacher forcing.
        decoder_input = inputs[:, 0].unsqueeze(1)
        for di in range(max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs_key, encoder_outputs,
                                                                     function=function, att_lengths=att_lengths ,advclss=advclss)
            step_output = decoder_output.squeeze(1)
            encoder_outputs_key = updateEncOutputsKeys(encoder_outputs_key, step_attn)
            left_value, _ = advclss(keys=encoder_outputs_key, hiddens=encoder_outputs)
            decode(di, step_output, step_attn, left_value)
            

        ret_dict[DecoderRNN.KEY_SCORE] = decoder_outputs
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        # ret_dict[DecoderRNN.KEY_LEFT_OVER] = self.getLeftOver(ret_dict[DecoderRNN.KEY_ATTN_SCORE], ret_dict[DecoderRNN.KEY_LENGTH], encoder_outputs)

        return ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            inputs = device.LongTensor([0] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length