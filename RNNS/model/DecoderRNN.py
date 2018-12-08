# import random

# import numpy as np

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F

# from .attention import Attention
# from .baseRNN import BaseRNN

# if torch.cuda.is_available():
#     import torch.cuda as device
# else:
#     import torch as device


# class DecoderRNN(BaseRNN):
#     r"""
#     Provides functionality for decoding in a seq2seq framework, with an option for attention.

#     Args:
#         vocab_size (int): size of the vocabulary
#         max_len (int): a maximum allowed length for the sequence to be processed
#         hidden_size (int): the number of features in the hidden state `h`
#         sos_id (int): index of the start of sentence symbol
#         eos_id (int): index of the end of sentence symbol
#         n_layers (int, optional): number of recurrent layers (default: 1)
#         rnn_cell (str, optional): type of RNN cell (default: gru)
#         bidirectional (bool, optional): if the encoder is bidirectional (default False)
#         input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
#         dropout_p (float, optional): dropout probability for the output sequence (default: 0)
#         use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

#     Attributes:
#         KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
#         KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
#         KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

#     Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
#         - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
#           each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
#         - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
#           hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
#         - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
#           Used for attention mechanism (default is `None`).
#         - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
#           (default is `torch.nn.functional.log_softmax`).
#         - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
#           drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
#           teacher forcing would be used (default is 0).

#     Outputs: decoder_outputs, decoder_hidden, ret_dict
#         - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
#           the outputs of the decoding function.
#         - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
#           state of the decoder.
#         - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
#           representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
#           predicted token IDs }.
#     """

#     KEY_ATTN_SCORE = 'attention_score'
#     KEY_LENGTH = 'length'
#     KEY_SEQUENCE = 'sequence'

#     def __init__(self, vocab_size, max_len, hidden_size,
#             sos_id, eos_id,
#             n_layers=1, rnn_cell='gru', bidirectional=False,
#             input_dropout_p=0, dropout_p=0, use_attention=False,embedding=None, update_embedding=False):
#         super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
#                 input_dropout_p, dropout_p,
#                 n_layers, rnn_cell)

#         self.bidirectional_encoder = bidirectional
#         self.rnn = self.rnn_cell(1324, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

#         self.output_size = vocab_size
#         self.max_length = max_len
#         self.use_attention = use_attention
#         self.eos_id = eos_id
#         self.sos_id = sos_id

#         self.init_input = None

#         self.embedding = nn.Embedding(self.output_size, 300)
#         if embedding is not None:
#             self.embedding.weight = nn.Parameter(embedding)
#         self.embedding.weight.requires_grad = update_embedding
#         if use_attention:
#             self.attention = Attention(self.hidden_size)

#         self.out = nn.Linear(1324+self.hidden_size, self.output_size)

#     def forward_step(self, input_var, hidden, encoder_outputs, function):
#         batch_size = input_var.size(0)
#         output_size = input_var.size(1)
#         # embedded = self.embedding(input_var)
#         embedded = self.input_dropout(input_var)#embedded)
#         output, hidden = self.rnn(embedded, hidden)

#         attn = None
#         if self.use_attention:
#             output, attn = self.attention(output, encoder_outputs)

#         # predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
#         # predicted_softmax = self.out(output.contiguous().view(-1, self.hidden_size)).view(batch_size, output_size, -1)
#         lnr_cat = torch.cat((embedded, output),2)
#         predicted_softmax = self.out(lnr_cat.contiguous().view(-1, 1324+self.hidden_size)).view(batch_size, output_size, -1)
#         return predicted_softmax, hidden, attn

#     def forward(self, inputs=None, target_inps=None, encoder_hidden=None, encoder_outputs=None,
#                     function=F.log_softmax, teacher_forcing_ratio=0, outputs_maxlen=None):
#         if outputs_maxlen:
#             self.max_length = outputs_maxlen
#         ret_dict = dict()
#         if self.use_attention:
#             ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

#         inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
#                                                              function, teacher_forcing_ratio)
#         decoder_hidden = self._init_state(encoder_hidden)

#         use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

#         decoder_outputs = []
#         sequence_symbols = []
#         lengths = np.array([max_length] * batch_size)

#         def decode(step, step_output, step_attn):
#             decoder_outputs.append(step_output)
#             if self.use_attention:
#                 ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
#             symbols = decoder_outputs[-1].topk(1)[1]
#             sequence_symbols.append(symbols)

#             eos_batches = symbols.data.eq(self.eos_id)
#             if eos_batches.dim() > 0:
#                 eos_batches = eos_batches.cpu().view(-1).numpy()
#                 update_idx = ((lengths > step) & eos_batches) != 0
#                 lengths[update_idx] = len(sequence_symbols)
#             return symbols

#         # Manual unrolling is used to support random teacher forcing.
#         # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
#         if use_teacher_forcing:
#             target_inps = self.embedding(target_inps[:,:-1])
#             decoder_input = torch.cat((target_inps, inputs[:, :-1]),2)
#             decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
#                                                                      function=function)

#             for di in range(decoder_output.size(1)):
#                 step_output = decoder_output[:, di, :]
#                 if attn is not None:
#                     step_attn = attn[:, di, :]
#                 else:
#                     step_attn = None
#                 decode(di, step_output, step_attn)
#         else:
#             decoder_input = inputs[:, 0].unsqueeze(1)
#             # cat sos to input(i.e. concatenated hiddens)
#             sos_ts = torch.LongTensor([[self.sos_id]]).repeat(batch_size,1)
#             if torch.cuda.is_available():
#                 sos_ts = sos_ts.cuda()
#             decoder_input = torch.cat((self.embedding(sos_ts),decoder_input), 2)
#             for di in range(max_length):
#                 decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
#                                                                          function=function)
#                 step_output = decoder_output.squeeze(1)
#                 symbols = decode(di, step_output, step_attn)
#                 decoder_input = torch.cat((self.embedding(symbols),inputs[:,di].unsqueeze(1)), 2)#symbols

#         ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
#         ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

#         return decoder_outputs, decoder_hidden, ret_dict

#     def _init_state(self, encoder_hidden):
#         """ Initialize the encoder hidden state. """
#         if encoder_hidden is None:
#             return None
#         if isinstance(encoder_hidden, tuple):
#             encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
#         else:
#             encoder_hidden = self._cat_directions(encoder_hidden)
#         return encoder_hidden

#     def _cat_directions(self, h):
#         """ If the encoder is bidirectional, do the following transformation.
#             (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
#         """
#         if self.bidirectional_encoder:
#             h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
#         return h

#     def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
#         if self.use_attention:
#             if encoder_outputs is None:
#                 raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

#         # inference batch size
#         if inputs is None and encoder_hidden is None:
#             batch_size = 1
#         else:
#             if inputs is not None:
#                 batch_size = inputs.size(0)
#             else:
#                 if self.rnn_cell is nn.LSTM:
#                     batch_size = encoder_hidden[0].size(1)
#                 elif self.rnn_cell is nn.GRU:
#                     batch_size = encoder_hidden.size(1)

#         # set default input and max decoding length
#         if inputs is None:
#             if teacher_forcing_ratio > 0:
#                 raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
#             inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
#             if torch.cuda.is_available():
#                 inputs = inputs.cuda()
#             max_length = self.max_length
#         else:
#             max_length = inputs.size(1) - 1 # minus the start of sequence symbol

#         return inputs, batch_size, max_length
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

EXTRA_LEN = 5

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
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id, unk_id, m_end_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, embedding=None,update_embedding=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(300, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.m_end_id = m_end_id

        self.init_input = None

        
        self.embedding = nn.Embedding(self.output_size, 300)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = self.out(output.contiguous().view(-1, self.hidden_size)).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, style_embd=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, outputs_maxlen=None):
        """Decode hidden states generated by encoder into sequence of probability distribution

        Args:
            inputs: contains sentene, tensor shape (batch_size, seq_len_batch)
                broken_sentence, tensor shape (batch_size, seq_len_broken)
                marker_length, tensor shape (batch_size)
            style_embd: embeded vector for the style of each sentence in the batch
            encoder_hidden: hidden state of encoder. torch.Size([direction * layers, batch, hidden_size])
            encoder_outputs: output of encoder. torch.Size([seq_len, batch, num_directions * hidden_size])
            output_maxlen: The maximum length that is allowed for decoder.

        Returns:
            decoder_outputs: A list corresponding to the sentence position. len() is the sentence length of this batch.
                Element tensor in the list is shape([batch_size, vocab_size])
            decoder_hidden: The hidden state without going through the linear output layer.
                Tensor with torch.Size([1, batch_size, 2 * (encoder_hidden_size+style_embd_size)])
                See ._cat_directions()
            ret_dict: A dict with keys, sequence and length.
                sequence: A list of generated token correspoing to the sentence position. Element is shape (batch_size, 1).
                length: A list corresponding to each sample in the batch. Each element is the length of this sample.
        """
        # inputs_bak: sentence, brk sentence, marker lengths
        inputs_bak = inputs.copy()
        # inputs: sentence
        inputs = inputs[0]
        # cat style embedding with encoder_hidden
        style_embd = style_embd.transpose(0,1).repeat(encoder_hidden.shape[0],1,1)
        encoder_hidden = torch.cat((style_embd, encoder_hidden),2)  

        # decide teacher forcing
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if outputs_maxlen:
            self.max_length = outputs_maxlen
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn, unk_state=None):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if not use_teacher_forcing and not self.training:
                for b in range(len(decoder_outputs[-1])):
                    if not unk_state[b]:
                        decoder_outputs[-1][b][self.m_end_id] = 0

            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input =  inputs_bak[0][:, 0].unsqueeze(1)

            # three variables used to manipulate decoder inputs
            cur = [0 for i in range(batch_size)] # cursors of brk_sentence
            unk_state = [False for i in range(batch_size)] # states: unk or not unk
            mk_cur = [0 for i in range(batch_size)] # cursors of marker
            mk_no = [0 for i in range(batch_size)] # num of marker

            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn, unk_state=unk_state)
                decoder_input = []

                # manipulate decoder inputs 
                for b in range(batch_size):
                    # if not in "unk" state, use words from broken sentence.
                    if not unk_state[b]:
                        if cur[b] >= len(inputs_bak[1][b]):
                            decoder_input.append(torch.tensor(0))
                            continue
                        decoder_input.append(inputs_bak[1][b][cur[b]])
                        if inputs_bak[1][b][cur[b]].data.eq(self.unk_id):
                            unk_state[b] = True
                        cur[b] += 1
                    else:# if in "unk" state, use the symbol generated at last time step.
                        decoder_input.append(symbols[b])
                        if self.training:
                            checkMK = (mk_cur[b]==(inputs_bak[2][b][mk_no[b]]-1))
                        else:
                            checkMK = symbols[b].data.eq(self.m_end_id)
                        if checkMK:
                            unk_state[b] = False
                            mk_no[b] += 1
                            mk_cur[b] = 0
                        mk_cur[b] += 1
                # print(mk_no)
                decoder_input = torch.tensor(decoder_input).unsqueeze(1)
                # decoder_input = symbols
                if torch.cuda.is_available():
                    decoder_input = decoder_input.cuda()

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

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

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
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
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol
            if not self.training: # when evaluating, add more budget to the length. 
                max_length += EXTRA_LEN
        return inputs, batch_size, max_length