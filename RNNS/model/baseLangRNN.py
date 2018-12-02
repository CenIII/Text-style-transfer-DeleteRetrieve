import torch.nn as nn

from .baseRNN import BaseRNN
import numpy as np

class baseLangRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=False):
        super(baseLangRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, 300)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(300, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.decoder = nn.Linear(hidden_size,self.vocab_size)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # import pdb;pdb.set_trace()
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
        # pdb.set_trace()
        output = self.decoder(output) # torch.Size([seq_len, batch, vocab_size])
        return output, hidden
