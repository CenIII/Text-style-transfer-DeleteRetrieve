import torch.nn as nn

from .baseRNN import BaseRNN
import numpy as np

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=False):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, 300)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding) # embedding.weight is torch.nn.Parameter.
        self.embedding.weight.requires_grad = update_embedding # whether to freeze the weight of embedding
        # self.rnn_sell is initialized in baseRNN, either nn.LSTM or nn.GRU
        # use batch_first when the input and output tensors are supposed to be as (batch, seq, feature)
        self.rnn = self.rnn_cell(300, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (Tensor of int, optional): A 1-D Tensor that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        # Sort sentences from long to short
        inds = np.argsort(-input_lengths) 
        input_var = input_var[inds]
        input_lengths = input_lengths[inds]
        # The index to reverse former transformation
        rev_inds = np.argsort(inds)

        embedded = self.embedding(input_var) # embedded is shape (batch, seq_len, embedding_dim)
        embedded = self.input_dropout(embedded) # a dropout layer

        # Deal with samples of different lengths.
        # https://pytorch.org/docs/stable/nn.html?highlight=nn%20utils%20rnn#torch.nn.utils.rnn.pack_padded_sequence
        if self.variable_lengths:
            # PackedSequence. pack data into batches for each character position.
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # output: packed sequence.
        # hidden: (h_n, c_n) for LSTM; h_n for GRU
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            # Revert to padded form, the length returned is ignored here.
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Revert the order of samples
        output = output[rev_inds] # torch.Size([seq_len, batch, num_directions * hidden_size])
        hidden = hidden[:,rev_inds] # torch.Size([direction * layers, batch, hidden_size])
        
        return output, hidden
