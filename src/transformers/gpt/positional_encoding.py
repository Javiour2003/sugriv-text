import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    Code from https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding slightly modified.
    """
    def __init__(self, d_model, max_length, dropout=None):
        """
        Positional Encoding with a maximum length of max_length.

        :param d_model: Dimensionality of the input embeddings
        :param max_length: Maximum length of source/target sentence
        :param dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_length, d_model)

        # get the position
        position = torch.arange(0, max_length).unsqueeze(1)

        # get the division
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encodings to the input embeddings.

        :param x: Input
        """
        x = x + self.pe[:, :x.size(1)].detach()
        if self.dropout is None:
            return x
        else:
            return self.dropout(x)