import torch
import torch.nn.functional as F

class LSTMFeatureExtractionModule(torch.nn.Module):
    """This module is the feature extraction part of the network. Could be anything, a CNN, LSTM, depending on the application"""
    def __init__(self,vocabulary_size,embedding_size,hidden_size,output_size,dropout_lstm=0,dropout_linear=0):
        super(LSTMFeatureExtractionModule, self).__init__()
        self.output_size = output_size
        self.layers = torch.nn.Sequential(
            torch.nn.Embedding(vocabulary_size, embedding_size),
            torch.nn.LSTM(embedding_size, hidden_size, 1, batch_first=True,dropout=dropout_lstm),
        )
        self.dropout1 = torch.nn.Dropout(dropout_linear)
        self.dropout2 = torch.nn.Dropout(dropout_linear)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self,input):
        _, rnn_hidden = self.layers(input)
        out = self.dropout1(F.relu(rnn_hidden[0][-1]))
        return self.dropout2(self.linear(out))