import argparse
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def load_architecture(device: torch.device, args: argparse.Namespace):
    """
    Initializes an RNNClassifier model with the specified parameters and sends
    it to the device.
    """
    model = RNNClassifier(arch=args.arch,
                          dynamic_input_size=args.dynamic_input_size,
                          hidden_size=args.hidden_size, dropout=args.dropout,
                          rnn_layers=args.rnn_layers,
                          bidirectional=args.bidirectional,
                          use_attention=args.use_attention,
                          attention_type=args.attention_type,
                          device=device, fc_layers=args.fc_layers)
    model.to(device)
    return model


class RNNClassifier(nn.Module):
    """
    Sequential model base class for prediction of brain state from optical
    imagine brain signals.
    """
    def __init__(self, arch: str, dynamic_input_size: int,
                 hidden_size: int, dropout: float, rnn_layers: int,
                 bidirectional: bool, use_attention: bool, attention_type: str,
                 device: str, fc_layers: int):
        """
        Initializes a new RNNClassifier module
        :param arch: A string 'lstm' or 'gru' that dictates recurrent unit used
        :param dynamic_input_size: Dimension of the input vital sign data
        :param hidden_size: Dimension of the hidden representation
        :param dropout: The probability of dropout for Dropout layers
        :param rnn_layers: Number of stacked RNN layers
        :param bidirectional: Boolean value indicating whether bidirectional
        :param use_attention: Boolean value indicating whether to use attention
        :param attention_type: A string that dictates (dot vs. general)
        :param device: String representation of the device
        :param fc_layers: Number of fully-connected layers used for
            classification, where each additional fc layer has half the number
            of nodes as its previous layer
        """
        super().__init__()
        assert arch in ['lstm', 'gru']

        self.arch = arch
        self.use_attention = use_attention
        self.dynamic_input_size = dynamic_input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.device = device
        self.fc_layers = fc_layers

        # Sequential model (RNN)
        rnn_dropout = 0.0 if self.rnn_layers == 1 else self.dropout
        self.rnn = None
        if arch == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.dynamic_input_size,
                hidden_size=self.hidden_size // self.directions,
                num_layers=self.rnn_layers, bidirectional=self.bidirectional,
                dropout=rnn_dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(
                input_size=self.dynamic_input_size,
                hidden_size=self.hidden_size // self.directions,
                num_layers=self.rnn_layers, bidirectional=self.bidirectional,
                dropout=rnn_dropout, batch_first=True)

        # Attention Mechanisms
        self.attn_dy_dy = Attention(self.hidden_size,
                                    attention_type=attention_type)

        # Fully-connected classifier layers
        in_layer_dim = self.hidden_size
        out_layer_dim = in_layer_dim // 2

        self.fc = [nn.Linear(in_layer_dim, out_layer_dim)]
        extra_layers = []
        last_out_layer_size = out_layer_dim
        for _ in range(1, self.fc_layers):
            extra_layers.append(
                nn.Linear(last_out_layer_size, last_out_layer_size // 2))
            last_out_layer_size = last_out_layer_size // 2

        self.fc = nn.ModuleList(self.fc + extra_layers)
        self.classifier = nn.Linear(last_out_layer_size, 1)
        self.classifier_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tuple, seq_idx: int):
        """
        Performs the forward computation for the sequential model
        :param x: Tuple of tensors loaded from the dataset in the format:
            (dynamic, lengths, previous hidden state, previous dynamic state)
        where previous_hidden_state is Tuple[torch.Tensor, torch.Tensor] for
        LSTM and torch.Tensor for GRU.
        """
        dynamic_data, lengths, initial_hidden, dynamic_previous = x

        # Packs a Tensor containing padded sequences of variable length
        dynamic_data = pack_padded_sequence(dynamic_data, lengths,
                                            batch_first=True,
                                            enforce_sorted=False)

        _, hidden_orig = self.rnn(dynamic_data, initial_hidden)
        if isinstance(hidden_orig, tuple):
            hidden = hidden_orig[0]
        else:
            hidden = hidden_orig.clone()
        if self.bidirectional:
            # Move values for section direction to hidden size dimension
            dynamic_output = hidden.view(
                self.rnn_layers, hidden.shape[1], hidden.shape[2] * 2)
        else:
            dynamic_output = hidden

        # Reverse batch ordering to match chronological order
        dynamic_output = dynamic_output[-1, :, :]

        if self.use_attention:
            # dynamic_output:
            # [batch_size, embedding_dim] => [batch_size, embedding_dim, 1]
            dynamic_output = torch.unsqueeze(dynamic_output, 2)
            # add new RNN dynamic_output to attention sequence dynamic_previous
            # (add to history)
            # dynamic_output:
            # [batch_size, embedding_dim] => [batch_size, embedding_dim]
            dynamic_prev = dynamic_previous.clone()

            if isinstance(dynamic_output.shape[0], torch.Tensor):
                dyn_out = self.add_dyn_history(
                    dynamic_prev, dynamic_output, self.device)
            else:
                dyn_out = torch.zeros(
                    dynamic_prev.shape[0], dynamic_prev.shape[2], 1
                    ).to(self.device)
                dyn_out[:dynamic_output.shape[0], :, :] = \
                    dynamic_output.clone()

            dynamic_prev[:, seq_idx, :] = torch.squeeze(dyn_out)

            attention_dynamic_dynamic, attention_w_dynamic_dynamic = \
                self.attn_dy_dy(torch.unsqueeze(
                    torch.squeeze(dyn_out, dim=2), 1), dynamic_prev)
            attention_dynamic_dynamic = torch.squeeze(
                attention_dynamic_dynamic, dim=1)
            if isinstance(dynamic_output.shape[0], torch.Tensor):
                attention_dynamic_dynamic = torch.squeeze(
                    attention_dynamic_dynamic[
                        :dynamic_output.shape[0].item(), :])
            else:
                attention_dynamic_dynamic = torch.squeeze(
                    attention_dynamic_dynamic[:dynamic_output.shape[0], :])
            if len(attention_dynamic_dynamic.shape) != 2:
                attention_dynamic_dynamic = torch.unsqueeze(
                    attention_dynamic_dynamic, 0)

            # Features consist of the dynamic/dynamic attention vector
            features = attention_dynamic_dynamic

        else:
            features = dynamic_output
            dynamic_output = torch.unsqueeze(dynamic_output, 2)
            dynamic_prev = dynamic_previous.clone()
            dyn_out = torch.zeros(
                dynamic_prev.shape[0], dynamic_prev.shape[2], 1
                ).to(self.device)
            dyn_out[:dynamic_output.shape[0], :, :] = dynamic_output.clone()
            dynamic_prev[:, seq_idx, :] = torch.squeeze(dyn_out)
            attention_w_dynamic_dynamic = torch.tensor(0)

        for fc_layer in self.fc:
            features = F.relu(fc_layer(features))
            features = self.classifier_dropout(features)

        output = self.classifier(features).squeeze()

        return (output, hidden_orig, torch.squeeze(dyn_out),
                attention_w_dynamic_dynamic)

    def init_hidden(self, batch_size: int):
        if self.arch == 'lstm':
            return (
                torch.zeros(self.rnn_layers * self.directions, batch_size,
                            self.hidden_size // self.directions
                            ).to(self.device),
                torch.zeros(self.rnn_layers * self.directions, batch_size,
                            self.hidden_size // self.directions
                            ).to(self.device))
        return torch.zeros(self.rnn_layers * self.directions, batch_size,
                           self.hidden_size // self.directions).to(self.device)


# Module taken from
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
class Attention(nn.Module):
    """
    Applies attention mechanism on the 'key' using the 'query'.

    :param dimensions: Dimensionality of the query and key
    :param attention_type: How to compute the attention score:
        * dot:      score(H_j, q) = H_j^T q
        * general:  score(H_j, q) = H_j^T W_a q
    """
    def __init__(self, dimensions, attention_type='dot'):
        super().__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        # Used to reduce dimensionality of concatenated vector back down to
        # dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, key):
        """
        :param query: Sequence of queries to query the key
        :type query: torch.FloatTensor [batch size, output length, dimensions]

        :param key: Data overwhich to apply the attention mechanism
        :type key: torch.FloatTensor [batch size, query length, dimensions])

        :rtype: Tuple[torch.LongTensor, torch.FloatTensor]
        :return: Tuple with 'output' and 'weights':
            * **output** (:class: torch.LongTensor
              [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class: torch.FloatTensor
              [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = key.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # Batch matrix multiplication between queries and keys to get
        # attention scores
        # (batch_size, output_len, dimensions) *
        # (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, key.transpose(1, 2).contiguous())

        # Compute weights across every key (context sequence)
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        # Batch matrix multiplication between attention weights and keys,
        # where keys are the values themselves
        # (batch_size, output_len, query_len) *
        # (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        weighted_values = torch.bmm(attention_weights, key)

        # Context vector = weighted_values, current RNN output = query
        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((weighted_values, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(
            batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
