import argparse
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def load_architecture(device: torch.device, args: argparse.Namespace):
    """
    Initializes an SpatiotemporalCNN model with the specified parameters and
    sends it to the device.
    """
    if args.arch == 'spatiotemporal_cnn':
        model = SpatiotemporalCNN(C=args.num_channels,
                                  F1=args.num_temporal_filters,
                                  D=args.num_spatial_filters,
                                  F2=args.num_pointwise_filters,
                                  p=args.dropout, fs=args.fs, T=args.seq_len)
    elif args.arch == 'dual_input_cnn':
        A_params = {
            'C': args.num_channels[0],
            'F1': args.num_temporal_filters[0],
            'D': args.num_spatial_filters[0],
            'F2': args.num_pointwise_filters[0],
            'p': args.dropout[0],
        }
        B_params = {
            'C': args.num_channels[1],
            'F1': args.num_temporal_filters[1],
            'D': args.num_spatial_filters[1],
            'F2': args.num_pointwise_filters[1],
            'p': args.dropout[1],
        }
        model = DualSpatiotemporalCNN(A_params=A_params,
                                      B_params=B_params,
                                      fs=args.fs, T=args.seq_len)
    elif args.arch == 'fcn':
        model = FCN(F1=args.num_temporal_filters,
                    S=args.num_depthwise_channels,
                    D=args.num_spatial_filters,
                    F2=args.num_pointwise_filters,
                    p=args.dropout, fs=args.fs, T=args.seq_len)
    model.to(device)
    return model


class SpatiotemporalCNN(nn.Module):
    """
    CNN based on EEGNet from https://arxiv.org/pdf/1611.08024.pdf which learns
    spatiotemporal filters from multi-variate time series data.
    """
    def __init__(self, C: int, F1: int, D: int, F2: int, p: float = 0.5,
                 fs: int = 40, T: int = 56):
        """
        Initializes a new SpatiotemporalCNN module.
        :param C: number of input channels
        :param F1: number of temporal filters
        :param D: number of spatial filters for each temporal filter
        :param F2: number of pointwise filters
        :param p: dropout probability (0.5 for within-subject classification
                  and 0.25 for cross-subject classification)
        :param fs: sampling frequency
        :param T: number of time points
        """
        super().__init__()

        # Temporal convolution capturing freq. above 2 Hz
        self.conv1 = torch.nn.Conv2d(1, F1, (1, fs // 2), padding=(0, fs // 4),
                                     bias=False)
        # Spatial BN over (N, C, T) slices
        self.bn1 = torch.nn.BatchNorm2d(F1)
        # Depthwise convolution to learn spatial filters for temporal filters
        self.conv2 = ConstrainedConv2d(1, F1, D * F1, (C, 1), padding=0,
                                       bias=False, groups=F1)
        # Temporal BN over (N, D) slices
        self.bn2 = torch.nn.BatchNorm2d(D * F1)
        self.dropout = torch.nn.Dropout(p)
        # Depthwise separable convolution representing 500 ms slices
        self.conv3 = SeparableConv2d(D * F1, F2, (1, fs // 2),
                                     padding=(0, fs // 4), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(F2)
        self.avgpool = torch.nn.AvgPool2d((1, 8))
        self.fc = ConstrainedDense(0.25, F2 * (T // 8), 1)

    def forward(self, x):

        x = x.type(torch.cuda.FloatTensor)

        # Spatiotemporal filtering
        x = x.unsqueeze(1)                                  # (1, C, T)
        x = self.bn1(self.conv1(x))                         # (F1, C, T)
        x = self.bn2(self.conv2(x))                         # (D * F1, 1, T)
        x = self.dropout(F.elu(x))                          # (D * F1, 1, T)

        # Combine filters with separable convolution
        x = self.bn3(self.conv3(x))                         # (F2, 1, T)
        x = self.dropout(self.avgpool(F.elu(x)))            # (F2, 1, T // 8)

        # Classification
        x = torch.flatten(x, 1, -1)                         # (F2 * (T // 8))
        x = self.fc(x)

        return x


class DualSpatiotemporalCNN(nn.Module):
    """
    Spatiotemporal CNN that lears separate spastiotemporal filters for
    each input data stream.
    """
    def __init__(self, A_params: dict, B_params: dict,
                 fs: int = 40, T: int = 56):
        """
        Initializes two SpatiotemporalCNN modules and combines their outputs.
        :param A_params: initialization parameters for the first input.
        :param B_params: initialization parameterse for the second input.
        :param fs: sampling frequency (same for both inputs).
        :param T: number of time points (same for both inputs).
        """
        super().__init__()

        self.CNN_A = SpatiotemporalCNN(**A_params, fs=fs, T=T)
        self.CNN_B = SpatiotemporalCNN(**B_params, fs=fs, T=T)

        # Remove the final dense layer from both models and define one for the
        # concatenated output.
        self.CNN_A.fc = nn.Identity()
        self.CNN_B.fc = nn.Identity()
        self.fc = ConstrainedDense(
            0.25, (A_params['F2'] + B_params['F2']) * (T // 8), 1)

    def forward(self, a, b):

        a, b = a.type(torch.cuda.FloatTensor), b.type(torch.cuda.FloatTensor)

        # Spatiotemporal filtering
        a, b = self.CNN_A(a), self.CNN_B(b)

        # Classification
        x = torch.cat((a, b), -1)
        x = self.fc(x)

        return x


class FCN(nn.Module):
    """
    Fully-convolutional version of the CNN based on EEGNet from
    https://arxiv.org/pdf/1611.08024.pdf which learns spatiotemporal filters
    from multi-variate time series data.
    """
    def __init__(self, F1: int, S: int, D: int, F2: int,
                 p: float = 0.5, fs: int = 40, T: int = 56):
        """
        Initializes a new SpatiotemporalCNN module.
        :param F1: number of temporal filters
        :param S: number of channels for depthwise channel convolution
        :param D: number of spatial filters for each temporal filter
        :param F2: number of pointwise filters
        :param p: dropout probability (0.5 for within-subject classification
                  and 0.25 for cross-subject classification)
        :param fs: sampling frequency
        :param T: number of time points
        """
        super().__init__()

        # Temporal convolution capturing freq. above 2 Hz
        self.conv1 = torch.nn.Conv2d(1, F1, (1, fs // 2), padding=(0, fs // 4),
                                     bias=False)
        # Spatial BN over (N, C, T) slices
        self.bn1 = torch.nn.BatchNorm2d(F1)
        # Depthwise convolution to learn spatial filters for temporal filters
        # Since the number of input channels is not fixed, a depthwise
        # convolution over S channels is defined to aggregate spatial
        # information and then max pooling takes the spatial region with the
        # highest activations
        self.conv2 = ConstrainedConv2d(1, F1, D * F1, (S, 1), padding=0,
                                       bias=False, groups=F1)
        # Temporal BN over (N, D) slices
        self.bn2 = torch.nn.BatchNorm2d(D * F1)
        self.dropout = torch.nn.Dropout(p)
        # Depthwise separable convolution representing 500 ms slices
        self.conv3 = SeparableConv2d(D * F1, F2, (1, fs // 2),
                                     padding=(0, fs // 4), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(F2)
        self.avgpool = torch.nn.AvgPool2d((1, 8))
        # Pointwise convolutions
        self.conv4 = torch.nn.Conv2d(F2, 1, (1, 1), bias=False)
        self.bn4 = torch.nn.BatchNorm2d(1)
        self.maxpool = torch.nn.MaxPool2d((1, (T // 8)))

    def forward(self, x):

        x = x.type(torch.cuda.FloatTensor)

        # Spatiotemporal filtering
        x = x.unsqueeze(1)                          # (1, C, T)
        x = self.bn1(self.conv1(x))                 # (F1, C, T)
        x = self.bn2(self.conv2(x))                 # (D * F1, C - S + 1, T)
        x = F.max_pool2d(
            x, kernel_size=(x.size(2), 1))          # (D * F1, 1, T)
        x = self.dropout(F.elu(x))                  # (D * F1, 1, T)

        # Combine filters with separable convolution
        x = self.bn3(self.conv3(x))                 # (F2, 1, T)
        x = self.dropout(self.avgpool(F.elu(x)))    # (F2, 1, T // 8)

        # Pointwise convolutions
        x = self.bn4(self.conv4(x))                 # (1, 1, T // 8)
        x = self.maxpool(x)                         # (1, 1, 1)
        x = x.squeeze(-1).squeeze(-1)

        return x


class ConstrainedConv2d(nn.Conv2d):
    """
    Allows the weights of a convolutional layer to be constrained by a
    maximum norm value.
    """
    def __init__(self, max_norm_value: float = 1.0, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._max_norm_val = max_norm_value
        self._eps = 1e-8

    def _max_norm(self, w):
        """ Scales the weights by the max norm value / norm """
        # L2 norm of spatial filters
        norm = w.norm(2, dim=0, keepdim=True)
        # Clamp all norms so that they fit the max norm constraint
        desired = torch.clamp(norm, 0, self._max_norm_val)
        # Scale each filter's weights by the max value / norm
        scaled = w * (desired / (self._eps + norm))
        return scaled

    def forward(self, x):

        return F.conv2d(x, self._max_norm(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], padding: str, bias: bool):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=bias)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ConstrainedDense(nn.Linear):
    """
    Allows the weights of a linear layer to be constrained by a maximum norm
    value.
    """
    def __init__(self, max_norm_value: float = 0.25, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._max_norm_val = max_norm_value
        self._eps = 1e-8

    def _max_norm(self, w):
        """ Scales the weights by the max norm value / norm """
        # L2 norm of spatial filters
        norm = w.norm(2, dim=0, keepdim=True)
        # Clamp all norms so that they fit the max norm constraint
        desired = torch.clamp(norm, 0, self._max_norm_val)
        # Scale each filter's weights by the max value / norm
        scaled = w * (desired / (self._eps + norm))
        return scaled

    def forward(self, x):

        return F.linear(x, self._max_norm(self.weight), self.bias)
