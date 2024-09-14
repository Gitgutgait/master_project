import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            activation,
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            activation,
            nn.Conv3d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv3d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv3d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv3d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = torch.relu(x)
        x = self._conv_2(x)
        x = torch.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(in_channels, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        
        self._conv_trans_1 = nn.ConvTranspose3d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose3d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_1(x)
        x = torch.relu(x)
        return self._conv_trans_2(x)
