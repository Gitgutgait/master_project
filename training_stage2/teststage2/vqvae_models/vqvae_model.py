import torch
import torch.nn as nn
from vqvae_models.network import Encoder, Decoder
from vqvae_models.quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv3d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self._vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
