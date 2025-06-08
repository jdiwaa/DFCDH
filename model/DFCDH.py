import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from torch.nn.functional import gumbel_softmax
from einops import rearrange


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Mahalanobis_mask(nn.Module):
    def __init__(self, seq_len, enc_in=7):
        super(Mahalanobis_mask, self).__init__()
        self.dec = series_decomp(kernel_size=25)
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.linears = nn.Linear(seq_len, 1, bias=False)
        self.lineart = nn.Linear(seq_len, 1, bias=False)
        self.linear = nn.Linear(seq_len, 1, bias=False)

        self.weight_s = torch.nn.Parameter(torch.tensor([0.45]))
        self.weight_t = torch.nn.Parameter(torch.tensor([0.45]))
        self.weight = torch.nn.Parameter(torch.tensor([0.3]))

    def calculate_prob_distance(self, x):
        sea, trend = self.dec(x)
        diffs = self.linears(torch.abs(sea.unsqueeze(2) - sea.unsqueeze(1))).squeeze(-1)
        difft = self.lineart(torch.abs(trend.unsqueeze(2) - trend.unsqueeze(1))).squeeze(-1)
        weighted_dists = self.weight_t * difft + self.weight_s * diffs
        p = (1 - torch.sigmoid(torch.abs(weighted_dists))) * 0.99 + 1e-10
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)

        sample = self.bernoulli_gumbel_rsample(p)

        inverse_eye = 1 - torch.eye(self.enc_in).to(X.device)
        diag = torch.eye(self.enc_in).to(X.device)

        sample = torch.einsum("bcd,cd->bcd", sample, inverse_eye) + diag

        mask = sample.unsqueeze(1)

        return mask


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.mask_generator = Mahalanobis_mask(seq_len=configs.seq_len, enc_in=configs.enc_in)
        self.w = nn.Parameter(torch.ones(configs.enc_in, configs.seq_len // 2 + 1))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, T, N = x_enc.shape  # B L N

        x_enc = x_enc.permute(0, 2, 1)

        channel_mask = self.mask_generator(x_enc)

        w = self.w
        x_enc = torch.fft.rfft(x_enc, dim=2, norm='ortho')
        x_enc = w * x_enc
        x_enc = torch.fft.irfft(x_enc, n=T, dim=2, norm="ortho")

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=channel_mask)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] #+ res.permute(0, 2, 1) # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

