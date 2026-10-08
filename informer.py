"""
Informer (Zhou et al., AAAI 2021), implementado desde cero. Es el primer "modelo
avanzado" de la Tabla 7 del paper de RFKAN ("parametros consistentes con la
literatura"). Hiperparametros del paper de Informer (config.INFORMER_*):
d_model 512, 8 cabezas, 2 capas de encoder, 1 de decoder, d_ff 2048, factor 5,
dropout 0.05, destilacion, lr 1e-4, batch 32.

Componentes:
  - embedding: conv1d (k=3, circular) de los 12 canales + posicional sinusoidal
    (sin embedding temporal: el dataset no trae fechas);
  - ProbSparse self-attention: solo las u = factor*ln(L) consultas mas "activas"
    (medida M = max - media de los scores sobre una muestra de claves) atienden;
    el resto toma la media de V (encoder) o la suma acumulada (decoder, con mascara);
  - destilacion entre capas del encoder: conv + BatchNorm + ELU + maxpool (L -> L/2);
  - decoder generativo: entrada = ultimas label_len horas + ceros para las 48 a
    predecir; salida de las 48 posiciones de una sola pasada (modo MS: 12 canales
    de entrada, 1 de salida, el caudal).
Se envuelve en Forecaster (salida residual) igual que los demas modelos.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ProbAttention(nn.Module):
    def __init__(self, mask_flag, factor, dropout):
        super().__init__()
        self.mask_flag, self.factor = mask_flag, factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):                                  # (B, H, L, D)
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        u_k = min(Lk, int(self.factor * math.ceil(math.log(Lk))))
        u_q = min(Lq, int(self.factor * math.ceil(math.log(Lq))))
        # medida de "esparsidad" de cada consulta sobre u_k claves muestreadas
        idx = torch.randint(Lk, (Lq, u_k), device=q.device)
        qk_s = torch.einsum("bhqd,bhqkd->bhqk", q, k[:, :, idx, :])
        M = qk_s.max(-1).values - qk_s.sum(-1) / Lk
        top = M.topk(u_q, dim=-1).indices                        # (B, H, u_q)
        q_red = torch.gather(q, 2, top.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = q_red @ k.transpose(-2, -1) / math.sqrt(D)      # (B, H, u_q, Lk)
        if self.mask_flag:                                       # decoder: causal
            context = v.cumsum(dim=2)
            pos = torch.arange(Lk, device=q.device)
            scores = scores.masked_fill(pos > top.unsqueeze(-1), float("-inf"))
        else:
            context = v.mean(dim=2, keepdim=True).expand(B, H, Lq, D).clone()
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, D), attn @ v)
        return context


class FullAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        return self.dropout(a) @ v


class AttentionLayer(nn.Module):
    def __init__(self, attn, d_model, n_heads):
        super().__init__()
        self.attn, self.h = attn, n_heads
        self.q, self.k, self.v, self.o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, xq, xk, xv):
        B, Lq, d = xq.shape
        split = lambda t: t.view(B, t.shape[1], self.h, d // self.h).transpose(1, 2)
        out = self.attn(split(self.q(xq)), split(self.k(xk)), split(self.v(xv)))
        return self.o(out.transpose(1, 2).reshape(B, Lq, d))


def _ffn(d_model, d_ff, dropout):
    return nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, dropout):
        super().__init__()
        self.attn = AttentionLayer(ProbAttention(False, factor, dropout), d_model, n_heads)
        self.ff = _ffn(d_model, d_ff, dropout)
        self.n1, self.n2, self.drop = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x)))
        return self.n2(x + self.drop(self.ff(x)))


class DistilLayer(nn.Module):
    """Destilacion: conv + BatchNorm + ELU + maxpool, reduce el largo a la mitad."""

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(F.elu(self.norm(self.conv(x.transpose(1, 2))))).transpose(1, 2)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, factor, dropout):
        super().__init__()
        self.self_attn = AttentionLayer(ProbAttention(True, factor, dropout), d_model, n_heads)
        self.cross = AttentionLayer(FullAttention(dropout), d_model, n_heads)
        self.ff = _ffn(d_model, d_ff, dropout)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem):
        x = self.n1(x + self.drop(self.self_attn(x, x, x)))
        x = self.n2(x + self.drop(self.cross(x, mem, mem)))
        return self.n3(x + self.drop(self.ff(x)))


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout, max_len=1024):
        super().__init__()
        self.token = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular")
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.token(x.transpose(1, 2)).transpose(1, 2) + self.pe[: x.shape[1]])


class Informer(nn.Module):
    def __init__(self, seq_len=config.INFORMER_SEQ_LEN, label_len=config.INFORMER_LABEL_LEN,
                 pred_len=config.FORECAST_HOURS, c_in=config.N_CHANNELS, d_model=config.INFORMER_D_MODEL,
                 n_heads=config.INFORMER_HEADS, e_layers=config.INFORMER_E_LAYERS,
                 d_layers=config.INFORMER_D_LAYERS, d_ff=config.INFORMER_D_FF,
                 factor=config.INFORMER_FACTOR, dropout=config.INFORMER_DROPOUT):
        super().__init__()
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.enc_emb = DataEmbedding(c_in, d_model, dropout)
        self.dec_emb = DataEmbedding(c_in, d_model, dropout)
        self.enc = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, factor, dropout) for _ in range(e_layers))
        self.distil = nn.ModuleList(DistilLayer(d_model) for _ in range(e_layers - 1))
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, factor, dropout) for _ in range(d_layers))
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):                                        # x: (B, 336, 12)
        enc = self.enc_emb(x[:, -self.seq_len:])
        for i, layer in enumerate(self.enc):
            enc = layer(enc)
            if i < len(self.distil):
                enc = self.distil[i](enc)
        enc = self.enc_norm(enc)
        dec_in = torch.cat([x[:, -self.label_len:], x.new_zeros(x.shape[0], self.pred_len, x.shape[2])], 1)
        dec = self.dec_emb(dec_in)
        for layer in self.dec:
            dec = layer(dec, enc)
        return self.proj(self.dec_norm(dec))[:, -self.pred_len:, 0]   # (B, 48)
