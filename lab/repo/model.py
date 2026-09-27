"""
RFKAN y sus variantes de ablacion (Tabla 3 del paper):

    KAN   : splines,  sin nodos recurrentes
    RKAN  : splines,  con nodos recurrentes
    FKAN  : Fourier,  sin nodos recurrentes
    RFKAN : Fourier,  con nodos recurrentes   <- propuesta

Estructura (Fig. 1): dos capas KAN.
  Capa 1: nodos normales x_{0,i}(t) (n = 12 canales) + nodos kernel recurrentes
          h_{0,i}(t) = W_hh h_{0,i}(t-1) + W_hz x_{0,i}(t)            (Ec. 3)
          -> 2n+1 nodos normales x_{1,j}(t) = sum_i phi(x_{0,i}(t), h_{0,i}(t))   (Ec. 4)
  Capa 2: nodos x_{1,j}(t) + nodos recurrentes h_{1,j}(t) (misma recurrencia)
          -> salida.
  Adaptacion de salida: el paper tiene un unico nodo de salida x_{2,1};
  aca la capa 2 tiene 48 nodos de salida (caudal de las 48 h siguientes),
  leidos en el ultimo paso de la ventana de entrada.

Variantes sin recurrencia (KAN / FKAN): se quitan los nodos kernel recurrentes
("removing the RNN structure at the nodes"); la ventana de entrada se aplana
(n = a x b nodos normales, como describe el paper) y se mantiene el mismo ancho
oculto 2*12+1 = 25 que RFKAN.
"""
import torch
import torch.nn as nn

import config
from kan_layers import make_kan_layer


class RecurrentKernelNodes(nn.Module):
    """h(t) = W_hh h(t-1) + W_hz x(t)   (Ec. 3, lineal, sin activacion)."""

    def __init__(self, dim):
        super().__init__()
        self.W_hh = nn.Linear(dim, dim, bias=False)
        self.W_hz = nn.Linear(dim, dim, bias=False)
        # espectro de W_hh < 1 al inicio para que la recurrencia lineal sea estable
        nn.init.orthogonal_(self.W_hh.weight, gain=0.9)

    def forward(self, x):                      # x: (B, T, dim) -> h: (B, T, dim)
        z = self.W_hz(x)
        h = torch.zeros_like(z[:, 0])
        out = []
        for t in range(x.shape[1]):
            h = self.W_hh(h) + z[:, t]
            out.append(h)
        return torch.stack(out, 1)


class RFKAN(nn.Module):
    def __init__(self, basis="fourier", recurrent=True, seq_len=config.SEQ_LEN,
                 n_in=config.INPUT_NODES, n_out=config.FORECAST_HOURS,
                 grid=config.GRID_SCHEDULE[0], k=config.SPLINE_ORDER):
        super().__init__()
        self.basis, self.recurrent, self.seq_len = basis, recurrent, seq_len
        hidden = 2 * n_in + 1                                   # 2n+1 (teorema K-A)
        if recurrent:
            self.rec0 = RecurrentKernelNodes(n_in)
            self.layer1 = make_kan_layer(basis, 2 * n_in, hidden, grid, k)
            self.rec1 = RecurrentKernelNodes(hidden)
            self.layer2 = make_kan_layer(basis, 2 * hidden, n_out, grid, k)
        else:
            self.layer1 = make_kan_layer(basis, n_in * seq_len, hidden, grid, k)
            self.layer2 = make_kan_layer(basis, hidden, n_out, grid, k)

    @property
    def grid(self):
        return self.layer1.grid

    def _layer1_input(self, x):
        x = x[:, -self.seq_len:]
        if self.recurrent:
            return torch.cat([x, self.rec0(x)], -1)            # (B, T, 2n)
        return x.reshape(x.shape[0], -1)                        # (B, T*n)

    def _layer2_input(self, x1):
        if self.recurrent:
            h1 = self.rec1(x1)                                  # (B, T, 2n+1)
            return torch.cat([x1[:, -1], h1[:, -1]], -1)        # ultimo paso
        return x1

    def forward(self, x):                                       # x: (B, 336, 12) normalizado
        x1 = self.layer1(self._layer1_input(x))
        return self.layer2(self._layer2_input(x1))

    @torch.no_grad()
    def extend_grid(self, new_grid, x_sample):
        """Refinamiento de grilla (Fig. 1 / Fig. 4). x_sample: lote de entradas."""
        in1 = self._layer1_input(x_sample)
        # por bloques de ventanas: con L=336 y grilla 100 la capa 1 sobre 4096 ventanas
        # de una vez pide ~14 GB (las bases spline son (B*L, 24, g+k))
        x1 = torch.cat([self.layer1(in1[i:i + 256]) for i in range(0, len(in1), 256)])
        in2 = self._layer2_input(x1)
        self.layer1.extend_grid(new_grid, in1)
        self.layer2.extend_grid(new_grid, in2)


class Forecaster(nn.Module):
    """Envuelve un nucleo y produce el caudal normalizado de las 48 h.

    Con RESIDUAL_OUTPUT: y_hat = q(t) + sigma_delta * core(x), donde q(t) es el
    ultimo caudal observado (canal 11, normalizado) y sigma_delta el desvio del
    cambio y(t+h) - q(t) en train. El nucleo arranca "cerca" de persistencia y
    aprende la correccion en una escala ~N(0,1). Se aplica igual a RFKAN, sus
    ablaciones y los baselines neuronales, asi la comparacion es justa.
    """

    def __init__(self, core, delta_std, residual=config.RESIDUAL_OUTPUT):
        super().__init__()
        self.core, self.delta_std, self.residual = core, delta_std, residual

    @property
    def grid(self):
        return getattr(self.core, "grid", None)

    def forward(self, x):
        out = self.core(x)
        if not self.residual:
            return out
        return x[:, -1:, config.TARGET_CHANNEL] + self.delta_std * out

    def extend_grid(self, new_grid, x_sample):
        self.core.extend_grid(new_grid, x_sample)


def build_model(name, seq_len=config.SEQ_LEN, grid=config.GRID_SCHEDULE[0]):
    variants = {
        "KAN": ("spline", False),
        "RKAN": ("spline", True),
        "FKAN": ("fourier", False),
        "RFKAN": ("fourier", True),
    }
    basis, rec = variants[name]
    return RFKAN(basis=basis, recurrent=rec, seq_len=seq_len, grid=grid)


def build_forecaster(name, delta_std, seq_len=config.SEQ_LEN, **kw):
    """Nucleo RFKAN/ablacion o baseline neuronal, envuelto en Forecaster."""
    if name in ("KAN", "RKAN", "FKAN", "RFKAN"):
        core = build_model(name, seq_len=seq_len, **kw)
    else:
        from baselines import build_nn_baseline
        core = build_nn_baseline(name, seq_len=seq_len, delta_std=delta_std, **kw)
    return Forecaster(core, delta_std)
