"""
Capas KAN implementadas desde cero (sin pykan / efficient-kan / FourierKAN).

- FourierKANLayer : phi(x) = sum_k a_k cos(k x) + b_k sin(k x)          (Ec. 8 del paper)
- SplineKANLayer  : phi(x) = w_b * silu(x) + w_s * sum_i c_i B_i(x)      (KAN original, Liu et al. 2024)

Ambas exponen `extend_grid(new_grid, x_sample)`: el refinamiento de grilla del
paper ("RFKAN hereda la curva de la funcion de activacion aprendida y se
entrena en una grilla mas fina", Fig. 1 / Fig. 4).

  * Fourier: extender de g a g' armonicos es EXACTO -- se copian los a_k, b_k
    existentes y los armonicos nuevos arrancan en 0. La funcion no cambia.
  * Spline: la grilla nueva no contiene a la vieja, asi que se reajustan los
    coeficientes por minimos cuadrados sobre una muestra de activaciones
    (el mismo procedimiento que el KAN original: grid extension por lstsq).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierKANLayer(nn.Module):
    """Capa KAN con funciones de activacion en base de Fourier (Ec. 8).

    y_o = sum_i sum_{k=1..g} a_{o,i,k} cos(k x_i) + b_{o,i,k} sin(k x_i) + bias_o
    """

    def __init__(self, in_dim, out_dim, grid):
        super().__init__()
        self.in_dim, self.out_dim, self.grid = in_dim, out_dim, grid
        k = torch.arange(1, grid + 1, dtype=torch.float32)
        # init "suave": la amplitud de cada armonico decae con k, para que la
        # funcion inicial no sea ruido de alta frecuencia.
        std = 1.0 / (math.sqrt(in_dim) * k)
        self.coef = nn.Parameter(torch.randn(2, out_dim, in_dim, grid) * std / math.sqrt(grid))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer("k", k)

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)
        kx = x.unsqueeze(-1) * self.k                     # (B, in, g)
        y = torch.einsum("big,oig->bo", torch.cos(kx), self.coef[0])
        y = y + torch.einsum("big,oig->bo", torch.sin(kx), self.coef[1])
        return (y + self.bias).reshape(*shape, self.out_dim)

    @torch.no_grad()
    def extend_grid(self, new_grid, x_sample=None):
        if new_grid <= self.grid:
            return
        coef = torch.zeros(2, self.out_dim, self.in_dim, new_grid, device=self.coef.device)
        coef[..., : self.grid] = self.coef.data
        self.coef = nn.Parameter(coef)
        self.k = torch.arange(1, new_grid + 1, dtype=torch.float32, device=self.coef.device)
        self.grid = new_grid


class SplineKANLayer(nn.Module):
    """Capa KAN original: B-splines de orden K sobre una grilla de g intervalos,
    mas una funcion base residual silu(x) (Liu et al. 2024, Ec. 2.10)."""

    def __init__(self, in_dim, out_dim, grid, k=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_dim, self.out_dim, self.grid, self.k = in_dim, out_dim, grid, k
        self.register_buffer("knots", self._make_knots(
            torch.tensor([grid_range] * in_dim, dtype=torch.float32), grid, k))
        self.base_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_w, a=math.sqrt(5))
        self.spline_w = nn.Parameter(torch.randn(out_dim, in_dim, grid + k) * 0.1 / math.sqrt(in_dim))

    @staticmethod
    def _make_knots(ranges, grid, k):
        # ranges: (in, 2) -> knots (in, grid + 2k + 1), uniformes, extendidos k pasos a cada lado
        lo, hi = ranges[:, :1], ranges[:, 1:]
        h = (hi - lo) / grid
        steps = torch.arange(-k, grid + k + 1, dtype=torch.float32, device=ranges.device)
        return lo + steps * h

    def b_splines(self, x, knots=None):
        """Cox-de Boor. x: (B, in) -> (B, in, grid + k)."""
        g = self.knots if knots is None else knots
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for p in range(1, self.k + 1):
            left = (x - g[:, : -(p + 1)]) / (g[:, p:-1] - g[:, : -(p + 1)]) * bases[..., :-1]
            right = (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p]) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)
        y = F.linear(F.silu(x), self.base_w)
        y = y + torch.einsum("bic,oic->bo", self.b_splines(x), self.spline_w)
        return y.reshape(*shape, self.out_dim)

    @torch.no_grad()
    def extend_grid(self, new_grid, x_sample, margin=0.01, max_rows=20000, chunk=256):
        """Refina la grilla a `new_grid` intervalos ajustando la grilla al rango
        de los datos y reajustando los coeficientes por minimos cuadrados, de
        modo que la parte spline de cada phi_{o,i} se preserve (KAN grid extension).
        Cada entrada i se ajusta por separado, asi que se procesa en bloques de
        `chunk` entradas (KAN aplanado con L=336 tiene 4032 entradas)."""
        x = x_sample.reshape(-1, self.in_dim)
        if len(x) > max_rows:
            x = x[torch.randperm(len(x), device=x.device)[:max_rows]]
        lo = x.min(0).values - margin
        hi = x.max(0).values + margin
        new_knots = self._make_knots(torch.stack([lo, hi], 1), new_grid, self.k)
        sols = []
        for a in range(0, self.in_dim, chunk):
            b = min(a + chunk, self.in_dim)
            xc = x[:, a:b]
            old = torch.einsum("bic,oic->bio", self.b_splines(xc, self.knots[a:b]), self.spline_w[:, a:b])
            A = self.b_splines(xc, new_knots[a:b]).permute(1, 0, 2)              # (in, B, g'+k)
            Bv = old.permute(1, 0, 2)                                             # (in, B, out)
            sols.append(torch.linalg.lstsq(A.cpu(), Bv.cpu(), driver="gelsd").solution)
        self.knots = new_knots
        self.grid = new_grid
        sol = torch.cat(sols, 0)                                                  # (in, g'+k, out)
        self.spline_w = nn.Parameter(sol.permute(2, 0, 1).contiguous().to(self.base_w.device))

def make_kan_layer(basis, in_dim, out_dim, grid, k=3):
    if basis == "fourier":
        return FourierKANLayer(in_dim, out_dim, grid)
    if basis == "spline":
        return SplineKANLayer(in_dim, out_dim, grid, k)
    raise ValueError(basis)
