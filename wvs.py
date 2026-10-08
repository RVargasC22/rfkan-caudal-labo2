"""
WVS = WOA-VMD-SCINet (Zhao et al., Energy Reports 12:3470-3487, 2024), segundo
"modelo avanzado" de la Tabla 7 del paper de RFKAN. Implementacion propia:

  1. Pearson: se eligen las WVS_N_FEATURES variables meteorologicas con mayor |r|
     respecto del caudal (sobre train). El paper usa entrada de dimension 4
     (1 subsecuencia + 3 variables) y salida 1 (Sec. 3.2).
  2. VMD (Dragomiretskiy & Zosso 2014): el caudal de la ventana de entrada se
     descompone en K modos. Se hace POR VENTANA y solo con la historia (sin mirar
     el futuro), en lotes en GPU, sobre (q - q(t)) / sigma_delta para que los
     modos queden en escala O(1). Se precalcula una vez (with_modes) y los modos
     entran como canales extra.
  3. WOA (Mirjalili & Lewis 2016): optimiza (K, alpha) de VMD con la entropia de
     envolvente como fitness (Sec. 2.3 del paper).
  4. SCINet (Liu et al., NeurIPS 2022): "cada subsecuencia se combina con las
     variables elegidas y entra a SCINet" y los resultados se suman: un SCINet por
     modo, con entrada [modo_k, variables] (dimension 4) y salida 1; la prediccion
     es la suma de los K modos. Hiperparametros de la Tabla 5 / Sec. 3.2:
     levels 3, stacks 1, hidden 1, kernel 4, dropout 0.5.
     Adaptacion: como la VMD se hace por ventana (sin mirar el futuro), no hay
     objetivo por modo; los K SCINet se entrenan juntos con la perdida del total.
Se envuelve en Forecaster (salida residual) como los demas modelos.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn

import config

PARAMS = config.OUTPUT_DIR / "experiments" / "WVS_params.json"


# ----------------------------------------------------------------------- VMD
@torch.no_grad()
def vmd(f, K, alpha, n_iter=config.WVS_VMD_ITERS, tau=0.0):
    """f: (B, T) real. Devuelve modos (B, K, T). VMD con extension espejo."""
    B, T = f.shape
    h = T // 2
    fm = torch.cat([f[:, :h].flip(1), f, f[:, h:].flip(1)], 1)           # (B, 2T)
    Tm = fm.shape[1]
    t = torch.arange(1, Tm + 1, device=f.device, dtype=torch.float32) / Tm
    freqs = t - 0.5 - 1.0 / Tm
    f_hat = torch.fft.fftshift(torch.fft.fft(fm), dim=-1)
    f_hat_plus = f_hat.clone()
    f_hat_plus[:, : Tm // 2] = 0                                            # solo frecuencias >= 0
    u = torch.zeros(B, K, Tm, dtype=torch.complex64, device=f.device)
    omega = (0.5 / K) * torch.arange(K, device=f.device, dtype=torch.float32).expand(B, K).clone()
    lam = torch.zeros(B, Tm, dtype=torch.complex64, device=f.device)
    pos = slice(Tm // 2, Tm)
    for _ in range(n_iter):
        for k in range(K):
            rest = u.sum(1) - u[:, k]
            u[:, k] = (f_hat_plus - rest - lam / 2) / (1 + alpha * (freqs - omega[:, k:k + 1]) ** 2)
            p = u[:, k, pos].abs() ** 2
            omega[:, k] = (freqs[pos] * p).sum(1) / p.sum(1).clamp_min(1e-12)
        lam = lam + tau * (u.sum(1) - f_hat_plus)
    # espectro analitico -> señal real: 2 Re(ifft), con el bin de DC contado una vez
    u[:, :, Tm // 2] = u[:, :, Tm // 2] / 2
    modes = 2 * torch.fft.ifft(torch.fft.ifftshift(u, dim=-1)).real
    return modes[:, :, h: h + T]                                            # quita el espejo


def envelope_entropy(modes):
    """Entropia de la envolvente (|Hilbert|) de cada modo. modes: (B, K, T) -> (B, K)."""
    T = modes.shape[-1]
    X = torch.fft.fft(modes)
    hmask = torch.zeros(T, device=modes.device)
    hmask[0] = 1
    hmask[1:(T + 1) // 2] = 2
    if T % 2 == 0:
        hmask[T // 2] = 1
    env = torch.fft.ifft(X * hmask).abs()
    p = env / env.sum(-1, keepdim=True).clamp_min(1e-12)
    return -(p * (p + 1e-12).log()).sum(-1)


# ----------------------------------------------------------------------- WOA
def woa(fitness, lo, hi, n_whales=config.WVS_WOA_POP, n_iter=config.WVS_WOA_ITERS, seed=config.SEED):
    """Whale Optimization Algorithm (minimiza). lo, hi: limites por dimension."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = lo + rng.random((n_whales, len(lo))) * (hi - lo)
    fit = np.array([fitness(x) for x in X])
    best, best_f = X[fit.argmin()].copy(), fit.min()
    history = [best_f]
    for it in range(n_iter):
        a = 2 - 2 * it / max(n_iter - 1, 1)                 # 2 -> 0
        for i in range(n_whales):
            A, C = 2 * a * rng.random() - a, 2 * rng.random()
            if rng.random() < 0.5:
                ref = best if abs(A) < 1 else X[rng.integers(n_whales)]   # encierro / busqueda
                X[i] = ref - A * np.abs(C * ref - X[i])
            else:                                            # espiral logaritmica (b = 1)
                l_ = rng.uniform(-1, 1)
                X[i] = np.abs(best - X[i]) * np.exp(l_) * np.cos(2 * np.pi * l_) + best
            X[i] = np.clip(X[i], lo, hi)
            f = fitness(X[i])
            if f < best_f:
                best, best_f = X[i].copy(), f
        history.append(best_f)
    return best, best_f, history


def select_features(X, n_features=config.WVS_N_FEATURES, n=20000, seed=config.SEED):
    """Pearson entre cada variable meteorologica y el caudal; devuelve las de mayor |r|."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Z = X[idx].reshape(-1, X.shape[-1])
    r = np.array([np.corrcoef(Z[:, c], Z[:, config.TARGET_CHANNEL])[0, 1] for c in range(X.shape[-1])])
    met = [c for c in np.argsort(-np.abs(np.nan_to_num(r))) if c != config.TARGET_CHANNEL]
    return sorted(int(c) for c in met[:n_features]), r


def fit_params(X_train_norm, delta_std, n_windows=512, seed=config.SEED):
    """Pearson + WOA sobre una muestra de ventanas de train; guarda WVS_params.json."""
    sel, r = select_features(X_train_norm)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train_norm), n_windows, replace=False)
    L = config.WVS_SEQ_LEN
    q = torch.tensor(X_train_norm[idx, -L:, config.TARGET_CHANNEL], device="cuda")
    f = (q - q[:, -1:]) / delta_std

    def fitness(x):
        K, alpha = int(round(x[0])), float(x[1])
        return envelope_entropy(vmd(f, K, alpha)).min(1).values.mean().item()

    best, best_f, hist = woa(fitness, lo=[2, 100], hi=[8, 4000])
    params = {"K": int(round(best[0])), "alpha": float(best[1]), "features": sel,
              "pearson": [float(v) for v in r], "woa_fitness": best_f, "woa_history": hist}
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(params, indent=1))
    return params


@torch.no_grad()
def with_modes(data, delta_std, chunk=4096):
    """Copia de un GPUData con X = [ultimas WVS_SEQ_LEN h de los 12 canales, K modos VMD].
    La VMD no tiene parametros entrenables: se calcula una vez por ventana (solo con su
    propia historia) en lugar de en cada paso de entrenamiento."""
    p = json.loads(PARAMS.read_text())
    L = config.WVS_SEQ_LEN
    parts = []
    for i in range(0, data.n, chunk):
        w = data.X[i:i + chunk, -L:]
        q = w[:, :, config.TARGET_CHANNEL]
        m = vmd(((q - q[:, -1:]) / delta_std).float(), p["K"], p["alpha"]).transpose(1, 2)
        parts.append(torch.cat([w, m], -1))
    out = object.__new__(type(data))
    out.__dict__.update(data.__dict__)
    out.X, out.n = torch.cat(parts), data.n
    return out


# -------------------------------------------------------------------- SCINet
# Los K SCINet (uno por modo) se implementan como UNA red con convoluciones
# agrupadas (groups = K): cada grupo de canales es un SCINet independiente, con
# sus propios pesos, pero todos corren en la misma pasada de GPU.
class Interactor(nn.Module):
    """phi / psi / rho / eta del bloque SCI: pad + conv(k) + LeakyReLU + dropout + conv(3) + tanh."""

    def __init__(self, c, hidden, k, dropout, groups):
        super().__init__()
        # padding como el repo de SCINet: conserva el largo tambien con kernel par
        self.net = nn.Sequential(nn.ReplicationPad1d(((k - 1) // 2 + 1, k // 2 + 1)),
                                 nn.Conv1d(c * groups, c * hidden * groups, k, groups=groups),
                                 nn.LeakyReLU(0.01), nn.Dropout(dropout),
                                 nn.Conv1d(c * hidden * groups, c * groups, 3, groups=groups), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class SCIBlock(nn.Module):
    def __init__(self, c, hidden, k, dropout, groups):
        super().__init__()
        self.phi, self.psi, self.rho, self.eta = (Interactor(c, hidden, k, dropout, groups) for _ in range(4))

    def forward(self, x):                                   # x: (B, G*C, T)
        even, odd = x[:, :, 0::2], x[:, :, 1::2]
        d = odd * torch.exp(self.phi(even))
        c = even * torch.exp(self.psi(odd))
        return c + self.eta(d), d - self.rho(c)             # (even', odd')


class SCITree(nn.Module):
    def __init__(self, level, c, hidden, k, dropout, groups):
        super().__init__()
        self.block = SCIBlock(c, hidden, k, dropout, groups)
        sub = lambda: SCITree(level - 1, c, hidden, k, dropout, groups) if level > 1 else None
        self.even, self.odd = sub(), sub()

    def forward(self, x):
        e, o = self.block(x)
        if self.even is not None:
            e, o = self.even(e), self.odd(o)
        return torch.stack([e, o], -1).reshape(*x.shape[:2], -1)   # intercalar de nuevo


class WVS(nn.Module):
    """Entrada: (B, WVS_SEQ_LEN, 12 + K), de with_modes(). Un SCINet por modo (grupos)."""

    def __init__(self, delta_std, seq_len=config.WVS_SEQ_LEN):
        super().__init__()
        p = json.loads(PARAMS.read_text())
        self.K, self.features = p["K"], p["features"]
        self.c = 1 + len(self.features)                      # entrada de dimension 4 por modo
        self.tree = SCITree(config.WVS_LEVELS, self.c, config.WVS_HIDDEN, config.WVS_KERNEL,
                            config.WVS_DROPOUT, self.K)
        H = config.FORECAST_HOURS
        # proyeccion temporal L -> 48 y salida de dimension 1, propias de cada modo
        self.proj = nn.Parameter(torch.randn(self.K, H, seq_len) / math.sqrt(seq_len))
        self.head = nn.Parameter(torch.randn(self.K, self.c) / math.sqrt(self.c))
        self.bias = nn.Parameter(torch.zeros(self.K))

    def forward(self, x):                                   # x: (B, L, 12 + K)
        B, L, _ = x.shape
        modes = x[:, :, config.N_CHANNELS:config.N_CHANNELS + self.K]            # (B, L, K)
        met = x[:, :, self.features]                                             # (B, L, 3)
        z = torch.cat([modes.unsqueeze(-1), met.unsqueeze(2).expand(B, L, self.K, len(self.features))], -1)
        z = z.reshape(B, L, self.K * self.c)                                     # grupos contiguos por modo
        z = self.tree(z.transpose(1, 2)).transpose(1, 2) + z                     # SCINet + residual
        z = z.reshape(B, L, self.K, self.c).permute(2, 1, 0, 3).reshape(self.K, L, B * self.c)
        z = (self.proj @ z).reshape(self.K, -1, B, self.c).permute(2, 1, 0, 3)  # (B, 48, K, c), L -> 48 por modo
        per_mode = (z * self.head).sum(-1) + self.bias                          # prediccion de cada modo
        return per_mode.sum(-1)                                                  # suma de modos (B, 48)
