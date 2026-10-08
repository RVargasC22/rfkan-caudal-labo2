"""
Modelos base (Tabla 5 del paper) + persistencia.

Neuronales (se entrenan con train.fit, mismo Forecaster residual que RFKAN):
  LSTM : 3 capas, Adam, lr 1e-3, batch 32                      [Tabla 5]
  GRU  : igual que LSTM con celdas GRU (Tabla 6 reporta GRU)    [Tabla 6]
  TCN  : 3 capas conv 1D, stride 2, padding "same", ReLU        [Tabla 5]
  (la "Activation function ReLU" de LSTM en la Tabla 5 se aplica en la cabeza MLP)

Clasicos (sin gradiente):
  Persistencia : y(t+h) = q(t)  -- el modelo base natural del problema
  ARIMA(2,1,1) : se ajusta sobre la serie de caudal de CADA muestra (336 h)
                 y se pronostica 48 h. Muy caro -> se evalua en un submuestreo.
  SVR RBF C=0.0043 : entrada = ultimas 24 h de los 12 canales (aplanado);
                 un SVR por hora de horizonte, entrenado en un submuestreo.
  ELM          : 3 capas ocultas aleatorias ReLU (no entrenadas) + salida por
                 regresion ridge cerrada (lambda = 0.001).
Los clasicos trabajan sobre el delta respecto del ultimo caudal (igual que el
Forecaster residual), en unidades normalizadas.
"""
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------- neuronales
class RNNForecaster(nn.Module):
    def __init__(self, cell="LSTM", seq_len=config.HISTORY_HOURS, hidden=config.BASELINE_HIDDEN,
                 layers=3, n_out=config.FORECAST_HOURS):
        super().__init__()
        self.seq_len = seq_len
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(config.N_CHANNELS, hidden, num_layers=layers, batch_first=True)
        if cell == "LSTM":
            # sesgo de la compuerta de olvido = 1 (Jozefowicz et al. 2015): con 336 pasos
            # y sesgo 0 el gradiente se desvanece y el LSTM se queda en persistencia
            for name, b in self.rnn.named_parameters():
                if name.startswith("bias_ih"):
                    b.data[hidden:2 * hidden] = 1.0
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        o, _ = self.rnn(x[:, -self.seq_len:])
        return self.head(o[:, -1])


class TCNForecaster(nn.Module):
    def __init__(self, seq_len=config.HISTORY_HOURS, hidden=config.BASELINE_HIDDEN,
                 layers=config.TCN_LAYERS, n_out=config.FORECAST_HOURS, kernel=3):
        super().__init__()
        self.seq_len = seq_len
        convs, c = [], config.N_CHANNELS
        for _ in range(layers):
            # stride 2 + padding "same" (k//2): cada capa reduce el largo a la mitad
            convs += [nn.Conv1d(c, hidden, kernel, stride=2, padding=kernel // 2), nn.ReLU()]
            c = hidden
        self.net = nn.Sequential(*convs)
        t = seq_len
        for _ in range(layers):
            t = (t + 1) // 2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden * t, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.head(self.net(x[:, -self.seq_len:].transpose(1, 2)))


def build_nn_baseline(name, seq_len=config.HISTORY_HOURS, **kw):
    if name in ("LSTM", "GRU"):
        return RNNForecaster(name, seq_len)
    if name == "TCN":
        return TCNForecaster(seq_len)
    if name == "Informer":
        from informer import Informer
        return Informer(seq_len=seq_len)
    if name == "WVS":
        from wvs import WVS
        return WVS(kw["delta_std"])
    raise ValueError(name)


# ---------------------------------------------------------------- clasicos
def persistence(X_norm):
    """X_norm: (N, T, 12) numpy normalizado -> (N, 48) normalizado."""
    q = X_norm[:, -1, config.TARGET_CHANNEL]
    return np.repeat(q[:, None], config.FORECAST_HOURS, 1)


def _arima_one(series):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = ARIMA(series, order=config.ARIMA_ORDER).fit()
            f = res.forecast(config.FORECAST_HOURS)
            if np.all(np.isfinite(f)):
                return f
        except Exception:
            pass
        # ajuste fallido o explosivo (p. ej. serie constante): persistencia
        return np.repeat(series[-1], config.FORECAST_HOURS)


def arima_forecast(X_norm, n_jobs=14):
    """ARIMA(2,1,1) independiente por muestra sobre la serie de caudal."""
    from concurrent.futures import ProcessPoolExecutor
    series = [X_norm[i, :, config.TARGET_CHANNEL].astype(np.float64) for i in range(len(X_norm))]
    with ProcessPoolExecutor(n_jobs) as ex:
        out = list(ex.map(_arima_one, series, chunksize=16))
    return np.stack(out).astype(np.float32)


def _flat_features(X_norm, lookback=24):
    return X_norm[:, -lookback:, :].reshape(len(X_norm), -1)


def _delta(X_norm, y_norm):
    return y_norm - X_norm[:, -1:, config.TARGET_CHANNEL]


class SVRBaseline:
    """SVR RBF, C=0.0043 (Tabla 5). Un SVR por hora de horizonte."""

    def __init__(self, delta_std, lookback=24):
        self.delta_std, self.lookback = delta_std, lookback

    def fit(self, X, y):
        from sklearn.svm import SVR
        F = _flat_features(X, self.lookback)
        D = _delta(X, y) / self.delta_std
        self.models = [SVR(kernel="rbf", C=config.SVR_C, gamma="scale").fit(F, D[:, h])
                       for h in range(D.shape[1])]
        return self

    def predict(self, X):
        F = _flat_features(X, self.lookback)
        D = np.stack([m.predict(F) for m in self.models], 1)
        return X[:, -1:, config.TARGET_CHANNEL] + D * self.delta_std


class ELMBaseline:
    """Extreme Learning Machine: 3 capas aleatorias ReLU + ridge (lambda 1e-3)."""

    def __init__(self, delta_std, lookback=24, hidden=512, seed=config.SEED):
        self.delta_std, self.lookback, self.hidden, self.seed = delta_std, lookback, hidden, seed

    def _hidden(self, F):
        H = F
        for W, b in self.layers:
            H = torch.relu(H @ W + b)
        return H

    @torch.no_grad()
    def fit(self, X, y):
        g = torch.Generator(device="cuda").manual_seed(self.seed)
        F = torch.from_numpy(_flat_features(X, self.lookback)).float().cuda()
        self.layers, d = [], F.shape[1]
        for _ in range(config.ELM_HIDDEN_LAYERS):
            W = torch.randn(d, self.hidden, generator=g, device="cuda") / np.sqrt(d)
            b = torch.randn(self.hidden, generator=g, device="cuda") * 0.1
            self.layers.append((W, b))
            d = self.hidden
        H = self._hidden(F)
        D = torch.from_numpy(_delta(X, y) / self.delta_std).float().cuda()
        A = H.T @ H + config.ELM_REG * torch.eye(H.shape[1], device="cuda")
        self.beta = torch.linalg.solve(A, H.T @ D)
        return self

    @torch.no_grad()
    def predict(self, X):
        F = torch.from_numpy(_flat_features(X, self.lookback)).float().cuda()
        D = (self._hidden(F) @ self.beta).cpu().numpy()
        return X[:, -1:, config.TARGET_CHANNEL] + D * self.delta_std


def timed(fn, *a, **k):
    t = time.time()
    out = fn(*a, **k)
    return out, time.time() - t
