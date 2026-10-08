"""
Carga de Rainfall-Runoff (train.h5 / test.h5) y preprocesamiento.

Preprocesamiento del paper (seccion "Data description"):
  1. relleno de faltantes por interpolacion lineal -> el dataset no tiene NaN
     (verificado; metadata.json: imputation_applied=false y 0 NaN en X/y), no aplica.
  2. outliers -> no hay limites fisicos analogos a "irradiancia 0-1000 W/m2"
     documentados para este dataset; se omite (ver GAP_ANALISIS.md).
  3. normalizacion min-max a [0, 1] por canal -> SE APLICA, con min/max
     calculados SOLO sobre la particion de entrenamiento (split == 0).
     El objetivo (caudal, canal 11) usa el mismo min/max del canal 11, y las
     predicciones se desnormalizan a mm/h antes de calcular metricas.

Todo el dataset se sube a GPU una vez (4.1 GB fp32) y se itera con indices
aleatorios; es mucho mas rapido que un DataLoader sobre h5.
"""
import json

import h5py
import numpy as np
import torch

import config


def load_split(split):
    """split: 'train' | 'validation' | 'test'. Devuelve dict de numpy arrays."""
    if split == "test":
        with h5py.File(config.TEST_H5, "r") as f:
            out = {"X": f["X"][:], "basin_id": f["basin_id"][:]}
        if config.TEST_TARGETS.exists():
            import pandas as pd
            y = pd.read_csv(config.TEST_TARGETS).sort_values("Id")
            out["y"] = y.drop(columns="Id").values.astype(np.float32)
        return out
    code = 0 if split == "train" else 1
    with h5py.File(config.TRAIN_H5, "r") as f:
        mask = f["split"][:] == code
        idx = np.flatnonzero(mask)
        # h5py lee mas rapido por bloques contiguos que con fancy indexing
        X = np.concatenate([f["X"][a:b] [mask[a:b]] for a, b in _chunks(len(mask))])
        y = f["y"][:][mask]
        basin = f["basin_id"][:][mask]
    return {"X": X, "y": y, "basin_id": basin, "row": idx}


def _chunks(n, size=20000):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


class MinMaxScaler:
    """Min-max por canal a [0, 1] (paper, paso 3)."""

    def __init__(self, X=None):
        if X is not None:
            self.lo = X.min(axis=(0, 1)).astype(np.float32)
            self.hi = X.max(axis=(0, 1)).astype(np.float32)

    def fit_delta(self, X, y):
        """sigma del cambio normalizado y(t+h) - q(t): escala de la salida residual."""
        c = config.TARGET_CHANNEL
        d = self.transform_y(y) - self.transform_y(X[:, -1:, c])
        self.delta_std = float(d.std())
        return self

    @property
    def span(self):
        return np.maximum(self.hi - self.lo, 1e-8)

    def transform_x(self, X):
        return (X - self.lo) / self.span

    def transform_y(self, y):
        c = config.TARGET_CHANNEL
        return (y - self.lo[c]) / self.span[c]

    def inverse_y(self, y):
        c = config.TARGET_CHANNEL
        return y * self.span[c] + self.lo[c]

    def save(self, path):
        json.dump({"lo": self.lo.tolist(), "hi": self.hi.tolist(), "delta_std": self.delta_std},
                  open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        s = cls()
        s.lo, s.hi = np.array(d["lo"], np.float32), np.array(d["hi"], np.float32)
        s.delta_std = d["delta_std"]
        return s


class GPUData:
    """Tensores normalizados en GPU + iterador de mini-lotes."""

    def __init__(self, d, scaler, device="cuda"):
        self.X = torch.from_numpy(scaler.transform_x(d["X"]).astype(np.float32)).to(device)
        self.y = None
        if "y" in d:
            self.y = torch.from_numpy(scaler.transform_y(d["y"]).astype(np.float32)).to(device)
            self.y_raw = d["y"]
        self.basin_id = d["basin_id"]
        self.n = len(self.X)

    def batches(self, batch_size, shuffle=True, generator=None):
        idx = torch.randperm(self.n, generator=generator, device="cpu") if shuffle else torch.arange(self.n)
        for s in range(0, self.n, batch_size):
            b = idx[s:s + batch_size].to(self.X.device)
            yield self.X[b], (self.y[b] if self.y is not None else None)


_CACHE = {}


def get_data(device="cuda"):
    """Devuelve (train, val, test, scaler) cacheados en memoria del proceso."""
    if "all" not in _CACHE:
        tr, va, te = load_split("train"), load_split("validation"), load_split("test")
        scaler = MinMaxScaler(tr["X"]).fit_delta(tr["X"], tr["y"])
        _CACHE["all"] = (GPUData(tr, scaler, device), GPUData(va, scaler, device),
                         GPUData(te, scaler, device), scaler, (tr, va, te))
    return _CACHE["all"]
