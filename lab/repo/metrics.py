"""
Metricas en mm/h (unidades originales).

Del paper (Ec. 9):
  MAE  = mean |y - y'|
  RMSE = sqrt(mean (y - y')^2)
  CORR = 1 - sum (y - y')^2 / sum (y - mean(y))^2      (el paper lo llama "R squared index")

Agregadas (estandar en hidrologia, no estan en el paper):
  NSE por cuenca (Nash-Sutcliffe) = CORR calculado dentro de cada cuenca;
  se reporta la mediana sobre cuencas (robusta a cuencas casi secas).
  KGE global (Kling-Gupta).
"""
import numpy as np


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def corr_r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def kge(y, p):
    y, p = y.ravel(), p.ravel()
    r = np.corrcoef(y, p)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (p.std() / y.std() - 1) ** 2 + (p.mean() / y.mean() - 1) ** 2))


def nse_per_basin(y, p, basin):
    out = {}
    for b in np.unique(basin):
        m = basin == b
        yb, pb = y[m], p[m]
        den = np.sum((yb - yb.mean()) ** 2)
        if den > 0:
            out[int(b)] = 1.0 - np.sum((yb - pb) ** 2) / den
    return out


def all_metrics(y, p, basin=None):
    m = {"RMSE": rmse(y, p), "MAE": mae(y, p), "CORR": corr_r2(y, p), "KGE": kge(y, p)}
    if basin is not None:
        nse = np.array(list(nse_per_basin(y, p, basin).values()))
        m["NSE_median"] = float(np.median(nse))
        m["NSE_mean"] = float(np.mean(nse))
    return m


def per_horizon(y, p):
    """RMSE / CORR por hora de horizonte (1..48)."""
    return {"RMSE": np.sqrt(((y - p) ** 2).mean(0)),
            "CORR": 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)}
